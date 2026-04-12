"""SHAP Interpreter — Interpretabilidade via SHapley Additive exPlanations.

Implementa a Secao 6.4 do TCC:
  - Interpretabilidade global (Tabela 4: ranking de features por tipo)
  - Interpretabilidade local (Force Plot / Waterfall por amostra)
  - Correspondencia com literatura de anti-spoofing

Fundamentacao: Lundberg & Lee (2017) — "A Unified Approach to Interpreting
Model Predictions" [ref. 15 do TCC].

Formula SHAP (Eq. 32 do TCC):
  phi_i = sum_{S subset F\{i}} (|S|!(|F|-|S|-1)! / |F|!) * [f(S u {i}) - f(S)]

Uso:
    from app.domain.services.shap_interpreter import SHAPInterpreter

    interp = SHAPInterpreter(model, feature_names=FEATURE_NAMES)
    global_table = interp.global_importance(X_test)   # -> Tabela 4
    local_exp    = interp.local_explanation(X_test[0]) # -> Force Plot data
    report       = interp.generate_report(X_test)      # -> JSON completo
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Union

import numpy as np

logger = logging.getLogger(__name__)

# Grupos de features alinhados com Tabela 4 do TCC
FEATURE_GROUPS = {
    "Espectrais": [
        "spectral_centroid", "spectral_bandwidth", "spectral_rolloff",
        "spectral_contrast", "spectral_flatness", "zcr",
        "spectral_flux", "spectral_entropy", "spectral_kurtosis",
        "spectral_skewness", "spectral_spread", "spectral_decrease",
        "spectral_slope", "spectral_crest",
    ],
    "MFCC": [f"mfcc_{i}" for i in range(1, 41)],
    "Prosodicas": [
        "f0_mean", "f0_std", "jitter", "shimmer", "hnr",
        "f0_range", "voiced_fraction",
    ],
    "Mel_Spectrogram": [f"mel_{i}" for i in range(80)],
    "CQT": [f"cqt_{i}" for i in range(84)],
}


class SHAPInterpreter:
    """Interpretabilidade SHAP para modelos de deteccao de deepfake.

    Suporta modelos sklearn (SVM, RF) e modelos Keras/TF via KernelExplainer.
    """

    def __init__(
        self,
        model,
        feature_names: Optional[List[str]] = None,
        feature_groups: Optional[Dict[str, List[str]]] = None,
        background_size: int = 100,
    ):
        """
        Args:
            model: modelo treinado (sklearn, TF/Keras, ou qualquer callable)
            feature_names: lista com nomes das features (ordem = colunas de X)
            feature_groups: dicionario mapeando grupo -> lista de nomes de features
            background_size: numero de amostras de background para KernelExplainer
        """
        self.model = model
        self.feature_names = feature_names or []
        self.feature_groups = feature_groups or FEATURE_GROUPS
        self.background_size = background_size
        self._explainer = None

    # ------------------------------------------------------------------
    # Setup do explainer
    # ------------------------------------------------------------------
    def _get_predict_fn(self):
        """Retorna funcao de predicao compativel com SHAP."""
        # Keras/TF
        if hasattr(self.model, 'predict'):
            def predict_fn(X):
                preds = self.model.predict(X, verbose=0)
                if preds.ndim > 1 and preds.shape[1] == 1:
                    return preds[:, 0]
                return preds.ravel()
            return predict_fn
        # sklearn
        if hasattr(self.model, 'predict_proba'):
            def predict_fn(X):
                return self.model.predict_proba(X)[:, 1]
            return predict_fn
        # generico
        return self.model

    def _build_explainer(self, X_background: np.ndarray):
        """Constroi o KernelExplainer com dados de background."""
        try:
            import shap
        except ImportError:
            raise ImportError(
                "Pacote 'shap' nao instalado. Execute: pip install shap"
            )

        n = min(self.background_size, len(X_background))
        idx = np.random.choice(len(X_background), n, replace=False)
        background = X_background[idx]

        predict_fn = self._get_predict_fn()

        # TreeExplainer para modelos de arvore (mais rapido)
        model_type = type(self.model).__name__.lower()
        if any(t in model_type for t in ['forest', 'tree', 'xgb', 'lgb', 'gradient']):
            try:
                self._explainer = shap.TreeExplainer(self.model)
                logger.info("SHAP: TreeExplainer selecionado")
                return
            except Exception:
                pass

        # LinearExplainer para SVM/LR linear
        if any(t in model_type for t in ['svc', 'svr', 'logistic', 'linear']):
            try:
                self._explainer = shap.LinearExplainer(self.model, background)
                logger.info("SHAP: LinearExplainer selecionado")
                return
            except Exception:
                pass

        # KernelExplainer (universal — mais lento)
        self._explainer = shap.KernelExplainer(predict_fn, background)
        logger.info(f"SHAP: KernelExplainer selecionado (background={n})")

    # ------------------------------------------------------------------
    # Interpretabilidade Global (Tabela 4 do TCC)
    # ------------------------------------------------------------------
    def global_importance(
        self,
        X_test: np.ndarray,
        max_samples: int = 200,
    ) -> Dict:
        """
        Calcula importancia global das features via SHAP (Tabela 4 do TCC).

        Agrega |phi_i| sobre todas as amostras de teste e agrupa
        por tipo de feature (Espectrais, MFCC, Prosodicas, etc.).

        Args:
            X_test: array (n_samples, n_features)
            max_samples: limitar amostras para KernelExplainer (performance)

        Returns:
            dict com:
              - 'feature_importance': {feature_name: mean_abs_shap}
              - 'group_importance': {group_name: total_shap_contribution_%}
              - 'ranking': lista ordenada de (feature_name, shap_value)
        """
        if self._explainer is None:
            self._build_explainer(X_test)

        n = min(max_samples, len(X_test))
        idx = np.random.choice(len(X_test), n, replace=False)
        X_sample = X_test[idx]

        logger.info(f"Calculando SHAP global para {n} amostras...")
        shap_values = self._explainer.shap_values(X_sample)

        # KernelExplainer pode retornar lista para classificacao binaria
        if isinstance(shap_values, list):
            shap_values = shap_values[1] if len(shap_values) > 1 else shap_values[0]

        # Importancia por feature: media dos valores absolutos
        mean_abs = np.mean(np.abs(shap_values), axis=0)

        # Mapear para nomes de features
        names = self.feature_names if len(self.feature_names) == len(mean_abs) else \
                [f"feature_{i}" for i in range(len(mean_abs))]

        feature_importance = {
            name: float(val) for name, val in zip(names, mean_abs)
        }

        # Agrupar por tipo (Tabela 4)
        group_importance = self._aggregate_by_group(feature_importance)

        # Ranking ordenado
        ranking = sorted(
            feature_importance.items(), key=lambda x: x[1], reverse=True
        )

        result = {
            "n_samples_analyzed": int(n),
            "feature_importance": feature_importance,
            "group_importance": group_importance,
            "ranking": ranking[:20],  # top 20
        }

        self._log_table4(group_importance)
        return result

    def _aggregate_by_group(self, feature_importance: Dict) -> Dict:
        """Agrega importancias por grupo e calcula percentuais (Tabela 4)."""
        group_sums = {g: 0.0 for g in self.feature_groups}
        group_sums["Outros"] = 0.0
        total = sum(feature_importance.values()) + 1e-10

        for feat, val in feature_importance.items():
            placed = False
            for group, members in self.feature_groups.items():
                if feat in members:
                    group_sums[group] += val
                    placed = True
                    break
            if not placed:
                group_sums["Outros"] += val

        return {
            g: {
                "shap_sum": round(v, 4),
                "contribution_pct": round(v / total * 100, 1),
            }
            for g, v in group_sums.items()
            if v > 0
        }

    def _log_table4(self, group_importance: Dict):
        """Loga a Tabela 4 do TCC no terminal."""
        logger.info("\n" + "=" * 55)
        logger.info("TABELA 4 — Importancia de Features via SHAP Global")
        logger.info("=" * 55)
        logger.info(f"  {'Tipo':<25} {'SHAP':>8} {'Contrib.':>10}")
        logger.info("  " + "-" * 45)

        sorted_groups = sorted(
            group_importance.items(),
            key=lambda x: x[1]["shap_sum"],
            reverse=True,
        )
        for group, info in sorted_groups:
            logger.info(
                f"  {group:<25} {info['shap_sum']:>8.3f} {info['contribution_pct']:>9.1f}%"
            )
        logger.info("=" * 55)

    # ------------------------------------------------------------------
    # Interpretabilidade Local (Force Plot / Waterfall por amostra)
    # ------------------------------------------------------------------
    def local_explanation(
        self,
        sample: np.ndarray,
        X_background: Optional[np.ndarray] = None,
    ) -> Dict:
        """
        Explicacao local para uma unica amostra.

        Gera dados equivalentes ao Force Plot / Waterfall Plot do TCC (Sec 6.4.3).

        Args:
            sample: array 1D (n_features,)
            X_background: necessario se explainer ainda nao foi construido

        Returns:
            dict com:
              - 'base_value': valor esperado do modelo (E[f(x)])
              - 'prediction': predicao para esta amostra
              - 'shap_values': {feature_name: phi_i}
              - 'top_positive': features que empurraram para FAKE
              - 'top_negative': features que empurraram para REAL
        """
        if self._explainer is None:
            if X_background is None:
                raise ValueError(
                    "Passe X_background na primeira chamada de local_explanation."
                )
            self._build_explainer(X_background)

        sample_2d = sample.reshape(1, -1)
        shap_vals = self._explainer.shap_values(sample_2d)

        if isinstance(shap_vals, list):
            shap_vals = shap_vals[1] if len(shap_vals) > 1 else shap_vals[0]

        shap_1d = shap_vals[0]

        names = self.feature_names if len(self.feature_names) == len(shap_1d) else \
                [f"feature_{i}" for i in range(len(shap_1d))]

        shap_map = {n: float(v) for n, v in zip(names, shap_1d)}

        # Base value
        base_value = float(
            self._explainer.expected_value[1]
            if isinstance(self._explainer.expected_value, (list, np.ndarray))
            else self._explainer.expected_value
        )

        # Predicao
        predict_fn = self._get_predict_fn()
        prediction = float(predict_fn(sample_2d))

        # Top features
        sorted_shap = sorted(shap_map.items(), key=lambda x: x[1], reverse=True)
        top_positive = [(n, v) for n, v in sorted_shap if v > 0][:5]
        top_negative = [(n, v) for n, v in sorted_shap if v < 0][-5:]

        return {
            "base_value": base_value,
            "prediction": prediction,
            "label": "FAKE" if prediction > 0.5 else "REAL",
            "confidence": round(abs(prediction - 0.5) * 2, 4),
            "shap_values": shap_map,
            "top_positive_features": top_positive,   # empurram para FAKE
            "top_negative_features": top_negative,   # empurram para REAL
        }

    # ------------------------------------------------------------------
    # Relatorio completo
    # ------------------------------------------------------------------
    def generate_report(
        self,
        X_test: np.ndarray,
        output_path: Optional[Union[str, Path]] = None,
        max_samples: int = 200,
    ) -> Dict:
        """
        Gera relatorio SHAP completo (global + amostras representativas).

        Args:
            X_test: conjunto de teste
            output_path: salvar JSON (opcional)
            max_samples: limite para calculo global

        Returns:
            dict com global_importance + exemplos locais (5 FAKE + 5 REAL)
        """
        logger.info("Gerando relatorio SHAP completo...")

        global_result = self.global_importance(X_test, max_samples=max_samples)

        report = {
            "global_importance": global_result,
            "local_examples": [],
            "summary": {
                "top_feature_group": max(
                    global_result["group_importance"].items(),
                    key=lambda x: x[1]["shap_sum"],
                    default=("unknown", {}),
                )[0],
                "n_features": len(global_result["feature_importance"]),
                "n_samples": global_result["n_samples_analyzed"],
            },
        }

        # Exemplos locais (5 amostras aleatorias)
        n_examples = min(5, len(X_test))
        for i in range(n_examples):
            try:
                local = self.local_explanation(X_test[i])
                report["local_examples"].append({
                    "sample_idx": i,
                    "prediction": local["prediction"],
                    "label": local["label"],
                    "confidence": local["confidence"],
                    "top_positive": local["top_positive_features"],
                    "top_negative": local["top_negative_features"],
                })
            except Exception as e:
                logger.warning(f"Explicacao local falhou para amostra {i}: {e}")

        if output_path:
            path = Path(output_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            with open(path, "w", encoding="utf-8") as f:
                json.dump(report, f, ensure_ascii=False, indent=2, default=str)
            logger.info(f"Relatorio SHAP salvo em: {path}")

        return report
