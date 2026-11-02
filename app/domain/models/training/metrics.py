"""Módulo de Cálculo de Métricas

Este módulo implementa calculadoras de métricas para avaliação de modelos.
"""

import logging
from typing import Dict, Optional, Tuple

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    auc,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)


class MetricsCalculator:
    """Calculadora de métricas para modelos de classificação."""

    def __init__(self):
        self.logger = logging.getLogger(__name__)

    def calculate_all_metrics(
        self,
        y_true: np.ndarray,
        y_pred_classes: np.ndarray,
        y_pred_proba: Optional[np.ndarray] = None
    ) -> Dict[str, float]:
        """Calcula todas as métricas disponíveis."""
        metrics = {}

        try:
            # Métricas básicas
            metrics.update(
                self.calculate_basic_metrics(
                    y_true, y_pred_classes))

            # Métricas baseadas em probabilidade
            if y_pred_proba is not None:
                metrics.update(
                    self.calculate_probability_metrics(
                        y_true, y_pred_proba))

            # Métricas de matriz de confusão
            metrics.update(
                self.calculate_confusion_metrics(
                    y_true, y_pred_classes))

        except Exception as e:
            self.logger.error(f"Erro ao calcular métricas: {str(e)}")

        return metrics

    def calculate_basic_metrics(
        self,
        y_true: np.ndarray,
        y_pred_classes: np.ndarray
    ) -> Dict[str, float]:
        """Calcula métricas básicas de classificação."""
        return {
            "accuracy": float(accuracy_score(y_true, y_pred_classes)),
            "precision": float(precision_score(y_true, y_pred_classes, average='weighted', zero_division=0)),
            "recall": float(recall_score(y_true, y_pred_classes, average='weighted', zero_division=0)),
            "f1_score": float(f1_score(y_true, y_pred_classes, average='weighted', zero_division=0)),
            "precision_macro": float(precision_score(y_true, y_pred_classes, average='macro', zero_division=0)),
            "recall_macro": float(recall_score(y_true, y_pred_classes, average='macro', zero_division=0)),
            "f1_score_macro": float(f1_score(y_true, y_pred_classes, average='macro', zero_division=0))
        }

    def calculate_probability_metrics(
        self,
        y_true: np.ndarray,
        y_pred_proba: np.ndarray
    ) -> Dict[str, float]:
        """Calcula métricas baseadas em probabilidades."""
        metrics = {}

        try:
            # Para classificação binária
            # Suporta saídas: (N,) sigmoid 1D, (N,1) sigmoid binário, (N,2) softmax
            if y_pred_proba.ndim == 1 or (y_pred_proba.ndim > 1 and y_pred_proba.shape[1] <= 2):
                if y_pred_proba.ndim == 1:
                    y_proba = y_pred_proba                      # já é P(classe=1)
                elif y_pred_proba.shape[1] == 2:
                    y_proba = y_pred_proba[:, 1]                # softmax: coluna 1
                else:
                    y_proba = y_pred_proba[:, 0]                # sigmoid (N,1): coluna 0

                # ROC AUC
                metrics["roc_auc"] = float(roc_auc_score(y_true, y_proba))

                # Precision-Recall AUC
                precision, recall, _ = precision_recall_curve(y_true, y_proba)
                metrics["pr_auc"] = float(auc(recall, precision))

                # Métricas primárias de anti-spoofing (protocolo ASVspoof):
                # EER + min t-DCF normalizado. y_proba = P(classe=1=fake/spoof).
                try:
                    eer, eer_thr = self.calculate_eer(y_true, y_proba)
                    metrics["eer"] = float(eer)
                    metrics["eer_threshold"] = float(eer_thr)
                    min_tdcf, _ = self.calculate_min_tdcf(y_true, y_proba)
                    metrics["min_tdcf"] = float(min_tdcf)
                except Exception as e:
                    self.logger.debug(f"EER/min-tDCF indisponível: {e}")

            # Para classificação multiclasse
            else:
                # ROC AUC multiclasse
                try:
                    metrics["roc_auc_ovr"] = float(roc_auc_score(
                        y_true, y_pred_proba, multi_class='ovr', average='weighted'
                    ))
                    metrics["roc_auc_ovo"] = float(roc_auc_score(
                        y_true, y_pred_proba, multi_class='ovo', average='weighted'
                    ))
                except ValueError:
                    # Pode falhar se houver apenas uma classe
                    pass

        except Exception as e:
            self.logger.warning(
                f"Erro ao calcular métricas de probabilidade: {str(e)}")

        return metrics

    def calculate_confusion_metrics(
        self,
        y_true: np.ndarray,
        y_pred_classes: np.ndarray
    ) -> Dict[str, float]:
        """Calcula métricas derivadas da matriz de confusão."""
        metrics = {}

        try:
            cm = confusion_matrix(y_true, y_pred_classes)

            # Para classificação binária
            if cm.shape == (2, 2):
                tn, fp, fn, tp = cm.ravel()

                metrics.update({
                    "true_positives": float(tp),
                    "true_negatives": float(tn),
                    "false_positives": float(fp),
                    "false_negatives": float(fn),
                    "specificity": float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0,
                    "sensitivity": float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0,
                    "false_positive_rate": float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0,
                    "false_negative_rate": float(fn / (fn + tp)) if (fn + tp) > 0 else 0.0,
                    "positive_predictive_value": float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0,
                    "negative_predictive_value": float(tn / (tn + fn)) if (tn + fn) > 0 else 0.0
                })

            # Métricas gerais da matriz de confusão
            total_samples = np.sum(cm)
            correct_predictions = np.trace(cm)

            metrics.update({
                "total_samples": float(total_samples),
                "correct_predictions": float(correct_predictions),
                "incorrect_predictions": float(total_samples - correct_predictions)
            })

        except Exception as e:
            self.logger.warning(
                f"Erro ao calcular métricas de confusão: {str(e)}")

        return metrics


    def calculate_eer(
        self,
        y_true: np.ndarray,
        y_scores: np.ndarray
    ) -> Tuple[float, float]:
        """Calcula Equal Error Rate (EER).

        Retorna (eer_value, eer_threshold).
        EER é o ponto onde FPR == FNR na curva DET.
        """
        try:
            fpr, tpr, thresholds = roc_curve(y_true, y_scores)
            fnr = 1 - tpr

            # sklearn >= 1.3 devolve `thresholds[0] = inf` (antes era
            # max(score)+1). Esse infinito entrava no eixo x do interp1d e
            # produzia NaN, o brentq abortava e TODO EER do projeto caía no
            # fallback — o caminho de interpolação nunca executava. Descartar os
            # thresholds não finitos restaura a interpolação; o ponto removido é
            # o extremo "nada classificado como positivo", que não carrega
            # informação sobre o cruzamento FPR=FNR.
            finite = np.isfinite(thresholds)
            fpr_f, fnr_f, thr_f = fpr[finite], fnr[finite], thresholds[finite]

            # Interpolação para encontrar interseção FPR = FNR
            try:
                if thr_f.size < 2:
                    raise ValueError("thresholds finitos insuficientes")

                from scipy.interpolate import interp1d
                from scipy.optimize import brentq

                # roc_curve devolve thresholds DECRESCENTES; interp1d exige x
                # crescente, daí o [::-1].
                fpr_interp = interp1d(thr_f[::-1], fpr_f[::-1],
                                      bounds_error=False, fill_value=(1, 0))
                fnr_interp = interp1d(thr_f[::-1], fnr_f[::-1],
                                      bounds_error=False, fill_value=(0, 1))

                # Encontrar threshold onde FPR(t) == FNR(t). No menor threshold
                # FPR=1 e FNR=0 (diferença > 0); no maior, o oposto — então há
                # troca de sinal e o brentq tem raiz garantida no intervalo.
                eer_threshold = brentq(
                    lambda t: float(fpr_interp(t)) - float(fnr_interp(t)),
                    float(thr_f.min()), float(thr_f.max())
                )
                eer = float(fpr_interp(eer_threshold))
            except Exception as exc:
                # Fallback: ponto mais próximo onde |FPR - FNR| é mínimo.
                # Registrado em debug para que uma nova quebra da interpolação
                # (como a do `inf` acima) não volte a passar despercebida.
                self.logger.debug(
                    "EER por interpolação falhou (%s); usando o ponto de "
                    "|FPR-FNR| mínimo", exc
                )
                diff = np.abs(fpr - fnr)
                eer_idx = np.argmin(diff)
                eer = float((fpr[eer_idx] + fnr[eer_idx]) / 2)
                eer_threshold = float(thresholds[eer_idx])

            return eer, eer_threshold

        except Exception as e:
            self.logger.error(f"Erro ao calcular EER: {str(e)}")
            return 0.5, 0.5

    @staticmethod
    def _det_curve(target_scores: np.ndarray, nontarget_scores: np.ndarray):
        """Curva DET (Pmiss, Pfa, thresholds) — algoritmo oficial ASVspoof.

        `target_scores` = scores da classe BONAFIDE (maior = mais bonafide);
        `nontarget_scores` = scores da classe SPOOF.
        """
        n = target_scores.size + nontarget_scores.size
        all_scores = np.concatenate((target_scores, nontarget_scores))
        labels = np.concatenate(
            (np.ones(target_scores.size), np.zeros(nontarget_scores.size))
        )
        idx = np.argsort(all_scores, kind="mergesort")
        labels = labels[idx]
        tar_sums = np.cumsum(labels)
        nontar_sums = nontarget_scores.size - (np.arange(1, n + 1) - tar_sums)
        pmiss = np.concatenate((np.atleast_1d(0), tar_sums / target_scores.size))
        pfa = np.concatenate((np.atleast_1d(1), nontar_sums / nontarget_scores.size))
        thr = np.concatenate(
            (np.atleast_1d(all_scores[idx[0]] - 1e-3), all_scores[idx])
        )
        return pmiss, pfa, thr

    def calculate_min_tdcf(
        self,
        y_true: np.ndarray,
        y_scores: np.ndarray,
        *,
        p_target: float = 0.9405,
        p_nontarget: float = 0.0095,
        p_spoof: float = 0.05,
        c_miss: float = 1.0,
        c_fa: float = 10.0,
        pfa_asv: float = 0.0524,
        pmiss_asv: float = 0.0104,
        pmiss_spoof_asv: float = 0.0058,
    ) -> Tuple[float, float]:
        """min t-DCF normalizado (Kinnunen et al., 2018 — protocolo ASVspoof2019).

        Métrica PRIMÁRIA de anti-spoofing. Versão **CM-only**: o sistema ASV é
        fixado num ponto de operação por meio das constantes (`pfa_asv`,
        `pmiss_asv`, `pmiss_spoof_asv` = defaults da partição LA do ASVspoof2019),
        tornando a métrica comparável entre execuções sem precisar de scores ASV.

        Args:
            y_true: 1 = fake/spoof, 0 = real/bonafide.
            y_scores: score de FAKE/spoof (ex.: p_fake). Internamente é invertido
                para a convenção do t-DCF (maior = mais bonafide).

        Returns:
            (min_tdcf, threshold). `min_tdcf` ~0 = ótimo; ~1 = sem poder de detecção.
        """
        y_true = np.asarray(y_true).astype(int).reshape(-1)
        y_scores = np.asarray(y_scores, dtype=float).reshape(-1)
        # Convenção t-DCF: score maior ⇒ mais BONAFIDE. Aqui score = p_fake → inverte.
        cm = -y_scores
        bona = cm[y_true == 0]
        spoof = cm[y_true == 1]
        if bona.size == 0 or spoof.size == 0:
            return 1.0, 0.0

        pmiss_cm, pfa_cm, thr = self._det_curve(bona, spoof)

        # Modelo de custo padrão do ASVspoof2019 (Cmiss_asv = Cmiss_cm = c_miss).
        c1 = p_target * (c_miss - c_miss * pmiss_asv) - p_nontarget * c_fa * pfa_asv
        c2 = c_fa * p_spoof * (1.0 - pmiss_spoof_asv)
        tdcf = c1 * pmiss_cm + c2 * pfa_cm
        denom = min(c1, c2)
        if denom <= 0:
            return 1.0, 0.0
        tdcf_norm = tdcf / denom
        min_idx = int(np.argmin(tdcf_norm))
        return float(tdcf_norm[min_idx]), float(thr[min_idx])


# Pontuação e EER de validação (saíram de trainer.py, P2-15): puro
# numpy/sklearn, sem TensorFlow — o callback ValidationEER e o
# TrainingService chamam ``validation_eer`` com o modelo já treinado.


def scores_from_predictions(bruto) -> "np.ndarray":
    """Converte a saída BRUTA do modelo na pontuação que ordena spoof>bonafide.

    SAÍDA DE 2 COLUNAS é o caso NORMAL no escopo, não a exceção: as
    arquiteturas emitem softmax/logits sobre {bonafide, spoof}. A coluna 1 é a
    do FAKE, a mesma convenção de `benchmarks/fit_neural.py` (`pred[:, 1]`).

    LOG-ODDS, nao a coluna 1 sozinha. Medido em 2026-08-19 no AASIST promovido:
    ele emite LOGITS CRUS (faixa -37,8 a +41,1; a soma das colunas tem media
    1,91 e desvio 4,51, ou seja NAO e softmax), e as duas colunas variam de
    forma independente. Ordenar so por `logit[1]` deu EER 0,0025 contra 0,0050
    pelo criterio correto -- OTIMISTA por um fator de 2.

    A diferenca serve aos DOIS casos e dispensa detectar o tipo de saida: com
    softmax as colunas somam 1, entao `col1 - col0 = 2*col1 - 1` e
    transformacao monotonica de col1 e o EER nao muda; com logits ela e a
    log-odds, que e o que ordena corretamente.
    """
    bruto = np.asarray(bruto)
    if bruto.ndim == 2 and bruto.shape[1] == 2:
        return (bruto[:, 1] - bruto[:, 0]).ravel()
    return bruto.ravel()


def eer_from_scores(y_true, scores) -> float:
    """EER a partir de rótulos e pontuações, com INTERPOLAÇÃO do cruzamento.

    Espelha `MetricsCalculator.calculate_eer` (que usa brentq). Usar só o ponto
    mais próximo (`argmin|fpr-fnr|`) fazia as duas implementações discordarem
    na terceira casa — num regime de EER de 0,1% é essa a casa que separa os
    modelos, e é esta métrica que SELECIONA o checkpoint. A interpolação linear
    entre os dois pontos que cercam a troca de sinal dá o mesmo resultado do
    brentq sem depender de scipy dentro de um callback de época.

    Levanta ValueError quando o EER não é definível (uma classe só, tamanhos
    incompatíveis, pontuações não-finitas) — quem chama decide se isso é aviso
    ou falha.
    """
    from sklearn.metrics import roc_curve

    y_true = np.asarray(y_true).ravel()
    scores = np.asarray(scores).ravel()
    if scores.size != y_true.size:
        raise ValueError(
            f"predição com {scores.size} scores para {y_true.size} rótulos"
        )
    if not np.isfinite(scores).all():
        raise ValueError("pontuações não-finitas (NaN/inf) na validação")
    if len(np.unique(y_true)) < 2:
        raise ValueError("validação tem uma classe só; EER indefinível")

    fpr, tpr, thr = roc_curve(y_true, scores)
    fnr = 1.0 - tpr
    finito = np.isfinite(thr)
    fpr, fnr = fpr[finito], fnr[finito]
    if fpr.size == 0:
        raise ValueError("curva ROC vazia após descartar limiares não-finitos")

    d = fpr - fnr
    troca = np.nonzero(np.sign(d[:-1]) * np.sign(d[1:]) < 0)[0]
    if troca.size:
        i = int(troca[0])
        passo = d[i] - d[i + 1]
        peso = float(d[i] / passo) if passo != 0 else 0.0
        eer = float(
            (fpr[i] + peso * (fpr[i + 1] - fpr[i]))
            + (fnr[i] + peso * (fnr[i + 1] - fnr[i]))
        ) / 2.0
    else:
        i = int(np.nanargmin(np.abs(d)))
        eer = float((fpr[i] + fnr[i]) / 2.0)
    return float(min(max(eer, 0.0), 1.0))


def validation_eer(model, validation_data, batch_size: int = 32) -> float:
    """EER do modelo no conjunto de validação, na convenção do projeto.

    Fonte ÚNICA do `val_eer`: o callback `ValidationEER` (que publica a métrica
    por época) e a restauração guardada de checkpoint (que decide se o
    checkpoint selecionado por ela fica) chamam esta função. Duas
    implementações do mesmo EER divergindo na terceira casa selecionariam
    épocas diferentes das que o log declara.
    """
    x, y = validation_data
    bruto = model.predict(x, verbose=0, batch_size=max(1, int(batch_size)))
    return eer_from_scores(np.asarray(y).ravel(), scores_from_predictions(bruto))
