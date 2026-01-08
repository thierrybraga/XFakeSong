"""Validação Cruzada Temporal para Prevenção de Data Leakage

Este módulo implementa validação cruzada temporal específica para dados de áudio
sequenciais, garantindo que não haja vazamento temporal entre folds.
"""

import numpy as np
import logging
from typing import Dict, List, Any, Optional, Tuple, Iterator, Union
from dataclasses import dataclass
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import tensorflow as tf
from pathlib import Path

from core.interfaces.base import ProcessingResult, ProcessingStatus
from .secure_training_pipeline import SecureFeatureScaler, SecureTrainingConfig

logger = logging.getLogger(__name__)


@dataclass
class TemporalCVConfig:
    """Configuração para validação cruzada temporal."""
    n_splits: int = 5
    test_size: Optional[int] = None  # Se None, usa divisão automática
    gap: int = 0  # Gap entre treino e teste para evitar vazamento
    # Limita tamanho do conjunto de treino
    max_train_size: Optional[int] = None
    scaler_type: str = "standard"
    random_state: int = 42


class TemporalCrossValidator:
    """Validador cruzado temporal que previne data leakage."""

    def __init__(self, config: TemporalCVConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.cv_splitter = TimeSeriesSplit(
            n_splits=config.n_splits,
            test_size=config.test_size,
            gap=config.gap,
            max_train_size=config.max_train_size
        )

    def split(self, X: np.ndarray, y: np.ndarray,
              timestamps: Optional[np.ndarray] = None) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        """Gera splits temporais para validação cruzada.

        Args:
            X: Features
            y: Labels
            timestamps: Timestamps opcionais para ordenação

        Yields:
            (train_indices, test_indices) para cada fold
        """
        # Ordenar por timestamp se fornecido
        if timestamps is not None:
            sorted_indices = np.argsort(timestamps)
            X_sorted = X[sorted_indices]
            y_sorted = y[sorted_indices]
        else:
            sorted_indices = np.arange(len(X))
            X_sorted = X
            y_sorted = y

        # Gerar splits temporais
        for train_idx, test_idx in self.cv_splitter.split(X_sorted):
            # Mapear de volta para índices originais se necessário
            if timestamps is not None:
                train_idx = sorted_indices[train_idx]
                test_idx = sorted_indices[test_idx]

            yield train_idx, test_idx

    def validate_model(self, model_factory, X: np.ndarray, y: np.ndarray,
                       timestamps: Optional[np.ndarray] = None,
                       compile_params: Optional[Dict] = None,
                       fit_params: Optional[Dict] = None) -> ProcessingResult[Dict[str, Any]]:
        """Executa validação cruzada temporal completa.

        Args:
            model_factory: Função que retorna um novo modelo
            X: Features
            y: Labels
            timestamps: Timestamps para ordenação temporal
            compile_params: Parâmetros para compilação do modelo
            fit_params: Parâmetros para treinamento

        Returns:
            Resultado com métricas de validação cruzada
        """
        try:
            self.logger.info(
                f"Iniciando validação cruzada temporal com {
                    self.config.n_splits} folds")

            # Parâmetros padrão
            compile_params = compile_params or {
                'optimizer': 'adam',
                'loss': 'binary_crossentropy',
                'metrics': ['accuracy']
            }

            fit_params = fit_params or {
                'epochs': 50,
                'batch_size': 32,
                'verbose': 0
            }

            # Armazenar resultados de cada fold
            fold_results = []
            fold_predictions = []
            fold_true_labels = []

            for fold_idx, (train_idx, test_idx) in enumerate(
                    self.split(X, y, timestamps)):
                self.logger.info(
                    f"Processando fold {fold_idx + 1}/{self.config.n_splits}")

                # Dividir dados para este fold
                X_train_fold = X[train_idx]
                y_train_fold = y[train_idx]
                X_test_fold = X[test_idx]
                y_test_fold = y[test_idx]

                # Normalizar dados de forma segura (scaler ajustado apenas no
                # treino)
                scaler = SecureFeatureScaler(
                    SecureTrainingConfig(scaler_type=self.config.scaler_type)
                )

                X_train_scaled = scaler.fit_transform_train(X_train_fold)
                X_test_scaled = scaler.transform_test(X_test_fold)

                # Criar e treinar modelo para este fold
                model = model_factory()
                model.compile(**compile_params)

                # Treinar modelo
                history = model.fit(
                    X_train_scaled, y_train_fold,
                    validation_data=(X_test_scaled, y_test_fold),
                    **fit_params
                )

                # Fazer predições
                y_pred_proba = model.predict(X_test_scaled, verbose=0)
                y_pred = (y_pred_proba > 0.5).astype(int).flatten()

                # Calcular métricas para este fold
                fold_metrics = self._calculate_fold_metrics(
                    y_test_fold, y_pred, y_pred_proba.flatten()
                )

                fold_result = {
                    'fold': fold_idx + 1,
                    'train_size': len(X_train_fold),
                    'test_size': len(X_test_fold),
                    'metrics': fold_metrics,
                    'history': history.history
                }

                fold_results.append(fold_result)
                fold_predictions.extend(y_pred)
                fold_true_labels.extend(y_test_fold)

                self.logger.info(
                    f"Fold {
                        fold_idx + 1} - Accuracy: {
                        fold_metrics['accuracy']:.4f}, F1: {
                        fold_metrics['f1']:.4f}")

            # Calcular métricas agregadas
            aggregated_metrics = self._calculate_aggregated_metrics(
                fold_results)

            # Calcular métricas globais (todas as predições)
            global_metrics = self._calculate_fold_metrics(
                np.array(fold_true_labels),
                np.array(fold_predictions),
                np.array(fold_predictions).astype(
                    float)  # Simplificado para exemplo
            )

            results = {
                'fold_results': fold_results,
                'aggregated_metrics': aggregated_metrics,
                'global_metrics': global_metrics,
                'config': self.config.__dict__,
                'total_samples': len(X),
                'n_folds': self.config.n_splits
            }

            self.logger.info(
                f"Validação cruzada concluída. Accuracy média: {
                    aggregated_metrics['accuracy']['mean']:.4f} ± {
                    aggregated_metrics['accuracy']['std']:.4f}")

            return ProcessingResult(
                success=True,
                data=results,
                status=ProcessingStatus.COMPLETED
            )

        except Exception as e:
            self.logger.error(f"Erro na validação cruzada temporal: {str(e)}")
            return ProcessingResult(
                success=False,
                error=str(e),
                status=ProcessingStatus.FAILED
            )

    def _calculate_fold_metrics(self, y_true: np.ndarray, y_pred: np.ndarray,
                                y_pred_proba: np.ndarray) -> Dict[str, float]:
        """Calcula métricas para um fold específico."""
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, average='binary', zero_division=0),
            'recall': recall_score(y_true, y_pred, average='binary', zero_division=0),
            'f1': f1_score(y_true, y_pred, average='binary', zero_division=0)
        }

        # AUC apenas se há ambas as classes
        if len(np.unique(y_true)) > 1:
            try:
                metrics['auc'] = roc_auc_score(y_true, y_pred_proba)
            except ValueError:
                metrics['auc'] = 0.0
        else:
            metrics['auc'] = 0.0

        return metrics

    def _calculate_aggregated_metrics(
            self, fold_results: List[Dict]) -> Dict[str, Dict[str, float]]:
        """Calcula métricas agregadas de todos os folds."""
        metrics_names = ['accuracy', 'precision', 'recall', 'f1', 'auc']
        aggregated = {}

        for metric_name in metrics_names:
            values = [fold['metrics'][metric_name] for fold in fold_results]
            aggregated[metric_name] = {
                'mean': np.mean(values),
                'std': np.std(values),
                'min': np.min(values),
                'max': np.max(values),
                'values': values
            }

        return aggregated

    def save_cv_results(
            self, results: Dict[str, Any], save_path: Union[str, Path]) -> None:
        """Salva resultados da validação cruzada."""
        import json

        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)

        # Converter numpy arrays para listas para serialização JSON
        serializable_results = self._make_json_serializable(results)

        with open(save_path, 'w') as f:
            json.dump(serializable_results, f, indent=2, default=str)

        self.logger.info(
            f"Resultados da validação cruzada salvos em: {save_path}")

    def _make_json_serializable(self, obj):
        """Converte objetos para formato serializável em JSON."""
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, dict):
            return {key: self._make_json_serializable(
                value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._make_json_serializable(item) for item in obj]
        else:
            return obj


class TemporalCVAnalyzer:
    """Analisador de resultados de validação cruzada temporal."""

    def __init__(self):
        self.logger = logging.getLogger(__name__)

    def analyze_stability(self, cv_results: Dict[str, Any]) -> Dict[str, Any]:
        """Analisa estabilidade do modelo através dos folds."""
        aggregated = cv_results['aggregated_metrics']

        stability_analysis = {
            'coefficient_of_variation': {},
            'stability_score': 0.0,
            'recommendations': []
        }

        # Calcular coeficiente de variação para cada métrica
        for metric_name, metric_data in aggregated.items():
            if metric_data['mean'] > 0:
                cv_coeff = metric_data['std'] / metric_data['mean']
                stability_analysis['coefficient_of_variation'][metric_name] = cv_coeff

        # Calcular score de estabilidade (baseado na variação da accuracy)
        acc_cv = stability_analysis['coefficient_of_variation'].get(
            'accuracy', 1.0)
        # Quanto menor a variação, maior a estabilidade
        stability_score = max(0, 1 - acc_cv)
        stability_analysis['stability_score'] = stability_score

        # Gerar recomendações
        if stability_score < 0.7:
            stability_analysis['recommendations'].append(
                "Modelo apresenta alta variabilidade entre folds. Considere:"
                "\n- Aumentar tamanho do dataset"
                "\n- Ajustar hiperparâmetros"
                "\n- Usar regularização mais forte"
            )

        if aggregated['accuracy']['std'] > 0.1:
            stability_analysis['recommendations'].append(
                "Alta variação na accuracy. Modelo pode estar instável."
            )

        return stability_analysis

    def detect_temporal_overfitting(
            self, cv_results: Dict[str, Any]) -> Dict[str, Any]:
        """Detecta overfitting temporal analisando tendências nos folds."""
        fold_results = cv_results['fold_results']

        # Extrair accuracy de cada fold em ordem temporal
        accuracies = [fold['metrics']['accuracy'] for fold in fold_results]

        # Calcular tendência (regressão linear simples)
        x = np.arange(len(accuracies))
        slope = np.polyfit(x, accuracies, 1)[0]

        analysis = {
            'temporal_trend': slope,
            'is_degrading': slope < -0.01,  # Threshold para degradação
            'performance_drop': max(accuracies) - min(accuracies),
            'recommendations': []
        }

        if analysis['is_degrading']:
            analysis['recommendations'].append(
                "Performance degrada ao longo do tempo. Possível overfitting temporal."
                "\nConsidere usar dados mais recentes para treinamento."
            )

        if analysis['performance_drop'] > 0.15:
            analysis['recommendations'].append(
                "Grande variação de performance entre folds temporais."
                "\nModelo pode não estar generalizando bem para dados futuros."
            )

        return analysis

    def generate_report(self, cv_results: Dict[str, Any]) -> str:
        """Gera relatório completo da validação cruzada."""
        stability = self.analyze_stability(cv_results)
        temporal_analysis = self.detect_temporal_overfitting(cv_results)
        aggregated = cv_results['aggregated_metrics']

        report = f"""
# Relatório de Validação Cruzada Temporal

## Configuração
- Número de folds: {cv_results['n_folds']}
- Total de amostras: {cv_results['total_samples']}
- Gap temporal: {cv_results['config']['gap']}

## Métricas Agregadas
"""

        for metric_name, metric_data in aggregated.items():
            report += f"""
### {metric_name.upper()}
- Média: {metric_data['mean']:.4f} ± {metric_data['std']:.4f}
- Mín/Máx: {metric_data['min']:.4f} / {metric_data['max']:.4f}
"""

        report += f"""

## Análise de Estabilidade
- Score de estabilidade: {stability['stability_score']:.3f}
- Coeficiente de variação (accuracy): {stability['coefficient_of_variation'].get('accuracy', 0):.3f}

## Análise Temporal
- Tendência temporal: {temporal_analysis['temporal_trend']:.4f}
- Performance degradando: {'Sim' if temporal_analysis['is_degrading'] else 'Não'}
- Variação de performance: {temporal_analysis['performance_drop']:.3f}

## Recomendações
"""

        all_recommendations = stability['recommendations'] + \
            temporal_analysis['recommendations']
        if all_recommendations:
            for i, rec in enumerate(all_recommendations, 1):
                report += f"\n{i}. {rec}"
        else:
            report += "\nModelo apresenta boa estabilidade temporal."

        return report

# Função de conveniência


def run_temporal_cross_validation(model_factory, X: np.ndarray, y: np.ndarray,
                                  timestamps: Optional[np.ndarray] = None,
                                  n_splits: int = 5,
                                  gap: int = 0,
                                  save_results: bool = True,
                                  results_path: str = "cv_results.json") -> ProcessingResult[Dict[str, Any]]:
    """Executa validação cruzada temporal completa com análise."""

    # Configurar validador
    config = TemporalCVConfig(n_splits=n_splits, gap=gap)
    validator = TemporalCrossValidator(config)

    # Executar validação
    cv_result = validator.validate_model(model_factory, X, y, timestamps)

    if not cv_result.success:
        return cv_result

    # Analisar resultados
    analyzer = TemporalCVAnalyzer()
    cv_results = cv_result.data

    cv_results['stability_analysis'] = analyzer.analyze_stability(cv_results)
    cv_results['temporal_analysis'] = analyzer.detect_temporal_overfitting(
        cv_results)
    cv_results['report'] = analyzer.generate_report(cv_results)

    # Salvar resultados se solicitado
    if save_results:
        validator.save_cv_results(cv_results, results_path)

    return ProcessingResult(
        success=True,
        data=cv_results,
        status=ProcessingStatus.COMPLETED
    )
