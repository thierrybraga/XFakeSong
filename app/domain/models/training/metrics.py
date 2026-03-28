"""Módulo de Cálculo de Métricas

Este módulo implementa calculadoras de métricas para avaliação de modelos.
"""

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    auc,
    classification_report,
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
            if y_pred_proba.shape[1] == 2 or len(y_pred_proba.shape) == 1:
                y_proba = y_pred_proba[:, 1] if len(
                    y_pred_proba.shape) > 1 else y_pred_proba

                # ROC AUC
                metrics["roc_auc"] = float(roc_auc_score(y_true, y_proba))

                # Precision-Recall AUC
                precision, recall, _ = precision_recall_curve(y_true, y_proba)
                metrics["pr_auc"] = float(auc(recall, precision))

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

    def get_classification_report(
        self,
        y_true: np.ndarray,
        y_pred_classes: np.ndarray,
        target_names: Optional[List[str]] = None
    ) -> str:
        """Retorna relatório de classificação detalhado."""
        try:
            return classification_report(
                y_true, y_pred_classes,
                target_names=target_names,
                zero_division=0
            )
        except Exception as e:
            self.logger.error(
                f"Erro ao gerar relatório de classificação: {str(e)}")
            return ""

    def get_confusion_matrix(
        self,
        y_true: np.ndarray,
        y_pred_classes: np.ndarray
    ) -> np.ndarray:
        """Retorna matriz de confusão."""
        try:
            return confusion_matrix(y_true, y_pred_classes)
        except Exception as e:
            self.logger.error(f"Erro ao calcular matriz de confusão: {str(e)}")
            return np.array([])

    def calculate_per_class_metrics(
        self,
        y_true: np.ndarray,
        y_pred_classes: np.ndarray,
        class_names: Optional[List[str]] = None
    ) -> Dict[str, Dict[str, float]]:
        """Calcula métricas por classe."""
        metrics_per_class = {}

        try:
            # Obter classes únicas
            unique_classes = np.unique(y_true)

            for i, class_label in enumerate(unique_classes):
                class_name = class_names[i] if class_names and i < len(
                    class_names) else f"class_{class_label}"

                # Criar máscaras binárias para a classe atual
                y_true_binary = (y_true == class_label).astype(int)
                y_pred_binary = (y_pred_classes == class_label).astype(int)

                # Calcular métricas para esta classe
                metrics_per_class[class_name] = {
                    "precision": float(precision_score(y_true_binary, y_pred_binary, zero_division=0)),
                    "recall": float(recall_score(y_true_binary, y_pred_binary, zero_division=0)),
                    "f1_score": float(f1_score(y_true_binary, y_pred_binary, zero_division=0)),
                    "support": float(np.sum(y_true_binary))
                }

        except Exception as e:
            self.logger.error(
                f"Erro ao calcular métricas por classe: {str(e)}")

        return metrics_per_class

    def calculate_threshold_metrics(
        self,
        y_true: np.ndarray,
        y_pred_proba: np.ndarray,
        thresholds: Optional[List[float]] = None
    ) -> Dict[float, Dict[str, float]]:
        """Calcula métricas para diferentes thresholds."""
        if thresholds is None:
            thresholds = [0.3, 0.4, 0.5, 0.6, 0.7]

        threshold_metrics = {}

        try:
            # Para classificação binária
            if len(y_pred_proba.shape) > 1 and y_pred_proba.shape[1] == 2:
                y_proba = y_pred_proba[:, 1]
            else:
                y_proba = y_pred_proba

            for threshold in thresholds:
                y_pred_threshold = (y_proba >= threshold).astype(int)

                threshold_metrics[threshold] = self.calculate_basic_metrics(
                    y_true, y_pred_threshold
                )

        except Exception as e:
            self.logger.error(
                f"Erro ao calcular métricas por threshold: {str(e)}")

        return threshold_metrics

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

            # Interpolação para encontrar interseção FPR = FNR
            try:
                from scipy.interpolate import interp1d
                from scipy.optimize import brentq

                fpr_interp = interp1d(thresholds[::-1], fpr[::-1],
                                      bounds_error=False, fill_value=(1, 0))
                fnr_interp = interp1d(thresholds[::-1], fnr[::-1],
                                      bounds_error=False, fill_value=(0, 1))

                # Encontrar threshold onde FPR(t) == FNR(t)
                t_min, t_max = thresholds.min(), thresholds.max()
                eer_threshold = brentq(
                    lambda t: float(fpr_interp(t)) - float(fnr_interp(t)),
                    t_min, t_max
                )
                eer = float(fpr_interp(eer_threshold))
            except Exception:
                # Fallback: ponto mais próximo onde |FPR - FNR| é mínimo
                diff = np.abs(fpr - fnr)
                eer_idx = np.argmin(diff)
                eer = float((fpr[eer_idx] + fnr[eer_idx]) / 2)
                eer_threshold = float(thresholds[eer_idx])

            return eer, eer_threshold

        except Exception as e:
            self.logger.error(f"Erro ao calcular EER: {str(e)}")
            return 0.5, 0.5

    def get_det_curve_data(
        self,
        y_true: np.ndarray,
        y_scores: np.ndarray
    ) -> Dict[str, np.ndarray]:
        """Retorna dados para plotagem da curva DET.

        Returns:
            Dict com 'fpr', 'fnr', 'thresholds' como np.ndarray.
        """
        try:
            fpr, tpr, thresholds = roc_curve(y_true, y_scores)
            fnr = 1 - tpr

            return {
                'fpr': fpr,
                'fnr': fnr,
                'thresholds': thresholds
            }
        except Exception as e:
            self.logger.error(f"Erro ao gerar dados DET: {str(e)}")
            return {
                'fpr': np.array([0, 1]),
                'fnr': np.array([1, 0]),
                'thresholds': np.array([1, 0])
            }
