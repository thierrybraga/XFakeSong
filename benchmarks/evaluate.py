"""Cálculo do conjunto de métricas a partir dos scores de FAKE."""

from __future__ import annotations

from typing import Dict

import numpy as np


def evaluate_scores(y_true: np.ndarray, p_fake: np.ndarray,
                    threshold: float = 0.5) -> Dict[str, float]:
    """Métricas de detecção a partir de y_true ∈ {0,1} e p_fake ∈ [0,1].

    Reaproveita o MetricsCalculator do pipeline para EER e min-tDCF (mesma
    metodologia do treino), e sklearn para AUC-ROC. Acurácia/precisão/recall/F1
    são medidas no limiar informado (default 0.5, como na decisão padrão).

    Retorna dict com: accuracy, precision, recall, f1, auc_roc, eer,
    eer_threshold, min_tdcf, n, n_pos, n_neg.
    """
    from sklearn.metrics import (
        accuracy_score,
        f1_score,
        precision_score,
        recall_score,
        roc_auc_score,
    )

    from app.domain.models.training.metrics import MetricsCalculator

    y_true = np.asarray(y_true).ravel().astype(int)
    p_fake = np.asarray(p_fake, dtype="float64").ravel()
    y_pred = (p_fake > threshold).astype(int)

    n_pos = int((y_true == 1).sum())
    n_neg = int((y_true == 0).sum())
    out: Dict[str, float] = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "n": int(len(y_true)),
        "n_pos": n_pos,
        "n_neg": n_neg,
    }

    # Métricas que exigem ambas as classes presentes
    if n_pos > 0 and n_neg > 0:
        try:
            out["auc_roc"] = float(roc_auc_score(y_true, p_fake))
        except Exception:
            out["auc_roc"] = float("nan")
        mc = MetricsCalculator()
        try:
            eer, eer_thr = mc.calculate_eer(y_true, p_fake)
            out["eer"] = float(eer)
            out["eer_threshold"] = float(eer_thr)
        except Exception:
            out["eer"] = float("nan")
            out["eer_threshold"] = float("nan")
        try:
            tdcf, _ = mc.calculate_min_tdcf(y_true, p_fake)
            out["min_tdcf"] = float(tdcf)
        except Exception:
            out["min_tdcf"] = float("nan")
    else:
        out["auc_roc"] = float("nan")
        out["eer"] = float("nan")
        out["eer_threshold"] = float("nan")
        out["min_tdcf"] = float("nan")

    return out
