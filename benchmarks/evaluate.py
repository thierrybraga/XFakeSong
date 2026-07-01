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

    P2 — também reporta `accuracy_at_eer`: a acurácia no LIMIAR ÓTIMO (ponto de
    EER), separando "falha de calibração/limiar" de "falha de separabilidade".
    Modelos cujos scores deslocam sob ruído podem despencar no limiar fixo 0.5
    mesmo quando permanecem parcialmente separáveis — `accuracy_at_eer` revela
    esse teto, enquanto `accuracy` mostra a decisão operável real. A distância
    entre os dois mede o quanto é só limiar.

    Retorna dict com: accuracy, accuracy_at_eer, precision, recall, f1, auc_roc,
    eer, eer_threshold, min_tdcf, n, n_pos, n_neg.
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
    nonfinite_scores = int((~np.isfinite(p_fake)).sum())
    if nonfinite_scores:
        p_fake = np.nan_to_num(p_fake, nan=0.5, posinf=1.0, neginf=0.0)
    p_fake = np.clip(p_fake, 0.0, 1.0)
    y_pred = (p_fake >= threshold).astype(int)

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
        "nonfinite_scores": nonfinite_scores,
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
            # Acurácia no limiar ótimo (ponto de EER) — teto de separabilidade,
            # independente da calibração do limiar fixo 0.5.
            if np.isfinite(eer_thr):
                out["accuracy_at_eer"] = float(
                    accuracy_score(y_true, (p_fake >= eer_thr).astype(int))
                )
            else:
                out["accuracy_at_eer"] = float("nan")
        except Exception:
            out["eer"] = float("nan")
            out["eer_threshold"] = float("nan")
            out["accuracy_at_eer"] = float("nan")
        try:
            tdcf, _ = mc.calculate_min_tdcf(y_true, p_fake)
            out["min_tdcf"] = float(tdcf)
        except Exception:
            out["min_tdcf"] = float("nan")
    else:
        out["auc_roc"] = float("nan")
        out["eer"] = float("nan")
        out["eer_threshold"] = float("nan")
        out["accuracy_at_eer"] = float("nan")
        out["min_tdcf"] = float("nan")

    return out
