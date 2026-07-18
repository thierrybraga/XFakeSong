"""Cálculo do conjunto de métricas a partir dos scores de FAKE."""

from __future__ import annotations

from typing import Dict

import numpy as np


def _expected_calibration_error(
    y_true: np.ndarray, p_fake: np.ndarray, n_bins: int = 15
) -> float:
    """ECE (Guo et al., 2017) sobre a probabilidade da classe predita.

    Para o caso binário com score p_fake: confiança = max(p, 1-p) e o acerto
    é medido na decisão argmax (limiar 0.5). Bins de largura igual.
    """
    p_fake = np.asarray(p_fake, dtype="float64").ravel()
    y_true = np.asarray(y_true).ravel().astype(int)
    conf = np.maximum(p_fake, 1.0 - p_fake)
    correct = ((p_fake >= 0.5).astype(int) == y_true).astype("float64")
    edges = np.linspace(0.5, 1.0, n_bins + 1)
    ece = 0.0
    n = len(y_true)
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (conf >= lo) & (conf < hi if hi < 1.0 else conf <= hi)
        if not mask.any():
            continue
        ece += (mask.sum() / n) * abs(correct[mask].mean() - conf[mask].mean())
    return float(ece)


def _bootstrap_cis(
    y_true: np.ndarray,
    p_fake: np.ndarray,
    threshold: float,
    n_bootstrap: int,
    seed: int = 12345,
) -> Dict[str, float]:
    """IC 95% percentil por bootstrap (reamostragem com reposição dos pares).

    Rigor acadêmico (2026-07-14): com n≈2250 o IC do EER é ~±0,5–1 pp —
    sem ele, diferenças finas entre modelos não são interpretáveis.
    """
    from sklearn.metrics import accuracy_score, roc_auc_score

    from app.domain.models.training.metrics import MetricsCalculator

    rng = np.random.default_rng(seed)
    mc = MetricsCalculator()
    n = len(y_true)
    eers, aucs, accs = [], [], []
    for _ in range(int(n_bootstrap)):
        idx = rng.integers(0, n, n)
        yb, pb = y_true[idx], p_fake[idx]
        if yb.min() == yb.max():  # reamostra sem ambas as classes: descarta
            continue
        try:
            eers.append(float(mc.calculate_eer(yb, pb)[0]))
            aucs.append(float(roc_auc_score(yb, pb)))
            accs.append(
                float(accuracy_score(yb, (pb >= threshold).astype(int)))
            )
        except Exception:
            continue
    out: Dict[str, float] = {}
    for name, values in (("eer", eers), ("auc_roc", aucs), ("accuracy", accs)):
        if values:
            lo, hi = np.percentile(values, [2.5, 97.5])
            out[f"{name}_ci95_low"] = float(lo)
            out[f"{name}_ci95_high"] = float(hi)
    if eers:
        out["bootstrap_samples"] = int(len(eers))
    return out


def evaluate_scores(y_true: np.ndarray, p_fake: np.ndarray,
                    threshold: float = 0.5,
                    n_bootstrap: int = 0) -> Dict[str, float]:
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

    Rigor acadêmico (2026-07-14): também reporta `ece` (calibração, 15 bins) e,
    quando `n_bootstrap > 0`, IC 95% percentil de EER/AUC/accuracy
    (`*_ci95_low`/`*_ci95_high`) — habilitado pelo benchmark (1000 amostras).

    Retorna dict com: accuracy, accuracy_at_eer, precision, recall, f1, auc_roc,
    eer, eer_threshold, min_tdcf, ece, n, n_pos, n_neg (+ ICs quando pedidos).
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

    out["ece"] = _expected_calibration_error(y_true, p_fake)

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

    if n_bootstrap and n_pos > 0 and n_neg > 0:
        out.update(_bootstrap_cis(y_true, p_fake, threshold, n_bootstrap))

    return out
