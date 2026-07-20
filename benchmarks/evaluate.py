"""Cálculo do conjunto de métricas a partir dos scores de FAKE."""

from __future__ import annotations

from typing import Any, Dict

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
    cluster_ids: np.ndarray | None = None,
) -> Dict[str, Any]:
    """IC 95% percentil por bootstrap de clusters ou, sem IDs, amostras.

    Rigor acadêmico (2026-07-14): com n≈2250 o IC do EER é ~±0,5–1 pp —
    sem ele, diferenças finas entre modelos não são interpretáveis.
    """
    from sklearn.metrics import accuracy_score, roc_auc_score

    from app.domain.models.training.metrics import MetricsCalculator

    rng = np.random.default_rng(seed)
    mc = MetricsCalculator()
    n = len(y_true)
    clusters = None
    unique_clusters = None
    if cluster_ids is not None:
        clusters = np.asarray(cluster_ids).astype(str).ravel()
        if len(clusters) != n:
            raise ValueError("cluster_ids desalinhado com y_true")
        unique_clusters = np.unique(clusters)
        if len(unique_clusters) < 2:
            raise ValueError("bootstrap por cluster exige pelo menos 2 clusters")
    eers, aucs, accs = [], [], []
    for _ in range(int(n_bootstrap)):
        if clusters is None:
            idx = rng.integers(0, n, n)
        else:
            chosen = rng.choice(
                unique_clusters, size=len(unique_clusters), replace=True
            )
            idx = np.concatenate([np.flatnonzero(clusters == group) for group in chosen])
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
    out: Dict[str, Any] = {}
    for name, values in (("eer", eers), ("auc_roc", aucs), ("accuracy", accs)):
        if values:
            lo, hi = np.percentile(values, [2.5, 97.5])
            out[f"{name}_ci95_low"] = float(lo)
            out[f"{name}_ci95_high"] = float(hi)
    if eers:
        out["bootstrap_samples"] = int(len(eers))
    out["bootstrap_unit"] = "cluster" if clusters is not None else "sample"
    if unique_clusters is not None:
        out["bootstrap_clusters"] = int(len(unique_clusters))
    return out


def evaluate_scores(y_true: np.ndarray, p_fake: np.ndarray,
                    threshold: float = 0.5,
                    n_bootstrap: int = 0,
                    cluster_ids: np.ndarray | None = None,
                    calibrated_threshold: float | None = None) -> Dict[str, Any]:
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
        "accuracy_at_fixed_threshold": float(accuracy_score(y_true, y_pred)),
        "decision_threshold": float(threshold),
        "metric_threshold_policy": "fixed_comparison",
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
                out["accuracy_at_eer_oracle"] = float(
                    accuracy_score(y_true, (p_fake >= eer_thr).astype(int))
                )
            else:
                out["accuracy_at_eer_oracle"] = float("nan")
        except Exception:
            out["eer"] = float("nan")
            out["eer_threshold"] = float("nan")
            out["accuracy_at_eer_oracle"] = float("nan")
        try:
            tdcf, _ = mc.calculate_min_tdcf(y_true, p_fake)
            out["min_tdcf"] = float(tdcf)
        except Exception:
            out["min_tdcf"] = float("nan")
    else:
        out["auc_roc"] = float("nan")
        out["eer"] = float("nan")
        out["eer_threshold"] = float("nan")
        out["accuracy_at_eer_oracle"] = float("nan")
        out["min_tdcf"] = float("nan")

    # Compatibilidade: este limiar é derivado do próprio conjunto avaliado e,
    # portanto, é um teto/oráculo, não o threshold operacional de validação.
    out["accuracy_at_eer"] = out.get("accuracy_at_eer_oracle", float("nan"))
    if calibrated_threshold is not None and np.isfinite(calibrated_threshold):
        out["calibrated_threshold"] = float(calibrated_threshold)
        out["accuracy_at_calibrated_threshold"] = float(
            accuracy_score(y_true, (p_fake >= calibrated_threshold).astype(int))
        )
    if n_bootstrap and n_pos > 0 and n_neg > 0:
        out.update(
            _bootstrap_cis(
                y_true,
                p_fake,
                threshold,
                n_bootstrap,
                cluster_ids=cluster_ids,
            )
        )

    return out


def evaluate_grouped_scores(
    y_true: np.ndarray,
    p_fake: np.ndarray,
    groups: np.ndarray,
    threshold: float = 0.5,
) -> Dict[str, Any]:
    """Report per-domain, macro and worst-group metrics without pooling bias."""
    y_true = np.asarray(y_true).ravel().astype(int)
    p_fake = np.asarray(p_fake, dtype="float64").ravel()
    groups = np.asarray(groups).astype(str).ravel()
    if not (len(y_true) == len(p_fake) == len(groups)):
        raise ValueError("y_true, p_fake e groups devem estar alinhados")

    per_group: Dict[str, Dict[str, Any]] = {}
    for group in sorted(set(groups.tolist())):
        mask = groups == group
        per_group[group] = evaluate_scores(
            y_true[mask], p_fake[mask], threshold=threshold, n_bootstrap=0
        )

    def _finite_values(metric: str) -> list[float]:
        values = []
        for result in per_group.values():
            value = result.get(metric)
            if value is not None and np.isfinite(value):
                values.append(float(value))
        return values

    accuracies = _finite_values("accuracy")
    eers = _finite_values("eer")
    aucs = _finite_values("auc_roc")
    return {
        "per_group": per_group,
        "n_groups": len(per_group),
        "macro_accuracy": float(np.mean(accuracies)) if accuracies else float("nan"),
        "macro_eer": float(np.mean(eers)) if eers else float("nan"),
        "macro_auc_roc": float(np.mean(aucs)) if aucs else float("nan"),
        "worst_group_accuracy": min(accuracies) if accuracies else float("nan"),
    }
