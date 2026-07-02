#!/usr/bin/env python3
"""Exporta a importância de características do Random Forest do benchmark.

Lê o artefato treinado ``app/models/bench_randomforest.pkl`` (sem retreinar),
extrai ``feature_importances_`` e gera figura + tabela LaTeX para o TCC.
Os nomes seguem a ordem de construção do vetor tabular de 63 descritores em
``benchmarks/data.py::_to_tabular_features`` (11 temporais + 26 MFCC +
26 RASTA-PLP).

Uso:
    python scripts/export_rf_feature_importance.py \
        --out-figure tcc_overleaf/figures/rf_feature_importance.png \
        --top-k 15
"""

from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def build_feature_names() -> list[str]:
    """Nomes na ordem exata de benchmarks/data.py::_to_tabular_features."""
    temporal = [
        "média",
        "desvio-padrão",
        "média |x|",
        "RMS",
        "mínimo",
        "máximo",
        "percentil 25",
        "percentil 50",
        "percentil 75",
        "energia da diferença",
        "ZCR",
    ]
    mfcc = [f"MFCC{i+1} (média)" for i in range(13)] + [
        f"MFCC{i+1} (desvio)" for i in range(13)
    ]
    rasta = [f"RASTA-PLP{i+1} (média)" for i in range(13)] + [
        f"RASTA-PLP{i+1} (desvio)" for i in range(13)
    ]
    return temporal + mfcc + rasta


def extract_estimator(obj):
    """Resolve o RandomForestClassifier dentro do artefato (pipeline/dict)."""
    if hasattr(obj, "feature_importances_"):
        return obj
    if hasattr(obj, "best_estimator_"):
        return extract_estimator(obj.best_estimator_)
    if hasattr(obj, "named_steps"):
        for step in reversed(list(obj.named_steps.values())):
            found = extract_estimator(step)
            if found is not None:
                return found
    if isinstance(obj, dict):
        for value in obj.values():
            found = extract_estimator(value)
            if found is not None:
                return found
    return None


def group_of(name: str) -> str:
    if name.startswith("MFCC"):
        return "MFCC"
    if name.startswith("RASTA"):
        return "RASTA-PLP"
    return "Temporal"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", default="app/models/bench_randomforest.pkl")
    parser.add_argument(
        "--out-figure", default="tcc_overleaf/figures/rf_feature_importance.png"
    )
    parser.add_argument("--top-k", type=int, default=15)
    args = parser.parse_args()

    artifact = ROOT / args.artifact
    try:
        import joblib

        obj = joblib.load(artifact)
    except Exception:
        with artifact.open("rb") as f:
            obj = pickle.load(f)  # nosec B301 — artefato local do benchmark
    est = extract_estimator(obj)
    if est is None:
        raise SystemExit(f"feature_importances_ não encontrado em {artifact}")

    importances = np.asarray(est.feature_importances_, dtype="float64")
    names = build_feature_names()
    if len(importances) != len(names):
        raise SystemExit(
            f"esperava {len(names)} features, artefato tem {len(importances)} — "
            "a ordem/nomes de benchmarks/data.py mudou?"
        )

    # Agregado por família (soma das importâncias)
    by_group: dict[str, float] = {}
    for name, imp in zip(names, importances):
        by_group[group_of(name)] = by_group.get(group_of(name), 0.0) + float(imp)
    print("Importância agregada por família:")
    for group, total in sorted(by_group.items(), key=lambda kv: -kv[1]):
        print(f"  {group:10s} {total*100:5.1f}%")

    order = np.argsort(importances)[::-1][: args.top_k]
    print(f"\nTop-{args.top_k} características:")
    for rank, idx in enumerate(order, 1):
        print(f"  {rank:2d}. {names[idx]:24s} {importances[idx]*100:5.2f}%")

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = {
        "Temporal": "#2980b9",
        "MFCC": "#27ae60",
        "RASTA-PLP": "#e67e22",
    }
    fig, ax = plt.subplots(figsize=(9, 5.2), dpi=200)
    top_names = [names[i] for i in order][::-1]
    top_vals = importances[order][::-1] * 100
    bar_colors = [colors[group_of(n)] for n in top_names]
    ax.barh(top_names, top_vals, color=bar_colors)
    ax.set_xlabel("Importância de Gini (%)")
    ax.set_title(
        f"Random Forest — top-{args.top_k} características "
        "(vetor tabular de 63 descritores)"
    )
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in colors.values()]
    ax.legend(handles, colors.keys(), loc="lower right", title="Família")
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    out_figure = ROOT / args.out_figure
    out_figure.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_figure)
    print(f"\nFigura: {out_figure}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
