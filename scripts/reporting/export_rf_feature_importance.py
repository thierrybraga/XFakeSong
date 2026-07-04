#!/usr/bin/env python3
"""Exporta a importância de características do Random Forest do benchmark.

Lê o artefato treinado ``app/models/bench_randomforest.pkl`` (sem retreinar),
extrai ``feature_importances_`` e gera figura + tabela LaTeX para o TCC.
Os nomes e a extração do estimador vêm da fonte única em
``app/core/xai/tabular.py`` (contrato de 63 descritores: 11 temporais +
26 MFCC + 26 RASTA-PLP, na ordem de ``benchmarks/data.py``).

Uso:
    python scripts/reporting/export_rf_feature_importance.py \
        --out-figure tcc_overleaf/figures/rf_feature_importance.png \
        --top-k 15
"""

from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.core.xai import tabular as _tabular  # noqa: E402  — fonte única

extract_estimator = _tabular.extract_sklearn_estimator
group_of = _tabular.feature_group
build_feature_names = _tabular.tabular_feature_names


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
