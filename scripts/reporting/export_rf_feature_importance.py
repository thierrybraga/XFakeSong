#!/usr/bin/env python3
"""Gera importância por permutação do Random Forest no teste canônico.

A redução média de impureza (MDI/Gini) não é usada na análise confirmatória.
O script reaplica o frontend tabular canônico (63 no v1, 183 no v2 com LFCC)
à partição de teste, calcula a queda de acurácia após permutações repetidas e salva figura e
CSV com média e desvio-padrão. No artigo vigente, esta é uma análise
exploratória: o frontend v2 foi definido após observar uma rodada de teste.
"""

from __future__ import annotations

import argparse
import csv
import pickle
import sys
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.domain.xai import tabular as _tabular  # noqa: E402
from app.domain.xai.tabular import N_FEATURES, N_FEATURES_V2  # noqa: E402
from benchmarks.data import prepare_input_for_architecture  # noqa: E402

# Os nomes saem da LARGURA do vetor preparado: `tabular_feature_names` fixo em
# 63 desalinharia os rótulos de um artefato v2 (183) sem erro nenhum.
build_feature_names = _tabular.feature_names_for_width
group_of = _tabular.feature_group


def _load_artifact(path: Path) -> Any:
    try:
        import joblib

        return joblib.load(path)
    except Exception:
        with path.open("rb") as stream:
            return pickle.load(stream)  # nosec B301 -- artefato local confiável


def _predictor(obj: Any) -> Any:
    if hasattr(obj, "predict"):
        return obj
    if isinstance(obj, dict):
        for key in ("pipeline", "model", "estimator", "classifier"):
            candidate = obj.get(key)
            if hasattr(candidate, "predict"):
                return candidate
    raise SystemExit("artefato não expõe um estimador/pipeline com predict()")


def _load_test(npz_path: Path) -> tuple[np.ndarray, np.ndarray]:
    with np.load(npz_path, allow_pickle=False) as data:
        if "X_test" not in data or "y_test" not in data:
            raise SystemExit("NPZ deve conter X_test e y_test materializados")
        X = np.asarray(data["X_test"], dtype="float32")
        y = np.asarray(data["y_test"]).reshape(-1).astype("int64")
    features, kind = prepare_input_for_architecture(X, "RandomForest")
    # A largura sai do front-end vigente (v1=63, v2=183), não de um literal: o
    # 63 fixo recusaria qualquer artefato treinado com o vetor v2.
    if kind != "tabular_audio_features" or features.shape[1] not in (
        N_FEATURES,
        N_FEATURES_V2,
    ):
        raise SystemExit(f"frontend inesperado: {kind}, shape={features.shape}")
    return features, y


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    # Defaults alinhados ao run que alimenta o artigo (2026-08-14).
    # O slug era `random_forest`, mas a promoção grava `randomforest`; e o
    # dataset apontava para `benchmark_audio_raw_balanced_15k.npz`, o corpus com
    # atalho de fonte de 87,6% descartado em 2026-08-09 e já apagado do disco.
    # Com os dois defaults quebrados, o script não rodava sem argumentos.
    parser.add_argument(
        "--artifact",
        default="data/models/benchmark_final/randomforest/bench_randomforest.pkl",
    )
    parser.add_argument(
        "--dataset", default="data/datasets/benchmark_dataset_15k.npz"
    )
    parser.add_argument(
        "--out-figure", default="data/results/paper/figures/rf_feature_importance.png"
    )
    parser.add_argument(
        "--out-csv",
        default="data/results/paper/consolidated/rf_permutation_importance.csv",
    )
    parser.add_argument("--top-k", type=int, default=15)
    parser.add_argument("--n-repeats", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--reuse-csv",
        action="store_true",
        help="redesenha a figura a partir do CSV existente, sem recomputar permutações",
    )
    parser.add_argument(
        "--acknowledge-exploratory-test-use",
        action="store_true",
        help=(
            "confirma ciência de que o frontend tabular v2 foi escolhido após "
            "uma observação da partição de teste"
        ),
    )
    args = parser.parse_args()
    if not args.acknowledge_exploratory_test_use:
        raise SystemExit(
            "use --acknowledge-exploratory-test-use e reporte a análise como "
            "exploratória, não confirmatória"
        )

    out_csv = (ROOT / args.out_csv).resolve()
    if args.reuse_csv:
        if not out_csv.is_file():
            raise SystemExit(f"CSV inexistente para --reuse-csv: {out_csv}")
        with out_csv.open(newline="", encoding="utf-8") as stream:
            cached = list(csv.DictReader(stream))
        names = [row["feature"] for row in cached]
        means = np.asarray(
            [float(row["importance_mean"]) for row in cached], dtype="float64"
        )
        stds = np.asarray(
            [float(row["importance_std"]) for row in cached], dtype="float64"
        )
        order = np.argsort(means)[::-1]
    else:
        from sklearn.inspection import permutation_importance

        artifact = (ROOT / args.artifact).resolve()
        dataset = (ROOT / args.dataset).resolve()
        estimator = _predictor(_load_artifact(artifact))
        X_test, y_test = _load_test(dataset)
        result = permutation_importance(
            estimator,
            X_test,
            y_test,
            scoring="accuracy",
            n_repeats=args.n_repeats,
            random_state=args.seed,
            n_jobs=-1,
        )

        names = build_feature_names(X_test.shape[1])
        if len(names) != X_test.shape[1]:
            raise SystemExit(f"nomes={len(names)}; colunas={X_test.shape[1]}")
        means = np.asarray(result.importances_mean, dtype="float64")
        stds = np.asarray(result.importances_std, dtype="float64")
        order = np.argsort(means)[::-1]

        out_csv.parent.mkdir(parents=True, exist_ok=True)
        with out_csv.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.writer(stream)
            writer.writerow(["feature", "group", "importance_mean", "importance_std"])
            for idx in order:
                writer.writerow(
                    [names[idx], group_of(names[idx]), means[idx], stds[idx]]
                )

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    selected = order[: args.top_k][::-1]
    # "LFCC" cobre também ΔLFCC/ΔΔLFCC (`feature_group` agrupa as três): sem a
    # entrada, o `colors[group_of(name)]` abaixo estouraria com KeyError em
    # qualquer artefato do vetor v2.
    colors = {
        "Temporal": "#2980b9",
        "MFCC": "#27ae60",
        "RASTA-PLP": "#e67e22",
        "LFCC": "#8e44ad",
    }
    fig, ax = plt.subplots(figsize=(6.8, 4.7))
    labels = [names[i] for i in selected]
    display_labels = [
        f"amplitude — {name}" if group_of(name) == "Temporal" else name
        for name in labels
    ]
    values = means[selected] * 100
    errors = stds[selected] * 100
    ax.barh(
        display_labels,
        values,
        xerr=errors,
        color=[colors[group_of(name)] for name in labels],
        capsize=2,
    )
    ax.set_xlabel("Queda de acurácia após permutação (p.p.)")
    ax.set_title(
        f"Random Forest — importância por permutação ({args.n_repeats} repetições)",
        fontsize=10.5,
    )
    handles = [plt.Rectangle((0, 0), 1, 1, color=color) for color in colors.values()]
    ax.legend(handles, colors.keys(), loc="lower right", title="Família")
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    out_figure = (ROOT / args.out_figure).resolve()
    out_figure.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_figure, dpi=300, bbox_inches="tight")
    print(f"Figura: {out_figure}")
    print(f"Tabela: {out_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
