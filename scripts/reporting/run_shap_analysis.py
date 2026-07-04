#!/usr/bin/env python3
"""Análise XAI do benchmark: SHAP (clássicos) + Grad-CAM (redes espectrais).

Materializa a etapa de explicabilidade prevista no projeto usando os
artefatos já treinados (``app/models/bench_*``), sem retreinar nada:

1. **SHAP** — Random Forest via ``TreeExplainer`` (exato) e SVM via
   ``KernelExplainer`` (agnóstico, custo controlado por k-means no
   background), ambos sobre o vetor tabular de 63 descritores
   (``app/core/xai/tabular.py``). Gera *beeswarm*, barras de importância
   média |SHAP| e CSV consolidado.
2. **Grad-CAM** — mapas de ativação das redes espectrais Keras (Res2Net,
   Conformer, CCT, AST) sobre espectrogramas Mel do conjunto de teste,
   com sobreposição espectrograma × heatmap por amostra.

As divisões usam a mesma semente do benchmark (42), portanto as amostras
explicadas pertencem ao MESMO conjunto de teste das métricas reportadas.

Uso:
    # análise completa com o dataset canônico:
    python scripts/reporting/run_shap_analysis.py \
        --dataset app/datasets/benchmark_audio_raw_balanced_15k.npz

    # smoke rápido sem dataset real (dados sintéticos, sem artefatos):
    python scripts/reporting/run_shap_analysis.py --synthetic --skip-gradcam

    # somente Grad-CAM de um subconjunto de arquiteturas:
    python scripts/reporting/run_shap_analysis.py --skip-shap --archs res2net conformer

Saídas (em --out, padrão results/xai/):
    shap_summary_random_forest.png, shap_bar_random_forest.png,
    shap_summary_svm.png, shap_bar_svm.png, shap_mean_abs.csv,
    gradcam_<arch>.png, xai_report.md
"""

from __future__ import annotations

import argparse
import csv
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts._bootstrap import setup_logging  # noqa: E402

logger = logging.getLogger("xai")

# Artefato Keras e nome de contrato de entrada por arquitetura espectral.
# Artefatos: preferimos os PROMOVIDOS em app/models/benchmark_final/<arch>/
# (fonte das metricas do TCC); app/models/bench_* fica como fallback.
SPECTRAL_ARCHS: dict[str, dict[str, object]] = {
    "res2net": {
        "artifacts": [
            "app/models/benchmark_final/res2net/bench_multiscalecnn.keras",
            "app/models/bench_multiscalecnn.keras",
        ],
        "contract": "multiscalecnn",
        "label": "Res2Net",
    },
    "conformer": {
        "artifacts": [
            "app/models/benchmark_final/conformer/bench_conformer.keras",
            "app/models/bench_conformer.keras",
        ],
        "contract": "conformer",
        "label": "Conformer",
    },
    "cct": {
        "artifacts": [
            "app/models/benchmark_final/cct/bench_hybrid_cnn_transformer.keras",
            "app/models/bench_hybrid_cnn_transformer.keras",
        ],
        "contract": "hybridcnntransformer",
        "label": "CCT",
    },
    "ast": {
        "artifacts": [
            "app/models/benchmark_final/ast/bench_spectrogramtransformer.keras",
            "app/models/bench_spectrogramtransformer.keras",
        ],
        "contract": "spectrogramtransformer",
        "label": "AST",
    },
}

CLASSICAL_ARTIFACTS = {
    "random_forest": [
        "app/models/benchmark_final/random_forest/bench_randomforest.pkl",
        "app/models/bench_randomforest.pkl",
    ],
    "svm": [
        "app/models/benchmark_final/svm/bench_svm.pkl",
        "app/models/bench_svm.pkl",
    ],
}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--dataset",
        default="app/datasets/benchmark_audio_raw_balanced_15k.npz",
        help="NPZ canônico de áudio bruto do benchmark.",
    )
    parser.add_argument(
        "--synthetic",
        action="store_true",
        help="Usa dados sintéticos (smoke) em vez do NPZ; requer modelos "
        "compatíveis ou --skip-gradcam.",
    )
    parser.add_argument("--out", default="results/xai", help="Diretório de saída.")
    parser.add_argument("--seed", type=int, default=42, help="Semente do split.")
    parser.add_argument(
        "--archs",
        nargs="+",
        choices=sorted(SPECTRAL_ARCHS),
        default=sorted(SPECTRAL_ARCHS),
        help="Arquiteturas espectrais para o Grad-CAM.",
    )
    parser.add_argument("--skip-shap", action="store_true")
    parser.add_argument("--skip-gradcam", action="store_true")
    parser.add_argument(
        "--shap-explain", type=int, default=100,
        help="Amostras de teste explicadas no Random Forest.",
    )
    parser.add_argument(
        "--svm-explain", type=int, default=40,
        help="Amostras explicadas no SVM (KernelExplainer é caro).",
    )
    parser.add_argument(
        "--background", type=int, default=200,
        help="Amostras de treino usadas como background do SHAP.",
    )
    parser.add_argument(
        "--gradcam-samples", type=int, default=6,
        help="Amostras de teste por arquitetura no Grad-CAM.",
    )
    parser.add_argument(
        "--copy-to-tcc", action="store_true",
        help="Copia as figuras geradas para tcc_overleaf/figures/.",
    )
    parser.add_argument(
        "--max-samples", type=int, default=2000,
        help="Subamostra estratificada (semente fixa) aplicada antes da "
        "extração de características, para custo tratável; 0 desativa.",
    )
    return parser.parse_args(argv)


def _subsample(data, max_samples: int, seed: int):
    """Subamostra balanceada e determinística de um BenchmarkData.

    A extração de características (tabular/espectrograma) é aplicada ao
    conjunto inteiro pelo ``prepare_for_architecture``; limitar o número de
    amostras mantém a análise em minutos. O protocolo de divisão (mesma
    semente do benchmark) é preservado dentro do subconjunto — registrado
    como ressalva no relatório.
    """
    import copy

    if max_samples <= 0 or len(data.y) <= max_samples:
        return data, False
    idx = _balanced_indices(np.asarray(data.y), max_samples, seed)
    sub = copy.copy(data)
    for field in ("X", "y", "groups", "speakers"):
        value = getattr(data, field, None)
        if value is not None:
            setattr(sub, field, np.asarray(value)[idx])
    logger.info(
        "Subamostra estratificada: %d → %d amostras (semente %d)",
        len(data.y), len(sub.y), seed,
    )
    return sub, True


def _load_benchmark_data(args: argparse.Namespace):
    from benchmarks.data import BenchmarkData

    if args.synthetic:
        logger.warning("Modo sintético: resultados servem apenas como smoke.")
        return BenchmarkData.synthetic(n=240)
    dataset = ROOT / args.dataset
    if not dataset.exists():
        raise SystemExit(f"Dataset não encontrado: {dataset}")
    return BenchmarkData.from_npz(str(dataset))


def _first_existing(candidates: list[str]):
    for rel_path in candidates:
        path = ROOT / rel_path
        if path.exists():
            return path
    return None


def _load_classical_artifact(candidates: list[str]):
    import joblib

    path = _first_existing(candidates)
    if path is None:
        return None, None
    try:
        return joblib.load(path), path
    except Exception:  # noqa: BLE001 - fallback p/ pickles antigos
        import pickle

        with path.open("rb") as handle:
            return pickle.load(handle), path  # nosec B301 — artefato local


def _balanced_indices(y: np.ndarray, n: int, seed: int) -> np.ndarray:
    """Seleciona até ``n`` índices balanceados entre classes, determinístico."""
    rng = np.random.default_rng(seed)
    picked: list[int] = []
    per_class = max(1, n // 2)
    for cls in (0, 1):
        idx = np.flatnonzero(y == cls)
        rng.shuffle(idx)
        picked.extend(idx[:per_class].tolist())
    return np.asarray(sorted(picked[:n]), dtype=int)


def run_shap(args: argparse.Namespace, data, out_dir: Path) -> list[str]:
    """Executa SHAP para RF e SVM; retorna nomes dos arquivos gerados."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import shap

    from app.core.xai import (
        explain_with_kernel_shap,
        explain_with_tree_shap,
        split_sklearn_pipeline,
        tabular_feature_names,
    )

    tabular = data.prepare_for_architecture("randomforest")
    X_train, y_train, X_val, y_val, X_test, y_test = tabular.stratified_split(
        seed=args.seed
    )
    del X_val, y_val
    names = tabular_feature_names()
    if X_test.shape[1] != len(names):
        raise SystemExit(
            f"Vetor tabular com {X_test.shape[1]} features (esperado "
            f"{len(names)}) — contrato de benchmarks/data.py mudou?"
        )

    bg_idx = _balanced_indices(y_train, args.background, args.seed)
    generated: list[str] = []
    mean_abs: dict[str, np.ndarray] = {}

    for model_key, candidates in CLASSICAL_ARTIFACTS.items():
        artifact, used_path = _load_classical_artifact(candidates)
        if artifact is None:
            logger.warning("Artefato ausente, pulando %s (%s)", model_key, candidates)
            continue
        logger.info("SHAP %s: artefato %s", model_key, used_path)
        transform, estimator = split_sklearn_pipeline(artifact)

        n_explain = args.shap_explain if model_key == "random_forest" else args.svm_explain
        ex_idx = _balanced_indices(y_test, n_explain, args.seed + 1)
        X_explain = X_test[ex_idx]

        logger.info("SHAP %s: explicando %d amostras…", model_key, len(ex_idx))
        try:
            if model_key == "random_forest":
                matrix = explain_with_tree_shap(estimator, transform(X_explain))
            else:
                def predict(X, _est=estimator, _tr=transform):
                    return _est.predict_proba(_tr(X))[:, 1]

                matrix = explain_with_kernel_shap(
                    predict, X_train[bg_idx], X_explain,
                )
        except Exception as exc:  # noqa: BLE001 - segue p/ os demais modelos
            logger.error("SHAP falhou para %s (%s): %s", model_key, used_path, exc)
            continue
        mean_abs[model_key] = np.abs(matrix).mean(axis=0)

        for kind in ("summary", "bar"):
            plt.figure()
            shap.summary_plot(
                matrix,
                features=X_explain,
                feature_names=names,
                plot_type="dot" if kind == "summary" else "bar",
                max_display=20,
                show=False,
            )
            plt.title(f"SHAP — {model_key.replace('_', ' ').title()} (classe spoof)")
            fname = f"shap_{kind}_{model_key}.png"
            plt.tight_layout()
            plt.savefig(out_dir / fname, dpi=180)
            plt.close("all")
            generated.append(fname)

    if mean_abs:
        csv_path = out_dir / "shap_mean_abs.csv"
        with csv_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(["feature", *mean_abs.keys()])
            for i, name in enumerate(names):
                writer.writerow(
                    [name, *(f"{mean_abs[m][i]:.6f}" for m in mean_abs)]
                )
        generated.append(csv_path.name)
    return generated


def _load_keras_model(rel_path: str):
    """Carrega um artefato Keras registrando as camadas customizadas."""
    import tensorflow as tf

    try:
        from app.domain.services.detection.model_loader import (
            _load_custom_architecture_modules,
        )

        _load_custom_architecture_modules()
    except Exception as exc:  # noqa: BLE001 - registro é melhor-esforço
        logger.debug("Registro de camadas customizadas incompleto: %s", exc)

    path = ROOT / rel_path
    if not path.exists():
        return None
    return tf.keras.models.load_model(
        str(path),
        custom_objects=tf.keras.utils.get_custom_objects(),
        safe_mode=False,
        compile=False,
    )


def _gradcam_sample_indices(
    model, X_test: np.ndarray, y_test: np.ndarray, n: int, seed: int
) -> np.ndarray:
    """Amostras de teste balanceadas, priorizando as menos saturadas.

    Modelos quase perfeitos concentram ``p_fake`` em 0/1, onde o gradiente
    (mesmo no espaço de logit) pode sofrer underflow em float32. Prioriza,
    por classe, as amostras com pontuação mais próxima do limiar.
    """
    pool = min(len(y_test), 256)
    rng = np.random.default_rng(seed)
    pool_idx = rng.permutation(len(y_test))[:pool]
    preds = np.asarray(
        model.predict(X_test[pool_idx].astype("float32"), verbose=0)
    ).reshape(len(pool_idx), -1)[:, -1]
    margin = np.abs(preds - 0.5)  # menor = menos saturada

    picked: list[int] = []
    per_class = max(1, n // 2)
    for cls in (0, 1):
        cls_positions = np.flatnonzero(y_test[pool_idx] == cls)
        ordered = cls_positions[np.argsort(margin[cls_positions])]
        picked.extend(pool_idx[ordered[:per_class]].tolist())
    return np.asarray(sorted(picked[:n]), dtype=int)


def run_gradcam(args: argparse.Namespace, data, out_dir: Path) -> list[str]:
    """Gera grades espectrograma×heatmap por arquitetura espectral."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from app.core.xai import compute_gradcam_auto, heatmap_to_input_grid

    generated: list[str] = []
    for arch in args.archs:
        spec = SPECTRAL_ARCHS[arch]
        artifact_path = _first_existing(list(spec["artifacts"]))
        if artifact_path is None:
            logger.warning("Artefato ausente, pulando %s (%s)", arch, spec["artifacts"])
            continue
        logger.info("Grad-CAM %s: artefato %s", arch, artifact_path)
        model = _load_keras_model(str(artifact_path.relative_to(ROOT)))
        if model is None:
            continue

        prepared = data.prepare_for_architecture(spec["contract"])
        _, _, _, _, X_test, y_test = prepared.stratified_split(seed=args.seed)
        idx = _gradcam_sample_indices(
            model, X_test, y_test, args.gradcam_samples, args.seed + 2
        )
        batch = X_test[idx].astype("float32")
        labels = y_test[idx]

        try:
            heat, layer = compute_gradcam_auto(model, batch)
        except (ValueError, RuntimeError) as exc:
            logger.error("Grad-CAM indisponível para %s: %s", arch, exc)
            continue
        heat = heatmap_to_input_grid(heat, batch.shape[1], batch.shape[2])
        p_fake = np.asarray(model.predict(batch, verbose=0)).reshape(len(idx), -1)
        p_fake = p_fake[:, -1]

        n = len(idx)
        fig, axes = plt.subplots(n, 2, figsize=(9.0, 2.2 * n), squeeze=False)
        for row in range(n):
            spec_img = batch[row, :, :, 0] if batch.ndim == 4 else batch[row]
            axes[row][0].imshow(spec_img, aspect="auto", origin="lower")
            axes[row][0].set_title(
                f"{'spoof' if labels[row] else 'bonafide'} — $p_{{fake}}$="
                f"{p_fake[row]:.2f}", fontsize=9,
            )
            axes[row][1].imshow(spec_img, aspect="auto", origin="lower")
            axes[row][1].imshow(
                heat[row], aspect="auto", origin="lower", cmap="jet", alpha=0.45
            )
            axes[row][1].set_title(f"Grad-CAM ({layer})", fontsize=9)
            for ax in axes[row]:
                ax.set_xticks([])
                ax.set_yticks([])
        fig.suptitle(f"Mapas de ativação — {spec['label']}")
        fig.tight_layout()
        fname = f"gradcam_{arch}.png"
        fig.savefig(out_dir / fname, dpi=180)
        plt.close(fig)
        generated.append(fname)
        logger.info("Grad-CAM %s: %s", arch, fname)
    return generated


def write_report(
    args: argparse.Namespace, out_dir: Path, shap_files: list[str],
    gradcam_files: list[str],
) -> None:
    lines = [
        "# Relatório XAI — SHAP + Grad-CAM",
        "",
        f"- Gerado em: {datetime.now(timezone.utc).isoformat(timespec='seconds')}",
        f"- Dataset: {'sintético (smoke)' if args.synthetic else args.dataset}",
        f"- Semente do split: {args.seed} (mesma do benchmark)",
        f"- Subamostragem estratificada: "
        f"{f'sim (máx. {args.max_samples} amostras)' if getattr(args, '_subsampled', False) else 'não'}",
        "",
        "## Arquivos gerados",
        "",
        *[f"- `{name}`" for name in (*shap_files, *gradcam_files)],
        "",
        "## Interpretação e ressalvas",
        "",
        "- Os valores SHAP explicam a pontuação da classe *spoof* no espaço "
        "tabular de 63 descritores; para o SVM, o `KernelExplainer` é uma "
        "aproximação amostral (background k-means), não um valor exato.",
        "- Nos modelos em que a última camada 4D é a tokenização "
        "convolucional (CCT/AST), o Grad-CAM reflete a atenção espacial na "
        "entrada dos blocos de atenção, não das camadas profundas.",
        "- As explicações herdam o escopo *in-domain* do benchmark "
        "(Seção de vazamento de domínio do TCC): padrões salientes podem "
        "incluir atalhos de fonte/canal, e não apenas artefatos de síntese.",
    ]
    (out_dir / "xai_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    setup_logging(name="xai")
    logging.getLogger("shap").setLevel(logging.WARNING)
    out_dir = ROOT / args.out
    out_dir.mkdir(parents=True, exist_ok=True)

    if not args.skip_shap:
        from app.core.xai import shap_available

        if not shap_available():
            raise SystemExit(
                "shap não instalado — use --skip-shap ou instale via "
                "requirements-dev.txt"
            )

    data = _load_benchmark_data(args)
    data, subsampled = _subsample(data, args.max_samples, args.seed)
    args._subsampled = subsampled
    shap_files: list[str] = []
    gradcam_files: list[str] = []
    if not args.skip_shap:
        shap_files = run_shap(args, data, out_dir)
    if not args.skip_gradcam:
        gradcam_files = run_gradcam(args, data, out_dir)

    write_report(args, out_dir, shap_files, gradcam_files)
    if args.copy_to_tcc:
        import shutil

        figures_dir = ROOT / "tcc_overleaf" / "figures"
        figures_dir.mkdir(parents=True, exist_ok=True)
        for name in (*shap_files, *gradcam_files):
            if name.endswith(".png"):
                shutil.copy2(out_dir / name, figures_dir / name)
        logger.info("Figuras copiadas para %s", figures_dir)

    logger.info("Relatório: %s", out_dir / "xai_report.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
