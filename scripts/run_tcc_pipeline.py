#!/usr/bin/env python3
"""Automação ponta a ponta do pipeline do TCC.

Executa, de forma reprodutível:
1. download/balanceamento do dataset;
2. validação, normalização e splits;
3. exportação dos WAVs para um `.npz` canônico em áudio bruto;
4. benchmark com treinamento, inferência, métricas e relatório.

Exemplos:
  python scripts/run_tcc_pipeline.py --smoke
  python scripts/run_tcc_pipeline.py --smoke --model SVM
  python scripts/run_tcc_pipeline.py --skip-download --target-per-class 500 --archs SVM RandomForest
  python scripts/run_tcc_pipeline.py --skip-download --skip-preprocess --model AASIST
  python scripts/run_tcc_pipeline.py --skip-download --skip-preprocess --archs SVM RandomForest
  python scripts/run_tcc_pipeline.py --tcc-full-dataset --download --full-benchmark
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

BASE_DIR = Path(__file__).resolve().parent.parent
SCRIPTS_DIR = BASE_DIR / "scripts"
DATASETS_DIR = BASE_DIR / "app" / "datasets"
SPLITS_DIR = DATASETS_DIR / "splits"

LOGGER = logging.getLogger("TCCPipeline")


def _run(cmd: list[str], description: str, log_path: Path) -> None:
    LOGGER.info("=" * 72)
    LOGGER.info(description)
    LOGGER.info("Comando: %s", " ".join(str(c) for c in cmd))
    LOGGER.info("=" * 72)
    with log_path.open("a", encoding="utf-8") as log:
        log.write(f"\n\n### {description}\n")
        log.write(" ".join(str(c) for c in cmd) + "\n")
        result = subprocess.run(
            cmd,
            cwd=str(BASE_DIR),
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
        )
    if result.returncode != 0:
        raise RuntimeError(
            f"Etapa falhou ({result.returncode}): {description}. "
            f"Veja o log em {log_path}"
        )


def _split_counts(splits_dir: Path) -> dict[str, dict[str, int]]:
    counts = {}
    for split in ("train", "val", "test"):
        counts[split] = {
            "real": len(list((splits_dir / split / "real").glob("*.wav"))),
            "fake": len(list((splits_dir / split / "fake").glob("*.wav"))),
        }
    return counts


def _load_wav(path: Path, sample_rate: int, samples: int) -> np.ndarray:
    import librosa

    y, _ = librosa.load(str(path), sr=sample_rate, mono=True)
    y = np.asarray(y, dtype="float32")
    if not np.all(np.isfinite(y)):
        y = np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)
    peak = float(np.max(np.abs(y))) if y.size else 0.0
    if peak > 1e-6:
        y = y / peak
    if len(y) >= samples:
        return y[:samples].astype("float32")
    return np.pad(y, (0, samples - len(y))).astype("float32")


def _collect_split(
    split_dir: Path,
    sample_rate: int,
    duration_sec: float,
    max_per_class: int | None,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    samples = int(sample_rate * duration_sec)
    X, y, paths = [], [], []
    for label, cls in ((0, "real"), (1, "fake")):
        files = sorted((split_dir / cls).glob("*.wav"))
        if max_per_class is not None:
            files = files[:max_per_class]
        for wav in files:
            try:
                X.append(_load_wav(wav, sample_rate, samples))
                y.append(label)
                paths.append(str(wav.relative_to(BASE_DIR)))
            except Exception as exc:  # noqa: BLE001
                LOGGER.warning("Falha ao carregar %s: %s", wav, exc)
    if not X:
        return np.empty((0, samples, 1), dtype="float32"), np.empty((0,), dtype="int64"), []
    return (
        np.asarray(X, dtype="float32")[..., np.newaxis],
        np.asarray(y, dtype="int64"),
        paths,
    )


def export_npz_from_splits(
    splits_dir: Path,
    out_npz: Path,
    sample_rate: int,
    duration_sec: float,
    max_per_class: int | None,
) -> Path:
    out_npz.parent.mkdir(parents=True, exist_ok=True)
    arrays = {}
    meta = {
        "source": str(splits_dir.relative_to(BASE_DIR)),
        "sample_rate": sample_rate,
        "duration_sec": duration_sec,
        "format": "raw_audio",
        "splits": {},
    }
    for split in ("train", "val", "test"):
        X, y, paths = _collect_split(
            splits_dir / split,
            sample_rate=sample_rate,
            duration_sec=duration_sec,
            max_per_class=max_per_class,
        )
        if len(y) == 0:
            raise RuntimeError(f"Split vazio: {splits_dir / split}")
        arrays[f"X_{split}"] = X
        arrays[f"y_{split}"] = y
        meta["splits"][split] = {
            "samples": int(len(y)),
            "real": int(np.sum(y == 0)),
            "fake": int(np.sum(y == 1)),
            "paths": paths,
        }
        LOGGER.info(
            "%s: %d amostras (%d real + %d fake), shape=%s",
            split,
            len(y),
            int(np.sum(y == 0)),
            int(np.sum(y == 1)),
            list(X.shape[1:]),
        )
    arrays["metadata_json"] = np.asarray(json.dumps(meta, ensure_ascii=False))
    np.savez_compressed(out_npz, **arrays)
    LOGGER.info("NPZ exportado: %s", out_npz)
    return out_npz


def create_synthetic_audio_npz(out_npz: Path, n_per_class: int, sample_rate: int) -> Path:
    out_npz.parent.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(42)
    duration = 1.0
    samples = int(sample_rate * duration)
    t = np.linspace(0, duration, samples, endpoint=False, dtype="float32")

    X, y = [], []
    for idx in range(n_per_class):
        freq = 220 + (idx % 5) * 20
        X.append(0.6 * np.sin(2 * np.pi * freq * t) + 0.02 * rng.standard_normal(samples))
        y.append(0)
        fake_freq = 440 + (idx % 7) * 25
        carrier = np.sin(2 * np.pi * fake_freq * t)
        modulation = 0.5 + 0.5 * np.sign(np.sin(2 * np.pi * 35 * t))
        X.append(0.6 * carrier * modulation + 0.03 * rng.standard_normal(samples))
        y.append(1)

    X = np.asarray(X, dtype="float32")[..., np.newaxis]
    y = np.asarray(y, dtype="int64")
    idx = rng.permutation(len(y))
    X, y = X[idx], y[idx]

    n = len(y)
    n_train = int(math.floor(n * 0.70))
    n_val = int(math.floor(n * 0.15))
    arrays = {
        "X_train": X[:n_train],
        "y_train": y[:n_train],
        "X_val": X[n_train:n_train + n_val],
        "y_val": y[n_train:n_train + n_val],
        "X_test": X[n_train + n_val:],
        "y_test": y[n_train + n_val:],
        "metadata_json": np.asarray(json.dumps({
            "source": "synthetic_audio_smoke",
            "sample_rate": sample_rate,
            "format": "raw_audio",
        })),
    }
    np.savez_compressed(out_npz, **arrays)
    LOGGER.info("NPZ sintético exportado: %s", out_npz)
    return out_npz


def _verify_outputs(output_dir: Path) -> list[str]:
    expected = [
        "results.json",
        "results.csv",
        "predictions_clean.csv",
        "summary.md",
        "tcc_report.md",
        "dataset_manifest.json",
        "dataset.md",
        "tables/tab_resultados.tex",
        "tables/tab_eficiencia.tex",
        "tables/tab_robustez.tex",
        "figures/roc.png",
        "figures/robustez.png",
        "figures/convergencia.png",
        "figures/eficiencia.png",
        "figures/confusion_matrices.png",
        "figures/score_distributions.png",
    ]
    results_path = output_dir / "results.json"
    if results_path.exists():
        results = json.loads(results_path.read_text(encoding="utf-8"))
        arch_files = [
            "metrics.json",
            "summary.md",
            "predictions_clean.csv",
            "robustness.csv",
            "confusion_matrix.png",
            "roc.png",
            "score_distribution.png",
            "convergence.png",
        ]
        for name, result in results.get("architectures", {}).items():
            if result.get("status") != "ok":
                continue
            slug = name.lower().replace(" ", "_").replace("-", "_")
            expected.extend(f"architectures/{slug}/{file}" for file in arch_files)
            model_artifact = result.get("model_artifact")
            if model_artifact:
                model_path = Path(model_artifact)
                if model_path.is_absolute():
                    try:
                        expected.append(str(model_path.relative_to(output_dir)))
                    except ValueError:
                        if not model_path.exists():
                            raise RuntimeError(
                                f"Artefato de modelo ausente: {model_path}"
                            )
                else:
                    if model_path.exists():
                        continue
                    expected.append(str(model_path))
            tuning = result.get("hyperparameter_tuning") or {}
            if tuning.get("enabled"):
                expected.append(f"architectures/{slug}/hyperparameter_tuning.json")
                if tuning.get("status") == "ok":
                    expected.append(f"architectures/{slug}/hyperparameter_tuning.csv")
    missing = [p for p in expected if not (output_dir / p).exists()]
    if missing:
        raise RuntimeError(f"Artefatos ausentes: {missing}")
    return expected


def _read_npz_metadata(npz_path: Path) -> dict:
    try:
        with np.load(npz_path, allow_pickle=False) as data:
            if "metadata_json" not in data.files:
                return {}
            raw = data["metadata_json"]
            if hasattr(raw, "item"):
                raw = raw.item()
            return json.loads(str(raw))
    except Exception as exc:
        return {"metadata_error": str(exc)}


def _summarize_npz_metadata(meta: dict) -> dict:
    summary = dict(meta or {})
    splits = {}
    for split, info in (meta.get("splits") or {}).items():
        split_info = dict(info)
        paths = split_info.pop("paths", [])
        split_info["path_count"] = len(paths)
        if paths:
            split_info["first_examples"] = paths[:5]
        splits[split] = split_info
    if splits:
        summary["splits"] = splits
    return summary


def _write_dataset_docs(
    output_dir: Path,
    npz_path: Path,
    args: argparse.Namespace,
    counts: dict | None,
) -> list[str]:
    meta = _read_npz_metadata(npz_path)
    dataset_config_path = DATASETS_DIR / "dataset_config.json"
    dataset_config = {}
    if dataset_config_path.exists():
        try:
            dataset_config = json.loads(dataset_config_path.read_text(encoding="utf-8"))
        except Exception as exc:
            dataset_config = {"parse_error": str(exc)}

    manifest = {
        "dataset_npz": str(npz_path),
        "target_per_class": int(args.target_per_class),
        "target_total": int(args.target_per_class) * 2,
        "requested_full_tcc_dataset": bool(args.tcc_full_dataset),
        "sample_rate": int(args.sample_rate),
        "duration_sec": float(args.duration_sec),
        "max_per_class_export": args.max_per_class_export,
        "split_counts_on_disk": counts or {},
        "npz_metadata": _summarize_npz_metadata(meta),
        "dataset_config": dataset_config,
        "benchmark": {
            "full_benchmark": bool(args.full_benchmark),
            "architectures": [args.model] if args.model else args.archs,
            "model": args.model,
            "epochs": int(args.epochs),
            "batch_size": int(args.batch_size),
            "device_profile": args.device_profile,
            "optimize_hyperparameters": not bool(args.no_optimize_hparams),
            "latency_runs": int(args.latency_runs),
            "snr": [int(v) for v in args.snr],
            "api": bool(args.api),
        },
    }
    (output_dir / "dataset_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )

    split_lines = []
    for split, info in (manifest["npz_metadata"].get("splits") or {}).items():
        split_lines.append(
            f"| {split} | {info.get('samples')} | {info.get('real')} | "
            f"{info.get('fake')} |"
        )
    if not split_lines and counts:
        for split, info in counts.items():
            split_lines.append(
                f"| {split} | {info.get('real', 0) + info.get('fake', 0)} | "
                f"{info.get('real')} | {info.get('fake')} |"
            )

    lines = [
        "# Dataset do Benchmark TCC",
        "",
        "## Composição Planejada",
        "",
        f"- Alvo: `{args.target_per_class}` amostras reais + "
        f"`{args.target_per_class}` amostras fake.",
        f"- Total planejado: `{args.target_per_class * 2}` amostras.",
        "- Fontes reais: BRSpeech-DF bonafide + CommonVoice/FLEURS PT-BR.",
        "- Fontes fake: BRSpeech-DF spoof + Fake Voices XTTS.",
        "- Split: estratificado 70/15/15.",
        "",
        "## Processamento",
        "",
        f"- Sample rate: `{args.sample_rate}` Hz",
        f"- Duração exportada por áudio: `{args.duration_sec}` s",
        "- Formato exportado para benchmark: áudio bruto `(samples, 1)` em `.npz`.",
        f"- Arquivo NPZ: `{npz_path}`",
        "",
        "## Splits Exportados",
        "",
        "| Split | Amostras | Real | Fake |",
        "|---|---:|---:|---:|",
        *split_lines,
        "",
        "## Hiperparâmetros Globais do Benchmark",
        "",
        f"- Full benchmark: `{args.full_benchmark}`",
        f"- Modelo individual: `{args.model}`",
        f"- Arquiteturas explícitas: `{args.archs}`",
        f"- Épocas: `{args.epochs}`",
        f"- Batch size: `{args.batch_size}`",
        f"- Perfil de dispositivo: `{args.device_profile}`",
        f"- Hiperparâmetros otimizados por arquitetura: `{not args.no_optimize_hparams}`",
        f"- SNRs de robustez: `{args.snr}`",
        f"- Rodadas de latência: `{args.latency_runs}`",
        f"- API probe: `{args.api}`",
        "",
        "## Arquivos Relacionados",
        "",
        "- `dataset_manifest.json`: manifesto estruturado do dataset e execução.",
        "- `results.json`: métricas completas por arquitetura.",
        "- `tcc_report.md`: relatório final com métricas, inferências e imagens PNG.",
        "- `figures/*.png`: gráficos agregados.",
        "- `architectures/<arquitetura>/*.png`: gráficos individuais.",
    ]
    (output_dir / "dataset.md").write_text("\n".join(lines), encoding="utf-8")
    return ["dataset_manifest.json", "dataset.md"]


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Executa download, treino, inferência e relatório do TCC.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--smoke", action="store_true", help="usa áudio sintético e não baixa datasets")
    parser.add_argument(
        "--tcc-full-dataset",
        action="store_true",
        help="preset de dataset ideal do TCC: 10.000 real + 10.000 fake, full benchmark e API",
    )
    parser.add_argument("--download", action="store_true", help="executa scripts/build_dataset.py antes do treino")
    parser.add_argument("--skip-download", action="store_true", help="usa datasets/splits já existentes")
    parser.add_argument(
        "--skip-preprocess",
        action="store_true",
        help="com --skip-download, não normaliza/recria splits; apenas exporta os splits existentes",
    )
    parser.add_argument("--target-per-class", type=int, default=10000)
    parser.add_argument("--skip-real-cv", action="store_true")
    parser.add_argument("--full-benchmark", action="store_true", help="usa preset completo do benchmark")
    parser.add_argument("--model", default=None, help="executa benchmark de uma única arquitetura")
    parser.add_argument("--archs", nargs="+", default=None)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device-profile", choices=["auto", "cpu", "gpu"], default="auto")
    parser.add_argument("--no-optimize-hparams", action="store_true")
    # P0 — repasse dos protocolos anti-vazamento ao benchmark.
    parser.add_argument("--group-split", action="store_true",
                        help="split disjunto por fonte/gerador (anti-vazamento)")
    parser.add_argument("--cross-generator", metavar="GERADOR", default=None,
                        help="segura este gerador fora do treino e testa nele "
                             "(ex.: fkvoice)")
    parser.add_argument("--skip-benchmark-preflight", action="store_true")
    parser.add_argument("--latency-runs", type=int, default=30)
    parser.add_argument("--snr", nargs="+", type=int, default=[30, 20, 10])
    parser.add_argument("--api", action="store_true")
    parser.add_argument("--sample-rate", type=int, default=16000)
    parser.add_argument("--duration-sec", type=float, default=5.0)
    parser.add_argument("--max-per-class-export", type=int, default=None)
    parser.add_argument("--npz", default="app/datasets/benchmark_audio_raw.npz")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    if args.model and args.archs:
        parser.error("Use --model para um único modelo ou --archs para vários, não ambos.")

    if args.tcc_full_dataset:
        args.target_per_class = 10000
        args.full_benchmark = True
        args.api = True
        if not args.download and not args.skip_download:
            args.download = True

    if not args.smoke and not args.download and not args.skip_download:
        parser.error("Use --download para baixar/preparar ou --skip-download para usar splits existentes.")

    output_dir = Path(args.out) if args.out else (
        BASE_DIR / "results" / "tcc_pipeline" / time.strftime("%Y%m%d_%H%M%S")
    )
    if not output_dir.is_absolute():
        output_dir = BASE_DIR / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / "pipeline.log"

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    LOGGER.info("Saída do pipeline: %s", output_dir)

    npz_path = Path(args.npz)
    if not npz_path.is_absolute():
        npz_path = BASE_DIR / npz_path

    counts = None
    if args.smoke:
        create_synthetic_audio_npz(
            npz_path,
            n_per_class=max(20, args.max_per_class_export or 40),
            sample_rate=args.sample_rate,
        )
    else:
        if args.download:
            cmd = [
                sys.executable,
                str(SCRIPTS_DIR / "build_dataset.py"),
                "--target",
                str(args.target_per_class),
            ]
            if args.skip_real_cv:
                cmd.append("--skip-real-cv")
            _run(cmd, "Download, balanceamento, validação e splits", log_path)
        else:
            if args.skip_preprocess:
                LOGGER.info("Pré-processamento pulado; usando splits existentes.")
            else:
                _run(
                    [
                        sys.executable,
                        str(SCRIPTS_DIR / "preprocess_dataset.py"),
                        "--full",
                    ],
                    "Validação, normalização e recriação de splits existentes",
                    log_path,
                )

        counts = _split_counts(SPLITS_DIR)
        if not all(counts[s]["real"] and counts[s]["fake"] for s in ("train", "val", "test")):
            raise RuntimeError(
                "Splits incompletos em app/datasets/splits. "
                "Execute com --download ou remova --skip-preprocess."
            )
        LOGGER.info("Splits encontrados: %s", counts)
        export_npz_from_splits(
            SPLITS_DIR,
            npz_path,
            sample_rate=args.sample_rate,
            duration_sec=args.duration_sec,
            max_per_class=args.max_per_class_export,
        )

    bench_cmd = [
        sys.executable,
        str(SCRIPTS_DIR / "run_benchmark.py"),
        "--dataset",
        str(npz_path),
        "--epochs",
        str(args.epochs),
        "--batch-size",
        str(args.batch_size),
        "--device-profile",
        args.device_profile,
        "--latency-runs",
        str(args.latency_runs),
        "--snr",
        *[str(v) for v in args.snr],
        "--out",
        str(output_dir),
    ]
    if args.full_benchmark:
        bench_cmd.insert(2, "--full")
    if args.no_optimize_hparams:
        bench_cmd.append("--no-optimize-hparams")
    if args.model:
        bench_cmd.extend(["--model", args.model])
    elif args.archs:
        bench_cmd.extend(["--archs", *args.archs])
    elif args.smoke:
        bench_cmd.extend(["--archs", "SVM", "RandomForest"])
    if args.api:
        bench_cmd.append("--api")
    else:
        bench_cmd.append("--no-api")
    if args.group_split:
        bench_cmd.append("--group-split")
    if args.cross_generator:
        bench_cmd.extend(["--cross-generator", args.cross_generator])

    if not args.skip_benchmark_preflight:
        _run(
            [*bench_cmd, "--plan-only"],
            "Preflight: preset, dataset e hiperparâmetros efetivos",
            log_path,
        )

    _run(bench_cmd, "Benchmark: treinamento, inferência e relatório", log_path)
    _write_dataset_docs(output_dir, npz_path, args, counts)
    artifacts = _verify_outputs(output_dir)

    summary = {
        "status": "ok",
        "output_dir": str(output_dir),
        "dataset_npz": str(npz_path),
        "artifacts": artifacts,
        "log": str(log_path),
    }
    (output_dir / "pipeline_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    LOGGER.info("Pipeline concluído com sucesso.")
    LOGGER.info("Resumo: %s", output_dir / "pipeline_summary.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
