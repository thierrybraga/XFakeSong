#!/usr/bin/env python3
"""Run a benchmark/training preset for a model family.

This wrapper keeps family entrypoints small and delegates the real work to
scripts/benchmark/run_models_sequential.py, which already handles per-model folders,
logs, resume, timeouts and benchmark artifacts.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.core.config.paths import resolve_results_output
DEFAULT_CONFIGS = {
    "classical-tabular": ROOT / "configs" / "training" / "classical.yaml",
    "spectral-convolutional": (
        ROOT / "configs" / "training" / "spectral_convolutional.yaml"
    ),
    "spectral-attention": ROOT / "configs" / "training" / "tensorflow.yaml",
    "waveform-end-to-end": ROOT / "configs" / "training" / "pytorch.yaml",
    "ssl-pretrained": ROOT / "configs" / "training" / "ssl.yaml",
    "extended": ROOT / "configs" / "training" / "extended.yaml",
}


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Invalid YAML mapping: {path}")
    return data


def _as_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item) for item in value]
    return [str(value)]


def build_command(args: argparse.Namespace) -> list[str]:
    config_path = Path(args.config) if args.config else DEFAULT_CONFIGS[args.family]
    if not config_path.is_absolute():
        config_path = ROOT / config_path
    cfg = _load_yaml(config_path)

    models = args.models or _as_list(cfg.get("models"))
    if not models:
        raise ValueError(f"No models defined for family {args.family}")

    dataset = args.dataset or cfg.get("dataset")
    if not dataset:
        raise ValueError("Dataset path is required")

    output_dir = args.out or resolve_results_output(
        cfg.get("output_dir"),
        default_subdir=f"{args.family}_benchmark",
        base_dir=ROOT,
    )
    epochs = args.epochs if args.epochs is not None else int(cfg.get("epochs", 100))
    batch_size = (
        args.batch_size
        if args.batch_size is not None
        else int(cfg.get("batch_size", 32))
    )
    device_profile = args.device_profile or str(cfg.get("device_profile", "auto"))
    latency_runs = (
        args.latency_runs
        if args.latency_runs is not None
        else int(cfg.get("latency_runs", 30))
    )
    # Sem default de 240 min: esse valor cabia num run de fumaca, mas mata
    # qualquer treino do orcamento de 100 epocas (RawGAT-ST leva ~54 h de GPU).
    # Omitido, o run_models_sequential deriva o limite por arquitetura a partir
    # do custo estimado (benchmarks.planning.EXPECTED_TRAINING_HOURS).
    timeout_min = (
        args.timeout_min
        if args.timeout_min is not None
        else cfg.get("timeout_min")
    )
    snr = args.snr or [str(item) for item in _as_list(cfg.get("snr") or [30, 20, 10])]

    cmd = [
        sys.executable,
        str(ROOT / "scripts" / "benchmark" / "run_models_sequential.py"),
        "--dataset",
        str(dataset),
        "--models",
        *models,
        "--out",
        str(output_dir),
        "--epochs",
        str(epochs),
        "--batch-size",
        str(batch_size),
        "--device-profile",
        device_profile,
        "--latency-runs",
        str(latency_runs),
        "--snr",
        *[str(item) for item in snr],
    ]
    if timeout_min is not None:
        cmd.extend(["--timeout-min", str(float(timeout_min))])
    if args.resume:
        cmd.append("--resume")
    if args.seeds:
        cmd.extend(["--seeds", *[str(seed) for seed in args.seeds]])
    if args.test_lock:
        cmd.extend(["--test-lock", args.test_lock])
    scope = str(cfg.get("scope", "official"))
    cmd.extend(["--scope", scope])
    academic = bool(cfg.get("academic_protocol", True))
    cmd.append("--academic-protocol" if academic else "--no-academic-protocol")
    if args.plan_only:
        cmd.append("--plan-only")
    if args.api:
        cmd.append("--api")
    optimize = cfg.get("optimize_hyperparameters", True)
    if args.no_optimize_hparams or not optimize:
        cmd.append("--no-optimize-hparams")
    if args.verbose:
        cmd.append("--verbose")
    return cmd


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Train/benchmark a family using the consolidated orchestrator.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--family",
        required=False,
        choices=sorted(DEFAULT_CONFIGS),
        help="Environment family to execute.",
    )
    parser.add_argument("--config", help="YAML config override.")
    parser.add_argument("--dataset", help="Dataset .npz override.")
    parser.add_argument("--models", nargs="+", help="Model list override.")
    parser.add_argument("--out", help="Output directory override.")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--device-profile", choices=["auto", "cpu", "gpu"])
    parser.add_argument("--latency-runs", type=int)
    parser.add_argument("--timeout-min", type=float)
    parser.add_argument("--snr", nargs="+")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--seeds", nargs="+", type=int)
    parser.add_argument("--test-lock")
    parser.add_argument("--plan-only", action="store_true")
    parser.add_argument("--api", action="store_true")
    parser.add_argument("--no-optimize-hparams", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args(argv)
    if not args.family:
        parser.error("--family is required when calling train_by_family.py directly")

    cmd = build_command(args)
    print("Executing:", " ".join(cmd), flush=True)
    result = subprocess.run(cmd, cwd=str(ROOT))
    return int(result.returncode)


if __name__ == "__main__":
    raise SystemExit(main())
