#!/usr/bin/env python3
"""Run the consolidated benchmark for all selected architectures."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run all-architecture benchmark through run_models_sequential.py.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--dataset",
        default="app/datasets/benchmark_audio_raw_balanced_20k.npz",
        help="Canonical dataset .npz.",
    )
    parser.add_argument("--out", default="results/all_architectures_benchmark")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device-profile", choices=["auto", "cpu", "gpu"], default="gpu")
    parser.add_argument("--timeout-min", type=float, default=240.0)
    parser.add_argument("--latency-runs", type=int, default=30)
    parser.add_argument("--snr", nargs="+", type=int, default=[30, 20, 10])
    parser.add_argument("--models", nargs="+", help="Optional explicit model list.")
    parser.add_argument("--classical-only", action="store_true")
    parser.add_argument("--neural-only", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--plan-only", action="store_true")
    parser.add_argument("--api", action="store_true")
    parser.add_argument("--no-optimize-hparams", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    cmd = [
        sys.executable,
        str(ROOT / "scripts" / "run_models_sequential.py"),
        "--dataset",
        args.dataset,
        "--out",
        args.out,
        "--epochs",
        str(args.epochs),
        "--batch-size",
        str(args.batch_size),
        "--device-profile",
        args.device_profile,
        "--timeout-min",
        str(args.timeout_min),
        "--latency-runs",
        str(args.latency_runs),
        "--snr",
        *[str(item) for item in args.snr],
    ]
    if args.models:
        cmd.extend(["--models", *args.models])
    if args.classical_only:
        cmd.append("--classical-only")
    if args.neural_only:
        cmd.append("--neural-only")
    if args.resume:
        cmd.append("--resume")
    if args.plan_only:
        cmd.append("--plan-only")
    if args.api:
        cmd.append("--api")
    if args.no_optimize_hparams:
        cmd.append("--no-optimize-hparams")
    if args.verbose:
        cmd.append("--verbose")

    print("Executing:", " ".join(cmd), flush=True)
    return int(subprocess.run(cmd, cwd=str(ROOT)).returncode)


if __name__ == "__main__":
    raise SystemExit(main())
