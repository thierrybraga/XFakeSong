#!/usr/bin/env python3
"""Clean benchmark orchestrator for XFakeSong.

This script coordinates a fresh benchmark run without mixing artifacts from
older experiments. It can clean model/result directories, build a named Docker
GPU image, write a manifest, and run the existing sequential benchmark harness
inside Docker so each model gets its own output folder and log.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Iterable

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmarks.config import (  # noqa: E402
    CLASSICAL_TCC_ARCHITECTURES,
    DOCKER_TRAINING_ARCHITECTURES,
    NEURAL_DOCKER_ARCHITECTURES,
    OFFICIAL_TCC_MODEL_MANIFEST,
)

DEFAULT_DATASET = ROOT / "app" / "datasets" / "benchmark_audio_raw_balanced_15k.npz"
DEFAULT_MODELS_DIR = ROOT / "app" / "models"
DEFAULT_RESULTS_DIR = ROOT / "results"
DEFAULT_IMAGE = "xfakesong:benchmark-gpu"

ALL_MODELS = list(DOCKER_TRAINING_ARCHITECTURES)
CLASSICAL_MODELS = list(CLASSICAL_TCC_ARCHITECTURES)
NEURAL_MODELS = list(NEURAL_DOCKER_ARCHITECTURES)

SMOKE_MODELS = ["SVM", "MultiscaleCNN"]


def _resolve_project_path(value: str | Path) -> Path:
    path = Path(value)
    if not path.is_absolute():
        path = ROOT / path
    resolved = path.resolve()
    try:
        resolved.relative_to(ROOT)
    except ValueError as exc:
        raise SystemExit(f"Path outside project root refused: {resolved}") from exc
    return resolved


def _run(cmd: list[str], *, cwd: Path = ROOT) -> None:
    print("\n$ " + " ".join(cmd), flush=True)
    proc = subprocess.Popen(
        cmd,
        cwd=str(cwd),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        bufsize=1,
        env={
            **os.environ,
            "PYTHONIOENCODING": "utf-8",
            "PYTHONUTF8": "1",
            "LANG": "C.UTF-8",
            "LC_ALL": "C.UTF-8",
        },
    )
    assert proc.stdout is not None
    for line in proc.stdout:
        print(line, end="", flush=True)
    if proc.wait() != 0:
        raise SystemExit(proc.returncode)


def _clean_directory(path: Path) -> None:
    resolved = _resolve_project_path(path)
    resolved.mkdir(parents=True, exist_ok=True)
    for item in resolved.iterdir():
        if item.is_dir():
            shutil.rmtree(item)
        else:
            item.unlink()


def _write_manifest(
    out_dir: Path,
    *,
    args: argparse.Namespace,
    models: list[str],
    phase: str,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    official_by_name = {
        item["benchmark_name"]: item for item in OFFICIAL_TCC_MODEL_MANIFEST
    }
    selected_manifest = [
        official_by_name.get(
            model,
            {
                "benchmark_name": model,
                "result_key": model,
                "display_name": model,
                "variant": "custom",
                "runner": "custom",
                "input_type": "unknown",
            },
        )
        for model in models
    ]
    payload = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "phase": phase,
        "dataset": str(args.dataset),
        "dataset_exists": Path(args.dataset).exists(),
        "dataset_size_bytes": Path(args.dataset).stat().st_size
        if Path(args.dataset).exists()
        else None,
        "models": models,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "device_profile": args.device_profile,
        "timeout_min": args.timeout_min,
        "latency_runs": args.latency_runs,
        "ssl_feature_batch_size": args.ssl_feature_batch_size,
        "snr": args.snr,
        "docker_image": args.image,
        "models_dir": str(args.models_dir),
        "output_dir": str(out_dir),
        "official_model_manifest": selected_manifest,
    }
    (out_dir / "execution_manifest.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    lines = [
        "# Clean Benchmark Execution",
        "",
        f"- Criado em: `{payload['created_at']}`",
        f"- Fase: `{phase}`",
        f"- Dataset: `{payload['dataset']}`",
        f"- Dataset bytes: `{payload['dataset_size_bytes']}`",
        f"- Device: `{args.device_profile}`",
        f"- Epochs: `{args.epochs}`",
        f"- Timeout por modelo: `{args.timeout_min}` min",
        f"- Docker image: `{args.image}`",
        "",
        "## Modelos",
        "",
    ]
    lines.extend(
        "- {display} (`{benchmark}`, variante `{variant}`, runner `{runner}`)".format(
            display=item["display_name"],
            benchmark=item["benchmark_name"],
            variant=item["variant"],
            runner=item["runner"],
        )
        for item in selected_manifest
    )
    (out_dir / "execution_manifest.md").write_text(
        "\n".join(lines),
        encoding="utf-8",
    )


def _select_models(args: argparse.Namespace) -> tuple[str, list[str]]:
    if args.models:
        return "custom", args.models
    if args.phase == "smoke":
        return "smoke", SMOKE_MODELS
    if args.phase == "classical":
        return "classical", CLASSICAL_MODELS
    if args.phase == "neural":
        return "neural", NEURAL_MODELS
    return "full", ALL_MODELS


def _docker_run_command(args: argparse.Namespace, models: Iterable[str]) -> list[str]:
    out_rel = Path(args.out).as_posix()
    dataset_rel = Path(args.dataset).as_posix()
    hf_cache = ROOT / "cache" / "huggingface"
    torch_cache = ROOT / "cache" / "torch"
    hf_cache.mkdir(parents=True, exist_ok=True)
    torch_cache.mkdir(parents=True, exist_ok=True)
    cmd = [
        "docker",
        "run",
        "--rm",
        "--no-healthcheck",
        "--gpus",
        "all",
        "--shm-size",
        "2g",
        "--name",
        args.container_name,
        "-v",
        f"{ROOT}:/app",
        "-v",
        f"{hf_cache}:/app/cache/huggingface",
        "-v",
        f"{torch_cache}:/app/cache/torch",
        "-w",
        "/app",
        "-e",
        "PYTHONIOENCODING=utf-8",
        "-e",
        "TF_FORCE_GPU_ALLOW_GROWTH=true",
        "-e",
        "GRADIO_ANALYTICS_ENABLED=false",
        "-e",
        "XFAKE_SYNC_MODELS_ON_BOOT=false",
        "-e",
        "XFAKE_CREATE_DEFAULT_MODELS=false",
        "-e",
        f"XFAKE_TRAIN_BATCH_LOG_INTERVAL_S={args.batch_log_interval_s}",
        "-e",
        "HF_HOME=/app/cache/huggingface",
        "-e",
        "TRANSFORMERS_CACHE=/app/cache/huggingface",
        "-e",
        "TORCH_HOME=/app/cache/torch",
        args.image,
        "python",
        "scripts/run_models_sequential.py",
        "--models",
        *models,
        "--dataset",
        dataset_rel,
        "--out",
        out_rel,
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
        "--ssl-feature-batch-size",
        str(args.ssl_feature_batch_size),
        "--snr",
        *[str(v) for v in args.snr],
    ]
    if args.detach:
        cmd.insert(3, "-d")
    if args.resume:
        cmd.append("--resume")
    if args.verbose:
        cmd.append("--verbose")
    if args.api:
        cmd.append("--api")
    return cmd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a clean XFakeSong benchmark/training pipeline.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--phase",
        choices=["smoke", "classical", "neural", "full"],
        default="smoke",
        help="Execution phase. Use smoke before long full/neural runs.",
    )
    parser.add_argument("--models", nargs="+", default=None)
    parser.add_argument("--dataset", default=str(DEFAULT_DATASET.relative_to(ROOT)))
    parser.add_argument("--out", default=None)
    parser.add_argument("--models-dir", default=str(DEFAULT_MODELS_DIR.relative_to(ROOT)))
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device-profile", choices=["auto", "cpu", "gpu"], default="gpu")
    parser.add_argument("--timeout-min", type=float, default=240.0)
    parser.add_argument("--latency-runs", type=int, default=30)
    parser.add_argument("--ssl-feature-batch-size", type=int, default=16)
    parser.add_argument("--snr", nargs="+", type=int, default=[30, 20, 10])
    parser.add_argument("--image", default=DEFAULT_IMAGE)
    parser.add_argument("--container-name", default="xfakesong_benchmark_run")
    parser.add_argument("--clean", action="store_true")
    parser.add_argument("--build", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--api", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--batch-log-interval-s", type=int, default=60)
    parser.add_argument(
        "--detach",
        action="store_true",
        help="inicia o container e retorna imediatamente; acompanhe com docker logs -f",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.dataset = str(_resolve_project_path(args.dataset).relative_to(ROOT))
    args.models_dir = str(_resolve_project_path(args.models_dir).relative_to(ROOT))

    phase, models = _select_models(args)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if args.out is None:
        args.out = f"results/clean_benchmark_{phase}_{stamp}"
    out_dir = _resolve_project_path(args.out)

    if not (ROOT / args.dataset).exists():
        raise SystemExit(f"Dataset not found: {ROOT / args.dataset}")

    if args.clean:
        print("[clean] Limpando app/models e results...")
        _clean_directory(ROOT / args.models_dir)
        _clean_directory(DEFAULT_RESULTS_DIR)
        out_dir.mkdir(parents=True, exist_ok=True)

    _write_manifest(out_dir, args=args, models=models, phase=phase)

    if args.build:
        _run([
            "docker",
            "build",
            "--build-arg",
            "TF_VARIANT=gpu",
            "-t",
            args.image,
            ".",
        ])

    started = time.time()
    _run(_docker_run_command(args, models))
    elapsed = time.time() - started
    if args.detach:
        print(f"\n[started] Pipeline {phase} iniciado em background")
        print(f"[started] Container: {args.container_name}")
        print(f"[started] Logs: docker logs -f {args.container_name}")
        print(f"[started] Saída: {out_dir}")
    else:
        print(f"\n[done] Pipeline {phase} concluído em {elapsed / 60:.1f} min")
        print(f"[done] Saída: {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
