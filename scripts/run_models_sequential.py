#!/usr/bin/env python3
"""Executa benchmark de modelos um por vez, com timeout e retomada.

Este orquestrador chama `scripts/run_benchmark.py --model <nome>` para cada
arquitetura. Cada modelo recebe uma pasta própria, log próprio e status próprio.

Exemplos:
  python scripts/run_models_sequential.py --dataset app/datasets/benchmark_audio_raw_balanced_15k.npz
  python scripts/run_models_sequential.py --models SVM RandomForest --timeout-min 20
  python scripts/run_models_sequential.py --neural-only --resume --device-profile gpu
  python scripts/run_models_sequential.py --neural-only --plan-only
"""

from __future__ import annotations

import argparse
import json
import os
import queue
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
SCRIPTS = ROOT / "scripts"

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmarks.config import (  # noqa: E402
    CLASSICAL_TCC_ARCHITECTURES,
    DOCKER_TRAINING_ARCHITECTURES,
    NEURAL_DOCKER_ARCHITECTURES,
)


SSL_ORIGINAL_MODELS = {
    "wavlm": {
        "display": "WavLM Original",
        "architecture": "wavlm",
        "runner": SCRIPTS / "run_wavlm_original_benchmark.py",
    },
    "wavlmoriginal": {
        "display": "WavLM Original",
        "architecture": "wavlm",
        "runner": SCRIPTS / "run_wavlm_original_benchmark.py",
    },
    "hubert": {
        "display": "HuBERT Original",
        "architecture": "hubert",
        "runner": SCRIPTS / "run_wavlm_original_benchmark.py",
    },
    "hubertoriginal": {
        "display": "HuBERT Original",
        "architecture": "hubert",
        "runner": SCRIPTS / "run_wavlm_original_benchmark.py",
    },
}


def _slug(name: str) -> str:
    return "".join(ch if ch.isalnum() else "_" for ch in name.lower()).strip("_")


def _compact(name: str) -> str:
    return "".join(ch for ch in name.lower() if ch.isalnum())


def _ssl_meta(model: str) -> dict[str, Any] | None:
    return SSL_ORIGINAL_MODELS.get(_compact(model))


def _load_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )


def _model_done(model_dir: Path) -> bool:
    results = model_dir / "results.json"
    if not results.exists():
        return False
    data = _load_json(results, {})
    archs = data.get("architectures") or {}
    return any(item.get("status") == "ok" for item in archs.values())


def _plan_done(model_dir: Path) -> bool:
    return (model_dir / "benchmark_plan.json").exists()


def _emit(message: str = "") -> None:
    print(message, flush=True)


def _tail_text(path: Path, max_lines: int = 40) -> str:
    if not path.exists():
        return ""
    try:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    except Exception:
        return ""
    return "\n".join(lines[-max_lines:])


def _build_command(args: argparse.Namespace, model: str, model_dir: Path) -> list[str]:
    ssl_meta = _ssl_meta(model)
    if ssl_meta is not None:
        return [
            sys.executable,
            str(ssl_meta["runner"]),
            "--architecture",
            ssl_meta["architecture"],
            "--dataset",
            str(Path(args.dataset).resolve()),
            "--out",
            str(model_dir),
            "--epochs",
            str(args.epochs),
            "--train-batch-size",
            str(args.batch_size),
            "--feature-batch-size",
            str(args.ssl_feature_batch_size),
            "--latency-runs",
            str(args.latency_runs),
            "--snr",
            *[str(v) for v in args.snr],
            "--freeze-backbone",
        ]

    cmd = [
        sys.executable,
        str(SCRIPTS / "run_benchmark.py"),
        "--model",
        model,
        "--dataset",
        str(Path(args.dataset).resolve()),
        "--out",
        str(model_dir),
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
    ]
    if args.api:
        cmd.append("--api")
    else:
        cmd.append("--no-api")
    if args.no_optimize_hparams:
        cmd.append("--no-optimize-hparams")
    if args.plan_only:
        cmd.append("--plan-only")
    if args.verbose:
        cmd.append("--verbose")
    if getattr(args, "speaker_split", False):
        cmd.append("--speaker-split")
    if getattr(args, "group_split", False):
        cmd.append("--group-split")
    return cmd


def _run_one(args: argparse.Namespace, model: str, root_out: Path) -> dict[str, Any]:
    slug = _slug(model)
    model_dir = root_out / slug
    model_dir.mkdir(parents=True, exist_ok=True)
    log_path = model_dir / "run.log"
    cmd = _build_command(args, model, model_dir)
    timeout_s = int(args.timeout_min * 60)
    started = time.time()
    ssl_meta = _ssl_meta(model)

    if args.plan_only and ssl_meta is not None:
        model_name = (
            "facebook/hubert-base-ls960"
            if ssl_meta["architecture"] == "hubert"
            else "microsoft/wavlm-base"
        )
        plan = {
            "model": ssl_meta["display"],
            "runner": str(ssl_meta["runner"]),
            "architecture": ssl_meta["architecture"],
            "model_name": model_name,
            "dataset": str(Path(args.dataset).resolve()),
            "output_dir": str(model_dir),
            "epochs": args.epochs,
            "train_batch_size": args.batch_size,
            "feature_batch_size": args.ssl_feature_batch_size,
            "latency_runs": args.latency_runs,
            "snr": args.snr,
            "freeze_backbone": True,
            "fit_strategy": "frozen_backbone_embedding_then_classifier_fit",
            "command": cmd,
        }
        _write_json(model_dir / "benchmark_plan.json", plan)
        model_dir.joinpath("benchmark_plan.md").write_text(
            "\n".join(
                [
                    f"# Plano de Benchmark - {ssl_meta['display']}",
                    "",
                    f"- Runner: `{ssl_meta['runner']}`",
                    f"- Backbone: `{model_name}`",
                    "- Pesos do backbone: congelados",
                    "- Treino: somente cabeca classificadora PyTorch",
                    f"- Dataset: `{plan['dataset']}`",
                    f"- Saida: `{model_dir}`",
                    "",
                    "## Comando",
                    "",
                    "```bash",
                    " ".join(cmd),
                    "```",
                ]
            ),
            encoding="utf-8",
        )
        log_path.write_text(
            "PLAN ONLY: runner SSL PyTorch preparado para WSL/Docker; "
            "backbone Hugging Face congelado por padrao.\n",
            encoding="utf-8",
        )
        elapsed = round(time.time() - started, 1)
        return {
            "model": model,
            "status": "ok",
            "error": None,
            "elapsed_s": elapsed,
            "output_dir": str(model_dir),
            "log": str(log_path),
            "returncode": 0,
            "clean": None,
            "efficiency": None,
            "model_artifact": None,
            "log_tail": "",
        }

    with log_path.open("w", encoding="utf-8", errors="replace") as log:
        log.write("COMMAND:\n")
        log.write(" ".join(cmd) + "\n\n")
        log.flush()
        try:
            proc = subprocess.Popen(
                cmd,
                cwd=str(ROOT),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                env={**os.environ, "PYTHONIOENCODING": "utf-8"},
            )
            output_queue: queue.Queue[str | None] = queue.Queue()

            def _reader() -> None:
                assert proc.stdout is not None
                for line in proc.stdout:
                    output_queue.put(line)
                output_queue.put(None)

            reader = threading.Thread(target=_reader, daemon=True)
            reader.start()
            reader_done = False

            while True:
                try:
                    line = output_queue.get(timeout=1.0)
                except queue.Empty:
                    line = ""

                if line is None:
                    reader_done = True
                elif line:
                    log.write(line)
                    log.flush()
                    print(line, end="", flush=True)

                if proc.poll() is not None and reader_done:
                    break
                if time.time() - started > timeout_s:
                    proc.kill()
                    raise subprocess.TimeoutExpired(cmd, timeout=timeout_s)

            returncode = proc.returncode
            done = _plan_done(model_dir) if args.plan_only else _model_done(model_dir)
            status = "ok" if returncode == 0 and done else "error"
            error = None if status == "ok" else f"returncode={returncode}"
        except subprocess.TimeoutExpired:
            returncode = None
            status = "timeout"
            error = f"timeout_min={args.timeout_min}"
            log.write(f"\n[TIMEOUT] {model}: {error}\n")
            log.flush()
            _emit(f"[TIMEOUT] {model}: {error}")

    elapsed = round(time.time() - started, 1)
    metrics = {}
    results_path = model_dir / "results.json"
    if results_path.exists():
        data = _load_json(results_path, {})
        metrics = (data.get("architectures") or {}).get(model, {})
        if not metrics:
            metrics = next(iter((data.get("architectures") or {}).values()), {})

    return {
        "model": model,
        "status": status,
        "error": error,
        "elapsed_s": elapsed,
        "output_dir": str(model_dir),
        "log": str(log_path),
        "returncode": returncode,
        "clean": metrics.get("clean"),
        "efficiency": metrics.get("efficiency"),
        "model_artifact": metrics.get("model_artifact"),
        "log_tail": _tail_text(log_path) if status != "ok" else "",
    }


def _write_summary(root_out: Path, summary: dict[str, Any]) -> None:
    lines = [
        "# Execução Sequencial de Benchmark",
        "",
        f"- Dataset: `{summary.get('dataset')}`",
        f"- Device profile: `{summary.get('device_profile')}`",
        f"- Timeout por modelo: `{summary.get('timeout_min')}` min",
        "",
        "| Modelo | Status | Accuracy | AUC | EER | Latência ms | Tempo s |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for item in summary.get("models", []):
        clean = item.get("clean") or {}
        eff = item.get("efficiency") or {}
        lines.append(
            f"| {item.get('model')} | {item.get('status')} | "
            f"{clean.get('accuracy', '')} | {clean.get('auc_roc', '')} | "
            f"{clean.get('eer', '')} | {eff.get('latency_ms', '')} | "
            f"{item.get('elapsed_s')} |"
        )
    root_out.joinpath("run_summary.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Roda benchmark de modelos um por vez com timeout e resume.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--dataset",
        default="app/datasets/benchmark_audio_raw_balanced_15k.npz",
        help="Dataset .npz usado por todos os modelos.",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=None,
        help=(
            "Lista de arquiteturas/modelos. Default: modelos documentados no "
            "artigo + WavLM/HuBERT Original no runner SSL Docker."
        ),
    )
    parser.add_argument("--neural-only", action="store_true",
                        help="roda arquiteturas neurais do artigo + WavLM/HuBERT SSL Docker")
    parser.add_argument("--classical-only", action="store_true",
                        help="roda somente SVM e RandomForest")
    parser.add_argument("--out", default="results/sequential_benchmark")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device-profile", choices=["auto", "cpu", "gpu"], default="auto")
    parser.add_argument("--latency-runs", type=int, default=30)
    parser.add_argument(
        "--ssl-feature-batch-size",
        type=int,
        default=16,
        help="batch para extracao de embeddings HuBERT/WavLM no runner SSL",
    )
    parser.add_argument("--snr", nargs="+", type=int, default=[30, 20, 10])
    parser.add_argument("--timeout-min", type=float, default=60.0)
    parser.add_argument("--resume", action="store_true", help="pula modelos já concluídos")
    parser.add_argument("--plan-only", action="store_true",
                        help="gera benchmark_plan.* por modelo e não inicia treino")
    parser.add_argument("--api", action="store_true")
    parser.add_argument("--no-optimize-hparams", action="store_true")
    parser.add_argument("--speaker-split", action="store_true",
                        help="split disjunto por falante (tier large; requer speaker_ids no .npz)")
    parser.add_argument("--group-split", action="store_true",
                        help="split por fonte/gerador (cross-generator)")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()
    if args.neural_only and args.classical_only:
        parser.error("Use apenas um entre --neural-only e --classical-only.")
    if args.models:
        selected_models = list(args.models)
    elif args.neural_only:
        selected_models = list(NEURAL_DOCKER_ARCHITECTURES)
    elif args.classical_only:
        selected_models = list(CLASSICAL_TCC_ARCHITECTURES)
    else:
        selected_models = list(DOCKER_TRAINING_ARCHITECTURES)

    dataset_path = Path(args.dataset)
    if not dataset_path.is_absolute():
        dataset_path = ROOT / dataset_path
    if not dataset_path.exists():
        parser.error(f"Dataset não encontrado: {dataset_path}")
    args.dataset = str(dataset_path)

    root_out = Path(args.out)
    if not root_out.is_absolute():
        root_out = ROOT / root_out
    root_out.mkdir(parents=True, exist_ok=True)

    summary_path = root_out / "run_summary.json"
    summary = _load_json(summary_path, {})
    completed = {
        item.get("model")
        for item in summary.get("models", [])
        if item.get("status") == "ok"
    }

    summary = {
        "status": "running",
        "dataset": str(dataset_path),
        "device_profile": args.device_profile,
        "timeout_min": args.timeout_min,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "ssl_feature_batch_size": args.ssl_feature_batch_size,
        "snr": args.snr,
        "models": summary.get("models", []),
    }

    existing_by_model = {item.get("model"): item for item in summary["models"]}
    for model in selected_models:
        model_dir = root_out / _slug(model)
        done = _plan_done(model_dir) if args.plan_only else _model_done(model_dir)
        if args.resume and model in completed and done:
            _emit(f"[SKIP] {model} ja concluido")
            continue

        _emit(f"[RUN] {model} -> {model_dir}")
        result = _run_one(args, model, root_out)
        existing_by_model[model] = result
        summary["models"] = [existing_by_model[m] for m in existing_by_model]
        _write_json(summary_path, summary)
        _write_summary(root_out, summary)
        _emit(f"[{result['status'].upper()}] {model} em {result['elapsed_s']}s")
        if result["status"] != "ok" and result.get("log_tail"):
            _emit(f"[LOG TAIL] {model}")
            _emit(result["log_tail"])

    statuses = [item.get("status") for item in summary["models"]]
    summary["status"] = "ok" if statuses and all(s == "ok" for s in statuses) else "partial"
    _write_json(summary_path, summary)
    _write_summary(root_out, summary)
    _emit(f"Resumo: {summary_path}")
    return 0 if summary["status"] == "ok" else 2


if __name__ == "__main__":
    raise SystemExit(main())
