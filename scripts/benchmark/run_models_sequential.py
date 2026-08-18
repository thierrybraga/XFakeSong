#!/usr/bin/env python3
"""Executa benchmark de modelos um por vez, com timeout e retomada.

Este orquestrador chama `scripts/benchmark/run_benchmark.py --model <nome>` para cada
arquitetura. Cada modelo recebe uma pasta própria, log próprio e status próprio.

Exemplos:
  python scripts/benchmark/run_models_sequential.py --dataset data/datasets/benchmark_dataset.npz
  python scripts/benchmark/run_models_sequential.py --models SVM RandomForest --timeout-min 20
  python scripts/benchmark/run_models_sequential.py --neural-only --resume --device-profile gpu
  python scripts/benchmark/run_models_sequential.py --neural-only --plan-only
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import queue
import subprocess
import sys
import threading
import time
import zipfile
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = ROOT / "scripts"

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.core.config.paths import resolve_results_output  # noqa: E402
from benchmarks.config import (  # noqa: E402
    CLASSICAL_TCC_ARCHITECTURES,
    DOCKER_TRAINING_ARCHITECTURES,
    MODEL_FAMILIES,
    NEURAL_DOCKER_ARCHITECTURES,
)

# Selo do teste: implementação compartilhada em benchmarks/test_lock.py, para
# que o entrypoint direto (run_benchmark.py) também consiga verificá-lo.
from benchmarks.test_lock import (  # noqa: E402
    inspect_npz as _inspect_npz,
)
from benchmarks.test_lock import (  # noqa: E402
    sha256_file as _sha256_file,
)
from benchmarks.test_lock import (  # noqa: E402
    validate_test_lock as _validate_test_lock,
)

#: Arquiteturas cujo grafo de treino (STFT/filterbank in-graph + laços de
#: sub-blocos, ex.: `Bottle2neck` do Res2Net) faz o auto-JIT do XLA
#: (`TF_XLA_FLAGS=--tf_xla_auto_jit=1`, ligado por padrão em
#: `app/core/performance.py`) compilar clusters caros o bastante para estourar
#: a RAM do HOST (não da GPU) já no 1º batch — confirmado via OOM-killer do
#: kernel para MultiscaleCNN (mata em ~33-39GB dependendo do teto do
#: container, sempre no mesmo ponto do treino, não crescendo com mais tempo
#: como um pico de dado real cresceria). Mesma classe de sintoma do
#: `Predictor._XLA_UNFRIENDLY_ARCHITECTURES` (inferência) — aqui é o
#: equivalente para o treino, aplicado via env var no subprocesso porque cada
#: modelo já roda isolado (`run_benchmark.py --model <nome>`).
#:
#: AJUSTE 2026-07-31: `aasist` e `rawgatst` (graph attention dinâmico, mesmo
#: motivo pelo qual já constam em `Predictor._XLA_UNFRIENDLY_ARCHITECTURES`
#: linha 522) morriam no run oficial exatamente com a mesma assinatura do
#: MultiscaleCNN pré-fix: processo encerrado logo após a config de GPU, sem
#: nenhuma linha de treino no log — nunca tinham sido adicionados aqui.
#: `rawnet2` morria por um caminho distinto (CUDA_ERROR_OUT_OF_MEMORY
#: explícito tentando alocar ~36 GB numa GPU de 12 GB, mesmo já no cap de
#: batch=16/float32 de `planning._fit_to_device`) — GRU(1024) sob auto-JIT
#: também é conhecido por gerar kernels fundidos com scratch buffer
#: desproporcional; mesma mitigação disponível, root cause de VRAM em vez de
#: RAM do host.
_XLA_UNFRIENDLY_TRAINING_ARCHITECTURES = {"multiscalecnn", "aasist", "rawgatst", "rawnet2"}

SSL_ORIGINAL_MODELS = {
    "wavlm": {
        "display": "WavLM Original",
        "architecture": "wavlm",
        "runner": SCRIPTS / "benchmark" / "run_wavlm_original_benchmark.py",
    },
    "wavlmoriginal": {
        "display": "WavLM Original",
        "architecture": "wavlm",
        "runner": SCRIPTS / "benchmark" / "run_wavlm_original_benchmark.py",
    },
    "hubert": {
        "display": "HuBERT Original",
        "architecture": "hubert",
        "runner": SCRIPTS / "benchmark" / "run_wavlm_original_benchmark.py",
    },
    "hubertoriginal": {
        "display": "HuBERT Original",
        "architecture": "hubert",
        "runner": SCRIPTS / "benchmark" / "run_wavlm_original_benchmark.py",
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


def _git_revision() -> dict[str, Any]:
    """Captura a revisão e o estado dirty sem alterar o repositório."""
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        dirty = (
            subprocess.run(["git", "diff", "--quiet"], cwd=ROOT, check=False).returncode
            != 0
        )
        return {"commit": commit, "dirty": dirty}
    except (OSError, subprocess.SubprocessError) as exc:
        return {"commit": None, "dirty": None, "error": str(exc)}


def _code_identity() -> dict[str, str]:
    files = (
        ROOT / "benchmarks" / "config.py",
        ROOT / "benchmarks" / "planning.py",
        ROOT / "benchmarks" / "runner.py",
        Path(__file__).resolve(),
    )
    return {str(path.relative_to(ROOT)): _sha256_file(path) for path in files}


def _run_fingerprint(args: argparse.Namespace, model: str) -> dict[str, Any]:
    payload = {
        "schema": "xfakesong-run-fingerprint-v1",
        "model": model,
        "scope": getattr(args, "scope", "official"),
        "dataset": str(Path(args.dataset).resolve()),
        "dataset_size": Path(args.dataset).stat().st_size,
        "test_lock_dataset_sha256": (
            (getattr(args, "validated_test_lock", None) or {}).get("dataset_sha256")
        ),
        "command": _build_command(args, model, Path("<MODEL_OUTPUT>")),
        "python": sys.version,
        "git": _git_revision(),
        "code_sha256": _code_identity(),
    }
    encoded = json.dumps(payload, sort_keys=True, ensure_ascii=True, default=str)
    return {**payload, "sha256": hashlib.sha256(encoded.encode("utf-8")).hexdigest()}


def _resume_matches(model_dir: Path, expected: dict[str, Any]) -> bool:
    existing = _load_json(model_dir / "run_fingerprint.json", {})
    return bool(existing and existing.get("sha256") == expected.get("sha256"))


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
        cmd = [
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
            str(args.ssl_train_batch_size),
            "--feature-batch-size",
            str(args.ssl_feature_batch_size),
            "--latency-runs",
            str(args.latency_runs),
            "--seed",
            str(args.seed),
            "--snr",
            *[str(v) for v in args.snr],
            "--train-aug-snr",
            *[str(v) for v in args.train_aug_snr],
            "--waveform-noise-batch-size",
            str(args.waveform_noise_batch_size),
            "--freeze-backbone",
            "--no-calibrate-under-noise",
            "--no-early-stopping",
        ]
        cmd.append(
            "--train-augmentation"
            if args.waveform_train_augmentation
            else "--no-train-augmentation"
        )
        return cmd

    cmd = [
        sys.executable,
        str(SCRIPTS / "benchmark" / "run_benchmark.py"),
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
        "--seed",
        str(args.seed),
        "--snr",
        *[str(v) for v in args.snr],
        "--train-aug-snr",
        *[str(v) for v in args.train_aug_snr],
        "--train-noise-copies",
        str(args.train_noise_copies),
        "--waveform-noise-batch-size",
        str(args.waveform_noise_batch_size),
    ]
    cmd.extend(["--experiment-scope", args.scope])
    cmd.append(
        "--waveform-train-augmentation"
        if args.waveform_train_augmentation
        else "--no-waveform-train-augmentation"
    )
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
    if getattr(args, "cross_generator", None):
        cmd.extend(["--cross-generator", str(args.cross_generator)])
    if getattr(args, "codec_eval", None):
        cmd.extend(["--codec-eval", *[str(c) for c in args.codec_eval]])
    if getattr(args, "academic_protocol", False):
        cmd.append("--fail-on-source-shortcut")
        if getattr(args, "source_shortcut_limit", None) is not None:
            cmd.extend(["--source-shortcut-limit", str(args.source_shortcut_limit)])
    return cmd


def _fit_samples(args: argparse.Namespace) -> int:
    """Amostras que o `fit` realmente ve, incluindo as copias de ruido.

    `expected_training_timeout_min` escala por isto, mas ninguem passava o
    valor: o default do parametro descreve o dataset de 40.980, entao QUALQUER
    outro tamanho herdava o timeout do completo. Num dataset 2,7x menor o limite
    ficava 2,7x mais generoso que o pretendido — falha segura, mas o timeout
    deixava de cumprir a funcao de matar um treino travado.

    Devolve 0 quando o tamanho e desconhecido; a funcao de planejamento cai no
    default nesse caso, que e o comportamento antigo.
    """
    contagens = getattr(args, "npz_split_counts", None) or {}
    treino = int(contagens.get("y_train") or 0)
    if treino <= 0:
        return 0
    copias = int(getattr(args, "train_noise_copies", 0) or 0)
    if not getattr(args, "waveform_train_augmentation", False):
        copias = 0
    return treino * (1 + copias)


def _expected_cost_min(args: argparse.Namespace, model: str) -> float:
    """Custo estimado deste modelo, em minutos (base da ordem de execucao)."""
    try:
        from benchmarks.planning import (
            _REFERENCE_FIT_SAMPLES,
            expected_training_timeout_min,
        )

        # safety_factor=1 e minimum_min=0: aqui interessa a ESTIMATIVA de custo,
        # nao o timeout (que aplica margem e piso e achataria os mais baratos).
        return expected_training_timeout_min(
            model,
            device_profile=getattr(args, "device_profile", "gpu") or "gpu",
            epochs=int(getattr(args, "epochs", 100) or 100),
            fit_samples=_fit_samples(args) or _REFERENCE_FIT_SAMPLES,
            safety_factor=1.0,
            minimum_min=0.0,
        )
    except Exception:  # noqa: BLE001 — sem estimativa, nao reordene
        return float("inf")


def _order_models(args: argparse.Namespace, models: list[str]) -> list[str]:
    """Ordena a suite do mais barato para o mais caro (default `cost`).

    A ordem do manifesto e de RELATORIO, nao de execucao: nela o
    SpectrogramTransformer (~28 h de GPU) roda em 4o, antes do Conformer
    (~1,4 h) e do Res2Net (~1,6 h). Uma falha de ambiente que atinja todos os
    modelos (OOM do container, checkpoint SSL ausente, dataset invalido) so
    apareceria depois de dias de GPU. Executando do mais barato para o mais
    caro, a mesma falha aparece na primeira hora e o run inteiro pode ser
    corrigido antes de queimar as arquiteturas longas.

    A ordem dos RESULTADOS nao depende disto: `consolidate_results.py` reordena
    por `OFFICIAL_TCC_RESULT_ORDER`.
    """
    if getattr(args, "order", "cost") != "cost":
        return list(models)
    indexed = list(enumerate(models))
    indexed.sort(key=lambda pair: (_expected_cost_min(args, pair[1]), pair[0]))
    return [model for _, model in indexed]


def _timeout_for(args: argparse.Namespace, model: str) -> float:
    """Timeout deste modelo: o do usuario, ou o derivado da estimativa.

    Um valor unico para todas as arquiteturas nao existe: no orcamento de 100
    epocas o Sonic Sleuth leva ~0,4 h de GPU e o RawGAT-ST ~54 h. Um timeout
    generoso para o primeiro mata o segundo; um generoso para o segundo deixa
    de proteger contra travamento no primeiro.
    """
    if getattr(args, "timeout_min", None):
        return float(args.timeout_min)
    try:
        from benchmarks.planning import (
            _REFERENCE_FIT_SAMPLES,
            expected_training_timeout_min,
        )

        return expected_training_timeout_min(
            model,
            device_profile=getattr(args, "device_profile", "gpu") or "gpu",
            epochs=int(getattr(args, "epochs", 100) or 100),
            fit_samples=_fit_samples(args) or _REFERENCE_FIT_SAMPLES,
        )
    except Exception:  # noqa: BLE001 — sem estimativa, nao estrangule o run
        return 24 * 60.0


def _run_one(args: argparse.Namespace, model: str, root_out: Path) -> dict[str, Any]:
    slug = _slug(model)
    model_dir = root_out / slug
    model_dir.mkdir(parents=True, exist_ok=True)
    log_path = model_dir / "run.log"
    cmd = _build_command(args, model, model_dir)
    timeout_min = _timeout_for(args, model)
    timeout_s = int(timeout_min * 60)
    started = time.time()
    ssl_meta = _ssl_meta(model)

    if args.plan_only and ssl_meta is not None:
        # Mesmos checkpoints que `run_wavlm_original_benchmark.py::arch_meta`
        # aplica de fato. Estava "microsoft/wavlm-base" enquanto o runner usa
        # base-PLUS desde 2026-07-15: o benchmark_plan.json anunciava um
        # backbone e o treino carregava outro.
        model_name = (
            "facebook/hubert-base-ls960"
            if ssl_meta["architecture"] == "hubert"
            else "microsoft/wavlm-base-plus"
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
            "seed": args.seed,
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

    # Append preserva o histórico quando o contêiner reinicia e o
    # BackupAndRestore retoma uma execução incompleta.
    with log_path.open("a", encoding="utf-8", errors="replace") as log:
        log.write("COMMAND:\n")
        log.write(" ".join(cmd) + "\n\n")
        log.flush()
        try:
            subprocess_env = {
                **os.environ,
                "PYTHONIOENCODING": "utf-8",
                # TF_USE_LEGACY_KERAS=0: transformers.modeling_tf_utils seta
                # =1 no processo que o importa; se o pai estiver poluído, o
                # filho carregaria tensorflow.keras como Keras 2 (tf_keras) e
                # o código Keras 3 do projeto quebraria. O runner SSL
                # (PyTorch) não usa tf.keras — pinar 0 é seguro p/ ambos.
                "TF_USE_LEGACY_KERAS": "0",
            }
            if slug in _XLA_UNFRIENDLY_TRAINING_ARCHITECTURES:
                # Ver _XLA_UNFRIENDLY_TRAINING_ARCHITECTURES: sem isso o
                # auto-JIT do XLA compila um cluster caro o bastante pra
                # estourar memoria (RAM do host para multiscalecnn/aasist/
                # rawgatst, confirmado via OOM-killer do kernel; VRAM da GPU
                # para rawnet2, confirmado via CUDA_ERROR_OUT_OF_MEMORY
                # tentando alocar dezenas de GB numa GPU de 12 GB) ja no 1o
                # batch, independente do teto de memoria do container.
                subprocess_env["XFAKE_ENABLE_XLA"] = "0"
                log.write(
                    f"[XLA] auto-JIT desligado para {model} "
                    "(arquitetura conhecida por estourar memoria na compilacao)\n"
                )
                log.flush()
            proc = subprocess.Popen(
                cmd,
                cwd=str(ROOT),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                env=subprocess_env,
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
            error = f"timeout_min={timeout_min:.0f}"
            log.write(f"\n[TIMEOUT] {model}: {error}\n")
            log.flush()
            _emit(f"[TIMEOUT] {model}: {error}")

    elapsed = round(time.time() - started, 1)
    metrics = {}

    results_path = model_dir / "results.json"
    if results_path.exists():
        data = _load_json(results_path, {})
        if args.academic_protocol and getattr(args, "validated_test_lock", None):
            data["academic_protocol_guard"] = {
                "test_lock": args.validated_test_lock,
                "test_split_sha256": (data.get("dataset") or {}).get(
                    "test_split_sha256"
                ),
                "validated_before_training": True,
            }
            _write_json(results_path, data)
        if not (data.get("persistence") or {}).get("run_uid"):
            try:
                from app.core.db.experiment_store import experiment_store

                experiment_store.ensure_schema()
                run_uid = experiment_store.persist_benchmark_results(
                    data, output_dir=model_dir, source=str(results_path)
                )
                data["persistence"] = {
                    "backend": "sqlite",
                    "run_uid": run_uid,
                }
                _write_json(results_path, data)
            except Exception as exc:
                _emit(f"[WARN] Falha ao consolidar {model} no SQLite: {exc}")
        metrics = (data.get("architectures") or {}).get(model, {})
        if not metrics:
            metrics = next(iter((data.get("architectures") or {}).values()), {})

    return {
        "model": model,
        "status": status,
        "error": error,
        "elapsed_s": elapsed,
        # timeout que valeu para ESTE modelo: sem ele, um status "timeout" no
        # resumo nao diz se o limite era generoso ou apertado demais.
        "timeout_min": round(timeout_min, 1),
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
        f"- Timeout por modelo: `{summary.get('timeout_min')}`",
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


def _aggregate_multiseed(base_out: Path, seeds: list[int]) -> dict[str, Any]:
    """Agrega métricas por modelo mantendo o teste congelado entre seeds."""
    by_model: dict[str, dict[str, list[float]]] = {}
    for seed in seeds:
        summary = _load_json(base_out / f"seed_{seed}" / "run_summary.json", {})
        for item in summary.get("models", []):
            if item.get("status") != "ok":
                continue
            target = by_model.setdefault(str(item.get("model")), {})
            for metric, value in (item.get("clean") or {}).items():
                supported = {"accuracy", "eer", "auc_roc", "f1", "ece"}
                if metric in supported and isinstance(value, (int, float)):
                    target.setdefault(metric, []).append(float(value))
    models: dict[str, Any] = {}
    for model, metrics in by_model.items():
        models[model] = {}
        for metric, values in metrics.items():
            arr = np.asarray(values, dtype="float64")
            models[model][metric] = {
                "n_seeds": int(len(arr)),
                "mean": float(np.mean(arr)),
                "std": float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0,
                "min": float(np.min(arr)),
                "max": float(np.max(arr)),
            }
    payload = {
        "schema": "xfakesong-multiseed-summary-v1",
        "seeds": seeds,
        "test_policy": "predefined_frozen_npz",
        "models": models,
    }
    _write_json(base_out / "multiseed_summary.json", payload)
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Roda benchmark de modelos um por vez com timeout e resume.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--dataset",
        default="data/datasets/benchmark_dataset.npz",
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
    parser.add_argument(
        "--neural-only",
        action="store_true",
        help="roda arquiteturas neurais do artigo + WavLM/HuBERT SSL Docker",
    )
    parser.add_argument(
        "--classical-only", action="store_true", help="roda somente SVM e RandomForest"
    )
    parser.add_argument("--out", default=None)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--scope", choices=["official", "extended"], default="official")
    parser.add_argument(
        "--test-lock",
        default=None,
        help="manifesto que sela o novo teste antes do treino (default: <dataset>.test-lock.json)",
    )
    parser.add_argument(
        "--academic-protocol",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="exige teste predefinido/congelado e controles acadêmicos padronizados",
    )
    parser.add_argument(
        "--source-shortcut-limit",
        type=float,
        default=None,
        help="sobrepoe o limite do oraculo fonte-rotulo (default: 0.55; "
             "use para datasets com confundimento fonte-classe documentado, "
             "ex.: 0.80 para um acervo cujo oraculo de fonte e 75%%)",
    )
    parser.add_argument(
        "--min-samples",
        type=int,
        default=15000,
        help="cardinalidade mínima do benchmark acadêmico",
    )
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument(
        "--device-profile", choices=["auto", "cpu", "gpu"], default="auto"
    )
    parser.add_argument("--latency-runs", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=None,
        help=(
            "Rigor acadêmico: roda a suíte COMPLETA uma vez por semente "
            "(ex.: --seeds 42 43 44), em subdiretórios seed_<n>/ — permite "
            "reportar média±desvio e testes pareados. Com splits predefinidos "
            "congelados, a semente só muda RNG de treino/ruído (teste fixo). "
            "Sobrepõe --seed."
        ),
    )
    parser.add_argument(
        "--cross-generator",
        metavar="GERADOR",
        default=None,
        help=(
            "Reteste cross-generator (ex.: fkvoice): segura o gerador fora do "
            "treino (repassado ao run_benchmark). Experimento separado do "
            "benchmark principal — não combine com --academic-protocol."
        ),
    )
    parser.add_argument(
        "--codec-eval",
        nargs="+",
        default=None,
        choices=["mp3", "opus"],
        metavar="CODEC",
        help=(
            "Robustez a codec com perdas (round-trip ffmpeg na forma de onda; "
            "repassado ao run_benchmark). Ex.: --codec-eval mp3 opus"
        ),
    )
    parser.add_argument(
        "--ssl-train-batch-size",
        type=int,
        default=128,
        help="batch customizado das cabeças SSL",
    )
    parser.add_argument(
        "--ssl-feature-batch-size",
        type=int,
        default=16,
        help="batch para extracao de embeddings HuBERT/WavLM no runner SSL",
    )
    parser.add_argument(
        "--snr",
        nargs="+",
        type=int,
        default=[30, 20, 10, 5],
        help=(
            "SNRs (dB) do teste de robustez. 30/20/10 casam com o augmentation "
            "de treino e medem condicao CASADA; 5 dB fica FORA do treino e e o "
            "unico nivel que mede generalizacao a ruido"
        ),
    )
    parser.add_argument(
        "--train-aug-snr",
        nargs="+",
        type=int,
        default=[30, 20, 10],
        help=(
            "SNRs balanceados na copia ruidosa de treino. NAO inclua 5 dB: e o "
            "nivel reservado para medir generalizacao na avaliacao"
        ),
    )
    parser.add_argument("--train-noise-copies", type=int, default=1)
    parser.add_argument("--waveform-noise-batch-size", type=int, default=64)
    parser.add_argument(
        "--waveform-train-augmentation",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--timeout-min",
        type=float,
        default=None,
        help=(
            "timeout POR MODELO em minutos. Omitido (recomendado), e derivado "
            "do custo estimado de cada arquitetura "
            "(benchmarks.planning.EXPECTED_TRAINING_HOURS x fator de seguranca "
            "3x), escalado por epocas e tamanho do treino. O default anterior "
            "era 60 min FIXO — menor que o treino de QUALQUER modelo neural no "
            "orcamento de 100 epocas, e portanto matava o run"
        ),
    )
    parser.add_argument(
        "--order",
        choices=["cost", "manifest"],
        default="cost",
        help=(
            "ordem de EXECUÇÃO: `cost` roda do mais barato para o mais caro "
            "(estimativa de benchmarks.planning.EXPECTED_TRAINING_HOURS), para "
            "que uma falha comum a todos apareça em minutos e não depois de "
            "dias de GPU; `manifest` preserva a ordem do manifesto oficial. A "
            "ordem das TABELAS não muda (consolidate_results reordena)"
        ),
    )
    parser.add_argument(
        "--resume", action="store_true", help="pula modelos já concluídos"
    )
    parser.add_argument(
        "--plan-only",
        action="store_true",
        help="gera benchmark_plan.* por modelo e não inicia treino",
    )
    parser.add_argument("--api", action="store_true")
    parser.add_argument("--no-optimize-hparams", action="store_true")
    parser.add_argument(
        "--speaker-split",
        action="store_true",
        help="split disjunto por falante (tier large; requer speaker_ids no .npz)",
    )
    parser.add_argument(
        "--group-split",
        action="store_true",
        help="split por fonte/gerador (cross-generator)",
    )
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

    allowed_models = (
        set(MODEL_FAMILIES["extended"])
        if args.scope == "extended"
        else set(DOCKER_TRAINING_ARCHITECTURES)
    )
    invalid_models = [model for model in selected_models if model not in allowed_models]
    if invalid_models:
        parser.error(
            f"modelos fora do escopo {args.scope}: {invalid_models}. "
            "Execute official e extended em suítes separadas."
        )
    if args.scope == "extended" and args.academic_protocol:
        parser.error("o escopo extended exige --no-academic-protocol")
    dataset_path = Path(args.dataset)
    if not dataset_path.is_absolute():
        dataset_path = ROOT / dataset_path
    if not dataset_path.exists():
        parser.error(f"Dataset não encontrado: {dataset_path}")
    args.dataset = str(dataset_path)
    try:
        npz_inspection = _inspect_npz(dataset_path)
    except (OSError, ValueError, zipfile.BadZipFile) as exc:
        parser.error(f"NPZ inválido: {exc}")
    sample_count = int(npz_inspection["sample_count"])
    # Consumido por `_fit_samples`, que alimenta o escalonamento do timeout pelo
    # tamanho real do treino em vez do default de 40.980 amostras.
    args.npz_split_counts = npz_inspection.get("split_counts") or {}
    test_lock = None
    lock_path = (
        Path(args.test_lock).resolve()
        if args.test_lock
        else dataset_path.with_suffix(dataset_path.suffix + ".test-lock.json")
    )
    if args.academic_protocol:
        try:
            test_lock = _validate_test_lock(dataset_path, npz_inspection, lock_path)
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            parser.error(f"teste acadêmico não selado: {exc}")
    args.validated_test_lock = test_lock
    if args.min_samples > 0 and sample_count < args.min_samples:
        parser.error(
            f"Dataset insuficiente: {sample_count} amostras; "
            f"mínimo acadêmico={args.min_samples}. Use o NPZ em data/datasets."
        )

    root_out = resolve_results_output(
        args.out,
        default_subdir="sequential_benchmark",
        base_dir=ROOT,
    )
    root_out.mkdir(parents=True, exist_ok=True)

    if args.epochs <= 0:
        parser.error("--epochs deve ser positivo")
    if args.train_noise_copies < 0:
        parser.error("--train-noise-copies deve ser >= 0")
    # O orçamento de épocas é independente da integridade dos dados: exigir
    # exatamente 100 no mesmo `if` que valida test-lock/source-shortcut/split
    # predefinido amarrava as duas coisas ao mesmo interruptor, forçando
    # `--no-academic-protocol` (que troca split_policy para
    # preserve_predefined_else_stratified_70_15_15 — arrisca re-split
    # diferente da partição locked) só para reduzir épocas. O que importa
    # para comparação justa entre arquiteturas é o MESMO orçamento para
    # todas (fixed_epoch_budget, já garantido em benchmark_plan), não que
    # esse orçamento seja especificamente 100. As demais garantias
    # (fail-on-source-shortcut, test-lock, predefined_frozen_npz)
    # permanecem obrigatórias sob academic_protocol, sem exceção.
    if args.academic_protocol:
        if not npz_inspection["predefined_splits"]:
            parser.error(
                "protocolo acadêmico exige X_train/y_train/X_val/y_val/X_test/y_test "
                "predefinidos; um split gerado por semente alteraria o teste"
            )
        # Avaliação em 30/20/10 (condição CASADA com o augmentation) MAIS 5 dB,
        # que fica deliberadamente fora do treino e é o único nível que mede
        # generalização a ruído — ver docs/evaluation/benchmark.md e o invariante
        # em tests/unit/test_benchmark_protocol_fixes.py::
        # test_default_protocol_includes_an_unseen_snr_level.
        # Até aqui o guard exigia `snr == [30, 20, 10]`, contradizendo o default
        # de BenchmarkConfig.snr_levels_db e a própria doc: quem passasse o 5 dB
        # documentado tomava parser.error, e quem não passasse rodava um
        # benchmark sem a coluna de generalização.
        if args.snr != [30, 20, 10, 5]:
            parser.error(
                "protocolo acadêmico exige SNRs de avaliação 30, 20, 10 e 5 dB "
                "nessa ordem"
            )
        if args.train_aug_snr != [30, 20, 10]:
            parser.error(
                "protocolo acadêmico exige augmentation de treino em 30, 20 e "
                "10 dB nessa ordem; 5 dB precisa continuar NÃO VISTO no treino"
            )
        if not npz_inspection["has_cluster_ids"]:
            parser.error("protocolo acadêmico exige cluster_ids para IC por cluster")
        if not npz_inspection["has_source_ids"]:
            parser.error(
                "protocolo acadêmico exige source_ids/groups para auditoria de domínio"
            )
        if not args.waveform_train_augmentation or args.train_noise_copies != 1:
            parser.error(
                "protocolo acadêmico exige uma cópia AWGN de treino por amostra"
            )
        if args.group_split or args.speaker_split:
            parser.error(
                "split por grupo/falante deve ser executado como experimento separado; "
                "não pode substituir o teste congelado do benchmark principal"
            )
        if getattr(args, "cross_generator", None):
            parser.error(
                "cross-generator altera o teste; execute como experimento "
                "separado, sem --academic-protocol"
            )

    selected_models = _order_models(args, selected_models)
    _emit(f"[ORDER:{args.order}] {' -> '.join(selected_models)}")

    seeds = list(args.seeds) if args.seeds else [int(args.seed)]
    if len(seeds) != len(set(seeds)):
        parser.error("--seeds contém sementes repetidas")
    base_out = root_out
    exit_codes: list[int] = []
    for sd in seeds:
        args.seed = int(sd)
        suite_out = base_out if len(seeds) == 1 else base_out / f"seed_{sd}"
        suite_out.mkdir(parents=True, exist_ok=True)
        if len(seeds) > 1:
            _emit(f"===== SEMENTE {sd} -> {suite_out} =====")
        exit_codes.append(
            _run_suite(
                args,
                selected_models,
                suite_out,
                npz_inspection,
                sample_count,
                dataset_path,
                test_lock,
            )
        )
    if len(seeds) > 1:
        _write_json(
            base_out / "seeds_manifest.json",
            {
                "seeds": seeds,
                "suite_dirs": [f"seed_{sd}" for sd in seeds],
                "note": (
                    "Teste idêntico entre sementes quando o NPZ traz splits "
                    "predefinidos; a semente varia inicialização/ordem/ruído "
                    "de treino. Reporte média±desvio e testes pareados."
                ),
            },
        )
        _aggregate_multiseed(base_out, seeds)
    return max(exit_codes) if exit_codes else 2


def _run_suite(
    args: argparse.Namespace,
    selected_models: list[str],
    root_out: Path,
    npz_inspection: dict[str, Any],
    sample_count: int,
    dataset_path: Path,
    test_lock: dict[str, Any] | None,
) -> int:
    protocol_manifest = {
        "protocol_version": "waveform-awgn-v2",
        "standardized_controls": {
            "epochs": int(args.epochs),
            "minimum_dataset_samples": int(args.min_samples),
            "fixed_epoch_budget": True,
            "early_stopping": False,
            "checkpoint_selection": "minimum_clean_validation_loss",
            "decision_threshold": 0.5,
            "seed": int(args.seed),
            "split_policy": (
                "predefined_frozen_npz"
                if args.academic_protocol
                else "preserve_predefined_else_stratified_70_15_15"
            ),
            "fail_on_exact_split_overlap": True,
            "waveform_awgn_before_frontend": True,
            "test_snr_db": [int(v) for v in args.snr],
            "train_aug_snr_db": [int(v) for v in args.train_aug_snr],
            "train_noise_copies": int(args.train_noise_copies),
            "waveform_noise_batch_size": int(args.waveform_noise_batch_size),
            "latency_runs": int(args.latency_runs),
        },
        "dataset_preflight": npz_inspection,
        "test_lock": test_lock,
        "academic_protocol": bool(args.academic_protocol),
        "experiment_scope": args.scope,
        "model_specific_hyperparameters_preserved": [
            "learning_rate",
            "batch_size",
            "optimizer",
            "scheduler",
            "dropout",
            "weight_decay",
            "l2",
            "architecture_parameters",
        ],
    }
    root_out.joinpath("benchmark_protocol.json").write_text(
        json.dumps(protocol_manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
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
        "dataset_samples": sample_count,
        "dataset_preflight": npz_inspection,
        "test_lock": test_lock,
        "academic_protocol": bool(args.academic_protocol),
        "experiment_scope": args.scope,
        "device_profile": args.device_profile,
        "timeout_min": args.timeout_min or "derivado por arquitetura",
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "ssl_train_batch_size": args.ssl_train_batch_size,
        "ssl_feature_batch_size": args.ssl_feature_batch_size,
        "seed": args.seed,
        "snr": args.snr,
        "train_aug_snr": args.train_aug_snr,
        "train_noise_copies": args.train_noise_copies,
        "waveform_train_augmentation": args.waveform_train_augmentation,
        "standardized_controls": protocol_manifest["standardized_controls"],
        "models": summary.get("models", []),
    }

    existing_by_model = {item.get("model"): item for item in summary["models"]}
    for model in selected_models:
        model_dir = root_out / _slug(model)
        done = _plan_done(model_dir) if args.plan_only else _model_done(model_dir)
        fingerprint = _run_fingerprint(args, model)
        if args.resume and model in completed and done:
            if _resume_matches(model_dir, fingerprint):
                _emit(f"[SKIP] {model} ja concluido; fingerprint confere")
                continue
            _emit(
                f"[RERUN] {model}: artefato antigo não corresponde ao protocolo atual"
            )
        _write_json(model_dir / "run_fingerprint.json", fingerprint)
        _write_json(
            model_dir / "effective_training_config.json",
            {
                "schema": "xfakesong-effective-training-config-v1",
                "sha256": fingerprint["sha256"],
                "model": model,
                "scope": args.scope,
                "command": fingerprint["command"],
                "dataset": fingerprint["dataset"],
                "test_lock_dataset_sha256": fingerprint["test_lock_dataset_sha256"],
            },
        )

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
    summary["status"] = (
        "ok" if statuses and all(s == "ok" for s in statuses) else "partial"
    )
    _write_json(summary_path, summary)
    _write_summary(root_out, summary)
    _emit(f"Resumo: {summary_path}")
    return 0 if summary["status"] == "ok" else 2


if __name__ == "__main__":
    raise SystemExit(main())
