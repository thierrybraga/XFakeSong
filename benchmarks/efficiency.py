"""Medições de eficiência: parâmetros, tamanho em disco e latência."""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Callable, Optional

import numpy as np

logger = logging.getLogger("benchmark")


def count_params(model) -> Optional[int]:
    """Nº de parâmetros (Keras). None para modelos clássicos (sklearn)."""
    try:
        return int(model.count_params())
    except Exception as exc:
        logger.warning("Contagem de parâmetros falhou: %s", exc)
        return None


def file_size_mb(path) -> Optional[float]:
    """Tamanho do artefato salvo em MB (None se ausente)."""
    try:
        p = Path(path)
        if p.exists():
            return round(p.stat().st_size / (1024 * 1024), 2)
    except Exception:
        pass
    return None


def describe_runtime(runtime: str) -> dict[str, object]:
    """Identifica o runtime que executou o forward medido.

    MOTIVAÇÃO 2026-08-09: o escopo oficial roda em TRÊS runtimes — Keras/TF para
    os 7 neurais, PyTorch para WavLM/HuBERT Original e scikit-learn para
    SVM/RandomForest. Uma latência de 17 ms contra 53 ms não separa arquitetura
    de runtime, e a figura de tradeoff acurácia × latência apresentava os três
    na mesma escala sem nada dizendo. O número continua o mesmo; o que muda é
    que o artefato passa a declarar em que pilha ele foi medido, para que a
    legenda possa ressalvar em vez de o leitor supor comparabilidade.
    """
    info: dict[str, object] = {"runtime": runtime}
    versions = {
        "keras": ("tensorflow", "__version__"),
        "pytorch": ("torch", "__version__"),
        "sklearn": ("sklearn", "__version__"),
    }
    module_name, attr = versions.get(runtime, (None, None))
    if module_name:
        try:  # import tardio: não força TF/torch em quem não usa
            import importlib

            info["runtime_version"] = str(
                getattr(importlib.import_module(module_name), attr, None)
            )
        except Exception:  # noqa: BLE001 — versão é informativa, não crítica
            info["runtime_version"] = None
    if runtime == "keras":
        try:
            import tensorflow as tf

            gpus = tf.config.list_physical_devices("GPU")
            info["device"] = "gpu" if gpus else "cpu"
        except Exception:  # noqa: BLE001
            info["device"] = None
    elif runtime == "pytorch":
        try:
            import torch

            info["device"] = "gpu" if torch.cuda.is_available() else "cpu"
        except Exception:  # noqa: BLE001
            info["device"] = None
    elif runtime == "sklearn":
        info["device"] = "cpu"
    info["cross_runtime_comparable"] = False
    return info


def measure_latency_profile(
    predict_fn: Callable[[np.ndarray], object],
    x_sample: np.ndarray,
    runs: int = 30,
    warmup: int = 2,
    runtime: str = "unknown",
) -> dict[str, object]:
    """Perfil de forward com protocolo explícito e estatísticas robustas.

    `runtime` identifica a pilha de execução ("keras", "pytorch", "sklearn").
    Medições de runtimes diferentes NÃO são diretamente comparáveis — ver
    `describe_runtime`.
    """
    x = np.asarray(x_sample, dtype="float32")[np.newaxis, ...]
    try:
        for _ in range(max(0, warmup)):
            predict_fn(x)
        times = []
        for _ in range(max(1, runs)):
            t0 = time.perf_counter()
            predict_fn(x)
            times.append((time.perf_counter() - t0) * 1000.0)
        values = np.asarray(times, dtype="float64")
        profile: dict[str, object] = {
            "status": "ok",
            "component": "model_forward_only",
            "batch_size": 1,
            "warmup_runs": int(max(0, warmup)),
            "measured_runs": int(max(1, runs)),
            "median_ms": round(float(np.median(values)), 2),
            "p95_ms": round(float(np.percentile(values, 95)), 2),
            "mean_ms": round(float(np.mean(values)), 2),
            "std_ms": round(float(np.std(values)), 2),
            "includes_frontend": False,
            "includes_postprocessing": False,
        }
        profile.update(describe_runtime(runtime))
        return profile
    except Exception as exc:
        return {"status": "error", "error": str(exc), "runtime": runtime}

def measure_latency_ms(
    predict_fn: Callable[[np.ndarray], object],
    x_sample: np.ndarray,
    runs: int = 30,
    warmup: int = 2,
) -> Optional[float]:
    """Latência mediana (ms) de uma inferência de 1 amostra.

    `predict_fn` recebe um batch (1, *shape) e roda a inferência. Faz `warmup`
    chamadas (descartadas) para amortizar JIT/alocação, depois mede `runs`.
    Mediana é mais robusta a outliers de SO que a média.
    """
    x = np.asarray(x_sample, dtype="float32")[np.newaxis, ...]
    try:
        for _ in range(max(0, warmup)):
            predict_fn(x)
        times = []
        for _ in range(max(1, runs)):
            t0 = time.perf_counter()
            predict_fn(x)
            times.append((time.perf_counter() - t0) * 1000.0)
        return round(float(np.median(times)), 2)
    except Exception:
        return None
