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
        logger.warning("Medição de latência falhou: %s", exc)
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
