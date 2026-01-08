import numpy as np
from typing import Optional, Dict


def compute_higuchi_fractal(y: np.ndarray, k_max: int = 10) -> float:
    """Calcula a dimensão fractal de Higuchi."""
    N = len(y)
    L = []

    for k in range(1, k_max + 1):
        Lk = []

        for m in range(k):
            # Construir subsequência
            indices = np.arange(m, N, k)
            if len(indices) < 2:
                continue

            subsequence = y[indices]

            # Calcular comprimento da curva
            length = 0
            for i in range(len(subsequence) - 1):
                length += abs(subsequence[i + 1] - subsequence[i])

            # Normalizar
            length = length * (N - 1) / (len(subsequence) - 1) / k
            Lk.append(length)

        if Lk:
            L.append(np.mean(Lk))
        else:
            L.append(np.nan)

    # Remover valores inválidos
    valid_indices = np.isfinite(L) & (np.array(L) > 0)

    if np.sum(valid_indices) < 3:
        return np.nan

    valid_k = np.arange(1, k_max + 1)[valid_indices]
    valid_L = np.array(L)[valid_indices]

    # Ajuste linear em escala log-log
    log_k = np.log(valid_k)
    log_L = np.log(valid_L)

    slope = np.polyfit(log_k, log_L, 1)[0]

    return -slope  # Dimensão fractal de Higuchi


def compute_higuchi_fractal_optimized(
        y: np.ndarray, k_max: int = 6, cache: Optional[Dict] = None) -> float:
    """Versão otimizada da dimensão fractal de Higuchi com menos k."""
    N = len(y)

    # Limitar tamanho
    if N > 2000:
        step = N // 2000
        y = y[::step][:2000]
        N = len(y)

    if N < 20:
        return np.nan

    # Cache key
    cache_key = f"higuchi_{hash(y.tobytes())}_{k_max}"
    if cache is not None and cache_key in cache:
        return cache[cache_key]

    # Reduzir k_max para performance
    k_max = min(k_max, N // 10)
    L = []

    for k in range(1, k_max + 1):
        Lk = []

        # Menos subsequências para performance
        for m in range(0, k, max(1, k // 4)):  # Pular algumas subsequências
            indices = np.arange(m, N, k)
            if len(indices) < 2:
                continue

            subsequence = y[indices]

            # Calcular comprimento da curva
            length = np.sum(np.abs(np.diff(subsequence)))

            # Normalizar
            length = length * (N - 1) / (len(subsequence) - 1) / k
            Lk.append(length)

        if Lk:
            L.append(np.mean(Lk))
        else:
            L.append(np.nan)

    # Remover valores inválidos
    valid_indices = np.isfinite(L) & (np.array(L) > 0)

    if np.sum(valid_indices) < 3:
        result = np.nan
    else:
        valid_k = np.arange(1, k_max + 1)[valid_indices]
        valid_L = np.array(L)[valid_indices]

        # Ajuste linear em escala log-log
        log_k = np.log(valid_k)
        log_L = np.log(valid_L)

        result = -np.polyfit(log_k, log_L, 1)[0]

    # Cache resultado
    if cache is not None:
        cache[cache_key] = result

    return result
