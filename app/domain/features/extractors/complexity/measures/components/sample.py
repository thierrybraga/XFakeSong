from typing import Dict, Optional

import numpy as np
from numba import njit, prange


@njit(fastmath=True)
def _phi_sampen_njit(y: np.ndarray, m: int, r_std: float) -> int:
    N = len(y)
    n_patterns = N - m + 1
    count = 0

    for i in range(n_patterns - 1):
        pattern_i = y[i:i + m]
        for j in range(i + 1, n_patterns):
            max_dist = 0.0
            for k in range(m):
                dist = abs(pattern_i[k] - y[j + k])
                if dist > max_dist:
                    max_dist = dist

            if max_dist <= r_std:
                count += 1

    return count


@njit(parallel=True, fastmath=True)
def _phi_sampen_parallel_njit(y: np.ndarray, m: int, r_std: float) -> int:
    N = len(y)
    n_patterns = N - m + 1

    # Numba prange requires manual reduction for counts
    counts = np.zeros(n_patterns - 1, dtype=np.int64)

    for i in prange(n_patterns - 1):
        pattern_i = y[i:i + m]
        c = 0
        for j in range(i + 1, n_patterns):
            max_dist = 0.0
            for k in range(m):
                dist = abs(pattern_i[k] - y[j + k])
                if dist > max_dist:
                    max_dist = dist

            if max_dist <= r_std:
                c += 1
        counts[i] = c

    return np.sum(counts)


def compute_sample_entropy(y: np.ndarray, m: int = 2, r: float = 0.2) -> float:
    """Calcula a entropia de amostra (SampEn) otimizada com Numba."""
    if len(y) < m + 2:
        return np.nan

    r_std = r * np.std(y)

    if len(y) > 500:
        A = _phi_sampen_parallel_njit(y, m, r_std)
        B = _phi_sampen_parallel_njit(y, m + 1, r_std)
    else:
        A = _phi_sampen_njit(y, m, r_std)
        B = _phi_sampen_njit(y, m + 1, r_std)

    if A == 0 or B == 0:
        return np.inf

    return -np.log(B / A)


def compute_sample_entropy_optimized(
        y: np.ndarray, m: int = 2, r: float = 0.2,
        cache: Optional[Dict] = None) -> float:
    """Versão otimizada da entropia de amostra."""
    N = len(y)

    # Limitar tamanho para performance
    if N > 1500:
        step = N // 1500
        y = y[::step][:1500]
        N = len(y)

    if N < 10:
        return np.nan

    # Cache key
    cache_key = f"sampen_{hash(y.tobytes())}_{m}_{r}"
    if cache is not None and cache_key in cache:
        return cache[cache_key]

    r_threshold = r * np.std(y)

    def _phi_optimized(m_val):
        patterns = np.array([y[i:i + m_val] for i in range(N - m_val + 1)])
        count = 0

        # Usar apenas metade das comparações (matriz triangular superior)
        for i in range(len(patterns) - 1):
            distances = np.max(np.abs(patterns[i + 1:] - patterns[i]), axis=1)
            count += np.sum(distances <= r_threshold)

        return count

    A = _phi_optimized(m)
    B = _phi_optimized(m + 1)

    if A == 0 or B == 0:
        result = np.inf
    else:
        result = -np.log(B / A)

    # Cache resultado
    if cache is not None:
        cache[cache_key] = result

    return result
