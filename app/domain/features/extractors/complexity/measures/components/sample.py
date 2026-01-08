import numpy as np
from typing import Optional, Dict


def compute_sample_entropy(y: np.ndarray, m: int = 2, r: float = 0.2) -> float:
    """Calcula a entropia de amostra (SampEn)."""
    N = len(y)

    def _maxdist(xi, xj, m):
        return max([abs(ua - va) for ua, va in zip(xi, xj)])

    def _phi(m):
        patterns = np.array([y[i:i + m] for i in range(N - m + 1)])
        C = 0

        for i in range(N - m):
            template_i = patterns[i]
            for j in range(i + 1, N - m + 1):
                if _maxdist(template_i, patterns[j], m) <= r * np.std(y):
                    C += 1

        return C

    A = _phi(m)
    B = _phi(m + 1)

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
