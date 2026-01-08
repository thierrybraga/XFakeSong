import numpy as np
from typing import Optional, Dict


def compute_approximate_entropy(
        y: np.ndarray, m: int = 2, r: float = 0.2) -> float:
    """Calcula a entropia aproximada (ApEn)."""
    N = len(y)

    def _maxdist(xi, xj, N, m):
        return max([abs(ua - va) for ua, va in zip(xi[0:m], xj[0:m])])

    def _phi(m):
        patterns = np.array([y[i:i + m] for i in range(N - m + 1)])
        C = np.zeros(N - m + 1)

        for i in range(N - m + 1):
            template_i = patterns[i]
            for j in range(N - m + 1):
                if _maxdist(template_i, patterns[j], N, m) <= r * np.std(y):
                    C[i] += 1.0

        phi = (N - m + 1.0) ** (-1) * \
            sum([np.log(c / (N - m + 1.0)) for c in C])
        return phi

    return _phi(m) - _phi(m + 1)


def compute_approximate_entropy_optimized(
        y: np.ndarray, m: int = 2, r: float = 0.2,
        cache: Optional[Dict] = None) -> float:
    """Versão otimizada da entropia aproximada com limitações de performance."""
    N = len(y)

    # Limitar tamanho para evitar complexidade O(N²)
    if N > 1000:
        step = N // 1000
        y = y[::step][:1000]
        N = len(y)

    if N < 20:
        return np.nan

    # Cache key
    cache_key = f"apen_{hash(y.tobytes())}_{m}_{r}"
    if cache is not None and cache_key in cache:
        return cache[cache_key]

    try:
        # Implementação ultra-otimizada com amostragem
        r_threshold = r * np.std(y)

        # Limitar ainda mais para performance extrema
        max_patterns = 200
        if N > max_patterns:
            indices = np.linspace(0, N - m - 1, max_patterns, dtype=int)
            y_sample = y[indices[0]:indices[-1] + m + 1]
            N_sample = len(y_sample)
        else:
            y_sample = y
            N_sample = N

        def _phi_fast(m_val):
            if N_sample < m_val + 1:
                return 0

            patterns = np.array([y_sample[i:i + m_val]
                                for i in range(N_sample - m_val + 1)])
            n_patterns = len(patterns)

            if n_patterns == 0:
                return 0

            phi_sum = 0
            # Usar broadcasting para acelerar
            for i in range(0, n_patterns, max(
                    1, n_patterns // 50)):  # Amostragem de padrões
                pattern = patterns[i]
                # Calcular distâncias usando broadcasting
                distances = np.max(np.abs(patterns - pattern), axis=1)
                matches = np.sum(distances <= r_threshold)

                if matches > 0:
                    phi_sum += np.log(matches / n_patterns)

            return phi_sum / min(50, n_patterns)

        phi_m = _phi_fast(m)
        phi_m1 = _phi_fast(m + 1)

        result = phi_m - phi_m1

    except Exception:
        result = np.nan

    # Cache resultado
    if cache is not None:
        cache[cache_key] = result

    return result
