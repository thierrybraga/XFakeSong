import numpy as np
from typing import Optional, Dict
from .sample import compute_sample_entropy, compute_sample_entropy_optimized


def compute_multiscale_entropy(
        y: np.ndarray, max_scale: int = 10) -> np.ndarray:
    """Calcula a entropia multiescala."""
    entropies = []

    for scale in range(1, max_scale + 1):
        # Coarse-graining
        if scale == 1:
            coarse_y = y
        else:
            coarse_length = len(y) // scale
            coarse_y = np.zeros(coarse_length)
            for i in range(coarse_length):
                coarse_y[i] = np.mean(y[i * scale:(i + 1) * scale])

        # Calcular entropia de amostra
        if len(coarse_y) > 10:
            se = compute_sample_entropy(coarse_y)
            entropies.append(se)
        else:
            entropies.append(np.nan)

    return np.array(entropies)


def compute_multiscale_entropy_optimized(
        y: np.ndarray, max_scale: int = 8,
        cache: Optional[Dict] = None) -> float:
    """Versão otimizada da entropia multiescala com menos escalas."""
    N = len(y)

    # Limitar tamanho e escalas para performance
    if N > 3000:
        step = N // 3000
        y = y[::step][:3000]
        N = len(y)

    # Reduzir número de escalas para performance
    max_scale = min(max_scale, N // 20)

    # Cache key
    cache_key = f"mse_{hash(y.tobytes())}_{max_scale}"
    if cache is not None and cache_key in cache:
        return cache[cache_key]

    entropies = []

    for scale in range(1, max_scale + 1):
        # Coarse-graining otimizado
        if scale == 1:
            coarse_y = y
        else:
            coarse_length = N // scale
            if coarse_length < 10:
                entropies.append(np.nan)
                continue

            # Usar reshape quando possível para melhor performance
            if N % scale == 0:
                coarse_y = y[:coarse_length *
                             scale].reshape(-1, scale).mean(axis=1)
            else:
                coarse_y = np.array(
                    [np.mean(y[i * scale:(i + 1) * scale])
                     for i in range(coarse_length)])

        # Calcular entropia de amostra otimizada
        if len(coarse_y) > 10:
            se = compute_sample_entropy_optimized(coarse_y)
            entropies.append(se)
        else:
            entropies.append(np.nan)

    # Retornar média das entropias válidas
    valid_entropies = [e for e in entropies if not np.isnan(e)]
    if valid_entropies:
        result = np.mean(valid_entropies)
    else:
        result = np.nan

    # Cache resultado
    if cache is not None:
        cache[cache_key] = result

    return result
