import numpy as np
from typing import Optional, Dict


def compute_lyapunov_exponent(y: np.ndarray) -> float:
    """Estima o maior expoente de Lyapunov."""
    # Implementação simplificada usando método de Wolf
    N = len(y)
    if N < 100:
        return np.nan

    # Parâmetros
    embedding_dim = 3
    delay = 1

    # Embedding
    embedded = np.zeros((N - embedding_dim + 1, embedding_dim))
    for i in range(N - embedding_dim + 1):
        embedded[i] = y[i:i + embedding_dim]

    # Calcular expoente
    lyap_sum = 0
    count = 0

    for i in range(len(embedded) - 10):
        # Encontrar vizinho mais próximo
        distances = np.linalg.norm(embedded - embedded[i], axis=1)
        distances[i] = np.inf  # Excluir o próprio ponto

        nearest_idx = np.argmin(distances)

        if distances[nearest_idx] > 0:
            # Evoluir por alguns passos
            steps = min(10, len(embedded) - max(i, nearest_idx) - 1)

            if steps > 1:
                initial_dist = distances[nearest_idx]
                final_dist = np.linalg.norm(
                    embedded[i + steps] - embedded[nearest_idx + steps])

                if final_dist > 0 and initial_dist > 0:
                    lyap_sum += np.log(final_dist / initial_dist) / steps
                    count += 1

    if count > 0:
        return lyap_sum / count
    else:
        return np.nan


def compute_lyapunov_exponent_optimized(
        y: np.ndarray, cache: Optional[Dict] = None) -> float:
    """Versão otimizada do expoente de Lyapunov com menos pontos."""
    N = len(y)

    # Limitar tamanho para performance
    if N > 800:
        step = N // 800
        y = y[::step][:800]
        N = len(y)

    if N < 50:
        return np.nan

    # Cache key
    cache_key = f"lyap_{hash(y.tobytes())}"
    if cache is not None and cache_key in cache:
        return cache[cache_key]

    # Parâmetros reduzidos
    embedding_dim = 2  # Reduzido de 3
    delay = 1

    # Embedding
    embedded = np.zeros((N - embedding_dim + 1, embedding_dim))
    for i in range(N - embedding_dim + 1):
        embedded[i] = y[i:i + embedding_dim]

    # Amostragem de pontos
    max_points = 200
    if len(embedded) > max_points:
        indices = np.random.choice(len(embedded), max_points, replace=False)
        embedded = embedded[indices]

    # Calcular expoente com menos iterações
    lyap_sum = 0
    count = 0

    for i in range(min(len(embedded) - 5, 50)):  # Menos iterações
        # Encontrar vizinho mais próximo
        distances = np.linalg.norm(embedded - embedded[i], axis=1)
        distances[i] = np.inf

        nearest_idx = np.argmin(distances)

        if distances[nearest_idx] > 0:
            # Evoluir por menos passos
            steps = min(5, len(embedded) - max(i, nearest_idx) - 1)

            if steps > 1:
                initial_dist = distances[nearest_idx]
                final_dist = np.linalg.norm(
                    embedded[i + steps] - embedded[nearest_idx + steps])

                if final_dist > 0 and initial_dist > 0:
                    lyap_sum += np.log(final_dist / initial_dist) / steps
                    count += 1

    if count > 0:
        result = lyap_sum / count
    else:
        result = np.nan

    # Cache resultado
    if cache is not None:
        cache[cache_key] = result

    return result
