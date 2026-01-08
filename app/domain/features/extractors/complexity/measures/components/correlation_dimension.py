import numpy as np
from typing import Optional, Dict


def compute_correlation_dimension(
        y: np.ndarray, embedding_dim: int = 5) -> float:
    """Calcula a dimensão de correlação."""
    # Embedding
    N = len(y)
    embedded = np.zeros((N - embedding_dim + 1, embedding_dim))

    for i in range(N - embedding_dim + 1):
        embedded[i] = y[i:i + embedding_dim]

    # Calcular distâncias
    distances = []
    for i in range(len(embedded)):
        for j in range(i + 1, len(embedded)):
            dist = np.linalg.norm(embedded[i] - embedded[j])
            distances.append(dist)

    distances = np.array(distances)

    # Calcular dimensão de correlação
    r_values = np.logspace(-3, 0, 20) * np.std(distances)
    correlations = []

    for r in r_values:
        correlation = np.sum(distances < r) / len(distances)
        correlations.append(correlation + 1e-10)  # Evitar log(0)

    # Ajuste linear em escala log-log
    log_r = np.log(r_values)
    log_c = np.log(correlations)

    # Encontrar região linear
    valid_indices = np.isfinite(log_c) & (log_c > -10)
    if np.sum(valid_indices) < 3:
        return np.nan

    slope = np.polyfit(log_r[valid_indices], log_c[valid_indices], 1)[0]

    return slope


def compute_correlation_dimension_optimized(
        y: np.ndarray, embedding_dim: int = 3,
        cache: Optional[Dict] = None) -> float:
    """Versão otimizada da dimensão de correlação com amostragem."""
    N = len(y)

    # Limitar tamanho drasticamente para performance
    if N > 1000:
        step = N // 1000
        y = y[::step][:1000]
        N = len(y)

    if N < 20:
        return np.nan

    # Cache key
    cache_key = f"corrdim_{hash(y.tobytes())}_{embedding_dim}"
    if cache is not None and cache_key in cache:
        return cache[cache_key]

    # Embedding com dimensão reduzida
    embedding_dim = min(embedding_dim, 3)  # Limitar dimensão
    embedded = np.zeros((N - embedding_dim + 1, embedding_dim))

    for i in range(N - embedding_dim + 1):
        embedded[i] = y[i:i + embedding_dim]

    # Amostragem de pontos para reduzir complexidade
    max_points = 500
    if len(embedded) > max_points:
        indices = np.random.choice(len(embedded), max_points, replace=False)
        embedded = embedded[indices]

    # Calcular distâncias usando broadcasting otimizado
    distances = np.linalg.norm(
        embedded[:, None, :] - embedded[None, :, :], axis=2)
    distances = distances[np.triu_indices_from(
        distances, k=1)]  # Apenas triangular superior

    # Calcular dimensão de correlação com menos pontos
    std_dist = np.std(distances)
    if std_dist < 1e-8:
        return 0.0

    r_values = np.logspace(-3, 0, 10) * std_dist
    correlations = []

    for r in r_values:
        correlation = np.sum(distances < r) / len(distances)
        correlations.append(correlation + 1e-10)

    # Ajuste linear em escala log-log
    log_r = np.log(r_values)
    log_c = np.log(correlations)

    valid_indices = np.isfinite(log_c) & (log_c > -10)
    if np.sum(valid_indices) < 3:
        result = np.nan
    else:
        result = np.polyfit(log_r[valid_indices], log_c[valid_indices], 1)[0]

    # Cache resultado
    if cache is not None:
        cache[cache_key] = result

    return result
