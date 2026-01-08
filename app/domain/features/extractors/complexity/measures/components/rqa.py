import numpy as np
from typing import Optional, Dict, List, Tuple


def _find_diagonal_lines(
        matrix: np.ndarray,
        min_length: int = 2) -> List[List[Tuple[int, int]]]:
    """Encontra linhas diagonais na matriz de recorrência."""
    M, N = matrix.shape
    lines = []

    # Verificar diagonais principais
    for offset in range(-M + 1, N):
        diagonal = []

        if offset >= 0:
            for i in range(min(M, N - offset)):
                if matrix[i, i + offset] == 1:
                    diagonal.append((i, i + offset))
                else:
                    if len(diagonal) >= min_length:
                        lines.append(diagonal)
                    diagonal = []
        else:
            for i in range(min(M + offset, N)):
                if matrix[i - offset, i] == 1:
                    diagonal.append((i - offset, i))
                else:
                    if len(diagonal) >= min_length:
                        lines.append(diagonal)
                    diagonal = []

        if len(diagonal) >= min_length:
            lines.append(diagonal)

    return lines


def _find_diagonal_lines_optimized(
        matrix: np.ndarray, min_length: int = 2) -> list:
    """Versão otimizada para encontrar linhas diagonais."""
    lines = []
    rows, cols = matrix.shape

    # Verificar apenas algumas diagonais para performance
    for offset in range(-min(rows, cols) //
                        4, min(rows, cols) // 4, 2):  # Pular algumas
        diagonal = np.diagonal(matrix, offset=offset)

        # Encontrar sequências de True
        current_line = []
        for i, val in enumerate(diagonal):
            if val:
                current_line.append(i)
            else:
                if len(current_line) >= min_length:
                    lines.append(current_line)
                current_line = []

        if len(current_line) >= min_length:
            lines.append(current_line)

    return lines


def _find_vertical_lines(
        matrix: np.ndarray,
        min_length: int = 2) -> List[List[Tuple[int, int]]]:
    """Encontra linhas verticais na matriz de recorrência."""
    M, N = matrix.shape
    lines = []

    for j in range(N):
        line = []
        for i in range(M):
            if matrix[i, j] == 1:
                line.append((i, j))
            else:
                if len(line) >= min_length:
                    lines.append(line)
                line = []

        if len(line) >= min_length:
            lines.append(line)

    return lines


def _find_vertical_lines_optimized(
        matrix: np.ndarray, min_length: int = 2) -> list:
    """Versão otimizada para encontrar linhas verticais."""
    lines = []
    rows, cols = matrix.shape

    # Verificar apenas algumas colunas para performance
    for col in range(0, cols, max(1, cols // 20)):  # Pular muitas colunas
        current_line = []
        for row in range(rows):
            if matrix[row, col]:
                current_line.append(row)
            else:
                if len(current_line) >= min_length:
                    lines.append(current_line)
                current_line = []

        if len(current_line) >= min_length:
            lines.append(current_line)

    return lines


def compute_rqa_features(y: np.ndarray, embedding_dim: int = 3,
                         threshold: float = 0.1) -> Dict[str, float]:
    """Calcula características de Recurrence Quantification Analysis."""
    N = len(y)

    # Embedding
    embedded = np.zeros((N - embedding_dim + 1, embedding_dim))
    for i in range(N - embedding_dim + 1):
        embedded[i] = y[i:i + embedding_dim]

    # Matriz de recorrência
    M = len(embedded)
    recurrence_matrix = np.zeros((M, M))

    threshold_value = threshold * np.std(y)

    for i in range(M):
        for j in range(M):
            dist = np.linalg.norm(embedded[i] - embedded[j])
            if dist < threshold_value:
                recurrence_matrix[i, j] = 1

    # Calcular medidas RQA
    features = {}

    # Recurrence Rate
    features['rqa_recurrence_rate'] = np.sum(recurrence_matrix) / (M * M)

    # Determinism
    diagonal_lines = _find_diagonal_lines(recurrence_matrix, min_length=2)
    total_diagonal_points = sum(len(line) for line in diagonal_lines)
    total_recurrence_points = np.sum(recurrence_matrix)

    if total_recurrence_points > 0:
        features['rqa_determinism'] = total_diagonal_points / \
            total_recurrence_points
    else:
        features['rqa_determinism'] = 0

    # Average Diagonal Line Length
    if diagonal_lines:
        features['rqa_avg_diagonal_length'] = np.mean(
            [len(line) for line in diagonal_lines])
        features['rqa_max_diagonal_length'] = max(
            len(line) for line in diagonal_lines)
    else:
        features['rqa_avg_diagonal_length'] = 0
        features['rqa_max_diagonal_length'] = 0

    # Laminarity
    vertical_lines = _find_vertical_lines(recurrence_matrix, min_length=2)
    total_vertical_points = sum(len(line) for line in vertical_lines)

    if total_recurrence_points > 0:
        features['rqa_laminarity'] = total_vertical_points / \
            total_recurrence_points
    else:
        features['rqa_laminarity'] = 0

    return features


def compute_rqa_features_optimized(y: np.ndarray, embedding_dim: int = 3,
                                   delay: int = 1, threshold: float = 0.1,
                                   cache: Optional[Dict] = None) -> dict:
    """Versão otimizada da análise de recorrência com menos pontos."""
    N = len(y)

    # Limitar tamanho drasticamente para RQA
    if N > 500:
        step = N // 500
        y = y[::step][:500]
        N = len(y)

    if N < 20:
        return {'rqa_recurrence_rate': np.nan,
                'rqa_determinism': np.nan, 'rqa_laminarity': np.nan}

    # Cache key
    cache_key = f"rqa_{hash(y.tobytes())}_{embedding_dim}_{delay}_{threshold}"
    if cache is not None and cache_key in cache:
        return cache[cache_key]

    # Embedding com menos pontos
    embedded = np.zeros((N - embedding_dim + 1, embedding_dim))
    for i in range(N - embedding_dim + 1):
        embedded[i] = y[i:i + embedding_dim]

    M = len(embedded)

    if M < 10:
        result = {
            'rqa_recurrence_rate': np.nan,
            'rqa_determinism': np.nan,
            'rqa_laminarity': np.nan}
    else:
        # Calcular matriz de recorrência com amostragem
        sample_size = min(M, 100)  # Limitar ainda mais
        indices = np.linspace(0, M - 1, sample_size, dtype=int)
        embedded_sample = embedded[indices]

        # Matriz de distâncias (apenas amostra)
        distances = np.linalg.norm(
            embedded_sample[:, None] - embedded_sample[None, :], axis=2)

        # Matriz de recorrência
        threshold_value = threshold * np.std(y)
        recurrence_matrix = distances < threshold_value

        # Taxa de recorrência
        recurrence_rate = np.sum(recurrence_matrix) / \
            (sample_size * sample_size)

        # Determinism (simplificado)
        diagonal_lines = _find_diagonal_lines_optimized(recurrence_matrix)
        determinism = np.sum([len(line) for line in diagonal_lines if len(
            line) >= 2]) / max(np.sum(recurrence_matrix), 1)

        # Laminaridade (simplificada)
        vertical_lines = _find_vertical_lines_optimized(recurrence_matrix)
        laminarity = np.sum([len(line) for line in vertical_lines if len(
            line) >= 2]) / max(np.sum(recurrence_matrix), 1)

        result = {
            'rqa_recurrence_rate': recurrence_rate,
            'rqa_determinism': determinism,
            'rqa_laminarity': laminarity}

    # Cache resultado
    if cache is not None:
        cache[cache_key] = result

    return result
