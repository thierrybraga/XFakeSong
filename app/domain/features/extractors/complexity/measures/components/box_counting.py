import numpy as np
from typing import Optional, Dict


def compute_fractal_dimension(y: np.ndarray) -> float:
    """Calcula a dimensão fractal usando box-counting."""
    # Normalizar para [0, 1]
    y_norm = (y - np.min(y)) / (np.max(y) - np.min(y) + 1e-8)

    # Diferentes tamanhos de caixa
    scales = np.logspace(0, np.log10(len(y) // 4), 20, dtype=int)
    scales = np.unique(scales)

    counts = []

    for scale in scales:
        # Dividir em caixas
        n_boxes_x = len(y) // scale
        n_boxes_y = int(1.0 / (1.0 / scale))

        if n_boxes_x < 1 or n_boxes_y < 1:
            continue

        boxes = set()

        for i in range(len(y)):
            box_x = i // scale
            box_y = int(y_norm[i] * n_boxes_y)

            if box_x < n_boxes_x and box_y < n_boxes_y:
                boxes.add((box_x, box_y))

        counts.append(len(boxes))

    if len(counts) < 3:
        return np.nan

    # Ajuste linear em escala log-log
    log_scales = np.log(scales[:len(counts)])
    log_counts = np.log(counts)

    slope = np.polyfit(log_scales, log_counts, 1)[0]

    return -slope  # Dimensão fractal


def compute_fractal_dimension_optimized(
        y: np.ndarray, cache: Optional[Dict] = None) -> float:
    """Versão otimizada da dimensão fractal com menos escalas."""
    N = len(y)

    # Limitar tamanho
    if N > 2000:
        step = N // 2000
        y = y[::step][:2000]
        N = len(y)

    if N < 20:
        return np.nan

    # Cache key
    cache_key = f"fractdim_{hash(y.tobytes())}"
    if cache is not None and cache_key in cache:
        return cache[cache_key]

    # Normalizar para [0, 1]
    y_norm = (y - np.min(y)) / (np.max(y) - np.min(y) + 1e-8)

    # Menos escalas para performance
    scales = np.unique(np.logspace(0, np.log10(N // 8), 10, dtype=int))
    counts = []

    for scale in scales:
        if scale < 1:
            continue

        # Dividir em caixas
        n_boxes_x = N // scale
        n_boxes_y = max(1, int(1.0 / (1.0 / scale)))

        if n_boxes_x < 1:
            continue

        boxes = set()

        for i in range(0, N, scale):  # Pular pontos para performance
            if i < N:
                box_x = i // scale
                box_y = min(int(y_norm[i] * n_boxes_y), n_boxes_y - 1)

                if box_x < n_boxes_x:
                    boxes.add((box_x, box_y))

        counts.append(len(boxes))

    if len(counts) < 3:
        result = np.nan
    else:
        # Ajuste linear em escala log-log
        log_scales = np.log(scales[:len(counts)])
        log_counts = np.log(np.maximum(counts, 1))  # Evitar log(0)

        result = -np.polyfit(log_scales, log_counts, 1)[0]

    # Cache resultado
    if cache is not None:
        cache[cache_key] = result

    return result
