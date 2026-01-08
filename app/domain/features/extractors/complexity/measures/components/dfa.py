import numpy as np
from typing import Optional, Dict


def compute_dfa_exponent(y: np.ndarray) -> float:
    """Calcula o expoente DFA (Detrended Fluctuation Analysis)."""
    N = len(y)
    if N < 16:
        return np.nan

    # Integrar série
    y_integrated = np.cumsum(y - np.mean(y))

    # Diferentes tamanhos de janela
    scales = np.unique(np.logspace(1, np.log10(N // 4), 15, dtype=int))
    fluctuations = []

    for scale in scales:
        if scale >= N:
            continue

        # Dividir em segmentos
        n_segments = N // scale
        segment_fluctuations = []

        for i in range(n_segments):
            segment = y_integrated[i * scale:(i + 1) * scale]

            # Ajustar tendência linear
            x = np.arange(len(segment))
            coeffs = np.polyfit(x, segment, 1)
            trend = np.polyval(coeffs, x)

            # Calcular flutuação
            detrended = segment - trend
            fluctuation = np.sqrt(np.mean(detrended ** 2))
            segment_fluctuations.append(fluctuation)

        if segment_fluctuations:
            fluctuations.append(np.mean(segment_fluctuations))
        else:
            fluctuations.append(np.nan)

    # Remover valores inválidos
    valid_indices = np.isfinite(fluctuations) & (np.array(fluctuations) > 0)

    if np.sum(valid_indices) < 3:
        return np.nan

    valid_scales = scales[:len(fluctuations)][valid_indices]
    valid_fluctuations = np.array(fluctuations)[valid_indices]

    # Ajuste linear em escala log-log
    log_scales = np.log(valid_scales)
    log_fluctuations = np.log(valid_fluctuations)

    alpha = np.polyfit(log_scales, log_fluctuations, 1)[0]

    return alpha


def compute_dfa_exponent_optimized(
        y: np.ndarray, cache: Optional[Dict] = None) -> float:
    """Versão otimizada do expoente DFA com menos escalas."""
    N = len(y)

    # Limitar tamanho
    if N > 2000:
        step = N // 2000
        y = y[::step][:2000]
        N = len(y)

    if N < 16:
        return np.nan

    # Cache key
    cache_key = f"dfa_{hash(y.tobytes())}"
    if cache is not None and cache_key in cache:
        return cache[cache_key]

    # Integrar série
    y_integrated = np.cumsum(y - np.mean(y))

    # Menos escalas para performance
    scales = np.unique(np.logspace(1, np.log10(N // 8), 10, dtype=int))
    fluctuations = []

    for scale in scales:
        if scale >= N or scale < 4:
            continue

        # Menos segmentos
        n_segments = min(N // scale, 10)
        segment_fluctuations = []

        for i in range(n_segments):
            start_idx = i * scale
            end_idx = min(start_idx + scale, N)
            segment = y_integrated[start_idx:end_idx]

            if len(segment) < 4:
                continue

            # Ajustar tendência linear simplificada
            x = np.arange(len(segment))

            # Cálculo manual de regressão linear simples para performance
            x_mean = np.mean(x)
            y_mean = np.mean(segment)
            numerator = np.sum((x - x_mean) * (segment - y_mean))
            denominator = np.sum((x - x_mean) ** 2)

            if denominator == 0:
                continue

            slope = numerator / denominator
            intercept = y_mean - slope * x_mean

            trend = slope * x + intercept

            # Calcular flutuação
            detrended = segment - trend
            fluctuation = np.sqrt(np.mean(detrended ** 2))
            segment_fluctuations.append(fluctuation)

        if segment_fluctuations:
            fluctuations.append(np.mean(segment_fluctuations))

    # Remover valores inválidos
    valid_indices = np.isfinite(fluctuations) & (np.array(fluctuations) > 0)

    if np.sum(valid_indices) < 3:
        result = np.nan
    else:
        valid_scales = scales[:len(fluctuations)][valid_indices]
        valid_fluctuations = np.array(fluctuations)[valid_indices]

        # Ajuste linear em escala log-log
        log_scales = np.log(valid_scales)
        log_fluctuations = np.log(valid_fluctuations)

        result = np.polyfit(log_scales, log_fluctuations, 1)[0]

    # Cache resultado
    if cache is not None:
        cache[cache_key] = result

    return result
