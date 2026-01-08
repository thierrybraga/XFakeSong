import numpy as np
from typing import Optional, Dict


def compute_hurst_exponent(y: np.ndarray) -> float:
    """Calcula o expoente de Hurst usando R/S analysis."""
    N = len(y)
    if N < 10:
        return np.nan

    # Diferentes tamanhos de janela
    scales = np.unique(np.logspace(1, np.log10(N // 4), 15, dtype=int))
    rs_values = []

    for scale in scales:
        if scale >= N:
            continue

        # Dividir série em segmentos
        n_segments = N // scale
        rs_segment = []

        for i in range(n_segments):
            segment = y[i * scale:(i + 1) * scale]

            # Calcular média
            mean_segment = np.mean(segment)

            # Desvios cumulativos
            deviations = segment - mean_segment
            cumulative_deviations = np.cumsum(deviations)

            # Range
            R = np.max(cumulative_deviations) - np.min(cumulative_deviations)

            # Standard deviation
            S = np.std(segment)

            if S > 0:
                rs_segment.append(R / S)

        if rs_segment:
            rs_values.append(np.mean(rs_segment))
        else:
            rs_values.append(np.nan)

    # Remover valores inválidos
    valid_indices = np.isfinite(rs_values) & (np.array(rs_values) > 0)

    if np.sum(valid_indices) < 3:
        return np.nan

    valid_scales = scales[:len(rs_values)][valid_indices]
    valid_rs = np.array(rs_values)[valid_indices]

    # Ajuste linear em escala log-log
    log_scales = np.log(valid_scales)
    log_rs = np.log(valid_rs)

    hurst = np.polyfit(log_scales, log_rs, 1)[0]

    return hurst


def compute_hurst_exponent_optimized(
        y: np.ndarray, cache: Optional[Dict] = None) -> float:
    """Versão otimizada do expoente de Hurst com menos escalas."""
    N = len(y)

    # Limitar tamanho
    if N > 2000:
        step = N // 2000
        y = y[::step][:2000]
        N = len(y)

    if N < 20:
        return np.nan

    # Cache key
    cache_key = f"hurst_{hash(y.tobytes())}"
    if cache is not None and cache_key in cache:
        return cache[cache_key]

    # Menos escalas para performance
    scales = np.unique(np.logspace(1, np.log10(N // 8), 10, dtype=int))
    rs_values = []

    for scale in scales:
        if scale >= N or scale < 4:
            continue

        # Menos segmentos para performance
        n_segments = min(N // scale, 10)
        rs_segment = []

        for i in range(n_segments):
            start_idx = i * scale
            end_idx = min(start_idx + scale, N)
            segment = y[start_idx:end_idx]

            if len(segment) < 4:
                continue

            # Calcular R/S
            mean_segment = np.mean(segment)
            deviations = segment - mean_segment
            cumulative_deviations = np.cumsum(deviations)

            R = np.max(cumulative_deviations) - np.min(cumulative_deviations)
            S = np.std(segment)

            if S > 1e-8:
                rs_segment.append(R / S)

        if rs_segment:
            rs_values.append(np.mean(rs_segment))

    # Remover valores inválidos
    valid_indices = np.isfinite(rs_values) & (np.array(rs_values) > 0)

    if np.sum(valid_indices) < 3:
        result = np.nan
    else:
        valid_scales = scales[:len(rs_values)][valid_indices]
        valid_rs = np.array(rs_values)[valid_indices]

        # Ajuste linear em escala log-log
        log_scales = np.log(valid_scales)
        log_rs = np.log(valid_rs)

        result = np.polyfit(log_scales, log_rs, 1)[0]

    # Cache resultado
    if cache is not None:
        cache[cache_key] = result

    return result
