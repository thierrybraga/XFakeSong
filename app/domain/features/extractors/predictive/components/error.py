"""
Extração de características de erro de predição.
"""

import numpy as np
import scipy.stats
from typing import Dict
from .utils import frame_signal, compute_prediction_error


def extract_prediction_error_features(
        y: np.ndarray, frame_length: int, hop_length: int,
        lpc_order: int) -> Dict:
    """Extrai características do erro de predição."""
    features = {}

    # Dividir sinal em frames
    frames = frame_signal(y, frame_length, hop_length)

    all_prediction_errors = []
    all_prediction_gains = []

    for frame in frames:
        if len(frame) > lpc_order + 1:
            # Calcular erro de predição
            prediction_error, prediction_gain = compute_prediction_error(
                frame, lpc_order)

            if prediction_error is not None:
                all_prediction_errors.append(prediction_error)
                all_prediction_gains.append(prediction_gain)

    if all_prediction_errors:
        # Concatenar todos os erros
        all_errors = np.concatenate(all_prediction_errors)
        all_gains = np.array(all_prediction_gains)

        # === CARACTERÍSTICAS DO ERRO DE PREDIÇÃO ===

        # Estatísticas do erro
        features['prediction_error_mean'] = np.mean(all_errors)
        features['prediction_error_std'] = np.std(all_errors)
        features['prediction_error_rms'] = np.sqrt(np.mean(all_errors ** 2))
        features['prediction_error_max'] = np.max(np.abs(all_errors))

        # Distribuição do erro
        features['prediction_error_skewness'] = scipy.stats.skew(all_errors)
        features['prediction_error_kurtosis'] = scipy.stats.kurtosis(
            all_errors)

        # Autocorrelação do erro (deve ser baixa para bom modelo)
        if len(all_errors) > 1:
            error_autocorr = np.correlate(all_errors, all_errors, mode='full')
            center_idx = len(error_autocorr) // 2

            # Autocorrelação normalizada no lag 1
            if center_idx + 1 < len(error_autocorr):
                autocorr_lag1 = error_autocorr[center_idx +
                                               1] / error_autocorr[center_idx]
                features['prediction_error_autocorr_lag1'] = autocorr_lag1

        # === CARACTERÍSTICAS DO GANHO DE PREDIÇÃO ===
        features['prediction_gain_mean'] = np.mean(all_gains)
        features['prediction_gain_std'] = np.std(all_gains)
        features['prediction_gain_range'] = np.max(
            all_gains) - np.min(all_gains)

        # Razão sinal-ruído de predição
        signal_power = np.mean(y ** 2)
        error_power = np.mean(all_errors ** 2)

        if error_power > 0:
            prediction_snr = 10 * np.log10(signal_power / error_power)
            features['prediction_snr'] = prediction_snr

        # Eficiência de predição
        if signal_power > 0:
            prediction_efficiency = 1 - (error_power / signal_power)
            features['prediction_efficiency'] = prediction_efficiency

    return features
