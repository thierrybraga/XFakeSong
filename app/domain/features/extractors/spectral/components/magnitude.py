"""
Extração de características de magnitude espectral.
"""
import numpy as np
from typing import List


def compute_spectral_slope(S: np.ndarray, freqs: np.ndarray) -> np.ndarray:
    """Computa a inclinação espectral."""
    slopes = []
    for frame in S.T:
        # Linear regression em escala log
        log_freqs = np.log(freqs[1:])  # Evitar log(0)
        log_mags = np.log(frame[1:] + 1e-10)

        # Calcular coeficiente angular
        if len(log_freqs) > 0 and len(log_mags) > 0:
            slope = np.polyfit(log_freqs, log_mags, 1)[0]
        else:
            slope = 0.0
        slopes.append(slope)

    return np.array(slopes)


def compute_spectral_kurtosis(S: np.ndarray, freqs: np.ndarray) -> np.ndarray:
    """Computa a curtose espectral."""
    kurtosis_values = []
    for frame in S.T:
        # Normalizar para formar distribuição de probabilidade
        prob_dist = frame / (np.sum(frame) + 1e-10)

        # Calcular momentos
        mean_freq = np.sum(prob_dist * freqs)
        variance = np.sum(prob_dist * (freqs - mean_freq) ** 2)

        if variance > 0:
            # Quarto momento central normalizado
            fourth_moment = np.sum(prob_dist * (freqs - mean_freq) ** 4)
            kurtosis = fourth_moment / (variance ** 2) - 3
        else:
            kurtosis = 0

        kurtosis_values.append(kurtosis)

    return np.array(kurtosis_values)


def compute_spectral_skewness(S: np.ndarray, freqs: np.ndarray) -> np.ndarray:
    """Computa a assimetria espectral."""
    skewness_values = []
    for frame in S.T:
        # Normalizar
        prob_dist = frame / (np.sum(frame) + 1e-10)

        # Calcular momentos
        mean_freq = np.sum(prob_dist * freqs)
        variance = np.sum(prob_dist * (freqs - mean_freq) ** 2)

        if variance > 0:
            # Terceiro momento central normalizado
            third_moment = np.sum(prob_dist * (freqs - mean_freq) ** 3)
            skewness = third_moment / (variance ** 1.5)
        else:
            skewness = 0

        skewness_values.append(skewness)

    return np.array(skewness_values)


def compute_spectral_spread(S: np.ndarray, freqs: np.ndarray,
                            centroid: np.ndarray) -> np.ndarray:
    """Computa o espalhamento espectral (variância)."""
    spread_values = []
    for i, frame in enumerate(S.T):
        # Normalizar
        prob_dist = frame / (np.sum(frame) + 1e-10)

        # Variância em relação ao centroide
        if i < len(centroid):
            spread = np.sum(prob_dist * (freqs - centroid[i]) ** 2)
        else:
            mean_freq = np.sum(prob_dist * freqs)
            spread = np.sum(prob_dist * (freqs - mean_freq) ** 2)

        spread_values.append(spread)

    return np.array(spread_values)


def compute_spectral_entropy(S: np.ndarray) -> np.ndarray:
    """Computa a entropia espectral."""
    entropy_values = []
    for frame in S.T:
        # Normalizar para distribuição de probabilidade
        prob_dist = frame / (np.sum(frame) + 1e-10)

        # Calcular entropia de Shannon
        entropy = -np.sum(prob_dist * np.log2(prob_dist + 1e-10))
        entropy_values.append(entropy)

    return np.array(entropy_values)
