import numpy as np
import librosa
from scipy.stats import kurtosis, skew
from typing import Dict


def compute_sign_change_rate(y: np.ndarray) -> float:
    """Calcula a taxa de mudança de sinal."""
    if len(y) < 2:
        return 0.0

    sign_changes = 0
    for i in range(1, len(y)):
        if (y[i] >= 0) != (y[i - 1] >= 0):
            sign_changes += 1

    return float(sign_changes / (len(y) - 1))


def compute_mean_crossing_rate(y: np.ndarray) -> float:
    """Calcula a taxa de cruzamento da média."""
    if len(y) < 2:
        return 0.0

    mean_val = np.mean(y)
    crossings = 0

    for i in range(1, len(y)):
        if (y[i] >= mean_val) != (y[i - 1] >= mean_val):
            crossings += 1

    return float(crossings / (len(y) - 1))


def compute_zcr(y: np.ndarray, frame_length: int,
                hop_length: int) -> np.ndarray:
    """Computa taxa de cruzamento por zero."""
    return librosa.feature.zero_crossing_rate(y, frame_length=frame_length,
                                              hop_length=hop_length)[0]


def compute_zcr_variance(y: np.ndarray, frame_length: int,
                         hop_length: int) -> float:
    """Computa variância da taxa de cruzamento por zero."""
    zcr = compute_zcr(y, frame_length, hop_length)
    return float(np.var(zcr))


def extract_signal_statistics(y: np.ndarray) -> Dict[str, float]:
    """Extrai estatísticas básicas do sinal."""
    features = {}

    # Estatísticas básicas
    features['signal_mean'] = float(np.mean(y))
    features['signal_std'] = float(np.std(y))
    features['signal_variance'] = float(np.var(y))
    features['signal_skewness'] = float(skew(y))
    features['signal_kurtosis'] = float(kurtosis(y))

    # Características de amplitude
    features['peak_amplitude'] = float(np.max(np.abs(y)))
    features['rms_amplitude'] = float(np.sqrt(np.mean(y**2)))
    features['crest_factor'] = float(
        features['peak_amplitude'] / (features['rms_amplitude'] + 1e-10))

    # Características de distribuição
    features['dynamic_range'] = float(np.max(y) - np.min(y))
    features['signal_energy'] = float(np.sum(y**2))

    return features
