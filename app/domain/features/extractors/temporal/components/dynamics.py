import numpy as np
import scipy.signal
from typing import Dict


def compute_amplitude_modulation(y: np.ndarray) -> float:
    """
    Computa o índice de modulação de amplitude.
    """
    # Calcular envelope
    envelope = np.abs(scipy.signal.hilbert(y))

    if len(envelope) == 0:
        return 0.0

    # Suavizar envelope
    window_size = max(1, len(envelope) // 100)
    if window_size > 1:
        envelope_smooth = scipy.signal.convolve(envelope,
                                                np.ones(window_size) /
                                                window_size,
                                                mode='same')
    else:
        envelope_smooth = envelope

    # Calcular modulação
    envelope_mean = np.mean(envelope_smooth)
    if envelope_mean > 0:
        envelope_variation = np.std(envelope_smooth)
        modulation_index = envelope_variation / envelope_mean
    else:
        modulation_index = 0.0

    return float(modulation_index)


def compute_tremolo_rate(envelope: np.ndarray, sr: int) -> float:
    """
    Computa a taxa de tremolo (modulação de amplitude).
    """
    if len(envelope) < 2:
        return 0.0

    # FFT do envelope para encontrar modulações
    envelope_fft = np.fft.fft(envelope - np.mean(envelope))
    envelope_freqs = np.fft.fftfreq(len(envelope), 1 / sr)
    envelope_magnitude = np.abs(envelope_fft)

    # Considerar apenas frequências positivas e na faixa de tremolo (0.5-20 Hz)
    valid_mask = (envelope_freqs > 0.5) & (envelope_freqs < 20)

    if np.any(valid_mask):
        valid_freqs = envelope_freqs[valid_mask]
        valid_magnitudes = envelope_magnitude[valid_mask]

        # Encontrar frequência dominante
        dominant_idx = np.argmax(valid_magnitudes)
        tremolo_rate = valid_freqs[dominant_idx]
    else:
        tremolo_rate = 0.0

    return float(tremolo_rate)


def extract_envelope_statistics(y: np.ndarray, sr: int) -> Dict[str, float]:
    """Extrai estatísticas do envelope e dinâmica."""
    features = {}

    # Envelope do sinal
    envelope = np.abs(scipy.signal.hilbert(y))

    # Características do envelope
    features['envelope_mean'] = float(np.mean(envelope))
    features['envelope_std'] = float(np.std(envelope))
    features['envelope_max'] = float(np.max(envelope))

    # Variação temporal do envelope
    if len(envelope) > 1:
        envelope_diff = np.diff(envelope)
        features['envelope_variation'] = float(np.std(envelope_diff))
        features['envelope_slope'] = float(np.mean(envelope_diff))
    else:
        features['envelope_variation'] = 0.0
        features['envelope_slope'] = 0.0

    # Características de modulação
    features['amplitude_modulation'] = compute_amplitude_modulation(y)
    features['tremolo_rate'] = compute_tremolo_rate(envelope, sr)

    return features
