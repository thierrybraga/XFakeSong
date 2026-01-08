import numpy as np
import librosa
from typing import Dict
from .utils import find_peaks


def compute_pitch_slope(f0_valid: np.ndarray) -> float:
    """Computa a inclinação do pitch ao longo do tempo."""
    if len(f0_valid) < 2:
        return 0

    # Regressão linear simples
    x = np.arange(len(f0_valid))
    slope = np.polyfit(x, f0_valid, 1)[0]
    return slope


def compute_pitch_contour_features(f0: np.ndarray) -> Dict:
    """Computa características do contorno de pitch."""
    features = {}

    # Remover NaN e zeros
    valid_mask = ~np.isnan(f0) & (f0 > 0)

    if np.sum(valid_mask) < 2:
        return {
            'pitch_contour_rises': 0,
            'pitch_contour_falls': 0,
            'pitch_contour_peaks': 0,
            'pitch_contour_valleys': 0
        }

    valid_f0 = f0[valid_mask]

    # Calcular derivada para detectar mudanças
    diff = np.diff(valid_f0)

    # Contar subidas e descidas
    rises = np.sum(diff > 0)
    falls = np.sum(diff < 0)

    # Detectar picos e vales
    peaks = find_peaks(valid_f0)
    valleys = find_peaks(-valid_f0)

    features['pitch_contour_rises'] = rises
    features['pitch_contour_falls'] = falls
    features['pitch_contour_peaks'] = len(peaks)
    features['pitch_contour_valleys'] = len(valleys)

    return features


def compute_voicing_probability(f0: np.ndarray) -> float:
    """Computa probabilidade de vocalização."""
    valid_frames = ~np.isnan(f0) & (f0 > 0)
    voicing_prob = np.sum(valid_frames) / len(f0) if len(f0) > 0 else 0
    return voicing_prob


def compute_pitch_strength(y: np.ndarray, f0: np.ndarray,
                           sr: int, frame_length: int, hop_length: int) -> np.ndarray:
    """Computa força/confiança do pitch."""
    # Usar autocorrelação como medida de confiança
    strengths = []

    frames = librosa.util.frame(y, frame_length=frame_length,
                                hop_length=hop_length)

    for i, frame in enumerate(frames.T):
        if i < len(f0) and not np.isnan(f0[i]) and f0[i] > 0:
            # Calcular autocorrelação
            autocorr = np.correlate(frame, frame, mode='full')
            autocorr = autocorr[len(autocorr) // 2:]

            if len(autocorr) > 1:
                # Normalizar
                autocorr = autocorr / autocorr[0]

                # Encontrar pico correspondente ao período
                period_samples = int(sr / f0[i])
                if period_samples < len(autocorr):
                    strength = autocorr[period_samples]
                else:
                    strength = 0
            else:
                strength = 0
        else:
            strength = 0

        strengths.append(strength)

    return np.array(strengths)
