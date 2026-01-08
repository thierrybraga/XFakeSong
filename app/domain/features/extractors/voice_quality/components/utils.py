"""
Funções utilitárias para extração de características de qualidade vocal.
"""
import numpy as np
from typing import Dict, Any


def get_default_voice_quality_features() -> Dict[str, float]:
    """Retorna dicionário com valores padrão (zeros) para todas as características."""
    return {
        # Perturbation
        'rap': 0.0, 'ppq': 0.0, 'apq': 0.0, 'vf0': 0.0, 'shimmer_db': 0.0,
        # Noise
        'nhr': 0.0, 'vti': 0.0, 'spi': 0.0, 'dfa_alpha': 0.0,
        # Quality
        'spectral_tilt': 0.0, 'breathiness_index': 0.0,
        'roughness_index': 0.0, 'voice_breaks': 0.0,
        # Mapped
        'shdb': 0.0, 'breathiness': 0.0, 'roughness': 0.0, 'hoarseness': 0.0
    }


def extract_pitch_periods(f0: np.ndarray, sr: int) -> np.ndarray:
    """
    Converte contorno de F0 em períodos de pitch.

    Args:
        f0: Array de frequências fundamentais
        sr: Taxa de amostragem

    Returns:
        Array de períodos de pitch em segundos
    """
    # Filtrar valores não vozeados (<= 0 ou NaN)
    voiced_mask = (f0 > 0) & (~np.isnan(f0))
    if np.sum(voiced_mask) == 0:
        return np.array([])

    # Converter F0 para períodos (T = 1/F)
    periods = 1.0 / f0[voiced_mask]
    return periods


def compute_amplitude_envelope(
        y: np.ndarray, frame_length: int = 2048,
        hop_length: int = 512) -> np.ndarray:
    """Calcula envelope de amplitude do sinal."""
    # Usar RMS por frame como aproximação da amplitude
    import librosa
    rms = librosa.feature.rms(
        y=y,
        frame_length=frame_length,
        hop_length=hop_length)[0]
    return rms
