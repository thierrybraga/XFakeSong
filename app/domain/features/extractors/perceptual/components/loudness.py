"""
Extração de características de loudness e sharpness.
"""
import numpy as np
import librosa
from typing import Dict
from .utils import hz_to_bark


def extract_loudness_features(
        y: np.ndarray, sr: int, frame_length: int, hop_length: int) -> Dict:
    """Extrai características de loudness (Stevens/Zwicker)."""
    features = {}

    # Calcular espectrograma
    stft = librosa.stft(y, n_fft=frame_length, hop_length=hop_length)
    magnitude = np.abs(stft)
    power = magnitude ** 2

    # Frequências dos bins
    freqs = librosa.fft_frequencies(sr=sr, n_fft=frame_length)

    # Modelo de loudness de Stevens (aproximação)
    # Loudness ∝ (Intensity)^0.3

    loudness_frames = []

    for frame_power in power.T:
        # Converter para intensidade (normalizada)
        intensity = frame_power / np.max(frame_power + 1e-10)

        # Aplicar função de loudness de Stevens
        loudness = intensity ** 0.3

        # Integrar sobre frequências (ponderado por frequência)
        total_loudness = np.sum(loudness * freqs) / np.sum(freqs + 1e-10)
        loudness_frames.append(total_loudness)

    loudness_frames = np.array(loudness_frames)

    # Características de loudness
    features['loudness_mean'] = np.mean(loudness_frames)
    features['loudness_std'] = np.std(loudness_frames)
    features['loudness_max'] = np.max(loudness_frames)
    features['loudness_range'] = np.max(
        loudness_frames) - np.min(loudness_frames)

    # Variação temporal de loudness
    if len(loudness_frames) > 1:
        loudness_variation = np.diff(loudness_frames)
        features['loudness_variation'] = np.std(loudness_variation)
    else:
        features['loudness_variation'] = 0

    return features


def extract_sharpness_features(
        y: np.ndarray, sr: int, frame_length: int, hop_length: int) -> Dict:
    """Extrai características de sharpness (agudeza)."""
    features = {}

    # Calcular espectrograma
    stft = librosa.stft(y, n_fft=frame_length, hop_length=hop_length)
    magnitude = np.abs(stft)

    # Frequências dos bins
    freqs = librosa.fft_frequencies(sr=sr, n_fft=frame_length)

    # Converter para escala Bark
    bark_freqs = hz_to_bark(freqs)

    sharpness_frames = []

    for frame in magnitude.T:
        # Calcular loudness específica por banda Bark
        specific_loudness = frame ** 2

        # Função de ponderação para sharpness (Zwicker & Fastl)
        # Sharpness aumenta com frequência
        weighting = np.ones_like(bark_freqs)

        # Aplicar ponderação crescente com frequência Bark
        for i, bark_freq in enumerate(bark_freqs):
            if bark_freq > 15.8:  # Acima de ~4 kHz
                weighting[i] = 0.066 * np.exp(0.171 * bark_freq)
            else:
                weighting[i] = 1.0

        # Calcular sharpness
        weighted_loudness = specific_loudness * weighting
        total_loudness = np.sum(specific_loudness)

        if total_loudness > 0:
            sharpness = 0.11 * np.sum(weighted_loudness) / total_loudness
        else:
            sharpness = 0

        sharpness_frames.append(sharpness)

    sharpness_frames = np.array(sharpness_frames)

    # Características de sharpness
    features['sharpness_mean'] = np.mean(sharpness_frames)
    features['sharpness_std'] = np.std(sharpness_frames)
    features['sharpness_max'] = np.max(sharpness_frames)
    features['sharpness_range'] = np.max(
        sharpness_frames) - np.min(sharpness_frames)

    return features
