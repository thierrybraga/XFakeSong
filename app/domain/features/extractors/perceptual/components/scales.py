"""
Extração de características baseadas em escalas perceptuais (Bark, ERB).
"""
import numpy as np
import librosa
from typing import Dict
from .utils import hz_to_bark, hz_to_erb


def extract_bark_features(y: np.ndarray, sr: int,
                          frame_length: int, hop_length: int) -> Dict:
    """Extrai características baseadas na escala Bark."""
    features = {}

    # Calcular espectrograma
    stft = librosa.stft(y, n_fft=frame_length, hop_length=hop_length)
    magnitude = np.abs(stft)

    # Frequências dos bins
    freqs = librosa.fft_frequencies(sr=sr, n_fft=frame_length)

    # Converter para escala Bark
    bark_freqs = hz_to_bark(freqs)

    # Criar banco de filtros Bark (24 bandas críticas)
    n_bark_bands = 24
    bark_bands = np.linspace(0, 24, n_bark_bands + 1)

    bark_energies = []

    for frame in magnitude.T:
        band_energies = []

        for i in range(n_bark_bands):
            # Encontrar bins nesta banda Bark
            band_mask = (bark_freqs >= bark_bands[i]) & (
                bark_freqs < bark_bands[i + 1])

            if np.any(band_mask):
                band_energy = np.sum(frame[band_mask] ** 2)
            else:
                band_energy = 0

            band_energies.append(band_energy)

        bark_energies.append(band_energies)

    bark_energies = np.array(bark_energies)

    # Características estatísticas das bandas Bark
    if bark_energies.size > 0:
        features['bark_energy_mean'] = np.mean(bark_energies)
        features['bark_energy_std'] = np.std(bark_energies)
        features['bark_energy_max'] = np.max(bark_energies)

        # Distribuição de energia entre bandas
        total_energy = np.sum(bark_energies, axis=1)
        valid_frames = total_energy > 0

        if np.any(valid_frames):
            normalized_energies = bark_energies[valid_frames] / \
                total_energy[valid_frames, np.newaxis]

            # Centroide Bark
            bark_centroids = np.sum(
                normalized_energies *
                np.arange(n_bark_bands),
                axis=1)
            features['bark_centroid_mean'] = np.mean(bark_centroids)
            features['bark_centroid_std'] = np.std(bark_centroids)

            # Spread Bark
            bark_spreads = np.sum(normalized_energies *
                                  (np.arange(n_bark_bands) -
                                   bark_centroids[:, np.newaxis]) ** 2, axis=1)
            features['bark_spread_mean'] = np.mean(bark_spreads)
            features['bark_spread_std'] = np.std(bark_spreads)

            # Entropia Bark
            bark_entropies = []
            for norm_energy in normalized_energies:
                entropy = -np.sum(norm_energy * np.log2(norm_energy + 1e-10))
                bark_entropies.append(entropy)

            features['bark_entropy_mean'] = np.mean(bark_entropies)
            features['bark_entropy_std'] = np.std(bark_entropies)

    return features


def extract_erb_features(y: np.ndarray, sr: int,
                         frame_length: int, hop_length: int) -> Dict:
    """Extrai características baseadas na escala ERB."""
    features = {}

    # Calcular espectrograma
    stft = librosa.stft(y, n_fft=frame_length, hop_length=hop_length)
    magnitude = np.abs(stft)

    # Frequências dos bins
    freqs = librosa.fft_frequencies(sr=sr, n_fft=frame_length)

    # Converter para escala ERB
    erb_freqs = hz_to_erb(freqs)

    # Criar banco de filtros ERB (40 bandas)
    n_erb_bands = 40
    max_erb = hz_to_erb(np.array([sr / 2]))[0]
    erb_bands = np.linspace(0, max_erb, n_erb_bands + 1)

    erb_energies = []

    for frame in magnitude.T:
        band_energies = []

        for i in range(n_erb_bands):
            # Encontrar bins nesta banda ERB
            band_mask = (erb_freqs >= erb_bands[i]) & (
                erb_freqs < erb_bands[i + 1])

            if np.any(band_mask):
                band_energy = np.sum(frame[band_mask] ** 2)
            else:
                band_energy = 0

            band_energies.append(band_energy)

        erb_energies.append(band_energies)

    erb_energies = np.array(erb_energies)

    # Características estatísticas das bandas ERB
    if erb_energies.size > 0:
        features['erb_energy_mean'] = np.mean(erb_energies)
        features['erb_energy_std'] = np.std(erb_energies)
        features['erb_energy_max'] = np.max(erb_energies)

        # Distribuição de energia entre bandas
        total_energy = np.sum(erb_energies, axis=1)
        valid_frames = total_energy > 0

        if np.any(valid_frames):
            normalized_energies = erb_energies[valid_frames] / \
                total_energy[valid_frames, np.newaxis]

            # Centroide ERB
            erb_centroids = np.sum(
                normalized_energies *
                np.arange(n_erb_bands),
                axis=1)
            features['erb_centroid_mean'] = np.mean(erb_centroids)
            features['erb_centroid_std'] = np.std(erb_centroids)

            # Spread ERB
            erb_spreads = np.sum(normalized_energies *
                                 (np.arange(n_erb_bands) -
                                  erb_centroids[:, np.newaxis]) ** 2, axis=1)
            features['erb_spread_mean'] = np.mean(erb_spreads)
            features['erb_spread_std'] = np.std(erb_spreads)

    return features
