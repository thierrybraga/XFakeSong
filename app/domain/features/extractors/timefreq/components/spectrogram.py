"""
Extração de características do espectrograma.
"""
import numpy as np
import scipy.signal
from typing import Dict


def extract_spectrogram_features(
        y: np.ndarray, sr: int, frame_length: int, hop_length: int) -> Dict:
    features = {}
    # Calcular espectrograma
    f, t, Sxx = scipy.signal.spectrogram(y, fs=sr,
                                         nperseg=frame_length,
                                         noverlap=frame_length - hop_length)
    # Magnitude e fase
    magnitude = np.abs(Sxx)
    # === CARACTERÍSTICAS DE MAGNITUDE ===
    spectral_energy = np.sum(magnitude ** 2, axis=0)
    features['spectral_energy_mean'] = np.mean(spectral_energy)
    features['spectral_energy_std'] = np.std(spectral_energy)
    features['spectral_energy_max'] = np.max(spectral_energy)
    # Variação temporal da energia
    if len(spectral_energy) > 1:
        energy_variation = np.diff(spectral_energy)
        features['spectral_energy_variation'] = np.std(energy_variation)
    else:
        features['spectral_energy_variation'] = 0
    # Centroide espectral temporal
    spectral_centroids = []
    for frame in magnitude.T:
        if np.sum(frame) > 0:
            centroid = np.sum(f * frame) / np.sum(frame)
            spectral_centroids.append(centroid)
    if spectral_centroids:
        features['spectral_centroid_temporal_mean'] = np.mean(
            spectral_centroids)
        features['spectral_centroid_temporal_std'] = np.std(spectral_centroids)
        if len(spectral_centroids) > 1:
            centroid_variation = np.diff(spectral_centroids)
            features['spectral_centroid_variation'] = np.std(
                centroid_variation)
        else:
            features['spectral_centroid_variation'] = 0
    # Bandwidth espectral temporal
    spectral_bandwidths = []
    for i, frame in enumerate(magnitude.T):
        if np.sum(frame) > 0 and i < len(spectral_centroids):
            centroid = spectral_centroids[i]
            bandwidth = np.sqrt(
                np.sum(
                    ((f - centroid) ** 2) * frame) / np.sum(frame))
            spectral_bandwidths.append(bandwidth)
    if spectral_bandwidths:
        features['spectral_bandwidth_temporal_mean'] = np.mean(
            spectral_bandwidths)
        features['spectral_bandwidth_temporal_std'] = np.std(
            spectral_bandwidths)
    # Rolloff espectral temporal
    spectral_rolloffs = []
    rolloff_threshold = 0.85
    for frame in magnitude.T:
        cumulative_energy = np.cumsum(frame ** 2)
        total_energy = cumulative_energy[-1]
        if total_energy > 0:
            rolloff_idx = np.where(
                cumulative_energy >= rolloff_threshold *
                total_energy)[0]
            if len(rolloff_idx) > 0:
                rolloff_freq = f[rolloff_idx[0]]
                spectral_rolloffs.append(rolloff_freq)
    if spectral_rolloffs:
        features['spectral_rolloff_temporal_mean'] = np.mean(spectral_rolloffs)
        features['spectral_rolloff_temporal_std'] = np.std(spectral_rolloffs)
    # Flux espectral
    if magnitude.shape[1] > 1:
        spectral_flux = []
        for i in range(1, magnitude.shape[1]):
            flux = np.sum((magnitude[:, i] - magnitude[:, i - 1]) ** 2)
            spectral_flux.append(flux)
        features['spectral_flux_mean'] = np.mean(spectral_flux)
        features['spectral_flux_std'] = np.std(spectral_flux)
    # Concentração tempo-frequência
    total_energy = np.sum(magnitude ** 2)
    if total_energy > 0:
        normalized_magnitude = (magnitude ** 2) / total_energy
        tf_entropy = -np.sum(normalized_magnitude *
                             np.log(normalized_magnitude + 1e-12))
        features['timefreq_entropy'] = tf_entropy
        temporal_energy = np.sum(magnitude ** 2, axis=0)
        temporal_concentration = np.max(
            temporal_energy) / np.mean(temporal_energy)
        features['temporal_concentration'] = temporal_concentration
        spectral_energy = np.sum(magnitude ** 2, axis=1)
        spectral_concentration = np.max(
            spectral_energy) / np.mean(spectral_energy)
        features['spectral_concentration'] = spectral_concentration
    return features
