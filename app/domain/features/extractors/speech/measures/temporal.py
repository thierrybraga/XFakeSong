import numpy as np
import scipy.signal
from typing import Dict, List, Optional
from . import common


def extract_temporal_speech_features(
        y: np.ndarray, sr: int, frame_length: int, hop_length: int) -> Dict:
    """Extrai características temporais específicas de fala."""
    features = {}

    # === ESTATÍSTICAS DE PAUSAS ===
    pause_features = extract_pause_statistics(y, sr, frame_length, hop_length)
    features.update(pause_features)

    # === CARACTERÍSTICAS DE RITMO ===
    rhythm_features = extract_rhythm_features(y, sr, frame_length, hop_length)
    features.update(rhythm_features)

    # === VARIABILIDADE TEMPORAL ===
    temporal_variability_features = extract_temporal_variability(
        y, sr, frame_length, hop_length)
    features.update(temporal_variability_features)

    return features


def extract_pause_statistics(
        y: np.ndarray, sr: int, frame_length: int, hop_length: int) -> Dict:
    """Extrai estatísticas de pausas."""
    features = {}

    # Detectar pausas (segmentos de silêncio)
    pause_segments = common.detect_pause_segments(
        y, sr, frame_length, hop_length)

    if not pause_segments:
        return features

    pause_durations = [(end - start) / sr for start, end in pause_segments]

    features['pause_count'] = len(pause_durations)
    features['pause_duration_mean'] = np.mean(pause_durations)
    features['pause_duration_std'] = np.std(pause_durations)
    features['pause_duration_median'] = np.median(pause_durations)
    features['pause_duration_max'] = np.max(pause_durations)

    # Razão de tempo em pausa
    total_pause_duration = sum(pause_durations)
    total_duration = len(y) / sr
    features['pause_ratio'] = total_pause_duration / total_duration

    # Classificar pausas por duração
    short_pauses = sum(1 for d in pause_durations if d < 0.2)
    medium_pauses = sum(1 for d in pause_durations if 0.2 <= d < 0.5)
    long_pauses = sum(1 for d in pause_durations if d >= 0.5)

    total_pauses = len(pause_durations)

    if total_pauses > 0:
        features['short_pause_ratio'] = short_pauses / total_pauses
        features['medium_pause_ratio'] = medium_pauses / total_pauses
        features['long_pause_ratio'] = long_pauses / total_pauses

    return features


def extract_rhythm_features(y: np.ndarray, sr: int,
                            frame_length: int, hop_length: int) -> Dict:
    """Extrai características de ritmo."""
    features = {}

    # Calcular envelope de energia
    energy_envelope = common.compute_energy_envelope(
        y, frame_length, hop_length)

    # Detectar picos de energia (batidas rítmicas)
    peaks, _ = scipy.signal.find_peaks(energy_envelope,
                                       height=np.mean(energy_envelope),
                                       distance=int(0.1 * sr / hop_length))

    if len(peaks) > 1:
        # Intervalos entre picos
        peak_intervals = np.diff(peaks) * hop_length / sr

        features['rhythm_interval_mean'] = np.mean(peak_intervals)
        features['rhythm_interval_std'] = np.std(peak_intervals)
        features['rhythm_regularity'] = 1 / \
            (1 + np.std(peak_intervals))  # Regularidade

        # Taxa de batidas por minuto
        if np.mean(peak_intervals) > 0:
            features['rhythm_bpm'] = 60 / np.mean(peak_intervals)

    # Periodicidade do envelope
    autocorr = np.correlate(energy_envelope, energy_envelope, mode='full')
    autocorr = autocorr[len(autocorr) // 2:]

    # Encontrar pico de autocorrelação (excluindo lag 0)
    if len(autocorr) > 1:
        peak_lag = np.argmax(autocorr[1:]) + 1
        peak_value = autocorr[peak_lag] / autocorr[0]  # Normalizar

        features['rhythm_periodicity'] = peak_value
        features['rhythm_period'] = peak_lag * hop_length / sr

    return features


def extract_temporal_variability(
        y: np.ndarray, sr: int, frame_length: int, hop_length: int) -> Dict:
    """Extrai características de variabilidade temporal."""
    features = {}

    # Calcular energia por frame
    frames = common.frame_signal(y, frame_length, hop_length)
    frame_energies = [np.sum(frame ** 2) for frame in frames]

    if len(frame_energies) > 1:
        # Variabilidade da energia
        energy_variation = np.std(frame_energies) / \
            (np.mean(frame_energies) + 1e-12)
        features['energy_variability'] = energy_variation

        # Mudanças abruptas de energia
        energy_changes = np.abs(np.diff(frame_energies))
        features['energy_change_mean'] = np.mean(energy_changes)
        features['energy_change_max'] = np.max(energy_changes)

    # Calcular características espectrais por frame
    spectral_centroids = []

    for frame in frames:
        if len(frame) > 0:
            spectrum = np.abs(np.fft.fft(frame))
            freqs = np.fft.fftfreq(len(spectrum), 1 / sr)

            # Centroide espectral
            positive_freqs = freqs[:len(freqs) // 2]
            positive_spectrum = spectrum[:len(spectrum) // 2]

            if np.sum(positive_spectrum) > 0:
                centroid = np.sum(
                    positive_freqs * positive_spectrum) / np.sum(positive_spectrum)
                spectral_centroids.append(centroid)

    if len(spectral_centroids) > 1:
        # Variabilidade do centroide espectral
        centroid_variation = np.std(
            spectral_centroids) / (np.mean(spectral_centroids) + 1e-12)
        features['spectral_variability'] = centroid_variation

    return features
