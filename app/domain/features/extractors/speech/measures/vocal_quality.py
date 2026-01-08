import numpy as np
from typing import Dict, List, Optional
from . import common


def extract_vocal_quality_features(
        y: np.ndarray, sr: int, frame_length: int, hop_length: int) -> Dict:
    """Extrai características de qualidade vocal específicas para fala."""
    features = {}

    # === CARACTERÍSTICAS DE RESPIRAÇÃO ===
    breathing_features = extract_breathing_features(
        y, sr, frame_length, hop_length)
    features.update(breathing_features)

    # === CARACTERÍSTICAS DE TENSÃO VOCAL ===
    tension_features = extract_vocal_tension_features(
        y, sr, frame_length, hop_length)
    features.update(tension_features)

    # === CARACTERÍSTICAS DE CLAREZA ===
    clarity_features = extract_speech_clarity_features(
        y, sr, frame_length, hop_length)
    features.update(clarity_features)

    return features


def extract_breathing_features(
        y: np.ndarray, sr: int, frame_length: int, hop_length: int) -> Dict:
    """Extrai características de respiração."""
    features = {}

    # Detectar pausas respiratórias
    breathing_pauses = common.detect_breathing_pauses(
        y, sr, frame_length, hop_length)

    if breathing_pauses:
        pause_durations = [
            (end - start) / sr for start,
            end in breathing_pauses]

        features['breathing_pause_count'] = len(pause_durations)
        features['breathing_pause_mean_duration'] = np.mean(pause_durations)

        # Regularidade da respiração
        if len(pause_durations) > 1:
            breathing_intervals = np.diff(
                [start for start, _ in breathing_pauses]) / sr
            features['breathing_regularity'] = 1 / \
                (1 + np.std(breathing_intervals))

    return features


def extract_vocal_tension_features(
        y: np.ndarray, sr: int, frame_length: int, hop_length: int) -> Dict:
    """Extrai características de tensão vocal."""
    features = {}

    # Tensão baseada em características espectrais
    frames = common.frame_signal(y, frame_length, hop_length)

    tension_indicators = []

    for frame in frames:
        if len(frame) > 0:
            # Calcular características indicativas de tensão

            # 1. Energia em altas frequências
            spectrum = np.abs(np.fft.fft(frame))
            freqs = np.fft.fftfreq(len(spectrum), 1 / sr)

            high_freq_mask = (freqs >= 2000) & (freqs <= sr / 2)
            high_freq_energy = np.sum(spectrum[high_freq_mask])
            total_energy = np.sum(spectrum[:len(spectrum) // 2])

            if total_energy > 0:
                high_freq_ratio = high_freq_energy / total_energy
                tension_indicators.append(high_freq_ratio)

    if tension_indicators:
        features['vocal_tension_mean'] = np.mean(tension_indicators)
        features['vocal_tension_std'] = np.std(tension_indicators)
        features['vocal_tension_max'] = np.max(tension_indicators)

    return features


def extract_speech_clarity_features(
        y: np.ndarray, sr: int, frame_length: int, hop_length: int) -> Dict:
    """Extrai características de clareza da fala."""
    features = {}

    # Clareza baseada em definição espectral
    frames = common.frame_signal(y, frame_length, hop_length)

    clarity_measures = []

    for frame in frames:
        if len(frame) > 0:
            # Calcular sharpness espectral
            spectrum = np.abs(np.fft.fft(frame))

            # Medida de sharpness (concentração espectral)
            if np.sum(spectrum) > 0:
                normalized_spectrum = spectrum / np.sum(spectrum)
                entropy = -np.sum(normalized_spectrum *
                                  np.log(normalized_spectrum + 1e-12))
                sharpness = 1 / (1 + entropy)  # Inverso da entropia
                clarity_measures.append(sharpness)

    if clarity_measures:
        features['speech_clarity_mean'] = np.mean(clarity_measures)
        features['speech_clarity_std'] = np.std(clarity_measures)
        features['speech_clarity_min'] = np.min(clarity_measures)

    return features
