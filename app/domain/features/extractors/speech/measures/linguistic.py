import numpy as np
from typing import Dict, List, Tuple, Optional
from . import common


def extract_linguistic_features(
        y: np.ndarray, sr: int, frame_length: int, hop_length: int) -> Dict:
    """Extrai características linguísticas."""
    features = {}

    # === VOICE ONSET TIME (VOT) ===
    vot_features = extract_vot_features(y, sr, frame_length)
    features.update(vot_features)

    # === DURAÇÃO DE VOGAIS ===
    vowel_features = extract_vowel_duration_features(
        y, sr, frame_length, hop_length)
    features.update(vowel_features)

    # === RAZÃO CONSOANTE-VOGAL ===
    cv_ratio_features = extract_consonant_vowel_ratio(
        y, sr, frame_length, hop_length)
    features.update(cv_ratio_features)

    # === CARACTERÍSTICAS DE TAXA DE FALA ===
    speaking_rate_features = extract_speaking_rate_features(
        y, sr, frame_length, hop_length)
    features.update(speaking_rate_features)

    return features


def extract_vot_features(y: np.ndarray, sr: int,
                         frame_length: int, hop_length: int = 512) -> Dict:
    """Extrai características de Voice Onset Time."""
    features = {}

    # Detectar segmentos de fala
    speech_segments = common.detect_speech_segments(
        y, sr, frame_length, hop_length)

    if not speech_segments:
        return features

    vot_estimates = []

    for start, end in speech_segments:
        segment = y[start:end]

        if len(segment) < frame_length:
            continue

        # Estimar VOT baseado em mudanças de energia e espectro
        vot = common.estimate_vot(segment, sr)

        if vot is not None:
            vot_estimates.append(vot)

    if vot_estimates:
        features['vot_mean'] = np.mean(vot_estimates)
        features['vot_std'] = np.std(vot_estimates)
        features['vot_median'] = np.median(vot_estimates)
        features['vot_range'] = np.max(vot_estimates) - np.min(vot_estimates)
        features['vot_count'] = len(vot_estimates)

    return features


def extract_vowel_duration_features(
        y: np.ndarray, sr: int, frame_length: int, hop_length: int) -> Dict:
    """Extrai características de duração de vogais."""
    features = {}

    # Detectar segmentos vocálicos baseado em características espectrais
    vowel_segments = common.detect_vowel_segments(
        y, sr, frame_length, hop_length)

    if not vowel_segments:
        return features

    vowel_durations = []

    for start, end in vowel_segments:
        duration = (end - start) / sr  # Duração em segundos
        vowel_durations.append(duration)

    if vowel_durations:
        features['vowel_duration_mean'] = np.mean(vowel_durations)
        features['vowel_duration_std'] = np.std(vowel_durations)
        features['vowel_duration_median'] = np.median(vowel_durations)
        features['vowel_duration_range'] = np.max(
            vowel_durations) - np.min(vowel_durations)
        features['vowel_count'] = len(vowel_durations)

        # Razão de duração (vogais longas vs curtas)
        threshold = np.median(vowel_durations)
        long_vowels = np.sum(np.array(vowel_durations) > threshold)
        features['long_vowel_ratio'] = long_vowels / len(vowel_durations)

    return features


def extract_consonant_vowel_ratio(
        y: np.ndarray, sr: int, frame_length: int, hop_length: int) -> Dict:
    """Extrai razão consoante-vogal."""
    features = {}

    # Detectar segmentos vocálicos e consonantais
    vowel_segments = common.detect_vowel_segments(
        y, sr, frame_length, hop_length)
    consonant_segments = common.detect_consonant_segments(
        y, sr, frame_length, hop_length)

    vowel_duration = sum((end - start) / sr for start, end in vowel_segments)
    consonant_duration = sum(
        (end - start) / sr for start,
        end in consonant_segments)

    total_speech_duration = vowel_duration + consonant_duration

    if total_speech_duration > 0:
        features['vowel_ratio'] = vowel_duration / total_speech_duration
        features['consonant_ratio'] = consonant_duration / \
            total_speech_duration

        if vowel_duration > 0:
            features['consonant_vowel_ratio'] = consonant_duration / \
                vowel_duration

    features['vowel_segment_count'] = len(vowel_segments)
    features['consonant_segment_count'] = len(consonant_segments)

    return features


def extract_speaking_rate_features(
        y: np.ndarray, sr: int, frame_length: int, hop_length: int) -> Dict:
    """Extrai características de taxa de fala."""
    features = {}

    # Detectar segmentos de fala
    speech_segments = common.detect_speech_segments(
        y, sr, frame_length, hop_length)

    if not speech_segments:
        return features

    # Calcular duração total de fala
    total_speech_duration = sum(
        (end - start) / sr for start,
        end in speech_segments)
    total_duration = len(y) / sr

    # Taxa de fala (proporção de tempo falando)
    features['speech_rate'] = total_speech_duration / total_duration

    # Detectar sílabas (aproximação baseada em picos de energia)
    syllable_count = common.estimate_syllable_count(
        y, sr, frame_length, hop_length)

    if total_speech_duration > 0:
        features['syllables_per_second'] = syllable_count / \
            total_speech_duration
        features['syllables_per_minute'] = syllable_count / \
            total_speech_duration * 60

    features['estimated_syllable_count'] = syllable_count

    # Variabilidade da taxa de fala
    segment_rates = []

    for start, end in speech_segments:
        segment_duration = (end - start) / sr

        if segment_duration > 0.5:  # Apenas segmentos longos o suficiente
            segment = y[start:end]
            segment_syllables = common.estimate_syllable_count(
                segment, sr, frame_length, hop_length)
            segment_rate = segment_syllables / segment_duration
            segment_rates.append(segment_rate)

    if segment_rates:
        features['speaking_rate_variability'] = np.std(segment_rates)
        features['speaking_rate_range'] = np.max(
            segment_rates) - np.min(segment_rates)

    return features
