import numpy as np
from typing import Dict, List, Optional
from . import common


def extract_fluency_features(
        y: np.ndarray, sr: int, frame_length: int, hop_length: int) -> Dict:
    """Extrai características de fluência e disfluência."""
    features = {}

    # === MEDIDAS DE DISFLUÊNCIA ===
    disfluency_features = extract_disfluency_measures(
        y, sr, frame_length, hop_length)
    features.update(disfluency_features)

    # === CARACTERÍSTICAS DE HESITAÇÃO ===
    hesitation_features = extract_hesitation_features(
        y, sr, frame_length, hop_length)
    features.update(hesitation_features)

    # === CONTINUIDADE DA FALA ===
    continuity_features = extract_speech_continuity(
        y, sr, frame_length, hop_length)
    features.update(continuity_features)

    return features


def extract_disfluency_measures(
        y: np.ndarray, sr: int, frame_length: int, hop_length: int) -> Dict:
    """Extrai medidas de disfluência."""
    features = {}

    # Detectar repetições baseadas em similaridade espectral
    repetitions = common.detect_repetitions(y)
    features['repetition_count'] = len(repetitions)

    # Detectar prolongamentos baseados em estabilidade espectral
    prolongations = common.detect_prolongations(y)
    features['prolongation_count'] = len(prolongations)

    # Detectar bloqueios baseados em pausas anômalas
    blocks = common.detect_blocks(y)
    features['block_count'] = len(blocks)

    # Taxa total de disfluência
    total_disfluencies = len(repetitions) + len(prolongations) + len(blocks)
    speech_duration = common.estimate_speech_duration(
        y, sr, frame_length, hop_length)

    if speech_duration > 0:
        features['disfluency_rate'] = total_disfluencies / speech_duration

    return features


def extract_hesitation_features(
        y: np.ndarray, sr: int, frame_length: int, hop_length: int) -> Dict:
    """Extrai características de hesitação."""
    features = {}

    # Detectar pausas preenchidas ("uh", "um")
    filled_pauses = common.detect_filled_pauses(y)
    features['filled_pause_count'] = len(filled_pauses)

    # Detectar hesitações baseadas em mudanças de pitch
    pitch_hesitations = common.detect_pitch_hesitations(y)
    features['pitch_hesitation_count'] = len(pitch_hesitations)

    # Detectar falsos inícios
    false_starts = common.detect_false_starts(y)
    features['false_start_count'] = len(false_starts)

    return features


def extract_speech_continuity(
        y: np.ndarray, sr: int, frame_length: int, hop_length: int) -> Dict:
    """Extrai características de continuidade da fala."""
    features = {}

    # Detectar segmentos de fala contínua
    continuous_segments = common.detect_continuous_speech_segments(
        y, sr, frame_length, hop_length)

    if continuous_segments:
        segment_durations = [
            (end - start) / sr for start,
            end in continuous_segments]

        features['continuous_segment_count'] = len(segment_durations)
        features['continuous_segment_mean_duration'] = np.mean(
            segment_durations)
        features['continuous_segment_max_duration'] = np.max(segment_durations)

        # Razão de fala contínua
        total_continuous_duration = sum(segment_durations)
        total_duration = len(y) / sr
        features['speech_continuity_ratio'] = total_continuous_duration / \
            total_duration

    return features
