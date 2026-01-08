"""
Extração de características de perturbação (Jitter, Shimmer, etc).
"""
import numpy as np
from typing import Dict, Optional
from .utils import extract_pitch_periods


def compute_rap(y: np.ndarray, f0: np.ndarray, sr: int = 22050) -> float:
    """
    Calcula Relative Average Perturbation (RAP).
    Média da diferença absoluta entre um período e a média dele com seus dois vizinhos.
    """
    try:
        periods = extract_pitch_periods(f0, sr)
        if len(periods) < 3:
            return 0.0

        # RAP = (1/N-2) * sum(|P(i) - (P(i-1) + P(i) + P(i+1))/3|) / mean(P)
        diffs = []
        for i in range(1, len(periods) - 1):
            avg_neighbors = (periods[i - 1] +
                             periods[i] + periods[i + 1]) / 3.0
            diffs.append(abs(periods[i] - avg_neighbors))

        if not diffs:
            return 0.0

        mean_period = np.mean(periods)
        if mean_period == 0:
            return 0.0

        rap = np.mean(diffs) / mean_period
        return float(rap)
    except BaseException:
        return 0.0


def compute_ppq(y: np.ndarray, f0: np.ndarray, sr: int = 22050) -> float:
    """
    Calcula Pitch Perturbation Quotient (PPQ).
    Variação suavizada de 5 pontos do período de pitch.
    """
    try:
        periods = extract_pitch_periods(f0, sr)
        if len(periods) < 5:
            return 0.0

        diffs = []
        for i in range(2, len(periods) - 2):
            # Média de 5 pontos
            avg_5 = np.mean(periods[i - 2:i + 3])
            diffs.append(abs(periods[i] - avg_5))

        if not diffs:
            return 0.0

        mean_period = np.mean(periods)
        if mean_period == 0:
            return 0.0

        ppq = np.mean(diffs) / mean_period
        return float(ppq)
    except BaseException:
        return 0.0


def compute_apq(y: np.ndarray, f0: np.ndarray,
                frame_length: int = 2048, hop_length: int = 512) -> float:
    """
    Calcula Amplitude Perturbation Quotient (APQ).
    Variação de amplitude suavizada (Shimmer local).
    """
    try:
        import librosa
        # Calcular amplitude pico a pico ou RMS por frame
        # Usando RMS como proxy para amplitude
        rms = librosa.feature.rms(
            y=y,
            frame_length=frame_length,
            hop_length=hop_length)[0]

        # Filtrar frames silenciosos (opcional, mas recomendado)
        rms = rms[rms > 1e-4]

        if len(rms) < 11:  # APQ geralmente usa janela de 11 pontos
            return 0.0

        diffs = []
        for i in range(5, len(rms) - 5):
            # Média de 11 pontos
            avg_11 = np.mean(rms[i - 5:i + 6])
            diffs.append(abs(rms[i] - avg_11))

        if not diffs:
            return 0.0

        mean_amp = np.mean(rms)
        if mean_amp == 0:
            return 0.0

        apq = np.mean(diffs) / mean_amp
        return float(apq)
    except BaseException:
        return 0.0


def compute_vf0(f0: np.ndarray) -> float:
    """Calcula variância da frequência fundamental."""
    try:
        voiced = f0[f0 > 0]
        if len(voiced) < 2:
            return 0.0
        return float(np.std(voiced) / np.mean(voiced)
                     ) if np.mean(voiced) > 0 else 0.0
    except BaseException:
        return 0.0


def compute_shimmer_db(y: np.ndarray, f0: np.ndarray,
                       frame_length: int = 2048, hop_length: int = 512) -> float:
    """Calcula Shimmer em dB."""
    try:
        import librosa
        # Amplitude por frame
        rms = librosa.feature.rms(
            y=y,
            frame_length=frame_length,
            hop_length=hop_length)[0]
        # Apenas frames vozeados (aproximado pela presença de F0 correspondente)
        # Precisamos alinhar f0 e rms temporalmente
        target_len = min(len(rms), len(f0))
        rms = rms[:target_len]
        f0_trunc = f0[:target_len]

        voiced_mask = f0_trunc > 0
        voiced_amp = rms[voiced_mask]

        if len(voiced_amp) < 2:
            return 0.0

        # Diferença absoluta entre amplitudes consecutivas (em dB)
        # Shimmer(dB) = 20 * |log10(A_i / A_{i+1})|
        shimmer_db_vals = []
        for i in range(len(voiced_amp) - 1):
            if voiced_amp[i] > 0 and voiced_amp[i + 1] > 0:
                ratio = voiced_amp[i] / voiced_amp[i + 1]
                val = 20 * abs(np.log10(ratio))
                shimmer_db_vals.append(val)

        if not shimmer_db_vals:
            return 0.0

        return float(np.mean(shimmer_db_vals))
    except BaseException:
        return 0.0


def extract_perturbation_features(y: np.ndarray, f0: np.ndarray, sr: int,
                                  frame_length: int, hop_length: int) -> Dict[str, float]:
    """Extrai todas as características de perturbação."""
    features = {}
    features['rap'] = compute_rap(y, f0, sr)
    features['ppq'] = compute_ppq(y, f0, sr)
    features['apq'] = compute_apq(y, f0, frame_length, hop_length)
    features['vf0'] = compute_vf0(f0)
    features['shimmer_db'] = compute_shimmer_db(
        y, f0, frame_length, hop_length)

    # Alias mapeado
    features['shdb'] = features['shimmer_db']

    return features
