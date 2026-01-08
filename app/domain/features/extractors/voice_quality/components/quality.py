"""
Extração de características adicionais de qualidade (Spectral Tilt, Breathiness, etc).
"""
import numpy as np
import scipy.stats
import librosa
from typing import Dict


def compute_spectral_tilt(y: np.ndarray, sr: int = 22050) -> float:
    """
    Calcula inclinação espectral (Spectral Tilt).
    Inclinação da regressão linear do espectro de potência logarítmico.
    """
    try:
        # Calcular espectro de potência
        S = np.abs(librosa.stft(y)) ** 2
        S_avg = np.mean(S, axis=1)  # Média temporal

        # Converter para dB
        S_db = librosa.power_to_db(S_avg, ref=np.max)

        # Frequências
        freqs = librosa.fft_frequencies(sr=sr)

        # Regressão linear (frequência vs magnitude dB)
        # Ignorar DC e frequências muito baixas/altas
        mask = (freqs > 100) & (freqs < 8000)
        if np.sum(mask) < 10:
            return 0.0

        slope, _, _, _, _ = scipy.stats.linregress(freqs[mask], S_db[mask])
        return float(slope)
    except BaseException:
        return 0.0


def compute_breathiness_index(y: np.ndarray, f0: np.ndarray) -> float:
    """
    Calcula índice de soprosidade (Breathiness Index).
    Baseado na energia em bandas de alta frequência vs total.
    """
    try:
        # Cepstral Peak Prominence (CPP) é uma medida melhor, mas complexa
        # Aproximação: Energia > 2kHz / Energia Total em regiões vozeadas

        S = np.abs(librosa.stft(y))
        freqs = librosa.fft_frequencies(sr=22050)

        split_idx = np.searchsorted(freqs, 2000)

        energy_high = np.sum(S[split_idx:, :] ** 2)
        energy_total = np.sum(S ** 2)

        if energy_total == 0:
            return 0.0

        return float(energy_high / energy_total)
    except BaseException:
        return 0.0


def compute_roughness_index(y: np.ndarray, f0: np.ndarray) -> float:
    """
    Calcula índice de rugosidade (Roughness Index).
    Relacionado à modulação de amplitude em frequências baixas (< ~70Hz).
    """
    try:
        # Aproximação baseada na variância do envelope de amplitude
        envelope = np.abs(scipy.signal.hilbert(y))

        # Filtrar envelope para capturar modulações de 20-70Hz
        # (Isso exigiria filtragem passa-banda do envelope)

        # Medida simplificada: desvio padrão do envelope normalizado
        mean_env = np.mean(envelope)
        if mean_env == 0:
            return 0.0

        return float(np.std(envelope) / mean_env)
    except BaseException:
        return 0.0


def compute_voice_breaks(f0: np.ndarray) -> float:
    """
    Calcula quebras de voz (Voice Breaks).
    Proporção de frames não vozeados entre frames vozeados.
    """
    try:
        # Identificar segmentos vozeados
        voiced = f0 > 0

        # Contar transições Vozeado -> Não Vozeado -> Vozeado
        # Difícil sem segmentação precisa.
        # Simplificação: razão de frames não vozeados / total (se houver voz
        # detectada)

        total_voiced = np.sum(voiced)
        if total_voiced == 0:
            return 0.0

        total_frames = len(f0)
        unvoiced = total_frames - total_voiced

        # Se for tudo unvoiced, não é quebra, é silêncio.
        # Voice breaks acontecem quando há fonação intermitente.

        return float(unvoiced / total_frames)
    except BaseException:
        return 0.0


def extract_additional_quality_features(
        y: np.ndarray, f0: np.ndarray, sr: int) -> Dict[str, float]:
    """Extrai características adicionais de qualidade."""
    features = {}
    features['spectral_tilt'] = compute_spectral_tilt(y, sr)
    features['breathiness_index'] = compute_breathiness_index(y, f0)
    features['roughness_index'] = compute_roughness_index(y, f0)
    features['voice_breaks'] = compute_voice_breaks(f0)

    # Aliases mapeados
    features['breathiness'] = features['breathiness_index']
    features['roughness'] = features['roughness_index']
    features['hoarseness'] = features['voice_breaks']

    return features
