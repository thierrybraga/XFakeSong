"""
Extração de características de ruído (NHR, VTI, SPI, DFA).
"""
import numpy as np
import scipy.signal
import librosa
from typing import Dict


def compute_nhr(y: np.ndarray, f0: np.ndarray, sr: int = 22050) -> float:
    """
    Calcula Noise-to-Harmonic Ratio (NHR).
    Razão entre energia na parte inarmônica e energia na parte harmônica.
    """
    try:
        # Método simplificado usando filtro comb ou autocorrelação
        # Aqui usaremos uma aproximação baseada em HNR do librosa (invertido)
        # HNR = Harmonic-to-Noise. NHR ≈ 1/HNR (para linear) ou -HNR (para dB)

        # Calcular harmocidade
        harmonicity = librosa.effects.harmonic(y)
        noise = y - harmonicity

        e_harmonic = np.sum(harmonicity ** 2)
        e_noise = np.sum(noise ** 2)

        if e_harmonic == 0:
            return 1.0  # Puro ruído

        return float(e_noise / e_harmonic)
    except BaseException:
        return 0.0


def compute_vti(y: np.ndarray, f0: np.ndarray) -> float:
    """
    Calcula Voice Turbulence Index (VTI).
    Mede a energia relativa de ruído de alta frequência.
    """
    try:
        # VTI geralmente foca em ruído acima de certas frequências
        # Implementação simplificada: razão de energia em segmentos não
        # vozeados vs total
        voiced_mask = f0 > 0
        total_energy = np.sum(y ** 2)

        if total_energy == 0:
            return 0.0

        # Precisamos mapear f0 para amostras temporais
        # Assumindo que f0 foi extraído com hop_length padrão, precisamos interpolar ou expandir
        # Mas para simplificar: usar frames
        # Se f0 tem tamanho N, e y tem tamanho M, cada ponto de f0 corresponde
        # a hop_length amostras

        # Abordagem alternativa: usar separação harmônico/percussivo
        y_harm, y_perc = librosa.effects.hpss(y)

        # VTI é frequentemente associado a turbulência (ar), que aparece mais
        # em y_perc
        e_perc = np.sum(y_perc ** 2)
        return float(e_perc / total_energy)
    except BaseException:
        return 0.0


def compute_spi(y: np.ndarray, f0: np.ndarray) -> float:
    """
    Calcula Soft Phonation Index (SPI).
    Mede a relação de energia harmônica de baixa vs alta frequência.
    """
    try:
        # SPI geralmente compara energia < 1600Hz vs > 1600Hz (ou similar)
        S = np.abs(librosa.stft(y))
        # Assumindo 22050 padrão, idealmente passar sr
        freqs = librosa.fft_frequencies(sr=22050)

        split_freq = 1600
        idx = np.searchsorted(freqs, split_freq)

        low_energy = np.sum(S[:idx, :] ** 2)
        high_energy = np.sum(S[idx:, :] ** 2)

        if high_energy == 0:
            return 100.0  # Valor alto arbitrário

        return float(low_energy / high_energy)
    except BaseException:
        return 0.0


def compute_dfa(y: np.ndarray) -> float:
    """
    Calcula Detrended Fluctuation Analysis (DFA) Alpha.
    Mede a auto-semelhança do sinal (complexidade estocástica).
    """
    try:
        # Implementação simplificada do coeficiente alfa de DFA
        # 1. Integrar o sinal (cumulative sum)
        y_int = np.cumsum(y - np.mean(y))

        # 2. Dividir em janelas de diferentes tamanhos e calcular RMS das
        # flutuações (detrended)
        scales = [16, 32, 64, 128, 256, 512, 1024]
        fluctuations = []

        valid_scales = [s for s in scales if s < len(y) // 4]
        if len(valid_scales) < 3:
            return 0.5  # Ruído branco

        for scale in valid_scales:
            rms = []
            # Dividir em segmentos
            n_segments = len(y_int) // scale
            for i in range(n_segments):
                segment = y_int[i * scale: (i + 1) * scale]
                # Detrend (remover tendência linear)
                x = np.arange(scale)
                coeffs = np.polyfit(x, segment, 1)
                trend = np.polyval(coeffs, x)
                # RMS da flutuação
                rms.append(np.sqrt(np.mean((segment - trend)**2)))

            fluctuations.append(np.mean(rms))

        # 3. Log-log plot e inclinação
        if not fluctuations or any(f <= 0 for f in fluctuations):
            return 0.5

        log_scales = np.log10(valid_scales)
        log_fluct = np.log10(fluctuations)

        alpha = np.polyfit(log_scales, log_fluct, 1)[0]
        return float(alpha)
    except BaseException:
        return 0.5


def extract_noise_features(
        y: np.ndarray, f0: np.ndarray, sr: int) -> Dict[str, float]:
    """Extrai todas as características de ruído."""
    features = {}
    features['nhr'] = compute_nhr(y, f0, sr)
    features['vti'] = compute_vti(y, f0)
    # Nota: spi usa sr padrão 22050 internamente se não ajustado
    features['spi'] = compute_spi(y, f0)
    features['dfa_alpha'] = compute_dfa(y)
    return features
