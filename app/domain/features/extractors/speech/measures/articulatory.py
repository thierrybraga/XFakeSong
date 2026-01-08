import numpy as np
import scipy.stats
from typing import Dict
from . import common


def extract_articulatory_features(
        y: np.ndarray, sr: int, frame_length: int, hop_length: int) -> Dict:
    """Extrai características articulatórias."""
    features = {}

    # === ESTIMATIVA DO COMPRIMENTO DO TRATO VOCAL ===
    vtl_features = extract_vocal_tract_length_features(
        y, sr, frame_length, hop_length)
    features.update(vtl_features)

    # === PARÂMETROS DE FLUXO GLOTAL ===
    glottal_features = extract_glottal_flow_features(y, sr)
    features.update(glottal_features)

    # === CARACTERÍSTICAS BASEADAS EM MODELOS ARTICULATÓRIOS ===
    articulatory_model_features = extract_articulatory_model_features(
        y, sr, frame_length, hop_length)
    features.update(articulatory_model_features)

    return features


def extract_vocal_tract_length_features(
        y: np.ndarray, sr: int, frame_length: int, hop_length: int) -> Dict:
    """Extrai estimativas do comprimento do trato vocal."""
    features = {}

    # Estimar comprimento do trato vocal baseado em formantes
    formant_frequencies = common.extract_formant_frequencies(
        y, sr, frame_length, hop_length)

    if formant_frequencies:
        # Usar fórmula aproximada: VTL ≈ (2n-1) * c / (4 * Fn)
        # onde c = velocidade do som (~35000 cm/s), n = número do formante

        c = 35000  # cm/s
        vtl_estimates = []

        for i, freq in enumerate(
                formant_frequencies[:4]):  # Primeiros 4 formantes
            if freq > 0:
                n = i + 1
                vtl = (2 * n - 1) * c / (4 * freq)
                vtl_estimates.append(vtl)

        if vtl_estimates:
            features['vocal_tract_length_mean'] = np.mean(vtl_estimates)
            features['vocal_tract_length_std'] = np.std(vtl_estimates)
            features['vocal_tract_length_median'] = np.median(vtl_estimates)

    return features


def extract_glottal_flow_features(y: np.ndarray, sr: int) -> Dict:
    """Extrai parâmetros de fluxo glotal."""
    features = {}

    # Estimar sinal glotal usando predição linear inversa
    glottal_signal = common.estimate_glottal_signal(y, sr)

    if glottal_signal is not None:
        # Características do fluxo glotal
        features['glottal_flow_mean'] = np.mean(glottal_signal)
        features['glottal_flow_std'] = np.std(glottal_signal)
        features['glottal_flow_skewness'] = scipy.stats.skew(glottal_signal)
        features['glottal_flow_kurtosis'] = scipy.stats.kurtosis(
            glottal_signal)

        # Características espectrais do fluxo glotal
        glottal_spectrum = np.abs(np.fft.fft(glottal_signal))

        # Inclinação espectral do fluxo glotal
        freqs = np.fft.fftfreq(len(glottal_spectrum), 1 / sr)
        positive_freqs = freqs[:len(freqs) // 2]
        positive_spectrum = glottal_spectrum[:len(glottal_spectrum) // 2]

        if len(positive_spectrum) > 1:
            # Ajustar linha em escala log
            log_spectrum = np.log(positive_spectrum + 1e-12)
            slope, _, _, _, _ = scipy.stats.linregress(
                positive_freqs, log_spectrum)
            features['glottal_spectral_tilt'] = slope

    return features


def extract_articulatory_model_features(
        y: np.ndarray, sr: int, frame_length: int, hop_length: int) -> Dict:
    """Extrai características baseadas em modelos articulatórios."""
    features = {}

    # Estimar parâmetros articulatórios baseados em características espectrais

    # === ABERTURA VOCAL (baseada em F1) ===
    formants = common.extract_formant_frequencies(
        y, sr, frame_length, hop_length)

    if formants and len(formants) > 0:
        f1 = formants[0]

        # F1 correlaciona com abertura da mandíbula
        # Normalizar para escala 0-1 (aproximada)
        f1_normalized = (f1 - 200) / (800 - 200)  # Faixa típica de F1
        f1_normalized = np.clip(f1_normalized, 0, 1)

        features['jaw_openness_estimate'] = f1_normalized

    if formants and len(formants) > 1:
        f2 = formants[1]

        # F2 correlaciona com posição da língua (anterior/posterior)
        # Normalizar para escala 0-1
        f2_normalized = (f2 - 800) / (2500 - 800)  # Faixa típica de F2
        f2_normalized = np.clip(f2_normalized, 0, 1)

        features['tongue_position_estimate'] = f2_normalized

    # === CARACTERÍSTICAS DE CONSTRIÇÃO ===
    # Baseadas na largura de banda dos formantes
    formant_bandwidths = common.extract_formant_bandwidths(y)

    if formant_bandwidths:
        # Largura de banda correlaciona com grau de constrição
        mean_bandwidth = np.mean(formant_bandwidths)
        features['constriction_degree_estimate'] = mean_bandwidth / \
            100  # Normalizar

    return features
