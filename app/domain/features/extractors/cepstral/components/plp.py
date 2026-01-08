import numpy as np
import librosa
import scipy.signal
import warnings
from typing import Dict
from .lpc_utils import solve_yule_walker, lpc_to_cepstral


def apply_bark_scale(S: np.ndarray, sr: int, frame_length: int) -> np.ndarray:
    """Aplica escala Bark ao espectrograma."""
    freqs = librosa.fft_frequencies(sr=sr, n_fft=frame_length)

    # Converter para escala Bark
    bark_freqs = 13 * np.arctan(0.00076 * freqs) + \
        3.5 * np.arctan((freqs / 7500)**2)

    # Criar banco de filtros Bark
    n_bark_bands = 24
    bark_filters = np.zeros((n_bark_bands, len(freqs)))

    bark_edges = np.linspace(0, np.max(bark_freqs), n_bark_bands + 1)

    for i in range(n_bark_bands):
        left = bark_edges[i]
        center = bark_edges[i + 1] if i + \
            1 < len(bark_edges) else bark_edges[i]
        right = bark_edges[i + 1] if i + 1 < len(bark_edges) else center

        # Filtro triangular
        for j, bark_freq in enumerate(bark_freqs):
            if left <= bark_freq <= center:
                bark_filters[i, j] = (bark_freq - left) / \
                    (center - left + 1e-10)
            elif center < bark_freq <= right:
                bark_filters[i, j] = (right - bark_freq) / \
                    (right - center + 1e-10)

    # Aplicar filtros
    bark_spectrum = np.dot(bark_filters, S)

    return bark_spectrum


def apply_equal_loudness_curve(spectrum: np.ndarray) -> np.ndarray:
    """Aplica curva de igual loudness."""
    # Simplificação da curva de igual loudness
    equal_loudness = np.ones_like(spectrum)

    # Aplicar pesos aproximados
    for i in range(spectrum.shape[0]):
        if i < spectrum.shape[0] // 4:  # Baixas frequências
            equal_loudness[i] *= 0.5
        elif i > 3 * spectrum.shape[0] // 4:  # Altas frequências
            equal_loudness[i] *= 0.3

    return spectrum * equal_loudness


def apply_rasta_filter(spectrum: np.ndarray) -> np.ndarray:
    """Aplica filtro RASTA."""
    filtered_spectrum = np.zeros_like(spectrum)

    for i in range(spectrum.shape[0]):
        signal = spectrum[i, :]

        # Aplicar filtro passa-banda simples
        if len(signal) > 4:
            # Filtro diferencial simples
            filtered = np.diff(signal, n=1)
            filtered = np.pad(filtered, (1, 0), mode='constant')

            # Suavização
            if len(filtered) > 3:
                filtered = scipy.signal.savgol_filter(filtered, 3, 1)

            filtered_spectrum[i, :] = filtered
        else:
            filtered_spectrum[i, :] = signal

    return filtered_spectrum


def extract_plp_features(y: np.ndarray, sr: int,
                         frame_length: int, hop_length: int,
                         n_plp: int) -> Dict:
    """Extrai características PLP (Perceptual Linear Prediction)."""
    features = {}

    try:
        # Calcular espectrograma
        S = np.abs(librosa.stft(y, n_fft=frame_length,
                                hop_length=hop_length))

        # Aplicar escala Bark
        bark_spectrum = apply_bark_scale(S, sr, frame_length)

        # Aplicar curva de igual loudness
        equal_loudness_spectrum = apply_equal_loudness_curve(bark_spectrum)

        # Compressão de intensidade (cube root)
        compressed_spectrum = np.power(equal_loudness_spectrum, 1 / 3)

        # Aplicar IDFT para obter autocorrelação
        autocorr = np.real(np.fft.ifft(compressed_spectrum, axis=0))

        # Análise LPC
        plp_coeffs = []
        for frame in range(autocorr.shape[1]):
            try:
                # Usar apenas parte positiva da autocorrelação
                r = autocorr[:n_plp + 1, frame]

                # Resolver equações de Yule-Walker
                if len(r) > 1 and np.any(r):
                    a = solve_yule_walker(r, n_plp)
                    plp_coeffs.append(a)
                else:
                    plp_coeffs.append(np.zeros(n_plp))
            except BaseException:
                plp_coeffs.append(np.zeros(n_plp))

        plp_coeffs = np.array(plp_coeffs).T

        # Converter para coeficientes cepstrais
        plp_cepstral = lpc_to_cepstral(plp_coeffs, n_plp)

        features['plp'] = plp_cepstral
        features['plp_delta'] = librosa.feature.delta(plp_cepstral)
        features['plp_delta2'] = librosa.feature.delta(plp_cepstral, order=2)

    except Exception as e:
        warnings.warn(f"Erro na extração PLP: {str(e)}")
        features['plp'] = np.zeros((n_plp, 1))
        features['plp_delta'] = np.zeros((n_plp, 1))
        features['plp_delta2'] = np.zeros((n_plp, 1))

    return features


def extract_rasta_plp_features(
        y: np.ndarray, sr: int, frame_length: int, hop_length: int,
        n_plp: int) -> Dict:
    """Extrai características RASTA-PLP."""
    features = {}

    try:
        # Primeiro extrair PLP normal
        plp_features = extract_plp_features(
            y, sr, frame_length, hop_length, n_plp)
        plp_spectrum = plp_features['plp']

        # Aplicar filtro RASTA
        rasta_spectrum = apply_rasta_filter(plp_spectrum)

        features['rasta_plp'] = rasta_spectrum
        features['rasta_plp_delta'] = librosa.feature.delta(rasta_spectrum)
        features['rasta_plp_delta2'] = librosa.feature.delta(
            rasta_spectrum, order=2)

    except Exception as e:
        warnings.warn(f"Erro na extração RASTA-PLP: {str(e)}")
        features['rasta_plp'] = np.zeros((n_plp, 1))
        features['rasta_plp_delta'] = np.zeros((n_plp, 1))
        features['rasta_plp_delta2'] = np.zeros((n_plp, 1))

    return features
