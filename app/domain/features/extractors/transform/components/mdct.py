import numpy as np
import librosa
import scipy.fft
import warnings
from typing import Dict, Any
from .utils import compute_spectral_centroid, compute_spectral_rolloff


def extract_mdct_features(y: np.ndarray, frame_length: int,
                          hop_length: int) -> Dict[str, Any]:
    """Extrai características do MDCT."""
    features = {}

    try:
        # MDCT simplificado usando DCT
        # Dividir sinal em frames
        frames = librosa.util.frame(y, frame_length=frame_length,
                                    hop_length=hop_length)

        mdct_coeffs = []
        # Iterar sobre colunas (se axis=-1 padrão)
        for frame in frames.T:
            # Aplicar janela
            windowed = frame * np.hanning(len(frame))
            # DCT como aproximação do MDCT
            dct_coeffs = scipy.fft.dct(windowed, type=2, norm='ortho')
            mdct_coeffs.append(dct_coeffs)

        mdct_coeffs = np.array(mdct_coeffs).T

        # Características
        features['mdct_centroid'] = compute_spectral_centroid(
            np.abs(mdct_coeffs))
        features['mdct_rolloff'] = compute_spectral_rolloff(
            np.abs(mdct_coeffs))
        features['mdct_energy'] = np.sum(mdct_coeffs**2, axis=0)

    except Exception as e:
        warnings.warn(f"Erro no MDCT: {str(e)}")
        features['mdct_centroid'] = np.array([0])
        features['mdct_rolloff'] = np.array([0])
        features['mdct_energy'] = np.array([0])

    return features
