import numpy as np
import librosa
import warnings
from typing import Dict
from .lpc_utils import compute_lpc, lpc_to_cepstral


def extract_lpcc_features(y: np.ndarray, frame_length: int,
                          hop_length: int, n_lpcc: int) -> Dict:
    """Extrai LPCC (Linear Predictive Cepstral Coefficients)."""
    features = {}

    try:
        # Dividir em frames
        frames = librosa.util.frame(y, frame_length=frame_length,
                                    hop_length=hop_length)

        lpcc_coeffs = []

        for frame in frames.T:
            if len(frame) > n_lpcc:
                # Análise LPC
                lpc_coeffs = compute_lpc(frame, n_lpcc)

                # Converter LPC para cepstral
                cepstral = lpc_to_cepstral(lpc_coeffs.reshape(-1, 1), n_lpcc)
                lpcc_coeffs.append(cepstral.flatten())
            else:
                lpcc_coeffs.append(np.zeros(n_lpcc))

        lpcc_coeffs = np.array(lpcc_coeffs).T

        features['lpcc'] = lpcc_coeffs
        features['lpcc_delta'] = librosa.feature.delta(lpcc_coeffs)
        features['lpcc_delta2'] = librosa.feature.delta(lpcc_coeffs, order=2)

    except Exception as e:
        warnings.warn(f"Erro na extração LPCC: {str(e)}")
        features['lpcc'] = np.zeros((n_lpcc, 1))
        features['lpcc_delta'] = np.zeros((n_lpcc, 1))
        features['lpcc_delta2'] = np.zeros((n_lpcc, 1))

    return features
