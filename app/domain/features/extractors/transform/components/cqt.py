import numpy as np
import librosa
import warnings
from typing import Dict, Any
from .utils import (
    compute_spectral_centroid,
    compute_spectral_rolloff,
    compute_octave_energy
)


def extract_cqt_features(y: np.ndarray, sr: int, hop_length: int,
                         bins_per_octave: int = 12) -> Dict[str, Any]:
    """Extrai características do Constant-Q Transform."""
    features = {}

    try:
        # Calcular CQT
        cqt = librosa.cqt(y, sr=sr, hop_length=hop_length)
        cqt_mag = np.abs(cqt)

        # Centroide CQT
        features['cqt_centroid'] = compute_spectral_centroid(cqt_mag)

        # Rolloff CQT
        features['cqt_rolloff'] = compute_spectral_rolloff(cqt_mag)

        # Energia por oitava
        features['cqt_octave_energy'] = compute_octave_energy(
            cqt_mag, bins_per_octave)

    except Exception as e:
        warnings.warn(f"Erro no CQT: {str(e)}")
        features['cqt_centroid'] = np.array([0])
        features['cqt_rolloff'] = np.array([0])
        features['cqt_octave_energy'] = np.array([0])

    return features
