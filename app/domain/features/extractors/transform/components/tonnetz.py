import numpy as np
import librosa
import warnings
from typing import Dict, Any


def extract_tonnetz_features(y: np.ndarray, sr: int) -> Dict[str, Any]:
    """Extrai características do tonnetz."""
    features = {}

    try:
        # Calcular tonnetz
        # Nota: tonnetz requer librosa >= 0.7
        tonnetz = librosa.feature.tonnetz(y=y, sr=sr)

        # Características estatísticas
        features['tonnetz_mean'] = np.mean(tonnetz, axis=1)
        features['tonnetz_std'] = np.std(tonnetz, axis=1)
        features['tonnetz_centroid'] = np.mean(tonnetz, axis=0)

    except Exception as e:
        warnings.warn(f"Erro no tonnetz: {str(e)}")
        features['tonnetz_mean'] = np.zeros(6)
        features['tonnetz_std'] = np.zeros(6)
        features['tonnetz_centroid'] = np.array([0])

    return features
