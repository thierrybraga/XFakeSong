import numpy as np
import librosa
import warnings
from typing import Dict, Any


def extract_chroma_features(y: np.ndarray, sr: int,
                            hop_length: int) -> Dict[str, Any]:
    """Extrai características do chromagram."""
    features = {}

    try:
        # Calcular chromagram
        chroma = librosa.feature.chroma_stft(y=y, sr=sr, hop_length=hop_length)

        # Características básicas
        features['chroma_mean'] = np.mean(chroma, axis=1)
        features['chroma_std'] = np.std(chroma, axis=1)
        features['chroma_centroid'] = compute_chroma_centroid(chroma)

        # Fluxo cromático
        features['chroma_flux'] = compute_chroma_flux(chroma)

    except Exception as e:
        warnings.warn(f"Erro no chromagram: {str(e)}")
        features['chroma_mean'] = np.zeros(12)
        features['chroma_std'] = np.zeros(12)
        features['chroma_centroid'] = np.array([0])
        features['chroma_flux'] = np.array([0])

    return features


def compute_chroma_centroid(chroma: np.ndarray) -> np.ndarray:
    """Computa centroide do chromagram."""
    chroma_indices = np.arange(chroma.shape[0])
    centroids = []

    for frame in range(chroma.shape[1]):
        chroma_frame = chroma[:, frame]
        if np.sum(chroma_frame) > 0:
            centroid = np.sum(
                chroma_indices * chroma_frame) / np.sum(chroma_frame)
        else:
            centroid = 0.0
        centroids.append(centroid)

    return np.array(centroids)


def compute_chroma_flux(chroma: np.ndarray) -> np.ndarray:
    """Computa flux do chromagram."""
    if chroma.shape[1] < 2:
        return np.array([0.0])

    flux = np.sum(np.diff(chroma, axis=1)**2, axis=0)
    return np.concatenate([[0.0], flux])
