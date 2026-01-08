import numpy as np
import librosa
import warnings
from typing import Dict


def extract_spectral_features(
        y: np.ndarray, sr: int, frame_length: int,
        hop_length: int) -> Dict[str, np.ndarray]:
    """Extrai características espectrais básicas."""
    try:
        # Centroide espectral
        spectral_centroids = librosa.feature.spectral_centroid(
            y=y, sr=sr, hop_length=hop_length, n_fft=frame_length
        )

        # Rolloff espectral
        spectral_rolloff = librosa.feature.spectral_rolloff(
            y=y, sr=sr, hop_length=hop_length, n_fft=frame_length
        )

        # Zero crossing rate
        zcr = librosa.feature.zero_crossing_rate(
            y, frame_length=frame_length, hop_length=hop_length
        )

        # Chroma
        chroma = librosa.feature.chroma_stft(
            y=y, sr=sr, n_fft=frame_length, hop_length=hop_length
        )

        return {
            'spectral_centroids': spectral_centroids,
            'spectral_rolloff': spectral_rolloff,
            'zero_crossing_rate': zcr,
            'chroma': chroma
        }

    except Exception as e:
        warnings.warn(f"Erro na extração espectral: {str(e)}")
        return {
            'spectral_centroids': np.zeros((1, 1)),
            'spectral_rolloff': np.zeros((1, 1)),
            'zero_crossing_rate': np.zeros((1, 1))
        }
