import numpy as np
import librosa
import warnings
from typing import Dict, Any
from .utils import compute_spectral_centroid, compute_spectral_rolloff, compute_spectral_flux


def extract_auditory_features(
        y: np.ndarray, sr: int, frame_length: int,
        hop_length: int) -> Dict[str, Any]:
    """Extrai características do espectro auditivo."""
    features = {}

    try:
        # Espectro auditivo simplificado
        auditory_spectrum = compute_auditory_spectrum(
            y, sr, frame_length, hop_length)

        # Características
        features['auditory_centroid'] = compute_spectral_centroid(
            auditory_spectrum)
        features['auditory_rolloff'] = compute_spectral_rolloff(
            auditory_spectrum)
        features['auditory_flux'] = compute_spectral_flux(auditory_spectrum)

    except Exception as e:
        warnings.warn(f"Erro no espectro auditivo: {str(e)}")
        features['auditory_centroid'] = np.array([0])
        features['auditory_rolloff'] = np.array([0])
        features['auditory_flux'] = np.array([0])

    return features


def compute_auditory_spectrum(
        y: np.ndarray, sr: int, frame_length: int, hop_length: int) -> np.ndarray:
    """Computa espectro auditivo simplificado."""
    # Calcular espectrograma
    S = np.abs(librosa.stft(y, n_fft=frame_length, hop_length=hop_length))**2

    # Aplicar escala mel como aproximação auditiva
    mel_spectrum = librosa.feature.melspectrogram(S=S, sr=sr, n_mels=128)

    return mel_spectrum
