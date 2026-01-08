import numpy as np
import librosa
import warnings


def compute_delta(features: np.ndarray, order: int = 1) -> np.ndarray:
    """Extrai características delta de primeira ou segunda ordem."""
    try:
        return librosa.feature.delta(features, order=order)
    except Exception as e:
        warnings.warn(f"Erro na extração delta: {str(e)}")
        return np.zeros_like(features)


class DeltaFeaturesExtractor:
    """Extrator de características delta (derivadas temporais)."""

    @staticmethod
    def extract_delta(features: np.ndarray, order: int = 1) -> np.ndarray:
        return compute_delta(features, order)
