import numpy as np
import librosa
import warnings
from .mfcc import FeatureExtractor


class MelSpectrogramExtractor(FeatureExtractor):
    """Extrator de espectrograma Mel otimizado para preservar características."""

    def __init__(self, sr: int = 22050, n_fft: int = 2048,
                 hop_length: int = 512, n_mels: int = 128):
        self.sr = sr
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_mels = n_mels

        # Faixa de frequência otimizada
        self.fmin = 80.0
        self.fmax = min(8000.0, sr // 2)

    def extract(self, y: np.ndarray) -> np.ndarray:
        """Extrai espectrograma Mel com processamento conservador."""
        try:
            # Espectrograma Mel
            mel_spec = librosa.feature.melspectrogram(
                y=y, sr=self.sr, n_fft=self.n_fft,
                hop_length=self.hop_length, n_mels=self.n_mels,
                fmin=self.fmin, fmax=self.fmax
            )

            # Conversão para escala logarítmica
            log_mel_spec = librosa.power_to_db(mel_spec, ref=np.max)

            # Suavização temporal conservadora
            log_mel_spec = self._apply_conservative_smoothing(log_mel_spec)

            return log_mel_spec

        except Exception as e:
            warnings.warn(f"Erro na extração Mel: {str(e)}")
            return np.zeros((self.n_mels, 1))

    def _apply_conservative_smoothing(
            self, spec: np.ndarray, sigma: float = 0.5) -> np.ndarray:
        """Aplica suavização temporal conservadora."""
        from scipy.ndimage import gaussian_filter1d

        # Suavização muito leve para preservar características
        smoothed = gaussian_filter1d(spec, sigma=sigma, axis=1)

        # Mistura conservadora: 80% original + 20% suavizado
        return 0.8 * spec + 0.2 * smoothed
