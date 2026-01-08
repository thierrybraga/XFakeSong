import numpy as np
import librosa
import warnings
from typing import Optional
from abc import ABC, abstractmethod


class FeatureExtractor(ABC):
    """Interface base para extratores de características."""

    @abstractmethod
    def extract(self, y: np.ndarray) -> np.ndarray:
        """Extrai características do sinal de áudio."""
        pass


class MFCCExtractor(FeatureExtractor):
    """Extrator de coeficientes MFCC otimizado para detecção de deepfakes."""

    def __init__(self, sr: int = 22050, n_mfcc: int = 13, n_fft: int = 2048,
                 hop_length: int = 512, n_mels: int = 40):
        self.sr = sr
        self.n_mfcc = n_mfcc
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_mels = n_mels

        # Faixa de frequência otimizada para voz humana
        self.fmin = 80.0
        self.fmax = min(8000.0, sr // 2)

    def extract(self, y: np.ndarray) -> np.ndarray:
        """Extrai coeficientes MFCC com processamento conservador."""
        try:
            # MFCC com parâmetros conservadores
            mfcc = librosa.feature.mfcc(
                y=y, sr=self.sr, n_mfcc=self.n_mfcc,
                n_fft=self.n_fft, hop_length=self.hop_length,
                n_mels=self.n_mels, fmin=self.fmin, fmax=self.fmax,
                window='hamming'
            )

            # Aplicar liftering conservador
            mfcc = self._apply_liftering(mfcc, lifter=22)

            # Normalização cepstral média (CMN)
            mfcc = self._apply_cepstral_mean_normalization(mfcc)

            return mfcc

        except Exception as e:
            warnings.warn(f"Erro na extração MFCC: {str(e)}")
            return np.zeros((self.n_mfcc, 1))

    def _apply_liftering(self, mfcc: np.ndarray,
                         lifter: int = 22) -> np.ndarray:
        """Aplica liftering aos coeficientes MFCC."""
        if lifter > 0:
            n_mfcc = mfcc.shape[0]
            lift = 1 + (lifter / 2) * np.sin(np.pi *
                                             np.arange(n_mfcc) / lifter)
            mfcc *= lift[:, np.newaxis]
        return mfcc

    def _apply_cepstral_mean_normalization(
            self, mfcc: np.ndarray) -> np.ndarray:
        """Aplica normalização cepstral média."""
        return mfcc - np.mean(mfcc, axis=1, keepdims=True)
