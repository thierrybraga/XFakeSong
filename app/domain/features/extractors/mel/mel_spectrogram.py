"""
Extração de Mel Spectrograma
===========================

Implementa extração de Mel Spectrograma para modelos de Deep Learning.
"""

import numpy as np
import librosa
import warnings
from typing import Dict, List, Any

from .....core.interfaces.audio import AudioData, AudioFeatures, FeatureType, IFeatureExtractor
from .....core.interfaces.base import ProcessingResult, ProcessingStatus


class MelSpectrogramExtractor(IFeatureExtractor):
    """Extrator de Mel Spectrograma."""

    def __init__(self, sr: int = 16000, n_mels: int = 128, n_fft: int = 2048,
                 hop_length: int = 512, win_length: int = None,
                 fmin: float = 0.0, fmax: float = None):
        """
        Inicializa o extrator de Mel Spectrograma.

        Args:
            sr: Taxa de amostragem (padrão: 16000Hz para maioria dos modelos DL)
            n_mels: Número de bandas Mel
            n_fft: Tamanho da FFT
            hop_length: Salto entre janelas
            win_length: Tamanho da janela (se None, igual a n_fft)
            fmin: Frequência mínima
            fmax: Frequência máxima
        """
        self.sr = sr
        self.n_mels = n_mels
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length if win_length else n_fft
        self.fmin = fmin
        self.fmax = fmax

    def extract(
            self, audio_data: AudioData) -> ProcessingResult[AudioFeatures]:
        """Extrai Mel Spectrograma do áudio."""
        try:
            return self.extract_features(audio_data)
        except Exception as e:
            return ProcessingResult(
                status=ProcessingStatus.ERROR,
                errors=[str(e)]
            )

    def extract_features(
            self, audio_data: AudioData) -> ProcessingResult[AudioFeatures]:
        """Implementação da interface IFeatureExtractor."""
        try:
            if audio_data.samples is None or len(audio_data.samples) == 0:
                return ProcessingResult(
                    status=ProcessingStatus.ERROR,
                    errors=["Dados de áudio vazios"]
                )

            y = audio_data.samples

            # Garantir 1D array
            if y.ndim > 1:
                y = y.flatten()

            # Calcular Mel Spectrograma
            mel_spec = librosa.feature.melspectrogram(
                y=y,
                sr=self.sr,
                n_fft=self.n_fft,
                hop_length=self.hop_length,
                win_length=self.win_length,
                n_mels=self.n_mels,
                fmin=self.fmin,
                fmax=self.fmax
            )

            # Converter para escala logarítmica (dB)
            mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)

            # Transpor para formato (tempo, features) que é o padrão
            # para modelos sequenciais/Keras
            # Original: (n_mels, tempo) -> Transposto: (tempo, n_mels)
            mel_spec_db_T = mel_spec_db.T

            # Criar dicionário de features
            features_dict = {
                'mel_spectrogram': mel_spec_db_T,
                # Flattened version for compatibility with some interfaces
                'mel_spectrogram_flat': mel_spec_db_T.flatten()
            }

            audio_features = AudioFeatures(
                features=features_dict,
                feature_type=FeatureType.MEL_SPECTROGRAM,
                extraction_params=self.get_extraction_params(),
                audio_metadata={
                    'sample_rate': audio_data.sample_rate,
                    'duration': audio_data.duration}
            )

            return ProcessingResult(
                status=ProcessingStatus.SUCCESS,
                data=audio_features
            )

        except Exception as e:
            return ProcessingResult(
                status=ProcessingStatus.ERROR,
                errors=[f"Erro na extração de Mel Spectrograma: {str(e)}"]
            )

    def get_feature_type(self) -> FeatureType:
        return FeatureType.MEL_SPECTROGRAM

    def get_feature_names(self) -> List[str]:
        return ['mel_spectrogram']

    def get_extraction_params(self) -> Dict[str, Any]:
        return {
            'sr': self.sr,
            'n_mels': self.n_mels,
            'n_fft': self.n_fft,
            'hop_length': self.hop_length,
            'win_length': self.win_length,
            'fmin': self.fmin,
            'fmax': self.fmax
        }
