"""
Extrator de características de qualidade vocal.
"""
import numpy as np
import warnings
import librosa
from typing import Dict, List, Optional
from app.core.interfaces.base import ProcessingResult, ProcessingStatus
from app.core.interfaces.audio import IFeatureExtractor, AudioFeatures, AudioData, FeatureType
from .components.perturbation import extract_perturbation_features
from .components.noise import extract_noise_features
from .components.quality import extract_additional_quality_features
from .components.utils import get_default_voice_quality_features


class VoiceQualityFeatureExtractor(IFeatureExtractor):
    """Extrator de características de qualidade vocal."""

    def __init__(self, sr: int = 22050, frame_length: int = 2048,
                 hop_length: int = 512, window: str = 'hann',
                 f0_min: float = 80.0, f0_max: float = 400.0):
        """Inicializa o extrator de qualidade vocal."""
        # Validação de parâmetros
        if sr <= 0:
            raise ValueError("Taxa de amostragem deve ser positiva")
        self.sr = sr
        self.frame_length = frame_length
        self.hop_length = hop_length
        self.window = window
        self.f0_min = f0_min
        self.f0_max = f0_max
        self.fmin = f0_min
        self.fmax = min(f0_max, sr // 4)

    def extract(
            self, audio_data: AudioData) -> ProcessingResult[AudioFeatures]:
        """Extrai características de qualidade vocal do áudio."""
        try:
            if audio_data.samples is None or len(audio_data.samples) == 0:
                return ProcessingResult(
                    status=ProcessingStatus.ERROR,
                    errors=["Dados de áudio vazios"]
                )

            samples = audio_data.samples

            if audio_data.sample_rate != self.sr:
                samples = librosa.resample(
                    samples, orig_sr=audio_data.sample_rate, target_sr=self.sr)

            features_dict = self._compute_features(samples)

            # Converter características para formato AudioFeatures
            features_arrays = {}
            for name, value in features_dict.items():
                if isinstance(value, (list, np.ndarray)):
                    if len(np.array(value).shape) == 0:
                        features_arrays[name] = np.array([float(value)])
                    else:
                        features_arrays[name] = np.array(value).flatten()
                else:
                    features_arrays[name] = np.array([float(value)])

            audio_features = AudioFeatures(
                features=features_arrays,
                feature_type=FeatureType.VOICE_QUALITY,
                extraction_params=self.get_extraction_params(),
                audio_metadata={'sample_rate': audio_data.sample_rate}
            )

            return ProcessingResult.success(
                data=audio_features
            )

        except Exception as e:
            return ProcessingResult(
                status=ProcessingStatus.ERROR,
                errors=[str(e)]
            )

    def extract_features(
            self, audio_data: AudioData) -> ProcessingResult[AudioFeatures]:
        """Implementação da interface IFeatureExtractor."""
        return self.extract(audio_data)

    def _compute_features(self, y: np.ndarray) -> Dict:
        """Extrai todas as características de qualidade vocal."""
        # Validação de entrada
        if not isinstance(y, np.ndarray):
            raise TypeError("Entrada deve ser um array numpy")

        if len(y) == 0:
            warnings.warn("Sinal de áudio vazio")
            return get_default_voice_quality_features()

        # Verificar se o sinal é muito curto
        min_length = self.frame_length
        if len(y) < min_length:
            warnings.warn(
                f"Sinal muito curto ({
                    len(y)} samples), fazendo padding para {min_length}")
            y = np.pad(y, (0, min_length - len(y)), mode='constant')

        # Normalizar sinal
        y_max = np.max(np.abs(y))
        if y_max > 0:
            y = y / y_max

        try:
            features = {}

            # Extrair F0 para análises baseadas em pitch
            f0 = librosa.yin(y, fmin=self.fmin, fmax=self.fmax, sr=self.sr,
                             frame_length=self.frame_length, hop_length=self.hop_length)

            # === MEDIDAS DE PERTURBAÇÃO ===
            perturbation_features = extract_perturbation_features(
                y, f0, self.sr, self.frame_length, self.hop_length
            )
            features.update(perturbation_features)

            # === MEDIDAS DE RUÍDO ===
            noise_features = extract_noise_features(
                y, f0, self.sr
            )
            features.update(noise_features)

            # === MEDIDAS ADICIONAIS DE QUALIDADE ===
            quality_features = extract_additional_quality_features(
                y, f0, self.sr
            )
            features.update(quality_features)

            return features

        except Exception as e:
            warnings.warn(
                f"Erro na extração de características de qualidade vocal: {e}")
            return get_default_voice_quality_features()

    def get_feature_type(self) -> FeatureType:
        """Retorna tipo de característica extraída."""
        return FeatureType.VOICE_QUALITY

    def get_feature_names(self) -> List[str]:
        """Retorna nomes das características."""
        return list(get_default_voice_quality_features().keys())

    def get_extraction_params(self) -> Dict:
        """Retorna parâmetros de extração."""
        return {
            'sr': self.sr,
            'frame_length': self.frame_length,
            'hop_length': self.hop_length,
            'window': self.window,
            'f0_min': self.f0_min,
            'f0_max': self.f0_max
        }
