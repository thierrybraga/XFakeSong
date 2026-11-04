"""Interfaces específicas para processamento de áudio e detecção de deepfake

Este módulo define interfaces especializadas para:
- Processamento de áudio
- Extração de características
- Detecção de deepfake
- Treinamento de modelos
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Union

import numpy as np

from .base import IExtractor, ProcessingResult


class AudioFormat(Enum):
    """Formatos de áudio suportados."""

    WAV = ".wav"
    MP3 = ".mp3"
    FLAC = ".flac"
    M4A = ".m4a"


class FeatureType(Enum):
    """Tipos de características de áudio."""

    SPECTRAL = "spectral"
    MEL_SPECTROGRAM = "mel_spectrogram"
    TEMPORAL = "temporal"
    PROSODIC = "prosodic"
    PERCEPTUAL = "perceptual"
    ADVANCED = "advanced"
    CEPSTRAL = "cepstral"
    FORMANT = "formant"
    VOICE_QUALITY = "voice_quality"
    COMPLEXITY = "complexity"
    CUSTOM = "custom"


@dataclass
class AudioData:
    """Dados de áudio padronizados."""

    samples: np.ndarray
    sample_rate: int
    duration: float
    channels: int = 1
    bit_depth: int = 16
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}

    @classmethod
    def from_file(
        cls, file_path: Union[str, Path], sr: int = 16000, mono: bool = True
    ) -> "AudioData":
        import librosa

        p = str(file_path)
        y, s = librosa.load(p, sr=sr, mono=mono)
        duration = float(len(y) / s) if s else 0.0
        channels = 1 if mono or y.ndim == 1 else (y.shape[1] if y.ndim > 1 else 1)
        return cls(
            samples=y.astype(np.float32),
            sample_rate=int(s),
            duration=duration,
            channels=int(channels),
            bit_depth=16,
            metadata={"source": str(file_path)},
        )


@dataclass
class AudioFeatures:
    """Características extraídas de áudio."""

    features: Dict[str, np.ndarray]
    feature_type: FeatureType
    extraction_params: Dict[str, Any]
    audio_metadata: Dict[str, Any] = None

    def __post_init__(self):
        if self.audio_metadata is None:
            self.audio_metadata = {}


@dataclass
class DeepfakeDetectionResult:
    """Resultado de detecção de deepfake."""

    is_fake: bool
    confidence: float
    probabilities: Dict[str, float]
    model_name: str
    features_used: List[str]
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class IFeatureExtractor(IExtractor[AudioData, AudioFeatures]):
    """Interface para extração de características."""

    @abstractmethod
    def extract_features(
        self, audio_data: AudioData
    ) -> ProcessingResult[AudioFeatures]:
        """Extrai características do áudio."""
        pass

    @abstractmethod
    def get_feature_type(self) -> FeatureType:
        """Retorna tipo de característica extraída."""
        pass

    @abstractmethod
    def get_feature_names(self) -> List[str]:
        """Retorna nomes das características."""
        pass

    @abstractmethod
    def get_extraction_params(self) -> Dict[str, Any]:
        """Retorna parâmetros de extração."""
        pass


class IModelTrainer(ABC):
    """Interface para treinamento de modelos."""

    @abstractmethod
    def train(
        self, model: Any, train_data: Any, validation_data: Any = None, **kwargs
    ) -> ProcessingResult[Dict[str, Any]]:
        """Treina modelo."""
        pass

    @abstractmethod
    def evaluate(
        self, model: Any, test_data: Any
    ) -> ProcessingResult[Dict[str, float]]:
        """Avalia modelo."""
        pass

    @abstractmethod
    def save_model(
        self, model: Any, save_path: Union[str, Path]
    ) -> ProcessingResult[str]:
        """Salva modelo treinado."""
        pass

    @abstractmethod
    def load_model(self, model_path: Union[str, Path]) -> ProcessingResult[Any]:
        """Carrega modelo salvo."""
        pass
