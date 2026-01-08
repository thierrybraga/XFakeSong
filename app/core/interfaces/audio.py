"""Interfaces específicas para processamento de áudio e detecção de deepfake

Este módulo define interfaces especializadas para:
- Processamento de áudio
- Extração de características
- Detecção de deepfake
- Treinamento de modelos
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional, Union, Tuple
from pathlib import Path
import numpy as np
from dataclasses import dataclass
from enum import Enum

from .base import IProcessor, IExtractor, IValidator, ProcessingResult, ProcessingStatus


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
    def from_file(cls, file_path: Union[str, Path],
                  sr: int = 16000, mono: bool = True) -> "AudioData":
        import librosa
        p = str(file_path)
        y, s = librosa.load(p, sr=sr, mono=mono)
        duration = float(len(y) / s) if s else 0.0
        channels = 1 if mono or y.ndim == 1 else (
            y.shape[1] if y.ndim > 1 else 1)
        return cls(
            samples=y.astype(np.float32),
            sample_rate=int(s),
            duration=duration,
            channels=int(channels),
            bit_depth=16,
            metadata={"source": str(file_path)}
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


class IAudioLoader(IProcessor[Union[str, Path], AudioData]):
    """Interface para carregamento de áudio."""

    @abstractmethod
    def load_audio(self, file_path: Union[str, Path]
                   ) -> ProcessingResult[AudioData]:
        """Carrega arquivo de áudio."""
        pass

    @abstractmethod
    def get_supported_formats(self) -> List[AudioFormat]:
        """Retorna formatos suportados."""
        pass


class IAudioValidator(IValidator[Union[str, Path, AudioData]]):
    """Interface para validação de áudio."""

    @abstractmethod
    def validate_format(
            self, file_path: Union[str, Path]) -> ProcessingResult[bool]:
        """Valida formato do arquivo."""
        pass

    @abstractmethod
    def validate_quality(
            self, audio_data: AudioData) -> ProcessingResult[bool]:
        """Valida qualidade do áudio."""
        pass

    @abstractmethod
    def validate_duration(self, audio_data: AudioData, min_duration: float = None,
                          max_duration: float = None) -> ProcessingResult[bool]:
        """Valida duração do áudio."""
        pass


class IFeatureExtractor(IExtractor[AudioData, AudioFeatures]):
    """Interface para extração de características."""

    @abstractmethod
    def extract_features(
            self, audio_data: AudioData) -> ProcessingResult[AudioFeatures]:
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


class IModelArchitecture(ABC):
    """Interface para arquiteturas de modelo."""

    @abstractmethod
    def create_model(
            self, input_shape: Tuple[int, ...], num_classes: int = 2, **kwargs) -> Any:
        """Cria modelo com arquitetura específica."""
        pass

    @abstractmethod
    def get_architecture_name(self) -> str:
        """Retorna nome da arquitetura."""
        pass

    @abstractmethod
    def get_default_params(self) -> Dict[str, Any]:
        """Retorna parâmetros padrão."""
        pass

    @abstractmethod
    def validate_input_shape(self, input_shape: Tuple[int, ...]) -> bool:
        """Valida formato de entrada."""
        pass


class IModelTrainer(ABC):
    """Interface para treinamento de modelos."""

    @abstractmethod
    def train(self, model: Any, train_data: Any, validation_data: Any = None,
              **kwargs) -> ProcessingResult[Dict[str, Any]]:
        """Treina modelo."""
        pass

    @abstractmethod
    def evaluate(self, model: Any,
                 test_data: Any) -> ProcessingResult[Dict[str, float]]:
        """Avalia modelo."""
        pass

    @abstractmethod
    def save_model(self, model: Any,
                   save_path: Union[str, Path]) -> ProcessingResult[str]:
        """Salva modelo treinado."""
        pass

    @abstractmethod
    def load_model(
            self, model_path: Union[str, Path]) -> ProcessingResult[Any]:
        """Carrega modelo salvo."""
        pass


class IDeepfakeDetector(IProcessor[AudioData, DeepfakeDetectionResult]):
    """Interface para detecção de deepfake."""

    @abstractmethod
    def detect(
            self, audio_data: AudioData) -> ProcessingResult[DeepfakeDetectionResult]:
        """Detecta se áudio é deepfake."""
        pass

    @abstractmethod
    def get_model_info(self) -> Dict[str, Any]:
        """Retorna informações do modelo."""
        pass

    @abstractmethod
    def get_confidence_threshold(self) -> float:
        """Retorna limiar de confiança."""
        pass

    @abstractmethod
    def set_confidence_threshold(
            self, threshold: float) -> ProcessingResult[bool]:
        """Define limiar de confiança."""
        pass


class IPipelineStage(IProcessor[Any, Any]):
    """Interface para estágios do pipeline."""

    @abstractmethod
    def get_stage_name(self) -> str:
        """Retorna nome do estágio."""
        pass

    @abstractmethod
    def get_dependencies(self) -> List[str]:
        """Retorna dependências do estágio."""
        pass

    @abstractmethod
    def can_run_parallel(self) -> bool:
        """Indica se pode executar em paralelo."""
        pass


class IPipelineOrchestrator(ABC):
    """Interface para orquestração do pipeline."""

    @abstractmethod
    def add_stage(self, stage: IPipelineStage) -> ProcessingResult[bool]:
        """Adiciona estágio ao pipeline."""
        pass

    @abstractmethod
    def remove_stage(self, stage_name: str) -> ProcessingResult[bool]:
        """Remove estágio do pipeline."""
        pass

    @abstractmethod
    def execute_pipeline(self, input_data: Any) -> ProcessingResult[Any]:
        """Executa pipeline completo."""
        pass

    @abstractmethod
    def get_pipeline_status(self) -> Dict[str, Any]:
        """Retorna status do pipeline."""
        pass
