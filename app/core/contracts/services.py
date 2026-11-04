"""Interfaces para serviços e repositórios do domínio

Este módulo define interfaces para:
- Repositórios de dados
- Serviços de domínio
- Casos de uso
- Integração externa
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from .audio import AudioData, AudioFeatures, DeepfakeDetectionResult
from .base import DatasetType, ProcessingResult


@dataclass
class DatasetMetadata:
    """Metadados de dataset."""

    name: str
    dataset_type: DatasetType
    description: str = ""
    file_count: int = 0
    total_size: int = 0
    total_duration: float = 0.0
    total_samples: int = 0
    real_samples: int = 0
    fake_samples: int = 0
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
    version: str = "1.0"
    tags: List[str] = None
    file_paths: List[str] = None

    def __post_init__(self):
        if self.tags is None:
            self.tags = []
        if self.file_paths is None:
            self.file_paths = []
        if self.file_count > 0 and self.total_samples == 0:
            self.total_samples = self.file_count


@dataclass
class ModelMetadata:
    """Metadados de modelo."""

    name: str
    architecture: str
    version: str
    created_at: datetime
    file_path: Path
    accuracy: float = 0.0
    precision: float = 0.0
    recall: float = 0.0
    f1_score: float = 0.0
    training_dataset: str = ""
    file_size: int = 0
    parameters: Dict[str, Any] = None
    metrics: Dict[str, float] = None

    def __post_init__(self):
        if self.parameters is None:
            self.parameters = {}
        if self.metrics is None:
            self.metrics = {}


class IUploadService(ABC):
    """Interface para serviço de upload."""

    @abstractmethod
    def upload_file(
        self, file_path: Union[str, Path], destination: str = None
    ) -> ProcessingResult[str]:
        """Faz upload de arquivo."""
        pass

    @abstractmethod
    def validate_upload(self, file_path: Union[str, Path]) -> ProcessingResult[bool]:
        """Valida arquivo para upload."""
        pass


class IFeatureExtractionService(ABC):
    """Interface para serviço de extração de características."""

    @abstractmethod
    def extract_single(
        self, audio_data: AudioData, feature_types: List[str]
    ) -> ProcessingResult[Dict[str, AudioFeatures]]:
        """Extrai características de um áudio."""
        pass

    @abstractmethod
    def extract_batch(
        self, audio_list: List[AudioData], feature_types: List[str]
    ) -> ProcessingResult[List[Dict[str, AudioFeatures]]]:
        """Extrai características em lote."""
        pass

    @abstractmethod
    def get_available_extractors(self) -> List[str]:
        """Retorna extratores disponíveis."""
        pass


class ITrainingService(ABC):
    """Interface para serviço de treinamento."""

    @abstractmethod
    def train_model(
        self, architecture: str, dataset_path: str, config: Dict[str, Any]
    ) -> ProcessingResult[ModelMetadata]:
        """Treina novo modelo."""
        pass


class IDetectionService(ABC):
    """Interface para serviço de detecção."""

    @abstractmethod
    def detect_single(
        self, audio_data: AudioData, model_name: str = None
    ) -> ProcessingResult[DeepfakeDetectionResult]:
        """Detecta deepfake em áudio único."""
        pass

    @abstractmethod
    def detect_batch(
        self, audio_list: List[AudioData], model_name: str = None
    ) -> ProcessingResult[List[DeepfakeDetectionResult]]:
        """Detecta deepfake em lote."""
        pass

    @abstractmethod
    def detect_from_file(
        self, file_path: Union[str, Path], model_name: str = None
    ) -> ProcessingResult[DeepfakeDetectionResult]:
        """Detecta deepfake de arquivo."""
        pass

    @abstractmethod
    def get_available_models(self) -> List[str]:
        """Retorna modelos disponíveis para detecção."""
        pass
