"""Interfaces para serviços e repositórios do domínio

Este módulo define interfaces para:
- Repositórios de dados
- Serviços de domínio
- Casos de uso
- Integração externa
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional, Union, Iterator
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass
from enum import Enum

from .base import IRepository, ProcessingResult, ProcessingStatus
from .audio import AudioData, AudioFeatures, DeepfakeDetectionResult


class DatasetType(Enum):
    """Tipos de dataset."""
    TRAINING = "training"
    VALIDATION = "validation"
    TEST = "test"
    PRODUCTION = "production"


class StorageType(Enum):
    """Tipos de armazenamento."""
    LOCAL = "local"
    CLOUD = "cloud"
    DATABASE = "database"
    MEMORY = "memory"


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


class IAudioRepository(IRepository[AudioData]):
    """Interface para repositório de áudio."""

    @abstractmethod
    def save_audio(self, audio_data: AudioData, filename: str,
                   metadata: Dict[str, Any] = None) -> ProcessingResult[str]:
        """Salva dados de áudio."""
        pass

    @abstractmethod
    def load_audio(self, identifier: str) -> ProcessingResult[AudioData]:
        """Carrega dados de áudio."""
        pass

    @abstractmethod
    def list_audio_files(
            self, dataset_type: DatasetType = None) -> ProcessingResult[List[str]]:
        """Lista arquivos de áudio."""
        pass

    @abstractmethod
    def get_audio_metadata(
            self, identifier: str) -> ProcessingResult[Dict[str, Any]]:
        """Retorna metadados do áudio."""
        pass


class IFeatureRepository(IRepository[AudioFeatures]):
    """Interface para repositório de características."""

    @abstractmethod
    def save_features(self, features: AudioFeatures,
                      identifier: str) -> ProcessingResult[str]:
        """Salva características extraídas."""
        pass

    @abstractmethod
    def load_features(
            self, identifier: str) -> ProcessingResult[AudioFeatures]:
        """Carrega características."""
        pass

    @abstractmethod
    def list_features(
            self, feature_type: str = None) -> ProcessingResult[List[str]]:
        """Lista características disponíveis."""
        pass

    @abstractmethod
    def batch_load_features(
            self, identifiers: List[str]) -> ProcessingResult[List[AudioFeatures]]:
        """Carrega múltiplas características."""
        pass


class IModelRepository(IRepository[Any]):
    """Interface para repositório de modelos."""

    @abstractmethod
    def save_model(self, model: Any,
                   metadata: ModelMetadata) -> ProcessingResult[str]:
        """Salva modelo treinado."""
        pass

    @abstractmethod
    def load_model(self, model_name: str,
                   version: str = "latest") -> ProcessingResult[Any]:
        """Carrega modelo."""
        pass

    @abstractmethod
    def list_models(
            self, architecture: str = None) -> ProcessingResult[List[ModelMetadata]]:
        """Lista modelos disponíveis."""
        pass

    @abstractmethod
    def get_model_metadata(
            self, model_name: str, version: str = "latest") -> ProcessingResult[ModelMetadata]:
        """Retorna metadados do modelo."""
        pass

    @abstractmethod
    def delete_model(self, model_name: str,
                     version: str = None) -> ProcessingResult[bool]:
        """Remove modelo."""
        pass


class IDatasetRepository(IRepository[DatasetMetadata]):
    """Interface para repositório de datasets."""

    @abstractmethod
    def create_dataset(
            self, metadata: DatasetMetadata) -> ProcessingResult[str]:
        """Cria novo dataset."""
        pass

    @abstractmethod
    def get_dataset_info(
            self, dataset_name: str) -> ProcessingResult[DatasetMetadata]:
        """Retorna informações do dataset."""
        pass

    @abstractmethod
    def list_datasets(
            self, dataset_type: DatasetType = None) -> ProcessingResult[List[DatasetMetadata]]:
        """Lista datasets disponíveis."""
        pass

    @abstractmethod
    def update_dataset_metadata(
            self, dataset_name: str, metadata: DatasetMetadata) -> ProcessingResult[bool]:
        """Atualiza metadados do dataset."""
        pass


class IUploadService(ABC):
    """Interface para serviço de upload."""

    @abstractmethod
    def upload_file(self, file_path: Union[str, Path],
                    destination: str = None) -> ProcessingResult[str]:
        """Faz upload de arquivo."""
        pass

    @abstractmethod
    def upload_dataset(self, dataset_path: Union[str, Path],
                       dataset_name: str) -> ProcessingResult[DatasetMetadata]:
        """Faz upload de dataset completo."""
        pass

    @abstractmethod
    def validate_upload(
            self, file_path: Union[str, Path]) -> ProcessingResult[bool]:
        """Valida arquivo para upload."""
        pass

    @abstractmethod
    def get_upload_progress(
            self, upload_id: str) -> ProcessingResult[Dict[str, Any]]:
        """Retorna progresso do upload."""
        pass


class IFeatureExtractionService(ABC):
    """Interface para serviço de extração de características."""

    @abstractmethod
    def extract_single(self, audio_data: AudioData,
                       feature_types: List[str]) -> ProcessingResult[Dict[str, AudioFeatures]]:
        """Extrai características de um áudio."""
        pass

    @abstractmethod
    def extract_batch(self, audio_list: List[AudioData], feature_types: List[str]
                      ) -> ProcessingResult[List[Dict[str, AudioFeatures]]]:
        """Extrai características em lote."""
        pass

    @abstractmethod
    def extract_from_dataset(self, dataset_name: str,
                             feature_types: List[str]) -> ProcessingResult[str]:
        """Extrai características de dataset completo."""
        pass

    @abstractmethod
    def get_available_extractors(self) -> List[str]:
        """Retorna extratores disponíveis."""
        pass


class ITrainingService(ABC):
    """Interface para serviço de treinamento."""

    @abstractmethod
    def train_model(self, architecture: str, dataset_name: str,
                    config: Dict[str, Any]) -> ProcessingResult[ModelMetadata]:
        """Treina novo modelo."""
        pass

    @abstractmethod
    def evaluate_model(self, model_name: str,
                       test_dataset: str) -> ProcessingResult[Dict[str, float]]:
        """Avalia modelo existente."""
        pass

    @abstractmethod
    def fine_tune_model(self, base_model: str, dataset_name: str,
                        config: Dict[str, Any]) -> ProcessingResult[ModelMetadata]:
        """Faz fine-tuning de modelo."""
        pass

    @abstractmethod
    def get_training_progress(
            self, training_id: str) -> ProcessingResult[Dict[str, Any]]:
        """Retorna progresso do treinamento."""
        pass


class IDetectionService(ABC):
    """Interface para serviço de detecção."""

    @abstractmethod
    def detect_single(self, audio_data: AudioData,
                      model_name: str = None) -> ProcessingResult[DeepfakeDetectionResult]:
        """Detecta deepfake em áudio único."""
        pass

    @abstractmethod
    def detect_batch(self, audio_list: List[AudioData],
                     model_name: str = None) -> ProcessingResult[List[DeepfakeDetectionResult]]:
        """Detecta deepfake em lote."""
        pass

    @abstractmethod
    def detect_from_file(
            self, file_path: Union[str, Path], model_name: str = None) -> ProcessingResult[DeepfakeDetectionResult]:
        """Detecta deepfake de arquivo."""
        pass

    @abstractmethod
    def get_available_models(self) -> List[str]:
        """Retorna modelos disponíveis para detecção."""
        pass


class INotificationService(ABC):
    """Interface para serviço de notificações."""

    @abstractmethod
    def send_notification(self, message: str, recipient: str,
                          notification_type: str = "info") -> ProcessingResult[bool]:
        """Envia notificação."""
        pass

    @abstractmethod
    def send_batch_notification(
            self, messages: List[Dict[str, str]]) -> ProcessingResult[List[bool]]:
        """Envia notificações em lote."""
        pass


class IMonitoringService(ABC):
    """Interface para serviço de monitoramento."""

    @abstractmethod
    def log_event(self, event_type: str,
                  data: Dict[str, Any]) -> ProcessingResult[bool]:
        """Registra evento."""
        pass

    @abstractmethod
    def get_system_metrics(self) -> ProcessingResult[Dict[str, Any]]:
        """Retorna métricas do sistema."""
        pass

    @abstractmethod
    def get_performance_metrics(
            self, component: str, time_range: str = "1h") -> ProcessingResult[Dict[str, Any]]:
        """Retorna métricas de performance."""
        pass
