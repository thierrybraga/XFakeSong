"""Módulo Core - Interfaces e utilitários centrais do sistema"""

# Importar interfaces principais
from .interfaces.base import (
    ProcessingStatus, ProcessingResult, IProcessor, IValidator,
    IExtractor, IRepository, IConfigurable, ILoggable, IMonitorable
)
from .interfaces.audio import (
    AudioFormat, FeatureType, AudioData, AudioFeatures,
    DeepfakeDetectionResult, IAudioLoader, IFeatureExtractor,
    IModelArchitecture, IDeepfakeDetector
)
from .interfaces.services import (
    DatasetType, StorageType, DatasetMetadata, ModelMetadata,
    IAudioRepository, IFeatureRepository, IModelRepository,
    IDatasetRepository, IUploadService, IFeatureExtractionService,
    ITrainingService, IDetectionService, INotificationService,
    IMonitoringService
)

# Importar configurações
from .config.settings import (
    PathConfig, AudioConfig, FeatureConfig,
    TrainingConfig, APIConfig, DatabaseConfig, LoggingConfig,
    MonitoringConfig, SecurityConfig, SystemConfig
)

# Importar exceções
from .exceptions.base import (
    DeepFakeSystemError, ValidationError, ConfigurationError,
    FileError, AudioError, FeatureExtractionError,
    ModelError, DatasetError, APIError, PipelineError
)

# Importar utilitários
from .utils.helpers import (
    ensure_directory, safe_filename, get_file_hash,
    format_file_size, format_duration, timing_decorator,
    retry_decorator, load_json, save_json
)

__version__ = "1.0.0"

__all__ = [
    # Status e Results
    "ProcessingStatus", "ProcessingResult",

    # Interfaces Base
    "IProcessor", "IValidator", "IExtractor", "IRepository",
    "IConfigurable", "ILoggable", "IMonitorable",

    # Interfaces Audio
    "AudioFormat", "FeatureType", "AudioData", "AudioFeatures",
    "DeepfakeDetectionResult", "IAudioLoader", "IFeatureExtractor",
    "IModelArchitecture", "IDeepfakeDetector",

    # Interfaces Services
    "DatasetType", "StorageType", "DatasetMetadata", "ModelMetadata",
    "IAudioRepository", "IFeatureRepository", "IModelRepository",
    "IDatasetRepository", "IUploadService", "IFeatureExtractionService",
    "ITrainingService", "IDetectionService", "INotificationService",
    "IMonitoringService",

    # Configurações
    "PathConfig", "AudioConfig", "FeatureConfig",
    "TrainingConfig", "APIConfig", "DatabaseConfig", "LoggingConfig",
    "MonitoringConfig", "SecurityConfig", "SystemConfig",

    # Exceções
    "DeepFakeSystemError", "ValidationError", "ConfigurationError",
    "FileError", "AudioError", "FeatureExtractionError",
    "ModelError", "DatasetError", "APIError", "PipelineError",

    # Utilitários
    "ensure_directory", "safe_filename", "get_file_hash",
    "format_file_size", "format_duration", "timing_decorator",
    "retry_decorator", "load_json", "save_json"
]
