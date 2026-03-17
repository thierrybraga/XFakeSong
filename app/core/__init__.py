"""Módulo Core - Interfaces e utilitários centrais do sistema"""

# Importar interfaces principais
# Importar configurações
from .config.settings import (
    APIConfig,
    AudioConfig,
    DatabaseConfig,
    FeatureConfig,
    LoggingConfig,
    MonitoringConfig,
    PathConfig,
    SecurityConfig,
    SystemConfig,
    TrainingConfig,
)

# Importar exceções
from .exceptions import (
    AppError,
    AudioProcessingError,
    ConflictError,
    DatasetNotFoundError,
    FileTooLargeError,
    ModelNotFoundError,
    NotFoundError,
    ProfileNotFoundError,
    ServiceUnavailableError,
    TrainingError,
    UnsupportedFormatError,
    ValidationError,
)
from .interfaces.audio import (
    AudioData,
    AudioFeatures,
    AudioFormat,
    DeepfakeDetectionResult,
    FeatureType,
    IAudioLoader,
    IDeepfakeDetector,
    IFeatureExtractor,
    IModelArchitecture,
)
from .interfaces.base import (
    IConfigurable,
    IExtractor,
    ILoggable,
    IMonitorable,
    IProcessor,
    IRepository,
    IValidator,
    ProcessingResult,
    ProcessingStatus,
)
from .interfaces.services import (
    DatasetMetadata,
    DatasetType,
    IAudioRepository,
    IDatasetRepository,
    IDetectionService,
    IFeatureExtractionService,
    IFeatureRepository,
    IModelRepository,
    IMonitoringService,
    INotificationService,
    ITrainingService,
    IUploadService,
    ModelMetadata,
    StorageType,
)

# Importar utilitários
from .utils.helpers import (
    ensure_directory,
    format_duration,
    format_file_size,
    get_file_hash,
    load_json,
    retry_decorator,
    safe_filename,
    save_json,
    timing_decorator,
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
    "AppError",
    "NotFoundError",
    "ValidationError",
    "ConflictError",
    "ServiceUnavailableError",
    "AudioProcessingError",
    "ModelNotFoundError",
    "DatasetNotFoundError",
    "ProfileNotFoundError",
    "TrainingError",
    "FileTooLargeError",
    "UnsupportedFormatError",

    # Utilitários
    "ensure_directory", "safe_filename", "get_file_hash",
    "format_file_size", "format_duration", "timing_decorator",
    "retry_decorator", "load_json", "save_json"
]
