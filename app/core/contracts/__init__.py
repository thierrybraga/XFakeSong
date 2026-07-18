"""Interfaces do sistema"""

from .audio import (
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
from .base import (
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
from .services import (
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

__all__ = [
    # Base
    "ProcessingStatus", "ProcessingResult", "IProcessor", "IValidator",
    "IExtractor", "IRepository", "IConfigurable", "ILoggable", "IMonitorable",

    # Audio
    "AudioFormat", "FeatureType", "AudioData", "AudioFeatures",
    "DeepfakeDetectionResult", "IAudioLoader", "IFeatureExtractor",
    "IModelArchitecture", "IDeepfakeDetector",

    # Services
    "DatasetType", "StorageType", "DatasetMetadata", "ModelMetadata",
    "IAudioRepository", "IFeatureRepository", "IModelRepository",
    "IDatasetRepository", "IUploadService", "IFeatureExtractionService",
    "ITrainingService", "IDetectionService", "INotificationService",
    "IMonitoringService"
]
