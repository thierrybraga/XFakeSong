"""Interfaces do sistema"""

from .base import (
    ProcessingStatus, ProcessingResult, IProcessor, IValidator,
    IExtractor, IRepository, IConfigurable, ILoggable, IMonitorable
)
from .audio import (
    AudioFormat, FeatureType, AudioData, AudioFeatures,
    DeepfakeDetectionResult, IAudioLoader, IFeatureExtractor,
    IModelArchitecture, IDeepfakeDetector
)
from .services import (
    DatasetType, StorageType, DatasetMetadata, ModelMetadata,
    IAudioRepository, IFeatureRepository, IModelRepository,
    IDatasetRepository, IUploadService, IFeatureExtractionService,
    ITrainingService, IDetectionService, INotificationService,
    IMonitoringService
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
