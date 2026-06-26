import logging
import os
from functools import lru_cache

from app.domain.services.detection_service import DetectionService
from app.domain.services.feature_extraction_service import AudioFeatureExtractionService
from app.domain.services.training_service import TrainingService
from app.domain.services.upload_service import AudioUploadService

logger = logging.getLogger(__name__)


def _env_path(*names: str, default: str, storage_subdir: str | None = None) -> str:
    for name in names:
        value = os.getenv(name)
        if value and value.strip():
            return value.strip()
    if storage_subdir:
        storage = os.getenv("XFAKE_STORAGE_DIR") or os.getenv("DEEPFAKE_STORAGE_DIR")
        if storage and storage.strip():
            return os.path.join(storage.strip(), storage_subdir)
    return default


def _env_flag(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


@lru_cache()
def get_detection_service() -> DetectionService:
    logger.info("Inicializando DetectionService singleton...")
    create_defaults = _env_flag("XFAKE_CREATE_DEFAULT_MODELS", True)
    models_dir = _env_path(
        "MODELS_DIR",
        "DEEPFAKE_MODELS_DIR",
        "XFAKE_MODELS_DIR",
        default="app/models",
        storage_subdir="models",
    )
    return DetectionService(
        models_dir=models_dir,
        create_default_models=create_defaults,
    )


@lru_cache()
def get_upload_service() -> AudioUploadService:
    logger.info("Inicializando AudioUploadService singleton...")
    upload_dir = _env_path(
        "UPLOAD_DIR",
        "DEEPFAKE_UPLOADS_DIR",
        "XFAKE_UPLOADS_DIR",
        default="data/uploads",
        storage_subdir="uploads",
    )
    return AudioUploadService(upload_directory=upload_dir)


@lru_cache()
def get_training_service() -> TrainingService:
    logger.info("Inicializando TrainingService singleton...")
    models_dir = _env_path(
        "MODELS_DIR",
        "DEEPFAKE_MODELS_DIR",
        "XFAKE_MODELS_DIR",
        default="app/models",
        storage_subdir="models",
    )
    return TrainingService(models_dir=models_dir)


@lru_cache()
def get_feature_extraction_service() -> AudioFeatureExtractionService:
    logger.info("Inicializando AudioFeatureExtractionService singleton...")
    return AudioFeatureExtractionService()
