from functools import lru_cache
from app.domain.services.detection_service import DetectionService
from app.domain.services.upload_service import AudioUploadService
from app.domain.services.training_service import TrainingService
import logging
import os

logger = logging.getLogger(__name__)


@lru_cache()
def get_detection_service() -> DetectionService:
    logger.info("Inicializando DetectionService singleton...")
    return DetectionService(models_dir="app/models")

@lru_cache()
def get_upload_service() -> AudioUploadService:
    logger.info("Inicializando AudioUploadService singleton...")
    # Usar diretório padrão 'uploads' na raiz ou configurável
    upload_dir = os.getenv("UPLOAD_DIR", "uploads")
    return AudioUploadService(upload_directory=upload_dir)

@lru_cache()
def get_training_service() -> TrainingService:
    logger.info("Inicializando TrainingService singleton...")
    models_dir = os.getenv("MODELS_DIR", "app/models")
    return TrainingService(models_dir=models_dir)
