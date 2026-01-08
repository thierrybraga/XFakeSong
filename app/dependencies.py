from functools import lru_cache
from app.domain.services.detection_service import DetectionService
import logging

logger = logging.getLogger(__name__)


@lru_cache()
def get_detection_service() -> DetectionService:
    logger.info("Inicializando DetectionService singleton...")
    return DetectionService(models_dir="app/models")
