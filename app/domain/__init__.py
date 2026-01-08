"""Camada de Domínio - Lógica de negócio e entidades"""

# Importar serviços de domínio
from .services.upload_service import AudioUploadService, UploadResult
from .services.feature_extraction_service import (
    AudioFeatureExtractionService, ExtractionConfig, ExtractionResult
)

__version__ = "1.0.0"

__all__ = [
    # Upload Services
    "AudioUploadService", "UploadResult",

    # Feature Extraction Services
    "AudioFeatureExtractionService", "ExtractionConfig", "ExtractionResult"
]
