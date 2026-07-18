"""Serviços de Domínio

NÃO importe DetectionService aqui de forma eager: seu carregamento traz
model_loader.py, que importa `tensorflow` no topo do arquivo. Qualquer import
de app.domain.* (incl. features tabulares para SVM/RandomForest, sem uso de
TF) executaria este __init__.py e herdaria TensorFlow como dependência
obrigatória — quebra o ambiente Docker "classical-ml". Todo consumidor real
já importa direto de `app.domain.services.detection_service`.
"""

from .feature_extraction_service import (
    AudioFeatureExtractionService,
    ExtractionConfig,
)
from .upload_service import AudioUploadService, UploadResult

__all__ = [
    "AudioUploadService", "UploadResult",
    "AudioFeatureExtractionService", "ExtractionConfig",
]
