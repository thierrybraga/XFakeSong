from pydantic import BaseModel
from typing import List, Dict, Any, Optional


class SystemStatus(BaseModel):
    status: str
    message: str
    version: str = "1.0.0"
    active_services: List[str]


class DetectionRequest(BaseModel):
    # Para upload de arquivo, usaremos Form/UploadFile, mas para requisições
    # JSON:
    model_name: Optional[str] = None
    feature_types: Optional[List[str]] = None
    normalize: bool = True
    segmented: bool = False


class PredictionResult(BaseModel):
    is_fake: bool
    confidence: float
    probabilities: Dict[str, float]
    model_name: str
    features_used: List[str]
    metadata: Dict[str, Any]


class ErrorResponse(BaseModel):
    error: str
    details: Optional[str] = None
