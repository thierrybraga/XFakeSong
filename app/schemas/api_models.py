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


class FeatureExtractionRequest(BaseModel):
    feature_types: List[str]
    normalize: bool = True
    n_mfcc: int = 13
    n_fft: int = 2048
    hop_length: int = 512


class FeatureExtractionResult(BaseModel):
    features: Dict[str, List[List[float]]]  # Simplificado para JSON serializable
    metadata: Dict[str, Any]


class TrainingRequest(BaseModel):
    architecture: str
    dataset_path: str
    model_name: str
    parameters: Dict[str, Any] = {}
    epochs: int = 10
    batch_size: int = 32


class TrainingResponse(BaseModel):
    job_id: str
    status: str
    message: str


class HistoryItem(BaseModel):
    id: int
    filename: str
    is_fake: bool
    confidence: float
    model_name: str
    created_at: str
