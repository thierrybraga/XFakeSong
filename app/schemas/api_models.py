"""Schemas Pydantic para validação de requests e responses da API.

Todas as validações de bounds, formatos e constraints ficam aqui.
"""

import re
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator


# ── System ─────────────────────────────────────────────────────────────

class SystemStatus(BaseModel):
    status: str
    message: str
    version: str = "1.0.0"
    active_services: List[str]


class HealthCheckResponse(BaseModel):
    """Resposta detalhada do health check."""
    status: str
    database: str = "unknown"
    models_loaded: int = 0
    storage_available: bool = True
    uptime_seconds: float = 0.0


# ── Detection ──────────────────────────────────────────────────────────

class DetectionRequest(BaseModel):
    model_name: Optional[str] = None
    feature_types: Optional[List[str]] = None
    normalize: bool = True
    segmented: bool = False


class PredictionResult(BaseModel):
    is_fake: bool
    confidence: float = Field(..., ge=0.0, le=1.0)
    probabilities: Dict[str, float]
    model_name: str
    features_used: List[str]
    metadata: Dict[str, Any]


# ── Features ───────────────────────────────────────────────────────────

class FeatureExtractionRequest(BaseModel):
    feature_types: List[str] = Field(..., min_length=1, max_length=20)
    normalize: bool = True
    n_mfcc: int = Field(13, ge=1, le=40)
    n_fft: int = Field(2048, ge=256, le=8192)
    hop_length: int = Field(512, ge=64, le=4096)


class FeatureExtractionResult(BaseModel):
    features: Dict[str, List[List[float]]]
    metadata: Dict[str, Any]


# ── Training ───────────────────────────────────────────────────────────

VALID_ARCHITECTURES = {
    "aasist", "conformer", "efficientnet_lstm", "ensemble",
    "hubert", "multiscale_cnn", "rawgat_st", "rawnet2",
    "sonic_sleuth", "spectrogram_transformer", "svm",
    "random_forest", "wavlm", "hybrid_cnn_transformer",
}


class TrainingRequest(BaseModel):
    architecture: str
    dataset_path: str = Field(..., min_length=1, max_length=500)
    model_name: str = Field(
        ..., min_length=1, max_length=200, pattern=r'^[\w\-. ]+$'
    )
    parameters: Dict[str, Any] = {}
    epochs: int = Field(10, ge=1, le=500)
    batch_size: int = Field(32, ge=1, le=512)
    learning_rate: float = Field(0.001, gt=0, le=1.0)

    @field_validator("architecture")
    @classmethod
    def validate_architecture(cls, v):
        v_lower = v.lower().strip()
        if v_lower not in VALID_ARCHITECTURES:
            raise ValueError(
                f"Arquitetura '{v}' inválida. "
                f"Válidas: {', '.join(sorted(VALID_ARCHITECTURES))}"
            )
        return v_lower


class TrainingResponse(BaseModel):
    job_id: str
    status: str
    message: str
    progress: int = Field(0, ge=0, le=100)
    metrics: Optional[Dict[str, Any]] = None


# ── History ────────────────────────────────────────────────────────────

class HistoryItem(BaseModel):
    id: int
    filename: str
    is_fake: bool
    confidence: float
    model_name: str
    created_at: str


class HistoryListResponse(BaseModel):
    """Resposta paginada do histórico."""
    items: List[HistoryItem]
    total: int
    limit: int
    offset: int


# ── Datasets ───────────────────────────────────────────────────────────

VALID_DATASET_TYPES = {"training", "validation", "test"}


class DatasetMetadata(BaseModel):
    name: str
    dataset_type: str
    description: str
    file_count: int = 0
    total_size: int = 0
    total_duration: float = 0.0
    created_at: Optional[str] = None
    file_paths: List[str] = []

    model_config = ConfigDict(from_attributes=True)


# ── Voice Profiles ─────────────────────────────────────────────────────

class ProfileCreate(BaseModel):
    name: str = Field(..., min_length=1, max_length=100)
    telegram_id: Optional[str] = Field(None, max_length=100)
    phone: Optional[str] = Field(None, max_length=20)
    email: Optional[str] = Field(None, max_length=120)
    description: Optional[str] = Field(None, max_length=500)
    architecture: str = Field("sonic_sleuth", max_length=100)

    @field_validator("email")
    @classmethod
    def validate_email(cls, v):
        if v is not None and v.strip():
            pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
            if not re.match(pattern, v.strip()):
                raise ValueError("Formato de email inválido")
        return v

    @field_validator("phone")
    @classmethod
    def validate_phone(cls, v):
        if v is not None and v.strip():
            cleaned = re.sub(r'[\s\-\(\)]', '', v)
            if not re.match(r'^\+?\d{8,15}$', cleaned):
                raise ValueError(
                    "Formato de telefone inválido. Use: +55 11 99999-9999"
                )
        return v


class ProfileUpdate(BaseModel):
    name: Optional[str] = Field(None, min_length=1, max_length=100)
    telegram_id: Optional[str] = Field(None, max_length=100)
    phone: Optional[str] = Field(None, max_length=20)
    email: Optional[str] = Field(None, max_length=120)
    description: Optional[str] = Field(None, max_length=500)
    architecture: Optional[str] = Field(None, max_length=100)

    @field_validator("email")
    @classmethod
    def validate_email(cls, v):
        if v is not None and v.strip():
            pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
            if not re.match(pattern, v.strip()):
                raise ValueError("Formato de email inválido")
        return v


class ProfileResponse(BaseModel):
    id: int
    name: str
    telegram_id: Optional[str] = None
    phone: Optional[str] = None
    email: Optional[str] = None
    status: str
    architecture: Optional[str] = None
    num_samples: int = 0
    total_duration_seconds: float = 0.0
    description: Optional[str] = None
    training_metrics: Optional[Dict[str, Any]] = None
    created_at: Optional[str] = None

    model_config = ConfigDict(from_attributes=True)


class ProfileTrainRequest(BaseModel):
    architecture: Optional[str] = None
    epochs: int = Field(30, ge=5, le=500)
    batch_size: int = Field(16, ge=4, le=128)
    learning_rate: float = Field(0.001, gt=1e-6, le=0.1)


class ProfileDetectionResult(BaseModel):
    success: bool
    is_authentic: Optional[bool] = None
    is_fake: Optional[bool] = None
    confidence: Optional[float] = Field(None, ge=0.0, le=100.0)
    raw_score: Optional[float] = None
    profile_name: Optional[str] = None
    profile_id: Optional[int] = None
    details: Optional[Dict[str, Any]] = None


# ── Error Response (RFC 7807) ──────────────────────────────────────────

class ErrorResponse(BaseModel):
    """RFC 7807 Problem Details."""
    type: str = "about:blank"
    title: str
    status: int
    detail: str
    error_code: Optional[str] = None
    request_id: Optional[str] = None
    errors: Optional[List[Dict[str, str]]] = None
