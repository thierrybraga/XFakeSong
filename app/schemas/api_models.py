"""Schemas Pydantic para validação de requests e responses da API.

Todas as validações de bounds, formatos e constraints ficam aqui.
"""

import re
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

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
    """Resultado de detecção de deepfake.

    API.4: Inclui campos novos dos Sprints 1.4, 2.5, 4.5:
    - `temperature_applied`: T de calibração (Sprint 1.4)
    - `ood_score` / `is_ood` / `ood_threshold`: detecção OOD (Sprint 2.5)
    - `classification_threshold`: EER threshold ou 0.5 (Sprint 4.5)
    Todos opcionais para compatibilidade retroativa com clientes antigos.
    """
    is_fake: bool
    confidence: float = Field(..., ge=0.0, le=1.0)
    probabilities: Dict[str, float]
    model_name: str
    features_used: List[str]
    metadata: Dict[str, Any]
    # Sprint 1.4 — Temperature scaling
    temperature_applied: Optional[float] = Field(
        None, description="Temperatura T usada para calibração pós-treino"
    )
    # Sprint 2.5 — OOD detection
    ood_score: Optional[float] = Field(
        None, description="Score de in-distribution (maior = mais confiante)"
    )
    is_ood: Optional[bool] = Field(
        None, description="True se o áudio é fora-da-distribuição"
    )
    ood_threshold: Optional[float] = Field(
        None, description="Threshold OOD calibrado no val set"
    )
    # Sprint 4.5 — EER threshold adaptativo
    classification_threshold: Optional[float] = Field(
        None,
        description="Threshold usado para classificar is_fake (EER ou 0.5)",
    )


# ── Multi-Model Fusion (Sprint 4.4 — API.5) ───────────────────────────

class MultiModelDetectionRequest(BaseModel):
    """Configuração para inferência via fusão de múltiplos modelos."""
    model_names: List[str] = Field(
        ..., min_length=2, max_length=10,
        description="Nomes dos modelos a fundir (≥2)",
    )
    fusion: str = Field(
        "weighted_avg",
        description=(
            "Estratégia: 'weighted_avg' | 'soft_voting' | "
            "'majority_vote' | 'max_conf'"
        ),
    )
    weights: Optional[List[float]] = Field(
        None, description="Pesos por modelo (default uniforme)"
    )
    use_tta: bool = Field(False, description="Aplica TTA em cada modelo")

    @field_validator("fusion")
    @classmethod
    def validate_fusion(cls, v: str) -> str:
        valid = {"weighted_avg", "soft_voting", "majority_vote", "max_conf"}
        if v not in valid:
            raise ValueError(
                f"fusion deve ser um de {sorted(valid)}, got '{v}'"
            )
        return v


class MultiModelPredictionResult(BaseModel):
    """Resultado de fusão multi-model."""
    is_fake: bool
    confidence: float = Field(..., ge=0.0, le=1.0)
    probabilities: Dict[str, float]
    fusion: str
    n_models: int
    fake_votes: int
    model_agreement: float = Field(..., ge=0.0, le=1.0)
    per_model: List[Dict[str, Any]]
    metadata: Dict[str, Any] = Field(default={}, description="Metadata adicional")
    
    model_config = ConfigDict(json_schema_extra={"metadata": {"type": "object", "additionalProperties": True}})


# ── Uncertainty (Sprint 5.4 — API.6) ──────────────────────────────────

class UncertaintyRequest(BaseModel):
    """Parâmetros para predição com MC Dropout."""
    model_name: Optional[str] = None
    n_samples: int = Field(
        20, ge=5, le=200,
        description="Número de forward passes MC (típico: 10-50)",
    )
    normalize: bool = True


class UncertaintyResult(BaseModel):
    """Predição com quantificação de incerteza (MC Dropout)."""
    is_fake: bool
    confidence: float = Field(..., ge=0.0, le=1.0)
    epistemic_uncertainty: float = Field(
        ..., description="Variância da classe predita entre MC samples"
    )
    predictive_entropy: float = Field(
        ..., description="Entropia da distribuição média"
    )
    is_uncertain: bool = Field(
        ..., description="Flag para decisão 'abstenha-se'"
    )
    n_mc_samples: int
    temperature_applied: Optional[float] = None
    classification_threshold: Optional[float] = None
    mc_fallback: bool = Field(
        False, description="True se MC Dropout falhou e usou predict padrão"
    )


# ── Cross Validation (Sprint 4.1 — API.7) ──────────────────────────────

class CrossValidationRequest(BaseModel):
    """Configuração para K-fold cross validation."""
    architecture: str
    dataset_path: str = Field(..., min_length=1, max_length=500)
    config: Dict[str, Any] = Field(
        default_factory=dict,
        description="Config de treinamento (epochs, batch_size, lr, etc.)",
    )
    n_folds: int = Field(5, ge=2, le=20)
    save_fold_models: bool = False

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


class CrossValidationResult(BaseModel):
    """Resultado agregado de K-fold CV."""
    architecture: str
    n_folds: int
    n_successful: int
    best_fold: int
    aggregated: Dict[str, Dict[str, float]] = Field(
        ..., description="{metric: {mean, std, min, max}}"
    )
    per_fold: List[Dict[str, Any]]


# ── ONNX Export (Sprint 3.4 — API.7) ───────────────────────────────────

class OnnxExportResponse(BaseModel):
    """Resposta do export ONNX."""
    success: bool
    onnx_path: Optional[str] = None
    onnx_int8_path: Optional[str] = None
    size_mb: Optional[float] = None
    size_int8_mb: Optional[float] = None
    message: str


# ── System version (API.10) ────────────────────────────────────────────

class SystemVersionResponse(BaseModel):
    """Informações de versão / build."""
    version: str
    python_version: str
    tensorflow_version: Optional[str] = None
    keras_version: Optional[str] = None
    gradio_version: Optional[str] = None
    sklearn_version: Optional[str] = None
    git_sha: Optional[str] = None
    platform: str


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

def _get_valid_architectures() -> set:
    """Carrega arquiteturas válidas dinamicamente do registry + ML clássico."""
    try:
        from app.domain.models.architectures.registry import get_valid_snake_names
        return get_valid_snake_names() | {"svm", "random_forest"}
    except Exception:
        # Fallback estático caso o registry não esteja disponível
        return {
            "aasist", "conformer", "efficientnet_lstm", "ensemble",
            "hubert", "multiscale_cnn", "rawgat_st", "rawnet2",
            "sonic_sleuth", "spectrogram_transformer", "svm",
            "random_forest", "wavlm", "hybrid_cnn_transformer",
        }


VALID_ARCHITECTURES = _get_valid_architectures()


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

    @field_validator("dataset_type", mode="before")
    @classmethod
    def coerce_dataset_type(cls, v: Any) -> str:
        """Accept DatasetType enum or plain string."""
        if hasattr(v, "value"):
            return v.value
        return str(v)


# ── Voice Profiles ─────────────────────────────────────────────────────

class ProfileCreate(BaseModel):
    name: str = Field(..., min_length=1, max_length=100)
    telegram_id: Optional[str] = Field(None, max_length=100)
    phone: Optional[str] = Field(None, max_length=20)
    email: Optional[str] = Field(None, max_length=120)
    description: Optional[str] = Field(None, max_length=500)
    architecture: str = Field("sonic_sleuth", max_length=100)

    @field_validator("architecture")
    @classmethod
    def validate_architecture(cls, v):
        v_lower = v.lower().strip().replace("-", "_").replace(" ", "_")
        if v_lower not in VALID_ARCHITECTURES:
            raise ValueError(
                f"Arquitetura '{v}' inválida para perfil. "
                f"Válidas: {', '.join(sorted(VALID_ARCHITECTURES))}"
            )
        return v_lower

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

    @field_validator("architecture")
    @classmethod
    def validate_architecture(cls, v):
        if v is None:
            return v
        v_lower = v.lower().strip().replace("-", "_").replace(" ", "_")
        if v_lower not in VALID_ARCHITECTURES:
            raise ValueError(
                f"Arquitetura '{v}' inválida. "
                f"Válidas: {', '.join(sorted(VALID_ARCHITECTURES))}"
            )
        return v_lower

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

    @field_validator("architecture")
    @classmethod
    def validate_architecture(cls, v):
        if v is None:
            return v
        v_lower = v.lower().strip().replace("-", "_").replace(" ", "_")
        if v_lower not in VALID_ARCHITECTURES:
            raise ValueError(
                f"Arquitetura '{v}' inválida. "
                f"Válidas: {', '.join(sorted(VALID_ARCHITECTURES))}"
            )
        return v_lower


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
