"""Endpoints de detecção de deepfake.

Melhorias: exceções de domínio, validação de formato, sanitização,
exposição completa de Sprints 1.4/2.5/4.4/4.5/5.4.
"""

import logging
import os
import shutil
import tempfile
import uuid
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, Depends, File, Form, Request, UploadFile

from app.core.exceptions import (
    AudioProcessingError,
    ModelNotFoundError,
    UnsupportedFormatError,
    ValidationError,
)
from app.core.interfaces.audio import AudioData
from app.core.interfaces.base import ProcessingStatus
from app.core.security import limiter, sanitize_filename
from app.dependencies import get_detection_service
from app.domain.services.detection_service import DetectionService
from app.domain.services.upload_service import AudioUploadService
from app.schemas.api_models import (
    MultiModelDetectionRequest,
    MultiModelPredictionResult,
    PredictionResult,
    UncertaintyRequest,
    UncertaintyResult,
)

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/v1/detection", tags=["Detection"])

SUPPORTED_FORMATS = list(AudioUploadService.SUPPORTED_FORMATS)


def _build_prediction_result(data, model_name_fallback: str = "unknown") -> PredictionResult:
    """Mapeia DeepfakeDetectionResult → PredictionResult preenchendo campos novos.

    Os campos extras (temperature/ood/eer) vêm do metadata['per_window_metadata']
    OU do próprio Predictor result via campos top-level.
    """
    meta = getattr(data, "metadata", None) or {}
    # Campos podem estar tanto em metadata quanto top-level no resultado
    return PredictionResult(
        is_fake=data.is_fake,
        confidence=data.confidence,
        probabilities=data.probabilities,
        model_name=getattr(data, "model_name", model_name_fallback),
        features_used=getattr(data, "features_used", []) or [],
        metadata=meta,
        # Sprint 1.4 / 2.5 / 4.5 — opcionais
        temperature_applied=meta.get("temperature_applied"),
        ood_score=meta.get("ood_score"),
        is_ood=meta.get("is_ood"),
        ood_threshold=meta.get("ood_threshold"),
        classification_threshold=meta.get("classification_threshold"),
    )


@router.post(
    "/analyze",
    response_model=PredictionResult,
    summary="Analisa um áudio para detecção de deepfake",
)
@limiter.limit("10/minute")
async def analyze_audio(
    request: Request,
    file: UploadFile = File(...),
    model_name: Optional[str] = Form(None),
    architecture: Optional[str] = Form(None),
    variant: Optional[str] = Form(None),
    normalize: bool = Form(True),
    segmented: bool = Form(False),
    service: DetectionService = Depends(get_detection_service),
):
    # Validar extensão do arquivo
    if not file.filename or not file.filename.lower().endswith(
        tuple(SUPPORTED_FORMATS)
    ):
        raise UnsupportedFormatError(
            Path(file.filename or "").suffix,
            SUPPORTED_FORMATS,
        )

    # Seleção de modelo
    selected_model_name = model_name

    if not selected_model_name and architecture:
        found_model = service.find_model(architecture, variant)
        if not found_model:
            variant_msg = f" variante '{variant}'" if variant else ""
            raise ModelNotFoundError(f"{architecture}{variant_msg}")
        selected_model_name = found_model

    if (
        selected_model_name
        and selected_model_name not in service.get_available_models()
    ):
        raise ModelNotFoundError(selected_model_name)

    # Salvar arquivo temporário com nome seguro
    temp_dir = tempfile.mkdtemp()
    safe_name = f"{uuid.uuid4()}_{sanitize_filename(file.filename)}"
    temp_path = Path(temp_dir) / safe_name

    try:
        with open(temp_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        result = service.detect_from_file(
            file_path=temp_path,
            model_name=selected_model_name,
            normalize=normalize,
            segmented=segmented,
        )

        if result.status == ProcessingStatus.SUCCESS:
            # API.4: preserva campos novos (temperature, OOD, EER threshold)
            return _build_prediction_result(
                result.data, model_name_fallback=selected_model_name or ""
            )
        else:
            detail = (
                result.errors[0] if result.errors
                else "Erro desconhecido na detecção"
            )
            raise AudioProcessingError(detail)

    finally:
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir, ignore_errors=True)


# ── API.5: Multi-Model Fusion (Sprint 4.4) ──────────────────────────────

@router.post(
    "/multi-model",
    response_model=MultiModelPredictionResult,
    summary="Detecção via fusão de múltiplos modelos",
    description=(
        "Executa N modelos no mesmo áudio e combina via "
        "weighted_avg / soft_voting / majority_vote / max_conf."
    ),
)
@limiter.limit("5/minute")
async def detect_multi_model(
    request: Request,
    file: UploadFile = File(...),
    model_names: str = Form(...,
        description='JSON-encoded lista: \'["AASIST_v1","Conformer_v1"]\''),
    fusion: str = Form("weighted_avg"),
    weights: Optional[str] = Form(None,
        description='Lista JSON de pesos: \'[0.4, 0.6]\''),
    use_tta: bool = Form(False),
    service: DetectionService = Depends(get_detection_service),
):
    # Parse JSON inputs (FastAPI Form não suporta List[str] direto)
    import json as _json
    try:
        names = _json.loads(model_names)
        if not isinstance(names, list) or len(names) < 2:
            raise ValueError("model_names precisa ser lista com ≥2 elementos")
    except (ValueError, _json.JSONDecodeError) as e:
        raise ValidationError(f"model_names inválido: {e}", field="model_names")

    weights_list = None
    if weights:
        try:
            weights_list = _json.loads(weights)
            if not isinstance(weights_list, list):
                raise ValueError("weights deve ser lista de números")
        except (ValueError, _json.JSONDecodeError) as e:
            raise ValidationError(f"weights inválido: {e}", field="weights")

    # Validar via schema (reaproveita validações)
    req_schema = MultiModelDetectionRequest(
        model_names=names, fusion=fusion,
        weights=weights_list, use_tta=use_tta,
    )

    # Validar extensão
    if not file.filename or not file.filename.lower().endswith(
        tuple(SUPPORTED_FORMATS)
    ):
        raise UnsupportedFormatError(
            Path(file.filename or "").suffix, SUPPORTED_FORMATS
        )

    temp_dir = tempfile.mkdtemp()
    try:
        safe_name = f"{uuid.uuid4()}_{sanitize_filename(file.filename)}"
        temp_path = Path(temp_dir) / safe_name
        with open(temp_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        audio_data = AudioData.from_file(temp_path)

        result = service.detect_multi_model(
            audio_data,
            model_names=req_schema.model_names,
            fusion=req_schema.fusion,
            weights=req_schema.weights,
            use_tta=req_schema.use_tta,
        )

        if result.status != ProcessingStatus.SUCCESS:
            raise AudioProcessingError(
                result.errors[0] if result.errors
                else "Erro na fusão multi-model"
            )

        data = result.data
        meta = data.metadata or {}
        return MultiModelPredictionResult(
            is_fake=data.is_fake,
            confidence=data.confidence,
            probabilities=data.probabilities,
            fusion=meta.get("fusion", req_schema.fusion),
            n_models=meta.get("n_models", len(req_schema.model_names)),
            fake_votes=meta.get("fake_votes", 0),
            model_agreement=meta.get("model_agreement", 0.0),
            per_model=meta.get("per_model", []),
            metadata={
                k: v for k, v in meta.items()
                if k not in {"fusion", "n_models", "fake_votes",
                             "model_agreement", "per_model"}
            },
        )
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


# ── API.6: MC Dropout Uncertainty (Sprint 5.4) ──────────────────────────

@router.post(
    "/uncertainty",
    response_model=UncertaintyResult,
    summary="Predição com quantificação de incerteza (MC Dropout)",
    description=(
        "Executa N forward passes com dropout ativo. Retorna confidence média "
        "+ epistemic_uncertainty + predictive_entropy + flag is_uncertain "
        "para decisão 'abstenha-se'."
    ),
)
@limiter.limit("5/minute")
async def detect_with_uncertainty(
    request: Request,
    file: UploadFile = File(...),
    model_name: Optional[str] = Form(None),
    n_samples: int = Form(20, ge=5, le=200),
    service: DetectionService = Depends(get_detection_service),
):
    if not file.filename or not file.filename.lower().endswith(
        tuple(SUPPORTED_FORMATS)
    ):
        raise UnsupportedFormatError(
            Path(file.filename or "").suffix, SUPPORTED_FORMATS
        )

    # Resolve modelo
    selected_model = model_name or service.default_model
    if not selected_model or selected_model not in service.get_available_models():
        raise ModelNotFoundError(selected_model or "default")

    model_info = service.loaded_models.get(selected_model)
    if model_info is None:
        raise ModelNotFoundError(selected_model)

    temp_dir = tempfile.mkdtemp()
    try:
        safe_name = f"{uuid.uuid4()}_{sanitize_filename(file.filename)}"
        temp_path = Path(temp_dir) / safe_name
        with open(temp_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        audio_data = AudioData.from_file(temp_path)

        # Usa o FeaturePreparer para preparar input no formato esperado
        from app.domain.models.architectures.registry import get_architecture_info
        try:
            arch_info = get_architecture_info(model_info.architecture)
        except Exception:
            arch_info = None
        prepared = service.feature_preparer.prepare_input(
            audio_data, model_info, arch_info
        )
        if prepared.get("status") != "ok":
            raise AudioProcessingError(
                prepared.get("error", "Falha ao preparar input")
            )

        result = service.predictor.predict_with_uncertainty(
            model_info, prepared["features"], n_samples=int(n_samples)
        )

        if result.status != ProcessingStatus.SUCCESS:
            raise AudioProcessingError(
                result.errors[0] if result.errors
                else "MC Dropout falhou"
            )

        data = result.data
        return UncertaintyResult(
            is_fake=bool(data["is_deepfake"]),
            confidence=float(data["confidence"]),
            epistemic_uncertainty=float(data["epistemic_uncertainty"]),
            predictive_entropy=float(data["predictive_entropy"]),
            is_uncertain=bool(data["is_uncertain"]),
            n_mc_samples=int(data["n_mc_samples"]),
            temperature_applied=data.get("temperature_applied"),
            classification_threshold=data.get("classification_threshold"),
            mc_fallback=bool(data.get("mc_fallback", False)),
        )
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


@router.get("/models", summary="Lista modelos disponíveis")
@limiter.limit("30/minute")
async def list_models(
    request: Request,
    service: DetectionService = Depends(get_detection_service),
):
    return {
        "available_models": service.get_available_models(),
        "default_model": service.default_model,
        "loaded_models": list(service.loaded_models.keys()),
    }


@router.get("/architectures", summary="Lista arquiteturas suportadas")
@limiter.limit("30/minute")
async def list_architectures(
    request: Request,
    service: DetectionService = Depends(get_detection_service),
):
    return {"architectures": service.get_available_architectures()}
