"""Endpoints de detecção de deepfake.

Melhorias: exceções de domínio, validação de formato, sanitização.
"""

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
)
from app.core.interfaces.base import ProcessingStatus
from app.core.security import limiter, sanitize_filename
from app.dependencies import get_detection_service
from app.domain.services.detection_service import DetectionService
from app.domain.services.upload_service import AudioUploadService
from app.schemas.api_models import PredictionResult

router = APIRouter(prefix="/api/v1/detection", tags=["Detection"])

SUPPORTED_FORMATS = list(AudioUploadService.SUPPORTED_FORMATS)


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
            return PredictionResult(
                is_fake=result.data.is_fake,
                confidence=result.data.confidence,
                probabilities=result.data.probabilities,
                model_name=result.data.model_name,
                features_used=result.data.features_used,
                metadata=result.data.metadata,
            )
        else:
            detail = (
                result.errors[0] if result.errors
                else "Erro desconhecido na detecção"
            )
            raise AudioProcessingError(detail)

    finally:
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)


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
