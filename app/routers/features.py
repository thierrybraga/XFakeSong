"""Endpoints de extração de features de áudio.

Melhorias: validação de JSON, exceções de domínio, rate limiting.
"""

import json
import os
import shutil
import tempfile
import uuid
from pathlib import Path

from fastapi import APIRouter, Depends, File, Form, Request, UploadFile

from app.core.exceptions import (
    AudioProcessingError,
    UnsupportedFormatError,
    ValidationError,
)
from app.core.interfaces.audio import FeatureType
from app.core.interfaces.base import ProcessingStatus
from app.core.security import limiter, sanitize_filename
from app.dependencies import get_feature_extraction_service
from app.domain.services.feature_extraction_service import AudioFeatureExtractionService
from app.domain.services.upload_service import AudioUploadService
from app.schemas.api_models import FeatureExtractionResult

router = APIRouter(prefix="/api/v1/features", tags=["Features"])

SUPPORTED_FORMATS = list(AudioUploadService.SUPPORTED_FORMATS)
VALID_FEATURE_TYPES = {ft.value for ft in FeatureType}


@router.post(
    "/extract",
    response_model=FeatureExtractionResult,
    summary="Extrai features de um arquivo de áudio",
)
@limiter.limit("10/minute")
async def extract_features(
    request: Request,
    file: UploadFile = File(...),
    feature_types: str = Form('["spectral"]'),
    normalize: bool = Form(True),
    service: AudioFeatureExtractionService = Depends(get_feature_extraction_service),
):
    # Validar JSON de feature_types
    try:
        types_list = json.loads(feature_types)
        if not isinstance(types_list, list) or not types_list:
            raise ValueError
    except (json.JSONDecodeError, ValueError):
        raise ValidationError(
            f"feature_types deve ser uma lista JSON válida. "
            f"Ex: {list(VALID_FEATURE_TYPES)[:3]}",
            field="feature_types",
        )

    # Rejeitar tipos completamente desconhecidos antes de chamar o serviço
    unknown = [t for t in types_list if t not in VALID_FEATURE_TYPES]
    if unknown and len(unknown) == len(types_list):
        raise ValidationError(
            f"Tipos de feature não reconhecidos: {unknown}. "
            f"Válidos: {sorted(VALID_FEATURE_TYPES)}",
            field="feature_types",
        )

    # Validar extensão do arquivo
    if not file.filename or not file.filename.lower().endswith(
        tuple(SUPPORTED_FORMATS)
    ):
        raise UnsupportedFormatError(
            Path(file.filename or "").suffix,
            SUPPORTED_FORMATS,
        )

    temp_dir = tempfile.mkdtemp()
    safe_name = f"{uuid.uuid4()}_{sanitize_filename(file.filename)}"
    temp_path = Path(temp_dir) / safe_name

    try:
        with open(temp_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # Carregar áudio
        from app.domain.services.audio_loading_service import AudioLoadingService
        loader = AudioLoadingService()
        load_result = loader.load_audio(temp_path)

        if load_result.status != ProcessingStatus.SUCCESS:
            raise AudioProcessingError(
                load_result.errors[0] if load_result.errors
                else "Erro ao carregar áudio"
            )

        audio_data = load_result.data

        # Extrair features
        result = service.extract_single(audio_data, types_list)

        if result.status == ProcessingStatus.SUCCESS:
            raw_features = (result.data or {}).get("features") or {}
            serializable_features: dict = {}
            for k, v in raw_features.items():
                if hasattr(v, "tolist"):
                    serializable_features[k] = v.tolist()
                elif isinstance(v, (list, dict, str, int, float, bool, type(None))):
                    serializable_features[k] = v
                else:
                    serializable_features[k] = str(v)

            meta: dict = {
                "duration": audio_data.duration,
                "sample_rate": audio_data.sample_rate,
            }
            if result.warnings:
                meta["warnings"] = result.warnings

            return FeatureExtractionResult(
                features=serializable_features,
                metadata=meta,
            )
        else:
            raise AudioProcessingError(
                result.errors[0] if result.errors
                else "Erro na extração de features"
            )

    finally:
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)


@router.get("/types", summary="Lista tipos de features disponíveis")
@limiter.limit("30/minute")
async def list_feature_types(request: Request):
    return {
        "available_types": sorted(VALID_FEATURE_TYPES),
        "active_extractors": ["spectral", "cepstral", "prosodic"],
    }
