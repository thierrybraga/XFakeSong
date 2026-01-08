from fastapi import APIRouter, UploadFile, File, Form, Depends, HTTPException
from typing import Optional, List
import shutil
import os
import tempfile
from pathlib import Path

from app.schemas.api_models import PredictionResult, ErrorResponse
from app.domain.services.detection_service import DetectionService
from app.dependencies import get_detection_service
from app.core.interfaces.base import ProcessingStatus

router = APIRouter(prefix="/api/v1/detection", tags=["Detection"])


@router.post("/analyze", response_model=PredictionResult)
async def analyze_audio(
    file: UploadFile = File(...),
    model_name: Optional[str] = Form(None),
    architecture: Optional[str] = Form(None),
    variant: Optional[str] = Form(None),
    normalize: bool = Form(True),
    segmented: bool = Form(False),
    service: DetectionService = Depends(get_detection_service)
):
    # Lógica de Seleção de Modelo
    selected_model_name = model_name

    if not selected_model_name and architecture:
        found_model = service.find_model(architecture, variant)
        if not found_model:
            variant_msg = f" e variante '{variant}'" if variant else ""
            raise HTTPException(
                status_code=404,
                detail=f"O modelo nao existe treinado para a arquitetura '{architecture}'{variant_msg}."
            )
        selected_model_name = found_model

    # Validar se o modelo selecionado (ou informado diretamente) existe
    if selected_model_name and selected_model_name not in service.get_available_models():
        raise HTTPException(
            status_code=404,
            detail=f"Modelo '{selected_model_name}' não encontrado.")

    # Salvar arquivo temporário
    temp_dir = tempfile.mkdtemp()
    temp_path = Path(temp_dir) / file.filename

    try:
        with open(temp_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # Executar detecção
        result = service.detect_from_file(
            file_path=temp_path,
            model_name=selected_model_name,
            normalize=normalize,
            segmented=segmented
        )

        if result.status == ProcessingStatus.SUCCESS:
            return PredictionResult(
                is_fake=result.data.is_fake,
                confidence=result.data.confidence,
                probabilities=result.data.probabilities,
                model_name=result.data.model_name,
                features_used=result.data.features_used,
                metadata=result.data.metadata
            )
        else:
            raise HTTPException(
                status_code=400,
                detail=result.errors[0] if result.errors else "Erro desconhecido na detecção")

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        # Limpar arquivo temporário
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)


@router.get("/models")
async def list_models(
        service: DetectionService = Depends(get_detection_service)):
    return {
        "available_models": service.get_available_models(),
        "default_model": service.default_model,
        "loaded_models": list(service.loaded_models.keys())
    }


@router.get("/architectures")
async def list_architectures(
        service: DetectionService = Depends(get_detection_service)):
    """Lista todas as arquiteturas suportadas para treinamento/inferência."""
    return {
        "architectures": service.get_available_architectures()
    }
