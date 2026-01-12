from fastapi import APIRouter, UploadFile, File, Form, Depends, HTTPException, Request
from typing import List, Optional
import shutil
import tempfile
import os
import json
import uuid
import numpy as np
from pathlib import Path

from app.schemas.api_models import FeatureExtractionResult
from app.domain.services.feature_extraction_service import AudioFeatureExtractionService
from app.core.interfaces.base import ProcessingStatus
from app.core.interfaces.audio import AudioData
from app.core.security import limiter, sanitize_filename
from app.domain.services.upload_service import AudioUploadService

router = APIRouter(prefix="/api/v1/features", tags=["Features"])


def get_feature_service():
    return AudioFeatureExtractionService()


@router.post("/extract", response_model=FeatureExtractionResult)
@limiter.limit("10/minute")
async def extract_features(
    request: Request,
    file: UploadFile = File(...),
    feature_types: str = Form("['mfcc']"),  # Recebe como string JSON
    normalize: bool = Form(True),
    service: AudioFeatureExtractionService = Depends(get_feature_service)
):
    try:
        types_list = json.loads(feature_types)
    except json.JSONDecodeError:
        types_list = ["mfcc"]
    
    # Validar Extensão
    if not file.filename.lower().endswith(tuple(AudioUploadService.SUPPORTED_FORMATS)):
         raise HTTPException(status_code=400, detail="Formato de arquivo não suportado.")

    temp_dir = tempfile.mkdtemp()
    safe_name = f"{uuid.uuid4()}_{sanitize_filename(file.filename)}"
    temp_path = Path(temp_dir) / safe_name

    try:
        with open(temp_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # 1. Carregar Áudio
        from app.domain.services.audio_loading_service import AudioLoadingService
        loader = AudioLoadingService()
        load_result = loader.load_audio(temp_path)
        
        if load_result.status != ProcessingStatus.SUCCESS:
             raise HTTPException(status_code=400, detail=load_result.errors[0])
        
        audio_data = load_result.data

        # 2. Extrair Features
        result = service.extract_single(audio_data, types_list)

        if result.status == ProcessingStatus.SUCCESS:
            # Converter numpy arrays para listas para serialização JSON
            serializable_features = {}
            for k, v in result.data.items():
                if hasattr(v.data, "tolist"):
                    serializable_features[k] = v.data.tolist()
                else:
                    serializable_features[k] = v.data # Fallback

            return FeatureExtractionResult(
                features=serializable_features,
                metadata={"duration": audio_data.duration, "sample_rate": audio_data.sample_rate}
            )
        else:
            raise HTTPException(status_code=500, detail=result.errors[0])

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)


@router.get("/types")
async def list_feature_types():
    return {
        "available_types": ["mfcc", "mel_spectrogram", "chroma", "spectral_contrast", "tonnetz", "raw"]
    }
