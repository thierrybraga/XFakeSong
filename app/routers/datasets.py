"""Endpoints de gerenciamento de datasets.

Melhorias: sanitização de filenames, rate limiting, validação de tipo,
não expor paths internos.
"""

import shutil
from typing import List, Optional

from fastapi import (
    APIRouter, Depends, File, Form, Query, Request, UploadFile,
)

from app.core.auth.auth_handler import get_api_key
from app.core.exceptions import DatasetNotFoundError, ValidationError
from app.core.interfaces.base import DatasetType, ProcessingStatus
from app.core.security import limiter, sanitize_filename
from app.dependencies import get_upload_service
from app.domain.services.upload_service import AudioUploadService
from app.schemas.api_models import DatasetMetadata

router = APIRouter(prefix="/api/v1/datasets", tags=["Datasets"])

VALID_DATASET_TYPES = {t.value for t in DatasetType}


def _validate_dataset_type(type_str: str) -> DatasetType:
    """Valida e converte string para DatasetType."""
    try:
        return DatasetType(type_str)
    except ValueError:
        raise ValidationError(
            f"Tipo de dataset inválido: '{type_str}'. "
            f"Válidos: {', '.join(sorted(VALID_DATASET_TYPES))}",
            field="type",
        )


@router.get(
    "/",
    response_model=List[DatasetMetadata],
    summary="Lista datasets disponíveis",
)
@limiter.limit("20/minute")
async def list_datasets(
    request: Request,
    type: Optional[str] = Query(
        None, description="Filtrar por tipo (training, validation, test)"
    ),
    service: AudioUploadService = Depends(get_upload_service),
):
    datasets = []
    base_dir = service.upload_directory

    if not base_dir.exists():
        return []

    # Validar tipo se fornecido
    if type and type not in VALID_DATASET_TYPES:
        raise ValidationError(
            f"Tipo '{type}' inválido. Válidos: {', '.join(sorted(VALID_DATASET_TYPES))}",
            field="type",
        )

    target_dirs = [type] if type else [t.value for t in DatasetType]

    for type_name in target_dirs:
        type_path = base_dir / type_name
        if type_path.exists():
            for item in type_path.iterdir():
                if item.is_dir():
                    file_count = sum(1 for f in item.glob("**/*") if f.is_file())
                    total_size = sum(
                        f.stat().st_size
                        for f in item.glob("**/*")
                        if f.is_file()
                    )

                    datasets.append(DatasetMetadata(
                        name=item.name,
                        dataset_type=type_name,
                        description=f"Dataset em {type_name}",
                        file_count=file_count,
                        total_size=total_size,
                        total_duration=0.0,
                        created_at=None,
                        file_paths=[],  # Não expor paths internos
                    ))

    return datasets


@router.post(
    "/",
    response_model=DatasetMetadata,
    dependencies=[Depends(get_api_key)],
    summary="Cria um novo dataset vazio",
)
@limiter.limit("10/minute")
async def create_dataset(
    request: Request,
    name: str = Form(...),
    type: str = Form("training"),
    description: Optional[str] = Form(None),
    service: AudioUploadService = Depends(get_upload_service),
):
    dataset_type = _validate_dataset_type(type)
    metadata = service.create_dataset(name, dataset_type, description)
    return metadata


@router.post(
    "/{name}/upload",
    dependencies=[Depends(get_api_key)],
    summary="Upload de arquivo para um dataset existente",
)
@limiter.limit("10/minute")
async def upload_to_dataset(
    request: Request,
    name: str,
    file: UploadFile = File(...),
    type: str = Form("training"),
    service: AudioUploadService = Depends(get_upload_service),
):
    _validate_dataset_type(type)

    dataset_dir = service.upload_directory / type / name
    if not dataset_dir.exists():
        raise DatasetNotFoundError(name)

    # Sanitizar filename antes de salvar
    safe_name = sanitize_filename(file.filename or "upload.wav")
    dest_path = dataset_dir / safe_name

    with open(dest_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    return {
        "status": "success",
        "filename": safe_name,
        "size_bytes": dest_path.stat().st_size,
    }


@router.delete(
    "/{name}",
    dependencies=[Depends(get_api_key)],
    summary="Exclui um dataset existente",
)
@limiter.limit("5/minute")
async def delete_dataset(
    request: Request,
    name: str,
    type: str = Query(
        "training",
        description="Tipo de dataset (training, validation, test)",
    ),
    service: AudioUploadService = Depends(get_upload_service),
):
    dataset_type = _validate_dataset_type(type)
    result = service.delete_dataset(name, dataset_type)

    if result.status == ProcessingStatus.ERROR:
        raise DatasetNotFoundError(name)

    return result.metadata
