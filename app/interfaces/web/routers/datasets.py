"""Endpoints de gerenciamento de datasets.

Melhorias: sanitização de filenames, rate limiting, validação de tipo,
não expor paths internos.
"""

from typing import List, Optional

from fastapi import (
    APIRouter,
    Depends,
    File,
    Form,
    Query,
    Request,
    UploadFile,
)

from app.core.auth.auth_handler import get_api_key
from app.core.contracts.base import DatasetType, ProcessingStatus
from app.core.exceptions import (
    AudioProcessingError,
    ConflictError,
    DatasetNotFoundError,
    FileTooLargeError,
    UnsupportedFormatError,
    ValidationError,
)
from app.core.security import limiter, sanitize_filename
from app.dependencies import get_upload_service
from app.domain.services.upload_service import AudioUploadService
from app.interfaces.web.schemas.api_models import DatasetMetadata
from app.utils.file_utils import resolve_within_directory, validate_path_segment

router = APIRouter(
    prefix="/api/v1/datasets",
    tags=["Datasets"],
    dependencies=[Depends(get_api_key)],
)

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
                        f.stat().st_size for f in item.glob("**/*") if f.is_file()
                    )

                    datasets.append(
                        DatasetMetadata(
                            name=item.name,
                            dataset_type=type_name,
                            description=f"Dataset em {type_name}",
                            file_count=file_count,
                            total_size=total_size,
                            total_duration=0.0,
                            created_at=None,
                            file_paths=[],  # Não expor paths internos
                        )
                    )

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
    result = service.create_dataset(name, dataset_type, description)
    if result.status == ProcessingStatus.ERROR:
        raise ValidationError(
            result.errors[0] if result.errors else "Erro ao criar dataset",
            field="name",
        )
    return result.data


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
    dataset_type = _validate_dataset_type(type)
    try:
        dataset_name = validate_path_segment(name, label="nome do dataset")
    except ValueError as exc:
        raise ValidationError(str(exc), field="name") from exc

    try:
        dataset_dir = resolve_within_directory(
            service.upload_directory,
            dataset_type.value,
            dataset_name,
            must_exist=True,
        )
    except (ValueError, FileNotFoundError):
        raise DatasetNotFoundError(dataset_name)
    if not dataset_dir.is_dir():
        raise DatasetNotFoundError(dataset_name)

    safe_name = sanitize_filename(file.filename or "upload.wav") or "upload.wav"
    suffix = "." + safe_name.rsplit(".", 1)[-1].lower() if "." in safe_name else ""
    if suffix not in service.SUPPORTED_FORMATS:
        raise UnsupportedFormatError(
            suffix or "(sem extensão)", sorted(service.SUPPORTED_FORMATS)
        )

    try:
        dest_path = resolve_within_directory(dataset_dir, safe_name)
    except ValueError as exc:
        raise ValidationError(str(exc), field="file") from exc
    if dest_path.exists():
        raise ConflictError(f"Arquivo já existe no dataset: {safe_name}")

    total_size = 0
    try:
        with open(dest_path, "xb") as buffer:
            while True:
                chunk = await file.read(1024 * 1024)
                if not chunk:
                    break
                total_size += len(chunk)
                if total_size > service.MAX_FILE_SIZE:
                    raise FileTooLargeError(service.MAX_FILE_SIZE // 1024 // 1024)
                buffer.write(chunk)
    except FileTooLargeError:
        dest_path.unlink(missing_ok=True)
        raise
    except FileExistsError as exc:
        raise ConflictError(f"Arquivo já existe no dataset: {safe_name}") from exc
    except OSError as exc:
        dest_path.unlink(missing_ok=True)
        raise AudioProcessingError(f"Erro ao salvar arquivo: {exc}") from exc
    finally:
        await file.close()

    return {
        "status": "success",
        "filename": safe_name,
        "size_bytes": total_size,
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
