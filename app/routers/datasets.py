from fastapi import (
    APIRouter, UploadFile, File, Form, Depends, HTTPException, Query
)
from typing import List, Optional
import shutil

from app.schemas.api_models import DatasetMetadata
from app.domain.services.upload_service import AudioUploadService
from app.dependencies import get_upload_service
from app.core.interfaces.base import ProcessingStatus, DatasetType
from app.core.auth.auth_handler import get_api_key

router = APIRouter(prefix="/api/v1/datasets", tags=["Datasets"])


@router.get("/", response_model=List[DatasetMetadata])
async def list_datasets(
    type: Optional[str] = Query(
        None,
        description="Filtrar por tipo (training, validation, test)"
    ),
    service: AudioUploadService = Depends(get_upload_service)
):
    """Lista todos os datasets disponíveis."""
    # Como o serviço não tem list_datasets implementado explicitamente,
    # vamos implementar uma lógica de varredura aqui.
    # Por segurança, vamos varrer o diretório de uploads.

    datasets = []
    base_dir = service.upload_directory

    if not base_dir.exists():
        return []

    # Varre diretórios (training, validation, test)
    target_dirs = [type] if type else [t.value for t in DatasetType]

    for type_name in target_dirs:
        type_path = base_dir / type_name
        if type_path.exists():
            for item in type_path.iterdir():
                if item.is_dir():  # ou arquivo se o dataset for um zip
                    # Calcular estatísticas básicas
                    file_count = len(list(item.glob("**/*")))
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
                        total_duration=0.0,  # Calcular seria custoso aqui
                        created_at=None,
                        file_paths=[]
                    ))

    return datasets


@router.post(
    "/",
    response_model=DatasetMetadata,
    dependencies=[Depends(get_api_key)]
)
async def create_dataset(
    name: str = Form(...),
    type: str = Form("training"),
    description: Optional[str] = Form(None),
    service: AudioUploadService = Depends(get_upload_service)
):
    """Cria um novo dataset vazio."""
    try:
        dataset_type = DatasetType(type)
        metadata = service.create_dataset(name, dataset_type, description)
        return metadata
    except ValueError:
        raise HTTPException(
            status_code=400,
            detail=f"Tipo de dataset inválido: {type}"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/{name}/upload", dependencies=[Depends(get_api_key)])
async def upload_to_dataset(
    name: str,
    file: UploadFile = File(...),
    type: str = Form("training"),
    service: AudioUploadService = Depends(get_upload_service)
):
    """Faz upload de um arquivo para um dataset existente."""
    try:
        # Verificar se dataset existe
        dataset_dir = service.upload_directory / type / name
        if not dataset_dir.exists():
            raise HTTPException(
                status_code=404,
                detail="Dataset não encontrado. Crie-o primeiro."
            )

        # Salvar arquivo
        dest_path = dataset_dir / file.filename
        with open(dest_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        return {
            "status": "success",
            "filename": file.filename,
            "path": str(dest_path)
        }

    except ValueError:
        raise HTTPException(
            status_code=400,
            detail=f"Tipo de dataset inválido: {type}"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/{name}", dependencies=[Depends(get_api_key)])
async def delete_dataset(
    name: str,
    type: str = Query(
        "training",
        description="Tipo de dataset (training, validation, test)"
    ),
    service: AudioUploadService = Depends(get_upload_service)
):
    """Exclui um dataset existente."""
    try:
        dataset_type = DatasetType(type)
        result = service.delete_dataset(name, dataset_type)

        if result.status == ProcessingStatus.ERROR:
            raise HTTPException(status_code=404, detail=result.errors[0])

        return result.metadata
    except ValueError:
        raise HTTPException(
            status_code=400,
            detail=f"Tipo de dataset inválido: {type}"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
