"""Serviço de Upload de Arquivos de Áudio

Implementa a lógica de negócio para upload e validação de arquivos de áudio.
"""

import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union

from ...core.contracts.audio import AudioFormat
from ...core.contracts.base import ProcessingResult, ProcessingStatus
from ...core.contracts.services import DatasetMetadata, DatasetType, IUploadService
from ...utils.file_utils import (
    ensure_directory,
    get_file_hash,
    resolve_within_directory,
    safe_filename,
    validate_path_segment,
)


@dataclass
class UploadResult:
    """Resultado do processo de upload"""

    file_path: str
    file_hash: str
    file_size: int
    duration: Optional[float] = None
    sample_rate: Optional[int] = None
    channels: Optional[int] = None
    format: Optional[AudioFormat] = None


class AudioUploadService(IUploadService):
    """Serviço para upload e validação de arquivos de áudio"""

    SUPPORTED_FORMATS = {".wav", ".mp3", ".flac", ".m4a", ".ogg"}
    MAX_FILE_SIZE = 100 * 1024 * 1024  # 100MB
    MIN_DURATION = 1.0  # 1 segundo
    MAX_DURATION = 300.0  # 5 minutos

    def __init__(self, upload_directory: str):
        self.upload_directory = Path(upload_directory)
        ensure_directory(str(self.upload_directory))

    def upload_file(
        self, file_path: str, dataset_type: DatasetType
    ) -> ProcessingResult:
        """Upload e validação de arquivo de áudio"""
        try:
            # Validar arquivo
            validation_result = self._validate_file(file_path)
            if validation_result.status != ProcessingStatus.SUCCESS:
                return validation_result

            # Processar upload
            upload_result = self._process_upload(file_path, dataset_type)

            return ProcessingResult(
                status=ProcessingStatus.SUCCESS,
                data=upload_result,
                metadata={
                    "message": (f"Arquivo {Path(file_path).name} enviado com sucesso")
                },
            )

        except Exception as e:
            return ProcessingResult(
                status=ProcessingStatus.ERROR,
                errors=[str(e)],
                metadata={"message": f"Erro no upload: {str(e)}"},
            )

    def create_dataset(
        self, name: str, dataset_type: DatasetType, description: Optional[str] = None
    ) -> ProcessingResult[DatasetMetadata]:
        """Criar novo dataset"""
        try:
            validated_name = validate_path_segment(name, label="nome do dataset")
            clean_name = safe_filename(validated_name)
            dataset_dir = resolve_within_directory(
                self.upload_directory, dataset_type.value, clean_name
            )
            if dataset_dir.exists():
                raise ValueError(f"Dataset já existe: {clean_name}")
            ensure_directory(str(dataset_dir))

            metadata = DatasetMetadata(
                name=clean_name,
                # Pass enum directly if model supports it, or .value
                dataset_type=dataset_type,
                description=description or f"Dataset {name}",
                file_count=0,
                total_size=0,
                total_duration=0.0,
                created_at=None,
                file_paths=[],
            )

            return ProcessingResult(
                status=ProcessingStatus.SUCCESS,
                data=metadata,
                metadata={"message": f"Dataset {name} criado com sucesso"},
            )
        except Exception as e:
            return ProcessingResult(status=ProcessingStatus.ERROR, errors=[str(e)])

    def delete_dataset(self, name: str, dataset_type: DatasetType) -> ProcessingResult:
        """Excluir dataset existente"""
        try:
            validated_name = validate_path_segment(name, label="nome do dataset")
            clean_name = safe_filename(validated_name)
            dataset_dir = resolve_within_directory(
                self.upload_directory, dataset_type.value, clean_name
            )
            if not dataset_dir.is_dir():
                return ProcessingResult(
                    status=ProcessingStatus.ERROR,
                    errors=[f"Dataset não encontrado: {name}"],
                )

            shutil.rmtree(dataset_dir)

            return ProcessingResult(
                status=ProcessingStatus.SUCCESS,
                metadata={"message": f"Dataset {name} excluído com sucesso"},
            )
        except Exception as e:
            return ProcessingResult(status=ProcessingStatus.ERROR, errors=[str(e)])

    def _validate_file(self, file_path: str) -> ProcessingResult:
        """Validar arquivo de áudio"""
        file_path = Path(file_path)

        # Verificar se arquivo existe
        if not file_path.exists():
            return ProcessingResult(
                status=ProcessingStatus.ERROR,
                errors=[f"Arquivo não encontrado: {file_path}"],
            )

        # Verificar extensão
        if file_path.suffix.lower() not in self.SUPPORTED_FORMATS:
            return ProcessingResult(
                status=ProcessingStatus.ERROR,
                errors=[f"Formato não suportado: {file_path.suffix}"],
            )

        # Verificar tamanho
        file_size = file_path.stat().st_size
        if file_size > self.MAX_FILE_SIZE:
            return ProcessingResult(
                status=ProcessingStatus.ERROR,
                errors=[f"Arquivo muito grande: " f"{file_size / 1024 / 1024:.1f}MB"],
            )

        # Validação adicional de áudio seria feita aqui
        # (usando librosa ou similar)

        return ProcessingResult(status=ProcessingStatus.SUCCESS)

    def _process_upload(
        self, file_path: str, dataset_type: DatasetType
    ) -> UploadResult:
        """Processar upload do arquivo"""
        source_path = Path(file_path)

        # Gerar nome único para o arquivo
        file_hash = get_file_hash(str(source_path))
        safe_name = safe_filename(source_path.stem)
        destination_name = f"{safe_name}_{file_hash[:8]}{source_path.suffix}"

        # Determinar diretório de destino
        type_dir = self.upload_directory / dataset_type.value
        ensure_directory(str(type_dir))

        destination_path = type_dir / destination_name

        # Copiar arquivo
        shutil.copy2(source_path, destination_path)

        # Criar resultado
        return UploadResult(
            file_path=str(destination_path),
            file_hash=file_hash,
            file_size=destination_path.stat().st_size,
        )

    def validate_upload(self, file_path: Union[str, Path]) -> ProcessingResult[bool]:
        """Valida arquivo para upload"""
        try:
            file_path = Path(file_path)

            # Verificar se arquivo existe
            if not file_path.exists():
                return ProcessingResult(
                    status=ProcessingStatus.ERROR,
                    data=False,
                    errors=["Arquivo não encontrado"],
                )

            # Verificar extensão
            if file_path.suffix.lower() not in self.SUPPORTED_FORMATS:
                return ProcessingResult(
                    status=ProcessingStatus.ERROR,
                    data=False,
                    errors=[f"Formato não suportado: {file_path.suffix}"],
                )

            # Verificar tamanho
            if file_path.stat().st_size > self.MAX_FILE_SIZE:
                return ProcessingResult(
                    status=ProcessingStatus.ERROR,
                    data=False,
                    errors=["Arquivo muito grande"],
                )

            return ProcessingResult(
                status=ProcessingStatus.SUCCESS,
                data=True,
                metadata={"message": "Arquivo válido para upload"},
            )

        except Exception as e:
            return ProcessingResult(
                status=ProcessingStatus.ERROR, data=False, errors=[str(e)]
            )
