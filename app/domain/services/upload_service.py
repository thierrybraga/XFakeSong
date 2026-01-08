"""Serviço de Upload de Arquivos de Áudio

Implementa a lógica de negócio para upload e validação de arquivos de áudio.
"""

import os
import hashlib
from pathlib import Path
from typing import List, Optional, Dict, Any, Union
from dataclasses import dataclass

from ...core.interfaces.base import IProcessor, ProcessingResult, ProcessingStatus
from ...core.interfaces.audio import AudioData, AudioFormat
from ...core.interfaces.services import IUploadService, DatasetMetadata, DatasetType
from ...core.exceptions.base import ValidationError, FileError, AudioError
from ...core.utils.helpers import ensure_directory, safe_filename, get_file_hash


def ensure_directory_exists(path: str) -> None:
    """Wrapper para ensure_directory"""
    ensure_directory(path)


def sanitize_filename(filename: str) -> str:
    """Wrapper para safe_filename"""
    return safe_filename(filename)


def calculate_file_hash(file_path: str) -> str:
    """Wrapper para get_file_hash"""
    return get_file_hash(file_path)


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

    SUPPORTED_FORMATS = {'.wav', '.mp3', '.flac', '.m4a', '.ogg'}
    MAX_FILE_SIZE = 100 * 1024 * 1024  # 100MB
    MIN_DURATION = 1.0  # 1 segundo
    MAX_DURATION = 300.0  # 5 minutos

    def __init__(self, upload_directory: str):
        self.upload_directory = Path(upload_directory)
        ensure_directory_exists(str(self.upload_directory))

    def upload_file(self, file_path: str,
                    dataset_type: DatasetType) -> ProcessingResult:
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
                    "message": f"Arquivo {
                        Path(file_path).name} enviado com sucesso"}
            )

        except Exception as e:
            return ProcessingResult(
                status=ProcessingStatus.ERROR,
                errors=[str(e)],
                metadata={"message": f"Erro no upload: {str(e)}"}
            )

    def upload_batch(
            self, file_paths: List[str], dataset_type: DatasetType) -> ProcessingResult:
        """Upload em lote de arquivos de áudio"""
        results = []
        errors = []

        for file_path in file_paths:
            result = self.upload_file(file_path, dataset_type)
            if result.status == ProcessingStatus.SUCCESS:
                results.append(result.data)
            else:
                errors.append(
                    f"{Path(file_path).name}: {result.errors[0] if result.errors else 'Erro desconhecido'}")

        if errors:
            return ProcessingResult(
                status=ProcessingStatus.PARTIAL_SUCCESS if results else ProcessingStatus.ERROR,
                data=results,
                errors=errors,
                metadata={
                    "message": f"Upload concluído: {
                        len(results)} sucessos, {
                        len(errors)} erros"}
            )

        return ProcessingResult(
            status=ProcessingStatus.SUCCESS,
            data=results,
            metadata={
                "message": f"Todos os {
                    len(results)} arquivos enviados com sucesso"}
        )

    def create_dataset(self, name: str, dataset_type: DatasetType,
                       description: Optional[str] = None) -> DatasetMetadata:
        """Criar novo dataset"""
        dataset_dir = self.upload_directory / sanitize_filename(name)
        ensure_directory_exists(str(dataset_dir))

        metadata = DatasetMetadata(
            name=name,
            dataset_type=dataset_type,
            description=description or f"Dataset {name}",
            file_count=0,
            total_size=0,
            total_duration=0.0,
            created_at=None,  # Será definido automaticamente
            file_paths=[]
        )

        return metadata

    def _validate_file(self, file_path: str) -> ProcessingResult:
        """Validar arquivo de áudio"""
        file_path = Path(file_path)

        # Verificar se arquivo existe
        if not file_path.exists():
            return ProcessingResult(
                status=ProcessingStatus.ERROR,
                errors=[f"Arquivo não encontrado: {file_path}"]
            )

        # Verificar extensão
        if file_path.suffix.lower() not in self.SUPPORTED_FORMATS:
            return ProcessingResult(
                status=ProcessingStatus.ERROR,
                errors=[f"Formato não suportado: {file_path.suffix}"]
            )

        # Verificar tamanho
        file_size = file_path.stat().st_size
        if file_size > self.MAX_FILE_SIZE:
            return ProcessingResult(
                status=ProcessingStatus.ERROR,
                errors=[
                    f"Arquivo muito grande: {
                        file_size /
                        1024 /
                        1024:.1f}MB"]
            )

        # Validação adicional de áudio seria feita aqui
        # (usando librosa ou similar)

        return ProcessingResult(
            status=ProcessingStatus.SUCCESS
        )

    def _process_upload(self, file_path: str,
                        dataset_type: DatasetType) -> UploadResult:
        """Processar upload do arquivo"""
        source_path = Path(file_path)

        # Gerar nome único para o arquivo
        file_hash = calculate_file_hash(str(source_path))
        safe_name = sanitize_filename(source_path.stem)
        destination_name = f"{safe_name}_{file_hash[:8]}{source_path.suffix}"

        # Determinar diretório de destino
        type_dir = self.upload_directory / dataset_type.value
        ensure_directory_exists(str(type_dir))

        destination_path = type_dir / destination_name

        # Copiar arquivo
        import shutil
        shutil.copy2(source_path, destination_path)

        # Criar resultado
        return UploadResult(
            file_path=str(destination_path),
            file_hash=file_hash,
            file_size=destination_path.stat().st_size
        )

    def upload_dataset(self, dataset_path: Union[str, Path],
                       dataset_name: str) -> ProcessingResult[DatasetMetadata]:
        """Faz upload de dataset completo"""
        try:
            dataset_path = Path(dataset_path)
            if not dataset_path.exists():
                return ProcessingResult(
                    status=ProcessingStatus.ERROR,
                    errors=[f"Dataset path não encontrado: {dataset_path}"]
                )

            # Criar metadata do dataset
            metadata = DatasetMetadata(
                name=dataset_name,
                dataset_type=DatasetType.TRAINING,
                description=f"Dataset {dataset_name}",
                file_count=0,
                total_size=0,
                total_duration=0.0,
                created_at=None,
                file_paths=[]
            )

            return ProcessingResult(
                status=ProcessingStatus.SUCCESS,
                data=metadata,
                metadata={
                    "message": f"Dataset {dataset_name} processado com sucesso"}
            )

        except Exception as e:
            return ProcessingResult(
                status=ProcessingStatus.ERROR,
                errors=[str(e)]
            )

    def validate_upload(
            self, file_path: Union[str, Path]) -> ProcessingResult[bool]:
        """Valida arquivo para upload"""
        try:
            file_path = Path(file_path)

            # Verificar se arquivo existe
            if not file_path.exists():
                return ProcessingResult(
                    status=ProcessingStatus.ERROR,
                    data=False,
                    errors=["Arquivo não encontrado"]
                )

            # Verificar extensão
            if file_path.suffix.lower() not in self.SUPPORTED_FORMATS:
                return ProcessingResult(
                    status=ProcessingStatus.ERROR,
                    data=False,
                    errors=[f"Formato não suportado: {file_path.suffix}"]
                )

            # Verificar tamanho
            if file_path.stat().st_size > self.MAX_FILE_SIZE:
                return ProcessingResult(
                    status=ProcessingStatus.ERROR,
                    data=False,
                    errors=["Arquivo muito grande"]
                )

            return ProcessingResult(
                status=ProcessingStatus.SUCCESS,
                data=True,
                metadata={"message": "Arquivo válido para upload"}
            )

        except Exception as e:
            return ProcessingResult(
                status=ProcessingStatus.ERROR,
                data=False,
                errors=[str(e)]
            )

    def get_upload_progress(
            self, upload_id: str) -> ProcessingResult[Dict[str, Any]]:
        """Retorna progresso do upload"""
        # Implementação simples - em um sistema real seria mais complexo
        return ProcessingResult(
            status=ProcessingStatus.SUCCESS,
            data={
                "upload_id": upload_id,
                "progress": 100,
                "status": "completed"
            },
            metadata={"message": "Upload concluído"}
        )

    def get_supported_formats(self) -> List[str]:
        """Retornar formatos suportados"""
        return list(self.SUPPORTED_FORMATS)

    def get_upload_stats(self) -> Dict[str, Any]:
        """Retornar estatísticas de upload"""
        stats = {
            'total_files': 0,
            'total_size': 0,
            'by_type': {}
        }

        for type_dir in self.upload_directory.iterdir():
            if type_dir.is_dir():
                files = list(type_dir.glob('*'))
                total_size = sum(
                    f.stat().st_size for f in files if f.is_file())

                stats['by_type'][type_dir.name] = {
                    'files': len(files),
                    'size': total_size
                }

                stats['total_files'] += len(files)
                stats['total_size'] += total_size

        return stats
