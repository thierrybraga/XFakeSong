"""Utilitários para manipulação de arquivos

Este módulo fornece funções utilitárias para operações com arquivos,
especialmente para o sistema de upload e processamento de datasets.
"""

import os
import hashlib
import shutil
import mimetypes
from pathlib import Path
from typing import List, Dict, Any, Optional, Union, Tuple
import tempfile
import zipfile
import tarfile
import logging
from datetime import datetime
import json


logger = logging.getLogger(__name__)


def ensure_directory(directory_path: Union[str, Path]) -> Path:
    """Garante que um diretório existe, criando-o se necessário"""
    path = Path(directory_path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def safe_filename(filename: str, max_length: int = 255) -> str:
    """Cria um nome de arquivo seguro removendo caracteres problemáticos"""
    # Caracteres não permitidos em nomes de arquivo
    invalid_chars = '<>:"/\\|?*'

    # Remover caracteres inválidos
    safe_name = ''.join(c for c in filename if c not in invalid_chars)

    # Remover espaços extras e pontos no início/fim
    safe_name = safe_name.strip(' .')

    # Limitar comprimento
    if len(safe_name) > max_length:
        name, ext = os.path.splitext(safe_name)
        max_name_length = max_length - len(ext)
        safe_name = name[:max_name_length] + ext

    # Garantir que não está vazio
    if not safe_name:
        safe_name = "unnamed_file"

    return safe_name


def get_file_hash(file_path: Union[str, Path], algorithm: str = 'md5') -> str:
    """Calcula hash de um arquivo"""
    hash_func = hashlib.new(algorithm)

    with open(file_path, 'rb') as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_func.update(chunk)

    return hash_func.hexdigest()


def get_file_info(file_path: Union[str, Path]) -> Dict[str, Any]:
    """Retorna informações detalhadas sobre um arquivo"""
    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(f"Arquivo não encontrado: {file_path}")

    stat = path.stat()

    info = {
        "name": path.name,
        "stem": path.stem,
        "suffix": path.suffix,
        "size_bytes": stat.st_size,
        "size_mb": round(stat.st_size / (1024 * 1024), 2),
        "created_at": datetime.fromtimestamp(stat.st_ctime),
        "modified_at": datetime.fromtimestamp(stat.st_mtime),
        "is_file": path.is_file(),
        "is_dir": path.is_dir(),
        "absolute_path": str(path.absolute()),
        "mime_type": mimetypes.guess_type(str(path))[0]
    }

    # Adicionar hash para arquivos pequenos (< 100MB)
    if path.is_file() and stat.st_size < 100 * 1024 * 1024:
        try:
            info["md5_hash"] = get_file_hash(path, 'md5')
        except Exception as e:
            logger.warning(f"Erro ao calcular hash de {path}: {e}")
            info["md5_hash"] = None

    return info


def find_files_by_extension(directory: Union[str, Path],
                            extensions: List[str],
                            recursive: bool = True) -> List[Path]:
    """Encontra arquivos por extensão em um diretório"""
    directory = Path(directory)
    extensions = [
        ext.lower() if ext.startswith('.') else f'.{
            ext.lower()}' for ext in extensions]

    files = []

    if recursive:
        pattern = '**/*'
    else:
        pattern = '*'

    for file_path in directory.glob(pattern):
        if file_path.is_file() and file_path.suffix.lower() in extensions:
            files.append(file_path)

    return files


def copy_file_safe(src: Union[str, Path], dst: Union[str, Path],
                   overwrite: bool = False) -> bool:
    """Copia arquivo de forma segura com verificações"""
    src_path = Path(src)
    dst_path = Path(dst)

    # Verificar se arquivo origem existe
    if not src_path.exists():
        raise FileNotFoundError(f"Arquivo origem não encontrado: {src}")

    # Verificar se destino já existe
    if dst_path.exists() and not overwrite:
        raise FileExistsError(f"Arquivo destino já existe: {dst}")

    # Criar diretório de destino se necessário
    dst_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        # Copiar arquivo preservando metadados
        shutil.copy2(src_path, dst_path)

        # Verificar se cópia foi bem-sucedida comparando tamanhos
        if src_path.stat().st_size != dst_path.stat().st_size:
            raise RuntimeError(
                "Falha na verificação da cópia: tamanhos diferentes")

        return True

    except Exception as e:
        logger.error(f"Erro ao copiar {src} para {dst}: {e}")
        # Limpar arquivo parcialmente copiado
        if dst_path.exists():
            try:
                dst_path.unlink()
            except BaseException:
                pass
        raise


def move_file_safe(src: Union[str, Path], dst: Union[str, Path],
                   overwrite: bool = False) -> bool:
    """Move arquivo de forma segura"""
    src_path = Path(src)
    dst_path = Path(dst)

    # Verificar se arquivo origem existe
    if not src_path.exists():
        raise FileNotFoundError(f"Arquivo origem não encontrado: {src}")

    # Verificar se destino já existe
    if dst_path.exists() and not overwrite:
        raise FileExistsError(f"Arquivo destino já existe: {dst}")

    # Criar diretório de destino se necessário
    dst_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        shutil.move(str(src_path), str(dst_path))
        return True
    except Exception as e:
        logger.error(f"Erro ao mover {src} para {dst}: {e}")
        raise


def extract_archive(archive_path: Union[str, Path],
                    extract_to: Union[str, Path],
                    password: Optional[str] = None) -> List[Path]:
    """Extrai arquivo comprimido (ZIP, TAR, etc.)"""
    archive_path = Path(archive_path)
    extract_to = Path(extract_to)

    if not archive_path.exists():
        raise FileNotFoundError(f"Arquivo não encontrado: {archive_path}")

    # Criar diretório de extração
    ensure_directory(extract_to)

    extracted_files = []

    try:
        # Detectar tipo de arquivo
        if archive_path.suffix.lower() == '.zip':
            with zipfile.ZipFile(archive_path, 'r') as zip_ref:
                # Listar arquivos antes da extração
                file_list = zip_ref.namelist()

                # Extrair com senha se fornecida
                if password:
                    zip_ref.setpassword(password.encode())

                # Extrair todos os arquivos
                zip_ref.extractall(extract_to)

                # Coletar caminhos dos arquivos extraídos
                for file_name in file_list:
                    if not file_name.endswith('/'):
                        extracted_files.append(extract_to / file_name)

        elif archive_path.suffix.lower() in ['.tar', '.tar.gz', '.tgz', '.tar.bz2']:
            with tarfile.open(archive_path, 'r:*') as tar_ref:
                # Listar arquivos
                file_list = tar_ref.getnames()

                # Extrair todos os arquivos
                tar_ref.extractall(extract_to)

                # Coletar caminhos dos arquivos extraídos
                for file_name in file_list:
                    file_path = extract_to / file_name
                    if file_path.is_file():
                        extracted_files.append(file_path)

        else:
            raise ValueError(
                f"Formato de arquivo não suportado: {
                    archive_path.suffix}")

        logger.info(
            f"Extraídos {
                len(extracted_files)} arquivos de {
                archive_path.name}")
        return extracted_files

    except zipfile.BadZipFile:
        raise ValueError(f"Arquivo ZIP corrompido: {archive_path}")
    except tarfile.TarError as e:
        raise ValueError(f"Erro no arquivo TAR: {e}")
    except Exception as e:
        logger.error(f"Erro ao extrair {archive_path}: {e}")
        raise


def create_temp_directory(prefix: str = "temp_", suffix: str = "") -> Path:
    """Cria diretório temporário"""
    temp_dir = Path(tempfile.mkdtemp(prefix=prefix, suffix=suffix))
    logger.debug(f"Diretório temporário criado: {temp_dir}")
    return temp_dir


def cleanup_temp_directory(
        temp_dir: Union[str, Path], force: bool = False) -> bool:
    """Remove diretório temporário"""
    temp_dir = Path(temp_dir)

    try:
        if temp_dir.exists() and (force or "temp" in temp_dir.name.lower()):
            shutil.rmtree(temp_dir)
            logger.debug(f"Diretório temporário removido: {temp_dir}")
            return True
        return False
    except Exception as e:
        logger.warning(f"Erro ao remover diretório temporário {temp_dir}: {e}")
        return False


def validate_file_type(file_path: Union[str, Path],
                       allowed_types: List[str],
                       check_content: bool = False) -> bool:
    """Valida tipo de arquivo por extensão e opcionalmente por conteúdo"""
    file_path = Path(file_path)

    # Verificar extensão
    extension = file_path.suffix.lower()
    if extension not in [t.lower() if t.startswith(
            '.') else f'.{t.lower()}' for t in allowed_types]:
        return False

    # Verificar conteúdo se solicitado
    if check_content:
        try:
            mime_type, _ = mimetypes.guess_type(str(file_path))
            if mime_type:
                # Mapear MIME types para extensões conhecidas
                mime_to_ext = {
                    'audio/wav': '.wav',
                    'audio/mpeg': '.mp3',
                    'audio/flac': '.flac',
                    'audio/mp4': '.m4a',
                    'audio/ogg': '.ogg',
                    'application/zip': '.zip'
                }

                expected_ext = mime_to_ext.get(mime_type)
                if expected_ext and expected_ext != extension:
                    return False
        except Exception as e:
            logger.warning(
                f"Erro na validação de conteúdo para {file_path}: {e}")

    return True


def get_directory_size(directory: Union[str, Path]) -> Tuple[int, int]:
    """Retorna tamanho total e número de arquivos em um diretório"""
    directory = Path(directory)
    total_size = 0
    file_count = 0

    for file_path in directory.rglob('*'):
        if file_path.is_file():
            total_size += file_path.stat().st_size
            file_count += 1

    return total_size, file_count


def create_file_manifest(directory: Union[str, Path],
                         output_file: Optional[Union[str, Path]] = None) -> Dict[str, Any]:
    """Cria manifesto com informações de todos os arquivos em um diretório"""
    directory = Path(directory)

    manifest = {
        "directory": str(directory.absolute()),
        "created_at": datetime.now().isoformat(),
        "total_size": 0,
        "file_count": 0,
        "files": []
    }

    for file_path in directory.rglob('*'):
        if file_path.is_file():
            try:
                file_info = get_file_info(file_path)
                relative_path = file_path.relative_to(directory)

                file_entry = {
                    "path": str(relative_path),
                    "name": file_info["name"],
                    "size": file_info["size_bytes"],
                    "extension": file_info["suffix"],
                    "modified_at": file_info["modified_at"].isoformat(),
                    "hash": file_info.get("md5_hash")
                }

                manifest["files"].append(file_entry)
                manifest["total_size"] += file_info["size_bytes"]
                manifest["file_count"] += 1

            except Exception as e:
                logger.warning(f"Erro ao processar arquivo {file_path}: {e}")

    # Salvar manifesto se caminho fornecido
    if output_file:
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(manifest, f, indent=2, ensure_ascii=False)

        logger.info(f"Manifesto salvo em: {output_path}")

    return manifest


def verify_file_integrity(file_path: Union[str, Path],
                          expected_hash: str,
                          algorithm: str = 'md5') -> bool:
    """Verifica integridade de arquivo comparando hash"""
    try:
        actual_hash = get_file_hash(file_path, algorithm)
        return actual_hash.lower() == expected_hash.lower()
    except Exception as e:
        logger.error(f"Erro na verificação de integridade de {file_path}: {e}")
        return False


def batch_rename_files(directory: Union[str, Path],
                       pattern: str = "{index:03d}_{original}",
                       extensions: Optional[List[str]] = None) -> List[Tuple[Path, Path]]:
    """Renomeia arquivos em lote seguindo um padrão"""
    directory = Path(directory)
    renamed_files = []

    # Filtrar arquivos por extensão se especificado
    if extensions:
        files = find_files_by_extension(directory, extensions, recursive=False)
    else:
        files = [f for f in directory.iterdir() if f.is_file()]

    # Ordenar arquivos
    files.sort(key=lambda x: x.name)

    for index, file_path in enumerate(files, 1):
        try:
            # Gerar novo nome
            new_name = pattern.format(
                index=index,
                original=file_path.stem,
                extension=file_path.suffix,
                timestamp=datetime.now().strftime("%Y%m%d_%H%M%S")
            )

            # Adicionar extensão se não incluída no padrão
            if not new_name.endswith(file_path.suffix):
                new_name += file_path.suffix

            new_path = file_path.parent / new_name

            # Renomear se nome for diferente
            if file_path.name != new_name:
                file_path.rename(new_path)
                renamed_files.append((file_path, new_path))
                logger.debug(f"Renomeado: {file_path.name} -> {new_name}")

        except Exception as e:
            logger.error(f"Erro ao renomear {file_path}: {e}")

    logger.info(f"Renomeados {len(renamed_files)} arquivos")
    return renamed_files


# Funções de conveniência para tipos específicos
def find_audio_files(
        directory: Union[str, Path], recursive: bool = True) -> List[Path]:
    """Encontra arquivos de áudio em um diretório"""
    audio_extensions = [
        '.wav',
        '.mp3',
        '.flac',
        '.m4a',
        '.ogg',
        '.aac',
        '.wma']
    return find_files_by_extension(directory, audio_extensions, recursive)


def find_archive_files(
        directory: Union[str, Path], recursive: bool = True) -> List[Path]:
    """Encontra arquivos comprimidos em um diretório"""
    archive_extensions = [
        '.zip',
        '.tar',
        '.tar.gz',
        '.tgz',
        '.tar.bz2',
        '.rar',
        '.7z']
    return find_files_by_extension(directory, archive_extensions, recursive)


def is_audio_file(file_path: Union[str, Path]) -> bool:
    """Verifica se arquivo é de áudio"""
    audio_extensions = [
        '.wav',
        '.mp3',
        '.flac',
        '.m4a',
        '.ogg',
        '.aac',
        '.wma']
    return validate_file_type(file_path, audio_extensions)


def is_archive_file(file_path: Union[str, Path]) -> bool:
    """Verifica se arquivo é comprimido"""
    archive_extensions = [
        '.zip',
        '.tar',
        '.tar.gz',
        '.tgz',
        '.tar.bz2',
        '.rar',
        '.7z']
    return validate_file_type(file_path, archive_extensions)
