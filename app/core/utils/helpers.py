"""Utilitários e helpers do sistema

Este módulo contém funções utilitárias reutilizáveis em todo o sistema.
"""

import os
import json
import hashlib
import time
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, Callable, TypeVar
from datetime import datetime, timezone
from functools import wraps
import numpy as np

T = TypeVar('T')


def ensure_directory(path: Union[str, Path]) -> Path:
    """Garante que um diretório existe, criando-o se necessário."""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def safe_filename(filename: str, max_length: int = 255) -> str:
    """Sanitiza nome de arquivo removendo caracteres perigosos."""
    # Remove caracteres perigosos
    unsafe_chars = '<>:"/\\|?*'
    for char in unsafe_chars:
        filename = filename.replace(char, '_')

    # Limita comprimento
    if len(filename) > max_length:
        name, ext = os.path.splitext(filename)
        max_name_length = max_length - len(ext)
        filename = name[:max_name_length] + ext

    return filename


def get_file_hash(file_path: Union[str, Path], algorithm: str = 'md5') -> str:
    """Calcula hash de arquivo."""
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"Arquivo não encontrado: {file_path}")

    hash_obj = hashlib.new(algorithm)
    with open(file_path, 'rb') as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_obj.update(chunk)

    return hash_obj.hexdigest()


def get_file_size(file_path: Union[str, Path]) -> int:
    """Retorna tamanho do arquivo em bytes."""
    return Path(file_path).stat().st_size


def format_file_size(size_bytes: int) -> str:
    """Formata tamanho de arquivo em formato legível."""
    if size_bytes == 0:
        return "0 B"

    size_names = ["B", "KB", "MB", "GB", "TB"]
    i = 0
    while size_bytes >= 1024 and i < len(size_names) - 1:
        size_bytes /= 1024.0
        i += 1

    return f"{size_bytes:.1f} {size_names[i]}"


def format_duration(seconds: float) -> str:
    """Formata duração em formato legível."""
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.1f}m"
    else:
        hours = seconds / 3600
        return f"{hours:.1f}h"


def get_timestamp(timezone_aware: bool = True) -> str:
    """Retorna timestamp atual em formato ISO."""
    if timezone_aware:
        return datetime.now(timezone.utc).isoformat()
    else:
        return datetime.now().isoformat()


def parse_timestamp(timestamp_str: str) -> datetime:
    """Converte string de timestamp para datetime."""
    return datetime.fromisoformat(timestamp_str)


def timing_decorator(func: Callable[..., T]) -> Callable[..., T]:
    """Decorator para medir tempo de execução."""
    @wraps(func)
    def wrapper(*args, **kwargs) -> T:
        start_time = time.time()
        try:
            result = func(*args, **kwargs)
            execution_time = time.time() - start_time
            logging.debug(
                f"{func.__name__} executado em {execution_time:.3f}s")
            return result
        except Exception as e:
            execution_time = time.time() - start_time
            logging.error(
                f"{func.__name__} falhou após {execution_time:.3f}s: {e}")
            raise
    return wrapper


def retry_decorator(max_attempts: int = 3,
                    delay: float = 1.0, backoff: float = 2.0):
    """Decorator para retry automático com backoff exponencial."""
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @wraps(func)
        def wrapper(*args, **kwargs) -> T:
            current_delay = delay
            last_exception = None

            for attempt in range(max_attempts):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    last_exception = e
                    if attempt < max_attempts - 1:
                        logging.warning(
                            f"Tentativa {attempt + 1} falhou para {func.__name__}: {e}. Tentando novamente em {current_delay}s")
                        time.sleep(current_delay)
                        current_delay *= backoff
                    else:
                        logging.error(
                            f"Todas as {max_attempts} tentativas falharam para {func.__name__}")

            raise last_exception
        return wrapper
    return decorator


def load_json(file_path: Union[str, Path], default: Any = None) -> Any:
    """Carrega arquivo JSON com tratamento de erro."""
    file_path = Path(file_path)
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        if default is not None:
            return default
        raise e


def save_json(data: Any, file_path: Union[str, Path], indent: int = 2) -> bool:
    """Salva dados em arquivo JSON."""
    file_path = Path(file_path)
    ensure_directory(file_path.parent)

    try:
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=indent, ensure_ascii=False, default=str)
        return True
    except Exception as e:
        logging.error(f"Erro ao salvar JSON em {file_path}: {e}")
        return False


def deep_merge_dicts(dict1: Dict[str, Any],
                     dict2: Dict[str, Any]) -> Dict[str, Any]:
    """Faz merge profundo de dois dicionários."""
    result = dict1.copy()

    for key, value in dict2.items():
        if key in result and isinstance(
                result[key], dict) and isinstance(value, dict):
            result[key] = deep_merge_dicts(result[key], value)
        else:
            result[key] = value

    return result


def flatten_dict(d: Dict[str, Any], parent_key: str = '',
                 sep: str = '.') -> Dict[str, Any]:
    """Achata dicionário aninhado."""
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def chunk_list(lst: List[T], chunk_size: int) -> List[List[T]]:
    """Divide lista em chunks de tamanho específico."""
    return [lst[i:i + chunk_size] for i in range(0, len(lst), chunk_size)]


def normalize_array(arr: np.ndarray, method: str = 'minmax') -> np.ndarray:
    """Normaliza array numpy."""
    if method == 'minmax':
        min_val = np.min(arr)
        max_val = np.max(arr)
        if max_val - min_val == 0:
            return np.zeros_like(arr)
        return (arr - min_val) / (max_val - min_val)

    elif method == 'zscore':
        mean_val = np.mean(arr)
        std_val = np.std(arr)
        if std_val == 0:
            return np.zeros_like(arr)
        return (arr - mean_val) / std_val

    elif method == 'robust':
        median_val = np.median(arr)
        mad = np.median(np.abs(arr - median_val))
        if mad == 0:
            return np.zeros_like(arr)
        return (arr - median_val) / mad

    else:
        raise ValueError(f"Método de normalização desconhecido: {method}")


def validate_audio_file(file_path: Union[str, Path]) -> bool:
    """Valida se arquivo é um áudio válido."""
    file_path = Path(file_path)

    # Verifica se arquivo existe
    if not file_path.exists():
        return False

    # Verifica extensão
    valid_extensions = {'.wav', '.mp3', '.flac', '.m4a', '.aac', '.ogg'}
    if file_path.suffix.lower() not in valid_extensions:
        return False

    # Verifica se não está vazio
    if file_path.stat().st_size == 0:
        return False

    return True


def get_system_info() -> Dict[str, Any]:
    """Retorna informações do sistema."""
    import platform
    import psutil

    return {
        'platform': platform.platform(),
        'python_version': platform.python_version(),
        'cpu_count': psutil.cpu_count(),
        'memory_total': psutil.virtual_memory().total,
        'memory_available': psutil.virtual_memory().available,
        'disk_usage': psutil.disk_usage('/').percent if os.name != 'nt' else psutil.disk_usage('C:\\').percent
    }


def create_unique_id(prefix: str = '', length: int = 8) -> str:
    """Cria ID único."""
    import uuid
    unique_part = str(uuid.uuid4()).replace('-', '')[:length]
    return f"{prefix}_{unique_part}" if prefix else unique_part


def memory_usage_mb() -> float:
    """Retorna uso de memória atual em MB."""
    import psutil
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024


def setup_logging(name: str, level: str = 'INFO',
                  format_str: Optional[str] = None) -> logging.Logger:
    """Configura logger padronizado."""
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level.upper()))

    if not logger.handlers:
        handler = logging.StreamHandler()
        if format_str is None:
            format_str = '%(asctime)s | %(levelname)s | %(name)s:%(funcName)s:%(lineno)d | %(message)s'

        formatter = logging.Formatter(format_str)
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    return logger


class ProgressTracker:
    """Classe para rastrear progresso de operações."""

    def __init__(self, total: int, description: str = ""):
        self.total = total
        self.current = 0
        self.description = description
        self.start_time = time.time()

    def update(self, increment: int = 1):
        """Atualiza progresso."""
        self.current += increment
        if self.current > self.total:
            self.current = self.total

    def get_progress(self) -> Dict[str, Any]:
        """Retorna informações de progresso."""
        elapsed = time.time() - self.start_time
        percentage = (self.current / self.total) * 100 if self.total > 0 else 0

        if self.current > 0 and elapsed > 0:
            rate = self.current / elapsed
            eta = (self.total - self.current) / rate if rate > 0 else 0
        else:
            rate = 0
            eta = 0

        return {
            'current': self.current,
            'total': self.total,
            'percentage': percentage,
            'elapsed': elapsed,
            'eta': eta,
            'rate': rate,
            'description': self.description
        }

    def __str__(self) -> str:
        progress = self.get_progress()
        return f"{self.description}: {progress['current']}/{progress['total']} ({progress['percentage']:.1f}%)"
