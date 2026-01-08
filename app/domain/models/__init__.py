# Arquivo de inicialização para modelos de domínio

from .base_model import BaseModel
from .analysis import AnalysisResult
from .architecture_config import ArchitectureConfig
from .user import User

__all__ = [
    'BaseModel',
    'AnalysisResult',
    'ArchitectureConfig',
    'User'
]
