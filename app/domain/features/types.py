"""Tipos e classes base para extração de características."""

from enum import Enum
from typing import Any, Dict, Optional, Generic, TypeVar
from dataclasses import dataclass

T = TypeVar('T')


class FeatureType(Enum):
    """Tipos de características de áudio."""
    SPECTRAL = "spectral"
    TEMPORAL = "temporal"
    PROSODIC = "prosodic"
    PERCEPTUAL = "perceptual"
    ADVANCED = "advanced"
    CEPSTRAL = "cepstral"
    FORMANT = "formant"
    VOICE_QUALITY = "voice_quality"


@dataclass
class ProcessingResult(Generic[T]):
    """Resultado de processamento com dados e metadados."""
    success: bool
    data: Optional[T] = None
    error_message: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None
    execution_time: Optional[float] = None

    @classmethod
    def success_result(
            cls, data: T = None, metadata: Dict[str, Any] = None, execution_time: float = None) -> 'ProcessingResult[T]':
        """Cria um resultado de sucesso."""
        return cls(
            success=True,
            data=data,
            metadata=metadata or {},
            execution_time=execution_time
        )

    @classmethod
    def error_result(cls, error_message: str,
                     metadata: Dict[str, Any] = None, execution_time: float = None) -> 'ProcessingResult[T]':
        """Cria um resultado de erro."""
        return cls(
            success=False,
            error_message=error_message,
            metadata=metadata or {},
            execution_time=execution_time
        )


class ProcessingStatus(Enum):
    """Status de processamento."""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
