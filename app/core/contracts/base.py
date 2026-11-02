"""Interfaces base do sistema seguindo princípios SOLID

Este módulo define as interfaces fundamentais que garantem:
- Single Responsibility Principle (SRP)
- Open/Closed Principle (OCP)
- Liskov Substitution Principle (LSP)
- Interface Segregation Principle (ISP)
- Dependency Inversion Principle (DIP)
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, Generic, List, Optional, TypeVar

# Type variables para generics
T = TypeVar('T')
R = TypeVar('R')


class ProcessingStatus(Enum):
    """Status de processamento padronizado."""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    SUCCESS = "success"
    PARTIAL_SUCCESS = "partial_success"
    ERROR = "error"
    WARNING = "warning"
    CANCELLED = "cancelled"


class DatasetType(Enum):
    """Tipos de datasets suportados."""
    TRAINING = "training"
    VALIDATION = "validation"
    TEST = "test"
    PRODUCTION = "production"


@dataclass
class ProcessingResult(Generic[T]):
    """Resultado padronizado de processamento."""
    status: ProcessingStatus
    data: Optional[T] = None
    metadata: Dict[str, Any] = None
    errors: List[str] = None
    warnings: List[str] = None
    execution_time: Optional[float] = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}
        if self.errors is None:
            self.errors = []
        if self.warnings is None:
            self.warnings = []

    @property
    def is_success(self) -> bool:
        return self.status == ProcessingStatus.SUCCESS

    @classmethod
    def success(cls, data: T = None, metadata: Dict[str, Any] = None,
                execution_time: float = None) -> 'ProcessingResult[T]':
        """Create a successful processing result."""
        return cls(
            status=ProcessingStatus.SUCCESS,
            data=data,
            metadata=metadata or {},
            execution_time=execution_time
        )

    @classmethod
    def error(cls, error_message: str,
              metadata: Dict[str, Any] = None,
              execution_time: float = None) -> 'ProcessingResult[T]':
        """Create an error processing result."""
        return cls(
            status=ProcessingStatus.ERROR,
            errors=[error_message],
            metadata=metadata or {},
            execution_time=execution_time
        )


class IExtractor(ABC, Generic[T, R]):
    """Interface para extratores (ISP)."""

    @abstractmethod
    def extract(self, source: T) -> ProcessingResult[R]:
        """Extrai dados da fonte."""
        pass
