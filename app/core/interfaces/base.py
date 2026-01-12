"""Interfaces base do sistema seguindo princípios SOLID

Este módulo define as interfaces fundamentais que garantem:
- Single Responsibility Principle (SRP)
- Open/Closed Principle (OCP)
- Liskov Substitution Principle (LSP)
- Interface Segregation Principle (ISP)
- Dependency Inversion Principle (DIP)
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Generic, TypeVar
from dataclasses import dataclass
from enum import Enum

# Type variables para generics
T = TypeVar('T')
R = TypeVar('R')


class ProcessingStatus(Enum):
    """Status de processamento padronizado."""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    SUCCESS = "success"
    ERROR = "error"
    WARNING = "warning"
    CANCELLED = "cancelled"


class DatasetType(Enum):
    """Tipos de datasets suportados."""
    TRAINING = "training"
    VALIDATION = "validation"
    TEST = "test"


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

    @property
    def has_errors(self) -> bool:
        return len(self.errors) > 0

    @property
    def has_warnings(self) -> bool:
        return len(self.warnings) > 0

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


class IProcessor(ABC, Generic[T, R]):
    """Interface base para processadores (SRP)."""

    @abstractmethod
    def process(self, input_data: T) -> ProcessingResult[R]:
        """Processa dados de entrada e retorna resultado."""
        pass

    @abstractmethod
    def validate_input(self, input_data: T) -> bool:
        """Valida dados de entrada."""
        pass


class IValidator(ABC, Generic[T]):
    """Interface para validadores (ISP)."""

    @abstractmethod
    def validate(self, data: T) -> ProcessingResult[bool]:
        """Valida dados."""
        pass


class IExtractor(ABC, Generic[T, R]):
    """Interface para extratores (ISP)."""

    @abstractmethod
    def extract(self, source: T) -> ProcessingResult[R]:
        """Extrai dados da fonte."""
        pass


class IRepository(ABC, Generic[T]):
    """Interface para repositórios (DIP)."""

    @abstractmethod
    def save(self, data: T, identifier: str) -> ProcessingResult[str]:
        """Salva dados."""
        pass

    @abstractmethod
    def load(self, identifier: str) -> ProcessingResult[T]:
        """Carrega dados."""
        pass

    @abstractmethod
    def exists(self, identifier: str) -> bool:
        """Verifica se dados existem."""
        pass

    @abstractmethod
    def delete(self, identifier: str) -> ProcessingResult[bool]:
        """Remove dados."""
        pass


class IConfigurable(ABC):
    """Interface para componentes configuráveis (ISP)."""

    @abstractmethod
    def configure(self, config: Dict[str, Any]) -> ProcessingResult[bool]:
        """Configura o componente."""
        pass

    @abstractmethod
    def get_config(self) -> Dict[str, Any]:
        """Retorna configuração atual."""
        pass


class ILoggable(ABC):
    """Interface para componentes que fazem logging (ISP)."""

    @abstractmethod
    def get_logger_name(self) -> str:
        """Retorna nome do logger."""
        pass


class IMonitorable(ABC):
    """Interface para componentes monitoráveis (ISP)."""

    @abstractmethod
    def get_metrics(self) -> Dict[str, Any]:
        """Retorna métricas do componente."""
        pass

    @abstractmethod
    def get_health_status(self) -> ProcessingResult[Dict[str, Any]]:
        """Retorna status de saúde."""
        pass
