"""Exceções base do sistema

Este módulo define a hierarquia de exceções seguindo boas práticas:
- Exceções específicas por domínio
- Informações detalhadas para debugging
- Códigos de erro padronizados
"""

from typing import Dict, Any, Optional, List
from enum import Enum


class ErrorCode(Enum):
    """Códigos de erro padronizados."""
    # Erros gerais (1000-1999)
    UNKNOWN_ERROR = 1000
    VALIDATION_ERROR = 1001
    CONFIGURATION_ERROR = 1002
    DEPENDENCY_ERROR = 1003

    # Erros de arquivo/IO (2000-2999)
    FILE_NOT_FOUND = 2000
    FILE_ACCESS_ERROR = 2001
    FILE_FORMAT_ERROR = 2002
    FILE_SIZE_ERROR = 2003
    DIRECTORY_ERROR = 2004

    # Erros de áudio (3000-3999)
    AUDIO_LOAD_ERROR = 3000
    AUDIO_FORMAT_ERROR = 3001
    AUDIO_QUALITY_ERROR = 3002
    AUDIO_DURATION_ERROR = 3003
    AUDIO_SAMPLE_RATE_ERROR = 3004

    # Erros de características (4000-4999)
    FEATURE_EXTRACTION_ERROR = 4000
    FEATURE_SAVE_ERROR = 4001
    FEATURE_LOAD_ERROR = 4002
    FEATURE_VALIDATION_ERROR = 4003

    # Erros de modelo/treinamento (5000-5999)
    MODEL_LOAD_ERROR = 5000
    MODEL_SAVE_ERROR = 5001
    MODEL_TRAINING_ERROR = 5002
    MODEL_PREDICTION_ERROR = 5003
    MODEL_ARCHITECTURE_ERROR = 5004

    # Erros de dataset (6000-6999)
    DATASET_NOT_FOUND = 6000
    DATASET_INVALID_STRUCTURE = 6001
    DATASET_EMPTY = 6002
    DATASET_CORRUPTED = 6003

    # Erros de API (7000-7999)
    API_REQUEST_ERROR = 7000
    API_AUTHENTICATION_ERROR = 7001
    API_AUTHORIZATION_ERROR = 7002
    API_RATE_LIMIT_ERROR = 7003
    API_TIMEOUT_ERROR = 7004

    # Erros de pipeline (8000-8999)
    PIPELINE_STAGE_ERROR = 8000
    PIPELINE_DEPENDENCY_ERROR = 8001
    PIPELINE_TIMEOUT_ERROR = 8002
    PIPELINE_RESOURCE_ERROR = 8003


class DeepFakeSystemError(Exception):
    """Exceção base do sistema XfakeSong."""

    def __init__(
        self,
        message: str,
        error_code: ErrorCode = ErrorCode.UNKNOWN_ERROR,
        details: Optional[Dict[str, Any]] = None,
        cause: Optional[Exception] = None
    ):
        super().__init__(message)
        self.message = message
        self.error_code = error_code
        self.details = details or {}
        self.cause = cause

    def __str__(self) -> str:
        return f"[{self.error_code.name}] {self.message}"

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(message='{self.message}', error_code={self.error_code}, details={self.details})"

    def to_dict(self) -> Dict[str, Any]:
        """Converte exceção para dicionário."""
        return {
            "error_type": self.__class__.__name__,
            "message": self.message,
            "error_code": self.error_code.value,
            "error_name": self.error_code.name,
            "details": self.details,
            "cause": str(self.cause) if self.cause else None
        }


class ValidationError(DeepFakeSystemError):
    """Erro de validação."""

    def __init__(
        self,
        message: str,
        field: Optional[str] = None,
        value: Optional[Any] = None,
        expected: Optional[str] = None,
        **kwargs
    ):
        details = kwargs.get('details', {})
        if field:
            details['field'] = field
        if value is not None:
            details['value'] = str(value)
        if expected:
            details['expected'] = expected

        super().__init__(
            message,
            error_code=ErrorCode.VALIDATION_ERROR,
            details=details,
            **{k: v for k, v in kwargs.items() if k != 'details'}
        )


class ConfigurationError(DeepFakeSystemError):
    """Erro de configuração."""

    def __init__(self, message: str,
                 config_key: Optional[str] = None, **kwargs):
        details = kwargs.get('details', {})
        if config_key:
            details['config_key'] = config_key

        super().__init__(
            message,
            error_code=ErrorCode.CONFIGURATION_ERROR,
            details=details,
            **{k: v for k, v in kwargs.items() if k != 'details'}
        )


class FileError(DeepFakeSystemError):
    """Erro relacionado a arquivos."""

    def __init__(
        self,
        message: str,
        file_path: Optional[str] = None,
        error_code: ErrorCode = ErrorCode.FILE_NOT_FOUND,
        **kwargs
    ):
        details = kwargs.get('details', {})
        if file_path:
            details['file_path'] = str(file_path)

        super().__init__(
            message,
            error_code=error_code,
            details=details,
            **{k: v for k, v in kwargs.items() if k != 'details'}
        )


class AudioError(DeepFakeSystemError):
    """Erro relacionado a processamento de áudio."""

    def __init__(
        self,
        message: str,
        audio_file: Optional[str] = None,
        error_code: ErrorCode = ErrorCode.AUDIO_LOAD_ERROR,
        **kwargs
    ):
        details = kwargs.get('details', {})
        if audio_file:
            details['audio_file'] = str(audio_file)

        super().__init__(
            message,
            error_code=error_code,
            details=details,
            **{k: v for k, v in kwargs.items() if k != 'details'}
        )


class FeatureExtractionError(DeepFakeSystemError):
    """Erro relacionado a extração de características."""

    def __init__(
        self,
        message: str,
        feature_type: Optional[str] = None,
        extractor_name: Optional[str] = None,
        **kwargs
    ):
        details = kwargs.get('details', {})
        if feature_type:
            details['feature_type'] = feature_type
        if extractor_name:
            details['extractor_name'] = extractor_name

        super().__init__(
            message,
            error_code=ErrorCode.FEATURE_EXTRACTION_ERROR,
            details=details,
            **{k: v for k, v in kwargs.items() if k != 'details'}
        )


class ModelError(DeepFakeSystemError):
    """Erro relacionado a modelos."""

    def __init__(
        self,
        message: str,
        model_name: Optional[str] = None,
        architecture: Optional[str] = None,
        error_code: ErrorCode = ErrorCode.MODEL_LOAD_ERROR,
        **kwargs
    ):
        details = kwargs.get('details', {})
        if model_name:
            details['model_name'] = model_name
        if architecture:
            details['architecture'] = architecture

        super().__init__(
            message,
            error_code=error_code,
            details=details,
            **{k: v for k, v in kwargs.items() if k != 'details'}
        )


class DatasetError(DeepFakeSystemError):
    """Erro relacionado a datasets."""

    def __init__(
        self,
        message: str,
        dataset_name: Optional[str] = None,
        dataset_path: Optional[str] = None,
        error_code: ErrorCode = ErrorCode.DATASET_NOT_FOUND,
        **kwargs
    ):
        details = kwargs.get('details', {})
        if dataset_name:
            details['dataset_name'] = dataset_name
        if dataset_path:
            details['dataset_path'] = str(dataset_path)

        super().__init__(
            message,
            error_code=error_code,
            details=details,
            **{k: v for k, v in kwargs.items() if k != 'details'}
        )


class APIError(DeepFakeSystemError):
    """Erro relacionado a API."""

    def __init__(
        self,
        message: str,
        status_code: Optional[int] = None,
        endpoint: Optional[str] = None,
        error_code: ErrorCode = ErrorCode.API_REQUEST_ERROR,
        **kwargs
    ):
        details = kwargs.get('details', {})
        if status_code:
            details['status_code'] = status_code
        if endpoint:
            details['endpoint'] = endpoint

        super().__init__(
            message,
            error_code=error_code,
            details=details,
            **{k: v for k, v in kwargs.items() if k != 'details'}
        )


class PipelineError(DeepFakeSystemError):
    """Erro relacionado ao pipeline."""

    def __init__(
        self,
        message: str,
        stage_name: Optional[str] = None,
        pipeline_id: Optional[str] = None,
        error_code: ErrorCode = ErrorCode.PIPELINE_STAGE_ERROR,
        **kwargs
    ):
        details = kwargs.get('details', {})
        if stage_name:
            details['stage_name'] = stage_name
        if pipeline_id:
            details['pipeline_id'] = pipeline_id

        super().__init__(
            message,
            error_code=error_code,
            details=details,
            **{k: v for k, v in kwargs.items() if k != 'details'}
        )


# Funções utilitárias para tratamento de exceções
def handle_exception(func):
    """Decorator para tratamento padronizado de exceções."""
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except DeepFakeSystemError:
            # Re-raise exceções do sistema
            raise
        except Exception as e:
            # Converte exceções genéricas
            raise DeepFakeSystemError(
                f"Erro inesperado em {func.__name__}: {str(e)}",
                cause=e,
                details={
                    'function': func.__name__,
                    'args': str(args),
                    'kwargs': str(kwargs)}
            )
    return wrapper


def create_error_response(error: DeepFakeSystemError) -> Dict[str, Any]:
    """Cria resposta padronizada de erro."""
    return {
        "success": False,
        "error": error.to_dict(),
        "timestamp": None  # Será preenchido pelo handler
    }


def log_error(error: DeepFakeSystemError, logger=None):
    """Registra erro no log."""
    if logger:
        logger.error(f"{error.error_code.name}: {error.message}",
                     extra=error.details)
    else:
        print(f"ERROR [{error.error_code.name}]: {error.message}")
        if error.details:
            print(f"Details: {error.details}")
