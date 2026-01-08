"""Exceções do sistema"""

from .base import (
    DeepFakeSystemError, ValidationError, ConfigurationError,
    FileError, AudioError, FeatureExtractionError,
    ModelError, DatasetError, APIError, PipelineError
)

__all__ = [
    "DeepFakeSystemError", "ValidationError", "ConfigurationError",
    "FileError", "AudioError", "FeatureExtractionError",
    "ModelError", "DatasetError", "APIError", "PipelineError"
]
