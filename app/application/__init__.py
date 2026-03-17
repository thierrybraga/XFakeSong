"""Camada de Aplicação - Casos de uso e orquestração"""

# Importar orquestrador do pipeline
from .pipeline.orchestrator import (
    DeepfakePipelineOrchestrator,
    FeatureExtractionStage,
    PipelineConfig,
    PipelineResult,
    PipelineStage,
    TrainingStage,
    UploadStage,
)

__version__ = "1.0.0"

__all__ = [
    "DeepfakePipelineOrchestrator", "PipelineConfig", "PipelineResult",
    "PipelineStage", "UploadStage", "FeatureExtractionStage", "TrainingStage"
]
