"""Camada de Aplicação - Casos de uso e orquestração"""

# Importar orquestrador do pipeline
from .pipeline.orchestrator import (
    DeepfakePipelineOrchestrator, PipelineConfig, PipelineResult,
    PipelineStage, UploadStage, FeatureExtractionStage, TrainingStage
)

__version__ = "1.0.0"

__all__ = [
    "DeepfakePipelineOrchestrator", "PipelineConfig", "PipelineResult",
    "PipelineStage", "UploadStage", "FeatureExtractionStage", "TrainingStage"
]
