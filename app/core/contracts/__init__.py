"""Contratos SOLID do sistema (inversão de dependência).

Cada interface aqui tem ao menos uma implementação no domínio:

- ``IFeatureExtractor`` — extratores em ``app/domain/features/``;
- ``IModelTrainer`` — ``app.domain.models.training.trainer.ModelTrainer``;
- ``IDetectionService``, ``ITrainingService``, ``IUploadService``,
  ``IFeatureExtractionService`` — serviços em ``app/domain/services/``.

Contratos sem implementação foram removidos em 2026-10-18 (YAGNI); crie o
contrato junto com a primeira implementação. Importe do submódulo
(``from app.core.contracts.base import ProcessingResult``).
"""
