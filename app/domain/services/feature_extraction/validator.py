import logging
from typing import Dict, Any, List
try:
    from app.domain.services.pipeline_validator import validate_pipeline
except ImportError:
    validate_pipeline = None

logger = logging.getLogger(__name__)


class FeatureExtractionValidator:
    """Valida o pipeline de extração."""

    def __init__(self, modular_enabled: bool,
                 extractor_cache: Dict[str, Any], extractor_specs: Dict[str, Any]):
        self.modular_enabled = modular_enabled
        self.extractor_cache = extractor_cache
        self.extractor_specs = extractor_specs

    def validate(self):
        """Valida o pipeline de extração."""
        if not self.modular_enabled or validate_pipeline is None:
            return

        try:
            extractors_for_validation = []
            for name, extractor in self.extractor_cache.items():
                if name in self.extractor_specs:
                    extractors_for_validation.append(
                        (extractor, self.extractor_specs[name]))

            if extractors_for_validation:
                validation_result = validate_pipeline(
                    extractors_for_validation, [])

                if not validation_result.is_valid:
                    logger.warning("Pipeline possui problemas de validação:")
                    for issue in validation_result.issues:
                        logger.warning(
                            f"  - {issue['component']}: {issue['message']}")

                if hasattr(validation_result,
                           'warnings') and validation_result.warnings:
                    for warning in validation_result.warnings:
                        logger.info(f"Aviso: {warning}")

                if hasattr(
                        validation_result, 'performance_score') and validation_result.performance_score is not None:
                    logger.info(
                        f"Score de performance: {
                            validation_result.performance_score:.1f}/100")

        except Exception as e:
            logger.error(f"Erro na validação do pipeline: {e}")
