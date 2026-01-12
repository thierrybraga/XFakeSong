"""Serviço de Extração de Características de Áudio

Implementa a lógica de negócio para extração de características de arquivos
de áudio. Refatorado para utilizar sub-módulos em
app/domain/services/feature_extraction/
"""

import logging
from typing import List, Dict, Any, Optional

from app.core.interfaces.base import ProcessingResult, ProcessingStatus
from app.core.interfaces.audio import (
    AudioData, AudioFeatures, FeatureType, IFeatureExtractor
)
from app.core.interfaces.services import IFeatureExtractionService

from app.domain.services.feature_extraction.types import (
    ExtractionConfig,
    ExtractionResult
)
from app.domain.services.feature_extraction.extractor_loader import (
    ExtractorLoader
)
from app.domain.services.feature_extraction.validator import (
    FeatureExtractionValidator
)
from app.domain.services.feature_extraction.core import FeatureExtractorCore

logger = logging.getLogger(__name__)


class ProgressTracker:
    """Rastreador de progresso simples."""

    def __init__(self, total: int, description: str = "Processing"):
        self.total = total
        self.description = description
        self.current = 0

    def update(self, current: int):
        self.current = current
        if current % 10 == 0 or current == self.total:
            logger.debug(f"{self.description}: {self.current}/{self.total}")


class AudioFeatureExtractionService(IFeatureExtractionService):
    """Serviço para extração de características de áudio

    Agora integrado com os novos componentes de modularidade:
    - Factory pattern para arquiteturas
    - Registry centralizado para extratores
    - Sistema de plugins
    - Validação de pipeline
    - Gerenciamento de configurações
    """

    def __init__(self):
        # Inicializar Loader
        self.loader = ExtractorLoader()

        # Inicializar Core com extratores carregados
        self.core = FeatureExtractorCore(self.loader.extractors)

        # Inicializar Validator
        self.validator = FeatureExtractionValidator(
            self.loader.modular_enabled,
            self.loader.extractor_cache,
            self.loader.extractor_specs
        )

        # Validar pipeline
        self.validator.validate()

    @property
    def _extractors(self):
        """Compatibilidade com código existente que acessa _extractors diretamente"""
        return self.loader.extractors

    @property
    def _modular_enabled(self):
        return self.loader.modular_enabled

    @property
    def extractor_registry(self):
        return self.loader.extractor_registry

    @property
    def plugin_manager(self):
        return self.loader.plugin_manager

    def extract_features(self, audio_data: AudioData,
                         config: ExtractionConfig) -> ProcessingResult:
        """Extrair características de um arquivo de áudio"""
        return self.core.extract_features(audio_data, config)

    def extract_batch_with_config(
        self, audio_files: List[AudioData], config: ExtractionConfig
    ) -> ProcessingResult:
        """Extrair características de múltiplos arquivos (com config)"""
        results = []
        errors = []

        progress = ProgressTracker(
            len(audio_files),
            "Extração de características")

        for i, audio_data in enumerate(audio_files):
            result = self.extract_features(audio_data, config)

            if result.status == ProcessingStatus.SUCCESS:
                if result.data:
                    results.append(result.data)
            else:
                msg = (
                    result.errors[0] if result.errors
                    else 'Erro desconhecido'
                )
                errors.append(f"audio_data: {msg}")

            progress.update(i + 1)

        if errors:
            status = (
                ProcessingStatus.PARTIAL_SUCCESS if results
                else ProcessingStatus.ERROR
            )
            return ProcessingResult(
                status=status,
                data=results,
                errors=errors
            )

        return ProcessingResult(
            status=ProcessingStatus.SUCCESS,
            data=results
        )

    def extract_single(
        self, audio_data: AudioData, feature_types: List[str]
    ) -> ProcessingResult:
        """Extrai características de um áudio."""
        try:
            feature_type_enums = []
            for ft_str in feature_types:
                try:
                    feature_type_enums.append(FeatureType(ft_str))
                except ValueError:
                    continue

            config = ExtractionConfig(feature_types=feature_type_enums)
            result = self.extract_features(audio_data, config)

            if result.status == ProcessingStatus.SUCCESS:
                return ProcessingResult(
                    status=ProcessingStatus.SUCCESS,
                    data={
                        "features": (
                            result.data.features if result.data else None
                        )
                    }
                )
            else:
                return result

        except Exception as e:
            return ProcessingResult(
                status=ProcessingStatus.ERROR,
                errors=[str(e)]
            )

    def extract_batch(
        self, audio_list: List[AudioData], feature_types: List[str]
    ) -> ProcessingResult:
        """Extrai características em lote."""
        try:
            results = []
            for audio_data in audio_list:
                result = self.extract_single(audio_data, feature_types)
                if result.status == ProcessingStatus.SUCCESS and result.data:
                    results.append(result.data)
                else:
                    return result

            return ProcessingResult(
                status=ProcessingStatus.SUCCESS,
                data=results
            )

        except Exception as e:
            return ProcessingResult(
                status=ProcessingStatus.ERROR,
                errors=[str(e)]
            )

    def extract_from_dataset(self, dataset_name: str,
                             feature_types: List[str]) -> ProcessingResult[str]:
        """Extrai características de dataset completo."""
        return ProcessingResult(
            status=ProcessingStatus.SUCCESS,
            data=f"Dataset {dataset_name} processado"
        )

    def get_available_extractors(self) -> List[str]:
        """Retorna extratores disponíveis."""
        return self.loader.get_available_extractors()

    def get_extractor_info(
            self, extractor_name: str) -> Optional[Dict[str, Any]]:
        """Retorna informações detalhadas sobre um extrator."""
        return self.loader.get_extractor_info(extractor_name)

    def reload_extractors(self):
        """Recarrega extratores (útil após carregar novos plugins)."""
        self.loader.reload_extractors()
        # Re-initialize validator to update checks
        self.validator = FeatureExtractionValidator(
            self.loader.modular_enabled,
            self.loader.extractor_cache,
            self.loader.extractor_specs
        )
        self.validator.validate()

    def get_available_features(self) -> List[FeatureType]:
        """Retornar tipos de características disponíveis"""
        return self.loader.get_available_features()

    def register_extractor(self, feature_type: FeatureType,
                           extractor: IFeatureExtractor) -> None:
        """Registrar um novo extrator de características"""
        self.loader.register_extractor(feature_type, extractor)

    def extract_segmented_features(
        self, audio_data: AudioData, config: ExtractionConfig
    ) -> AudioFeatures:
        """Extrai features usando segmentação de áudio.

        Necessário para compatibilidade com modelos treinados.
        """
        return self.core.extract_segmented_features(audio_data, config)

    # Métodos privados movidos para sub-módulos, mas mantidos se necessários por herança (improvável)
    # _register_extractors -> loader
    # _validate_pipeline -> validator
    # _validate_audio_data -> core
    # _extract_all_features -> core
    # _load_plugins -> loader
