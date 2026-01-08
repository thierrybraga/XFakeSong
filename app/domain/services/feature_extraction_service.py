"""Serviço de Extração de Características de Áudio

Implementa a lógica de negócio para extração de características de arquivos de áudio.
Refatorado para utilizar sub-módulos em app/domain/services/feature_extraction/
"""

import logging
from typing import List, Dict, Any, Optional, Union
import numpy as np

from app.core.interfaces.base import ProcessingResult, ProcessingStatus
from app.core.interfaces.audio import (
    AudioData, AudioFeatures, FeatureType, IFeatureExtractor
)
from app.core.interfaces.services import IFeatureExtractionService

from app.domain.services.feature_extraction.types import ExtractionConfig, ExtractionResult
from app.domain.services.feature_extraction.extractor_loader import ExtractorLoader
from app.domain.services.feature_extraction.validator import FeatureExtractionValidator
from app.domain.services.feature_extraction.core import FeatureExtractorCore
from app.domain.features.extractors.segmented_feature_extractor import SegmentedFeatureExtractor, SegmentedExtractionConfig

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
            self, audio_files: List[AudioData], config: ExtractionConfig) -> ProcessingResult:
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
                errors.append(
                    f"audio_data: {
                        result.errors[0] if result.errors else 'Erro desconhecido'}")

            progress.update(i + 1)

        if errors:
            return ProcessingResult(
                status=ProcessingStatus.PARTIAL_SUCCESS if results else ProcessingStatus.ERROR,
                data=results,
                errors=errors
            )

        return ProcessingResult(
            status=ProcessingStatus.SUCCESS,
            data=results
        )

    def extract_segmented_features(self, audio_data: AudioData, config: ExtractionConfig) -> AudioFeatures:
        """Extrai características segmentadas e agrega estatisticamente."""
        try:
            # Configuração
            seg_config = SegmentedExtractionConfig(
                target_sample_rate=audio_data.sample_rate if audio_data.sample_rate else 16000,
                extract_spectral=FeatureType.SPECTRAL in config.feature_types,
                extract_cepstral=FeatureType.CEPSTRAL in config.feature_types,
                extract_temporal=FeatureType.TEMPORAL in config.feature_types,
                extract_prosodic=FeatureType.PROSODIC in config.feature_types,
                extract_formant=FeatureType.FORMANT in config.feature_types,
                extract_voice_quality=FeatureType.VOICE_QUALITY in config.feature_types,
                extract_perceptual=FeatureType.PERCEPTUAL in config.feature_types,
                extract_complexity=FeatureType.COMPLEXITY in config.feature_types,
                extract_transform=False,
                extract_timefreq=False,
                extract_predictive=False,
                extract_speech=False
            )

            # Extração
            extractor = SegmentedFeatureExtractor(seg_config)
            result = extractor.extract(audio_data)

            if result.status != ProcessingStatus.SUCCESS or not result.data:
                # Retornar vazio em caso de falha ou sem dados
                return AudioFeatures(features={'combined_features': np.array([])}, feature_type=FeatureType.ADVANCED)

            segments = result.data  # List[SegmentedFeatures]
            
            # Agregar features
            # Stack all combined_features: shape (n_segments, n_features)
            feature_matrix = np.vstack([seg.combined_features for seg in segments])
            
            # Calcular estatísticas (média e desvio padrão)
            means = np.mean(feature_matrix, axis=0)
            stds = np.std(feature_matrix, axis=0)
            
            # Concatenar: [mean_1, ..., mean_n, std_1, ..., std_n]
            aggregated_features = np.concatenate([means, stds])
            
            return AudioFeatures(
                features={'combined_features': aggregated_features},
                feature_type=FeatureType.ADVANCED,
                extraction_params={'aggregated': True, 'stats': ['mean', 'std']}
            )

        except Exception as e:
            logger.error(f"Erro na extração segmentada: {e}")
            return AudioFeatures(features={'combined_features': np.array([])}, feature_type=FeatureType.ADVANCED)

    def extract_single(self, audio_data: AudioData,
                       feature_types: List[str]) -> ProcessingResult[Dict[str, AudioFeatures]]:
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
                        "features": result.data.features if result.data else None}
                )
            else:
                return result

        except Exception as e:
            return ProcessingResult(
                status=ProcessingStatus.ERROR,
                errors=[str(e)]
            )

    def extract_batch(self, audio_list: List[AudioData], feature_types: List[str]
                      ) -> ProcessingResult[List[Dict[str, AudioFeatures]]]:
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
            self, audio_data: AudioData, config: ExtractionConfig) -> AudioFeatures:
        """Extrai features usando segmentação de áudio para compatibilidade com modelos treinados."""
        return self.core.extract_segmented_features(audio_data, config)

    # Métodos privados movidos para sub-módulos, mas mantidos se necessários por herança (improvável)
    # _register_extractors -> loader
    # _validate_pipeline -> validator
    # _validate_audio_data -> core
    # _extract_all_features -> core
    # _load_plugins -> loader
