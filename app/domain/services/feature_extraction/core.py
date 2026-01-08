import logging
import time
import numpy as np
from typing import Dict, List, Any
from app.core.interfaces.base import ProcessingResult, ProcessingStatus
from app.core.interfaces.audio import AudioData, AudioFeatures, FeatureType
from .types import ExtractionConfig, ExtractionResult

logger = logging.getLogger(__name__)


class FeatureExtractorCore:
    """Núcleo de extração de características."""

    def __init__(self, extractors: Dict[FeatureType, Any]):
        self.extractors = extractors

    def extract_features(self, audio_data: AudioData,
                         config: ExtractionConfig) -> ProcessingResult:
        """Extrair características de um arquivo de áudio"""
        try:
            start_time = time.time()

            # Validar dados de áudio
            validation_result = self._validate_audio_data(audio_data)
            if validation_result.status != ProcessingStatus.SUCCESS:
                return validation_result

            # Extrair características
            features = self._extract_all_features(audio_data, config)

            extraction_time = time.time() - start_time

            # Verificar se features foi extraído corretamente
            if features is None:
                return ProcessingResult(
                    status=ProcessingStatus.ERROR,
                    errors=["Falha na extração de características: features é None"]
                )

            # Verificar se features.features existe e não é None
            feature_shape = (0,)
            if isinstance(features, AudioFeatures):
                if features.features is not None and isinstance(
                        features.features, dict):
                    # Calcular shape baseado nas features disponíveis
                    total_features = 0
                    for feature_array in features.features.values():
                        if hasattr(feature_array, 'shape'):
                            total_features += feature_array.shape[0] if len(
                                feature_array.shape) > 0 else 1
                    feature_shape = (total_features,)

            result = ExtractionResult(
                file_path="unknown",
                features=features,
                extraction_time=extraction_time,
                feature_shape=feature_shape,
                metadata={
                    'sample_rate': audio_data.sample_rate,
                    'duration': audio_data.duration,
                    'channels': audio_data.channels,
                    'feature_types': [ft.value for ft in config.feature_types]
                }
            )

            return ProcessingResult(
                status=ProcessingStatus.SUCCESS,
                data=result,
                execution_time=extraction_time
            )

        except Exception as e:
            return ProcessingResult(
                status=ProcessingStatus.ERROR,
                errors=[str(e)]
            )

    def _validate_audio_data(self, audio_data: AudioData) -> ProcessingResult:
        """Validar dados de áudio"""
        if audio_data.samples is None or len(audio_data.samples) == 0:
            return ProcessingResult(status=ProcessingStatus.ERROR, errors=[
                                    "Dados de áudio vazios"])
        if audio_data.sample_rate <= 0:
            return ProcessingResult(status=ProcessingStatus.ERROR, errors=[
                                    "Taxa de amostragem inválida"])
        if audio_data.duration <= 0:
            return ProcessingResult(status=ProcessingStatus.ERROR, errors=[
                                    "Duração de áudio inválida"])
        return ProcessingResult(
            status=ProcessingStatus.SUCCESS, data=audio_data)

    def _extract_all_features(self, audio_data: AudioData,
                              config: ExtractionConfig) -> AudioFeatures:
        """Extrair todas as características solicitadas"""
        all_features = []
        feature_names = []
        features_dict = {}

        logger.debug(f"Extratores disponíveis: {list(self.extractors.keys())}")
        logger.debug(f"Tipos solicitados: {config.feature_types}")

        # Extrair features dos tipos solicitados (FeatureType enum)
        for feature_type in config.feature_types:
            if feature_type in self.extractors:
                extractor = self.extractors[feature_type]
                logger.debug(f"Extraindo {feature_type} com {type(extractor)}")

                result = extractor.extract(audio_data)

                if result.status == ProcessingStatus.SUCCESS:
                    features = result.data
                    if isinstance(features, AudioFeatures):
                        if features.features:
                            for key, feature_array in features.features.items():
                                if isinstance(feature_array, np.ndarray):
                                    if feature_array.ndim > 1:
                                        all_features.append(
                                            feature_array.flatten())
                                    else:
                                        all_features.append(feature_array)
                                else:
                                    all_features.append(
                                        np.array([feature_array]))

                                name = f"{key}"
                                feature_names.append(name)
                                features_dict[name] = feature_array
                    else:
                        if isinstance(features, np.ndarray):
                            if features.ndim > 1:
                                all_features.append(features.flatten())
                            else:
                                all_features.append(features)
                        else:
                            all_features.append(np.array([features]))

                        name = feature_type.value if hasattr(
                            feature_type, 'value') else str(feature_type)
                        feature_names.append(name)
                        features_dict[name] = features
                else:
                    logger.debug(
                        f"Erro na extração de {feature_type}: {
                            result.errors}")
            else:
                logger.debug(f"Extrator não encontrado para {feature_type}")

        # Extrair features dos extratores do sistema antigo (strings)
        legacy_extractors = [
            'formant',
            'voice_quality',
            'perceptual',
            'complexity',
            'transform',
            'speech',
            'cepstral']
        for extractor_name in legacy_extractors:
            if extractor_name in self.extractors:
                extractor = self.extractors[extractor_name]
                try:
                    result = extractor.extract(audio_data)
                    if result.status == ProcessingStatus.SUCCESS:
                        features = result.data
                        if isinstance(features, AudioFeatures):
                            if features.features:
                                for key, feature_array in features.features.items():
                                    if isinstance(feature_array, np.ndarray):
                                        if feature_array.ndim > 1:
                                            all_features.append(
                                                feature_array.flatten())
                                        else:
                                            all_features.append(feature_array)
                                    else:
                                        all_features.append(
                                            np.array([feature_array]))
                                    name = f"{extractor_name}_{key}"
                                    feature_names.append(name)
                                    features_dict[name] = feature_array
                        else:
                            if isinstance(features, np.ndarray):
                                if features.ndim > 1:
                                    all_features.append(features.flatten())
                                else:
                                    all_features.append(features)
                            else:
                                all_features.append(np.array([features]))
                            all_features.append(features)
                            feature_names.append(extractor_name)
                            features_dict[extractor_name] = features
                except Exception as e:
                    logger.debug(
                        f"Exceção na extração de {extractor_name}: {e}")

        if all_features:
            try:
                combined_features = np.concatenate(all_features, axis=-1)
            except Exception as e:
                logger.debug(f"Erro na concatenação: {e}")
                combined_features = np.array([])
        else:
            combined_features = np.array([])

        features_dict['combined_features'] = combined_features
        features_dict['feature_names'] = feature_names

        return AudioFeatures(
            features=features_dict,
            feature_type=FeatureType.SPECTRAL,
            extraction_params={'sample_rate': audio_data.sample_rate},
            audio_metadata={'sample_rate': audio_data.sample_rate}
        )

    def extract_segmented_features(
            self, audio_data: AudioData, config: ExtractionConfig) -> AudioFeatures:
        """Extrai features usando segmentação de áudio para compatibilidade com modelos treinados."""
        try:
            from app.domain.features.extractors.segmented_feature_extractor import (
                SegmentedFeatureExtractor, SegmentedExtractionConfig
            )

            segmented_config = SegmentedExtractionConfig(
                segment_duration=1.0,
                overlap_ratio=0.0,
                target_sample_rate=audio_data.sample_rate,
                extract_spectral=True,
                extract_cepstral=True,
                extract_temporal=True,
                extract_prosodic=True,
                extract_formant=True,
                extract_voice_quality=True,
                extract_perceptual=True,
                extract_complexity=True,
                extract_transform=True,
                extract_timefreq=True,
                extract_predictive=True,
                extract_speech=True,
                normalize_segments=True,
                remove_silence=False,
                export_csv=False
            )

            segmented_extractor = SegmentedFeatureExtractor(segmented_config)
            segments = segmented_extractor._segment_audio(
                audio_data.samples, audio_data.sample_rate)

            all_segment_features = []
            for i, segment in enumerate(segments):
                segment_features = segmented_extractor._extract_segment_features(
                    segment, audio_data.sample_rate, i
                )
                if len(segment_features.combined_features) > 0:
                    all_segment_features.append(
                        segment_features.combined_features)

            if not all_segment_features:
                logger.warning("Nenhuma feature segmentada foi extraída")
                empty_features = AudioFeatures(
                    features={
                        'combined_features': np.array(
                            []), 'feature_names': []},
                    feature_type=FeatureType.SPECTRAL,
                    extraction_params={'sample_rate': audio_data.sample_rate},
                    audio_metadata={'sample_rate': audio_data.sample_rate}
                )
                # Return AudioFeatures wrapped in result
                return empty_features

            import pandas as pd
            df = pd.DataFrame(all_segment_features)

            # Aggregation logic based on config.aggregate_method
            method = getattr(config, 'aggregate_method', 'mean')

            if method == 'mean':
                aggregated_features = df.mean().fillna(0).values
            elif method == 'median':
                aggregated_features = df.median().fillna(0).values
            elif method == 'std':
                aggregated_features = df.std().fillna(0).values
            elif method == 'all':
                mean_vals = df.mean().fillna(0).values
                std_vals = df.std().fillna(0).values
                min_vals = df.min().fillna(0).values
                max_vals = df.max().fillna(0).values
                aggregated_features = np.concatenate(
                    [mean_vals, std_vals, min_vals, max_vals])
            else:
                logger.warning(
                    f"Método de agregação desconhecido: {method}, usando mean")
                aggregated_features = df.mean().fillna(0).values

            feature_names = [
                f"segmented_feature_{i}" for i in range(
                    len(aggregated_features))]

            return AudioFeatures(
                features={
                    'combined_features': aggregated_features,
                    'feature_names': feature_names},
                feature_type=FeatureType.SPECTRAL,
                extraction_params={
                    'sample_rate': audio_data.sample_rate,
                    'method': 'segmented'},
                audio_metadata={'sample_rate': audio_data.sample_rate}
            )
        except ImportError as e:
            logger.error(f"Erro ao importar SegmentedFeatureExtractor: {e}")
            raise
