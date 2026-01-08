import logging
from typing import Dict, List, Optional, Any
from app.core.interfaces.audio import FeatureType, IFeatureExtractor, AudioData, AudioFeatures
from app.core.interfaces.base import ProcessingResult
from app.domain.features.extractor_registry import ExtractorSpec

logger = logging.getLogger(__name__)

try:
    from app.domain.features.extractor_registry import get_extractor_registry
    from app.domain.services.plugin_system import get_plugin_manager
    MODULAR_COMPONENTS_AVAILABLE = True
except ImportError:
    MODULAR_COMPONENTS_AVAILABLE = False


class ExtractorLoader:
    """Responsável por carregar e registrar extratores."""

    def __init__(self):
        self._extractors: Dict[FeatureType, IFeatureExtractor] = {}
        self._extractor_cache: Dict[str, IFeatureExtractor] = {}
        self._extractor_specs: Dict[str, ExtractorSpec] = {}
        self._modular_enabled = False
        self.extractor_registry = None
        self.plugin_manager = None

        if MODULAR_COMPONENTS_AVAILABLE:
            try:
                self.extractor_registry = get_extractor_registry()
                self.plugin_manager = get_plugin_manager()
                self._modular_enabled = True
                self._load_plugins()
            except Exception as e:
                logger.warning(
                    f"Erro ao inicializar componentes modulares: {e}")
                self._modular_enabled = False

        self._register_extractors()

    @property
    def extractors(self) -> Dict[FeatureType, IFeatureExtractor]:
        return self._extractors

    @property
    def extractor_cache(self) -> Dict[str, IFeatureExtractor]:
        return self._extractor_cache

    @property
    def extractor_specs(self) -> Dict[str, ExtractorSpec]:
        return self._extractor_specs

    @property
    def modular_enabled(self) -> bool:
        return self._modular_enabled

    def get_available_extractors(self) -> List[str]:
        """Retorna extratores disponíveis."""
        extractors = [ft.value for ft in self._extractors.keys()]

        if self._modular_enabled and self.extractor_registry:
            try:
                modular_extractors = self.extractor_registry.list_extractors()
                extractors.extend(modular_extractors)
                extractors = list(set(extractors))
            except Exception as e:
                logger.warning(f"Erro ao obter extratores modulares: {e}")

        return extractors

    def get_extractor_info(
            self, extractor_name: str) -> Optional[Dict[str, Any]]:
        """Retorna informações detalhadas sobre um extrator."""
        if self._modular_enabled and extractor_name in self._extractor_specs:
            spec = self._extractor_specs[extractor_name]
            return {
                'name': spec.name,
                'description': spec.description,
                'feature_type': spec.feature_type,
                'dependencies': spec.dependencies,
                'is_cached': extractor_name in self._extractor_cache
            }
        return None

    def reload_extractors(self):
        """Recarrega extratores."""
        if self._modular_enabled:
            self._extractor_cache.clear()
            self._extractor_specs.clear()
            self._load_plugins()

        self._register_extractors()

    def get_available_features(self) -> List[FeatureType]:
        """Retornar tipos de características disponíveis"""
        return list(self._extractors.keys())

    def register_extractor(self, feature_type: FeatureType,
                           extractor: IFeatureExtractor) -> None:
        """Registrar um novo extrator de características"""
        self._extractors[feature_type] = extractor

    def _load_plugins(self):
        """Carrega plugins de extratores."""
        if not self._modular_enabled:
            return

        try:
            if self.plugin_manager:
                extractor_plugins = self.plugin_manager.get_plugins_by_type(
                    'extractor')
                for plugin in extractor_plugins:
                    logger.info(
                        f"Plugin de extrator carregado: {
                            plugin.metadata.name}")

                for plugin in extractor_plugins:
                    if hasattr(plugin, 'get_extractors'):
                        extractors = plugin.get_extractors()
                        for name, extractor_class in extractors.items():
                            if not self.extractor_registry.is_registered(name):
                                spec = ExtractorSpec(
                                    name=name,
                                    description=f"Extrator de plugin: {
                                        plugin.metadata.name}",
                                    extractor_class=extractor_class,
                                    feature_type=plugin.metadata.feature_types[
                                        0] if plugin.metadata.feature_types else "unknown",
                                    dependencies=plugin.metadata.dependencies
                                )
                                self.extractor_registry.register_extractor(
                                    name, spec)
                                logger.info(
                                    f"Extrator de plugin registrado: {name}")
        except Exception as e:
            logger.error(f"Erro ao carregar plugins: {e}")

    def _register_extractors(self) -> None:
        """Registrar extratores padrão"""
        # Importar e registrar extratores das diferentes categorias
        try:
            from app.domain.features.extractors.spectral.spectral_features import SpectralFeatureExtractor
            self._extractors[FeatureType.SPECTRAL] = SpectralFeatureExtractor()
            logger.debug(f"SpectralFeatureExtractor registrado com sucesso")

            from app.domain.features.extractors.mel.mel_spectrogram import MelSpectrogramExtractor
            self._extractors[FeatureType.MEL_SPECTROGRAM] = MelSpectrogramExtractor(
            )
            logger.debug(f"MelSpectrogramExtractor registrado com sucesso")

            if self._modular_enabled and self.extractor_registry:
                try:
                    spec = ExtractorSpec(
                        name="spectral",
                        description="Extrator de características espectrais",
                        extractor_class=SpectralFeatureExtractor,
                        feature_type="spectral",
                        dependencies=[]
                    )
                    self.extractor_registry.register_extractor(
                        "spectral", spec)
                    self._extractor_specs["spectral"] = spec
                    self._extractor_cache["spectral"] = SpectralFeatureExtractor(
                    )
                except Exception as e:
                    logger.warning(f"Erro ao registrar no novo sistema: {e}")

        except ImportError as e:
            logger.warning(f"Could not import SpectralFeatureExtractor: {e}")

        try:
            from app.domain.features.extractors.temporal.temporal_features import TemporalFeatureExtractor
            from app.domain.features.adapters.temporal_adapter import TemporalExtractorWrapper

            self._extractors[FeatureType.TEMPORAL] = TemporalExtractorWrapper()
            logger.debug(f"TemporalFeatureExtractor registrado com sucesso")

            if self._modular_enabled and self.extractor_registry:
                try:
                    spec = ExtractorSpec(
                        name="temporal",
                        description="Extrator de características temporais",
                        extractor_class=TemporalFeatureExtractor,
                        feature_type="temporal",
                        dependencies=[]
                    )
                    self.extractor_registry.register_extractor(
                        "temporal", spec)
                    self._extractor_specs["temporal"] = spec
                    self._extractor_cache["temporal"] = TemporalFeatureExtractor(
                    )
                except Exception as e:
                    logger.warning(f"Erro ao registrar no novo sistema: {e}")

        except ImportError as e:
            logger.warning(f"Could not import TemporalFeatureExtractor: {e}")

        try:
            from app.domain.features.extractors.prosodic.prosodic_features import ProsodicFeatureExtractor
            from app.domain.features.extractors.predictive.predictive_features import PredictiveFeatureExtractor
            from app.domain.features.extractors.timefreq.timefreq_features import TimeFrequencyFeatureExtractor
            from app.domain.features.adapters.prosodic_adapter import ProsodicExtractorWrapper

            self._extractors[FeatureType.PROSODIC] = ProsodicExtractorWrapper()
            logger.debug(f"ProsodicFeatureExtractor registrado com sucesso")

            self._extractors[FeatureType.ADVANCED] = PredictiveFeatureExtractor(
                sr=22050)
            self._extractors[FeatureType.PERCEPTUAL] = TimeFrequencyFeatureExtractor(
                sr=22050)
            logger.debug(
                f"PredictiveFeatureExtractor e TimeFrequencyFeatureExtractor registrados com sucesso")

            if self._modular_enabled and self.extractor_registry:
                try:
                    spec = ExtractorSpec(
                        name="prosodic",
                        description="Extrator de características prosódicas",
                        extractor_class=ProsodicFeatureExtractor,
                        feature_type="prosodic",
                        dependencies=[]
                    )
                    self.extractor_registry.register_extractor(
                        "prosodic", spec)
                    self._extractor_specs["prosodic"] = spec
                    self._extractor_cache["prosodic"] = ProsodicFeatureExtractor(
                    )
                except Exception as e:
                    logger.warning(f"Erro ao registrar no novo sistema: {e}")

            from app.domain.features.adapters.formant_adapter import FormantExtractorWrapper
            self.register_extractor("formant", FormantExtractorWrapper())

            from app.domain.features.extractors.voice_quality.voice_quality_features import VoiceQualityFeatureExtractor
            import numpy as np
            from app.core.interfaces.base import ProcessingStatus

            class VoiceQualityExtractorWrapper(IFeatureExtractor):
                def __init__(self):
                    self.extractor = VoiceQualityFeatureExtractor(sr=22050)

                def extract(self, audio_data: AudioData) -> ProcessingResult:
                    try:
                        features_dict = self.extractor.extract_features(
                            audio_data.samples)
                        normalized_features = {}
                        for key, value in features_dict.items():
                            if isinstance(value, (int, float)):
                                normalized_features[key] = np.array([value])
                            elif isinstance(value, np.ndarray):
                                normalized_features[key] = value.flatten()
                            else:
                                normalized_features[key] = np.array(
                                    [float(value)])

                        audio_features = AudioFeatures(
                            features=normalized_features,
                            feature_type=FeatureType.PROSODIC,
                            extraction_params=self.get_extraction_params(),
                            audio_metadata={"extractor": "voice_quality"}
                        )
                        return ProcessingResult(
                            status=ProcessingStatus.SUCCESS, data=audio_features)
                    except Exception as e:
                        return ProcessingResult(status=ProcessingStatus.ERROR, errors=[
                                                f"Erro na extração de qualidade vocal: {str(e)}"])

                def extract_features(
                        self, audio_data: AudioData) -> ProcessingResult:
                    return self.extract(audio_data)

                def get_feature_type(self) -> FeatureType:
                    return FeatureType.PROSODIC

                def get_feature_names(self) -> List[str]:
                    return self.extractor.get_feature_names()

                def get_extraction_params(self) -> Dict[str, Any]:
                    return {"sr": self.extractor.sr, "f0_min": self.extractor.f0_min,
                            "f0_max": self.extractor.f0_max}

            self.register_extractor(
                "voice_quality", VoiceQualityExtractorWrapper())

            from app.domain.features.adapters.perceptual_adapter import PerceptualExtractorWrapper
            self.register_extractor("perceptual", PerceptualExtractorWrapper())

            from app.domain.features.adapters.complexity_adapter import ComplexityExtractorWrapper
            self.register_extractor("complexity", ComplexityExtractorWrapper())

            from app.domain.features.adapters.transform_adapter import TransformExtractorWrapper
            self.register_extractor("transform", TransformExtractorWrapper())

            from app.domain.features.adapters.speech_adapter import SpeechExtractorWrapper
            self.register_extractor("speech", SpeechExtractorWrapper())

            from app.domain.features.extractors.cepstral.cepstral_features import CepstralFeatureExtractor

            class CepstralExtractorWrapper:
                def __init__(self):
                    self.extractor = CepstralFeatureExtractor(sr=22050)

                def extract(self, audio_data):
                    return self.extractor.extract(audio_data)

                def get_extraction_params(self):
                    return self.extractor.get_extraction_params()

            self.register_extractor("cepstral", CepstralExtractorWrapper())

            class PredictiveExtractorWrapper:
                def __init__(self):
                    self.extractor = PredictiveFeatureExtractor(sr=22050)

                def extract(self, audio_data):
                    try:
                        features_dict = self.extractor.extract_features(
                            audio_data.samples)
                        features_list = []
                        for key, value in features_dict.items():
                            if isinstance(value, np.ndarray):
                                if value.ndim > 1:
                                    features_list.extend(value.flatten())
                                else:
                                    features_list.extend(value)
                            else:
                                features_list.append(float(value))
                        features_array = np.array(features_list)
                        audio_features = AudioFeatures(
                            features={"predictive_features": features_array},
                            feature_type=FeatureType.ADVANCED,
                            extraction_params={"sr": 22050}
                        )
                        return ProcessingResult(
                            status=ProcessingStatus.SUCCESS, data=audio_features)
                    except Exception as e:
                        return ProcessingResult(
                            status=ProcessingStatus.ERROR, errors=[str(e)])

            self.register_extractor(
                FeatureType.ADVANCED,
                PredictiveExtractorWrapper())

            from app.domain.features.adapters.timefreq_adapter import TimeFreqExtractorWrapper
            self.register_extractor(
                FeatureType.PERCEPTUAL,
                TimeFreqExtractorWrapper())

        except ImportError as e:
            logger.warning(f"Could not import ProsodicFeatureExtractor: {e}")

        logger.debug(
            f"Total de extratores registrados: {len(self._extractors)}")
