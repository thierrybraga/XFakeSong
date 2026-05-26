import logging
from typing import Any, Dict, List, Optional

from app.core.interfaces.audio import (
    AudioData,
    AudioFeatures,
    FeatureType,
    IFeatureExtractor,
)
from app.core.interfaces.base import ProcessingResult
from app.domain.features.extractor_registry import ExtractorComplexity, ExtractorSpec

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
                        f"Plugin de extrator carregado: {plugin.metadata.name}")

                for plugin in extractor_plugins:
                    if hasattr(plugin, 'get_extractors'):
                        extractors = plugin.get_extractors()
                        for name, extractor_class in extractors.items():
                            if not self.extractor_registry.is_registered(name):
                                spec = ExtractorSpec(
                                    name=name,
                                    description=f"Extrator de plugin: {plugin.metadata.name}",
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
        """Registrar extratores usando exclusivamente chaves FeatureType enum."""
        _pairs = [
            (FeatureType.SPECTRAL,
             "app.domain.features.extractors.spectral.spectral_features",
             "SpectralFeatureExtractor"),
            (FeatureType.MEL_SPECTROGRAM,
             "app.domain.features.extractors.mel.mel_spectrogram",
             "MelSpectrogramExtractor"),
            (FeatureType.TEMPORAL,
             "app.domain.features.adapters.temporal_adapter",
             "TemporalExtractorWrapper"),
            (FeatureType.PROSODIC,
             "app.domain.features.adapters.prosodic_adapter",
             "ProsodicExtractorWrapper"),
            (FeatureType.CEPSTRAL,
             "app.domain.features.adapters.cepstral_adapter",
             "CepstralExtractorWrapper"),
            (FeatureType.FORMANT,
             "app.domain.features.adapters.formant_adapter",
             "FormantExtractorWrapper"),
            (FeatureType.VOICE_QUALITY,
             "app.domain.features.adapters.voice_quality_adapter",
             "VoiceQualityExtractorWrapper"),
            (FeatureType.PERCEPTUAL,
             "app.domain.features.adapters.perceptual_adapter",
             "PerceptualExtractorWrapper"),
            (FeatureType.COMPLEXITY,
             "app.domain.features.adapters.complexity_adapter",
             "ComplexityExtractorWrapper"),
            (FeatureType.ADVANCED,
             "app.domain.features.adapters.predictive_adapter",
             "PredictiveExtractorWrapper"),
        ]

        import importlib
        for feature_type, module_path, class_name in _pairs:
            try:
                mod = importlib.import_module(module_path)
                cls = getattr(mod, class_name)
                self._extractors[feature_type] = cls()
                logger.debug(f"{class_name} registrado para {feature_type}")
            except (ImportError, AttributeError) as e:
                logger.warning(f"{class_name} não disponível: {e}")

        if self._modular_enabled and self.extractor_registry:
            self._register_modular_specs()

        logger.debug(f"Total de extratores registrados: {len(self._extractors)}")

    def _register_modular_specs(self) -> None:
        """Registra specs no registry modular para descoberta de extratores."""
        for feature_type, extractor in self._extractors.items():
            name = feature_type.value
            try:
                spec = ExtractorSpec(
                    name=name,
                    description=f"Extrator de features {name}",
                    extractor_class=type(extractor),
                    feature_type=feature_type,
                    complexity=ExtractorComplexity.MEDIUM,
                    default_params={},
                    input_requirements={"sample_rate": 16000},
                    dependencies=[]
                )
                if not self.extractor_registry.is_registered(name):
                    self.extractor_registry.register(spec)
                self._extractor_specs[name] = spec
                self._extractor_cache[name] = extractor
            except Exception as e:
                logger.warning(f"Erro ao registrar spec para {feature_type}: {e}")
