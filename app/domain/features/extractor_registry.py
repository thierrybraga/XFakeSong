"""Registry centralizado para extratores de features.

Este módulo implementa um sistema de registry para extratores de features,
permitindo registro dinâmico e descoberta de diferentes tipos de extratores.
"""

from typing import Dict, List, Optional, Type, Any
from dataclasses import dataclass
from enum import Enum
import logging

# Import da interface
from app.domain.features.interfaces import IFeatureExtractor

# Imports dos extratores existentes (comentados temporariamente)
# from app.domain.models.features.spectral.cepstral_features import (
#     MFCCExtractor
# )
# from app.domain.models.features.spectral.transform_features import (
#     SpectralPhaseExtractor,
#     WaveletTransformExtractor,
#     DCTExtractor,
#     HilbertTransformExtractor
# )
# from app.domain.models.features.advanced.prosodic_features import (
#     PitchExtractor,
#     EnergyExtractor,
#     FormantExtractor
# )
# from app.domain.models.features.advanced.perceptual_features import (
#     SpectralCentroidExtractor,
#     SpectralRolloffExtractor,
#     ZeroCrossingRateExtractor,
#     ChromaExtractor
# )

# Imports dos novos extratores implementados
from app.domain.features.extractors.spectral.spectral_features import (
    SpectralFeatureExtractor,
)
from app.domain.features.extractors.cepstral.cepstral_features import (
    CepstralFeatureExtractor,
)
from app.domain.features.extractors.prosodic.prosodic_features import (
    ProsodicFeatureExtractor,
)
# from app.domain.features.extractors.prosodic.formant_features import (
#     FormantFeatureExtractor
# )
# from app.domain.features.extractors.prosodic.voice_quality_features import (
#     VoiceQualityFeatureExtractor
# )

logger = logging.getLogger(__name__)


class FeatureType(Enum):
    """Tipos de features suportados."""
    SPECTRAL = "spectral"
    CEPSTRAL = "cepstral"
    PROSODIC = "prosodic"
    PERCEPTUAL = "perceptual"
    TEMPORAL = "temporal"
    ADVANCED = "advanced"
    CUSTOM = "custom"


class ExtractorComplexity(Enum):
    """Níveis de complexidade computacional."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    VERY_HIGH = "very_high"


@dataclass
class ExtractorSpec:
    """Especificação de um extrator de features."""
    name: str
    feature_type: FeatureType
    complexity: ExtractorComplexity
    description: str
    extractor_class: Type
    default_params: Dict[str, Any]
    input_requirements: Dict[str, Any]  # sample_rate, channels, etc.
    output_shape: Optional[tuple] = None
    dependencies: List[str] = None
    version: str = "1.0.0"
    author: str = "DeepFake Detection System"

    def __post_init__(self):
        if self.dependencies is None:
            self.dependencies = []


# Interface IFeatureExtractor agora importada de
# app.domain.features.interfaces


class FeatureExtractorRegistry:
    """Registry centralizado para extratores de features."""

    def __init__(self):
        self._extractors: Dict[str, ExtractorSpec] = {}
        self._type_index: Dict[FeatureType, List[str]] = {}
        self._complexity_index: Dict[ExtractorComplexity, List[str]] = {}
        self._initialized = False

    def register(self, spec: ExtractorSpec) -> None:
        """Registra um extrator de features.

        Args:
            spec: Especificação do extrator
        """
        if spec.name in self._extractors:
            logger.warning(
                f"Extrator '{spec.name}' já registrado. Sobrescrevendo."
            )

        self._extractors[spec.name] = spec

        # Atualizar índices
        if spec.feature_type not in self._type_index:
            self._type_index[spec.feature_type] = []
        if spec.name not in self._type_index[spec.feature_type]:
            self._type_index[spec.feature_type].append(spec.name)

        if spec.complexity not in self._complexity_index:
            self._complexity_index[spec.complexity] = []
        if spec.name not in self._complexity_index[spec.complexity]:
            self._complexity_index[spec.complexity].append(spec.name)

        logger.info(f"Extrator '{spec.name}' registrado com sucesso")

    def get_extractor(self, name: str, **params) -> IFeatureExtractor:
        """Cria instância de um extrator.

        Args:
            name: Nome do extrator
            **params: Parâmetros para o extrator

        Returns:
            Instância do extrator
        """
        if name not in self._extractors:
            raise ValueError(f"Extrator '{name}' não encontrado")

        spec = self._extractors[name]

        # Mesclar parâmetros padrão com os fornecidos
        final_params = {**spec.default_params, **params}

        try:
            return spec.extractor_class(**final_params)
        except Exception as e:
            logger.error(f"Erro ao criar extrator '{name}': {e}")
            raise

    def list_extractors(
        self,
        feature_type: Optional[FeatureType] = None,
        complexity: Optional[ExtractorComplexity] = None
    ) -> List[str]:
        """Lista extratores disponíveis.

        Args:
            feature_type: Filtrar por tipo de feature
            complexity: Filtrar por complexidade

        Returns:
            Lista de nomes de extratores
        """
        extractors = list(self._extractors.keys())

        if feature_type:
            extractors = [
                name for name in extractors
                if name in self._type_index.get(feature_type, [])
            ]

        if complexity:
            extractors = [
                name for name in extractors
                if name in self._complexity_index.get(complexity, [])
            ]

        return sorted(extractors)

    def get_spec(self, name: str) -> ExtractorSpec:
        """Obtém especificação de um extrator.

        Args:
            name: Nome do extrator

        Returns:
            Especificação do extrator
        """
        if name not in self._extractors:
            raise ValueError(f"Extrator '{name}' não encontrado")
        return self._extractors[name]

    def get_extractors_by_type(self, feature_type: FeatureType) -> List[str]:
        """Obtém extratores por tipo de feature.

        Args:
            feature_type: Tipo de feature

        Returns:
            Lista de nomes de extratores
        """
        return self._type_index.get(feature_type, [])

    def get_extractors_by_complexity(
        self, complexity: ExtractorComplexity
    ) -> List[str]:
        """Obtém extratores por complexidade.

        Args:
            complexity: Nível de complexidade

        Returns:
            Lista de nomes de extratores
        """
        return self._complexity_index.get(complexity, [])

    def is_registered(self, name: str) -> bool:
        """Verifica se um extrator está registrado.

        Args:
            name: Nome do extrator

        Returns:
            True se registrado
        """
        return name in self._extractors

    def unregister(self, name: str) -> bool:
        """Remove um extrator do registry.

        Args:
            name: Nome do extrator

        Returns:
            True se removido com sucesso
        """
        if name not in self._extractors:
            return False

        spec = self._extractors[name]

        # Remover dos índices
        if spec.feature_type in self._type_index:
            if name in self._type_index[spec.feature_type]:
                self._type_index[spec.feature_type].remove(name)

        if spec.complexity in self._complexity_index:
            if name in self._complexity_index[spec.complexity]:
                self._complexity_index[spec.complexity].remove(name)

        del self._extractors[name]
        logger.info(f"Extrator '{name}' removido do registry")
        return True

    def get_stats(self) -> Dict[str, Any]:
        """Obtém estatísticas do registry.

        Returns:
            Dicionário com estatísticas
        """
        return {
            "total_extractors": len(self._extractors),
            "by_type": {
                ft.value: len(extractors)
                for ft, extractors in self._type_index.items()
            },
            "by_complexity": {
                comp.value: len(extractors)
                for comp, extractors in self._complexity_index.items()
            },
            "registered_extractors": list(self._extractors.keys())
        }


# Instância global do registry
extractor_registry = FeatureExtractorRegistry()


def get_extractor_registry() -> FeatureExtractorRegistry:
    """Retorna a instância global do registry."""
    return extractor_registry


def register_default_extractors():
    """Registra extratores padrão no registry."""

    # Extratores antigos comentados temporariamente
    # # Extratores Cepstrais
    # extractor_registry.register(ExtractorSpec(
    #     name="mfcc",
    #     feature_type=FeatureType.CEPSTRAL,
    #     complexity=ExtractorComplexity.MEDIUM,
    #     description="Mel-Frequency Cepstral Coefficients",
    #     extractor_class=MFCCExtractor,
    #     default_params={"n_mfcc": 13, "n_fft": 2048, "hop_length": 512},
    #     input_requirements={"sample_rate": 16000, "channels": 1},
    #     output_shape=(13,)
    # ))
    #
    # # Extratores Espectrais
    # extractor_registry.register(ExtractorSpec(
    #     name="spectral_phase",
    #     feature_type=FeatureType.SPECTRAL,
    #     complexity=ExtractorComplexity.LOW,
    #     description="Spectral Phase Features",
    #     extractor_class=SpectralPhaseExtractor,
    #     default_params={},
    #     input_requirements={"sample_rate": 16000}
    # ))
    #
    # extractor_registry.register(ExtractorSpec(
    #     name="wavelet_transform",
    #     feature_type=FeatureType.SPECTRAL,
    #     complexity=ExtractorComplexity.HIGH,
    #     description="Wavelet Transform Features",
    #     extractor_class=WaveletTransformExtractor,
    #     default_params={"wavelet": "db4", "levels": 5},
    #     input_requirements={"sample_rate": 16000}
    # ))
    #
    # extractor_registry.register(ExtractorSpec(
    #     name="dct",
    #     feature_type=FeatureType.SPECTRAL,
    #     complexity=ExtractorComplexity.LOW,
    #     description="Discrete Cosine Transform",
    #     extractor_class=DCTExtractor,
    #     default_params={"n_coeffs": 13},
    #     input_requirements={"sample_rate": 16000}
    # ))
    #
    # extractor_registry.register(ExtractorSpec(
    #     name="hilbert_transform",
    #     feature_type=FeatureType.SPECTRAL,
    #     complexity=ExtractorComplexity.MEDIUM,
    #     description="Hilbert Transform Features",
    #     extractor_class=HilbertTransformExtractor,
    #     default_params={},
    #     input_requirements={"sample_rate": 16000}
    # ))

    # # Extratores Prosódicos
    # extractor_registry.register(ExtractorSpec(
    #     name="pitch",
    #     feature_type=FeatureType.PROSODIC,
    #     complexity=ExtractorComplexity.MEDIUM,
    #     description="Pitch/F0 Features",
    #     extractor_class=PitchExtractor,
    #     default_params={"fmin": 50, "fmax": 400},
    #     input_requirements={"sample_rate": 16000}
    # ))
    #
    # extractor_registry.register(ExtractorSpec(
    #     name="energy",
    #     feature_type=FeatureType.PROSODIC,
    #     complexity=ExtractorComplexity.LOW,
    #     description="Energy Features",
    #     extractor_class=EnergyExtractor,
    #     default_params={"frame_length": 2048, "hop_length": 512},
    #     input_requirements={"sample_rate": 16000}
    # ))
    #
    # extractor_registry.register(ExtractorSpec(
    #     name="formant",
    #     feature_type=FeatureType.PROSODIC,
    #     complexity=ExtractorComplexity.HIGH,
    #     description="Formant Features",
    #     extractor_class=FormantExtractor,
    #     default_params={"n_formants": 4},
    #     input_requirements={"sample_rate": 16000}
    # ))
    #
    # # Extratores Perceptuais
    # extractor_registry.register(ExtractorSpec(
    #     name="spectral_centroid",
    #     feature_type=FeatureType.PERCEPTUAL,
    #     complexity=ExtractorComplexity.LOW,
    #     description="Spectral Centroid",
    #     extractor_class=SpectralCentroidExtractor,
    #     default_params={},
    #     input_requirements={"sample_rate": 16000}
    # ))
    #
    # extractor_registry.register(ExtractorSpec(
    #     name="spectral_rolloff",
    #     feature_type=FeatureType.PERCEPTUAL,
    #     complexity=ExtractorComplexity.LOW,
    #     description="Spectral Rolloff",
    #     extractor_class=SpectralRolloffExtractor,
    #     default_params={"roll_percent": 0.85},
    #     input_requirements={"sample_rate": 16000}
    # ))
    #
    # extractor_registry.register(ExtractorSpec(
    #     name="zero_crossing_rate",
    #     feature_type=FeatureType.TEMPORAL,
    #     complexity=ExtractorComplexity.LOW,
    #     description="Zero Crossing Rate",
    #     extractor_class=ZeroCrossingRateExtractor,
    #     default_params={"frame_length": 2048, "hop_length": 512},
    #     input_requirements={"sample_rate": 16000}
    # ))
    #
    # extractor_registry.register(ExtractorSpec(
    #     name="chroma",
    #     feature_type=FeatureType.PERCEPTUAL,
    #     complexity=ExtractorComplexity.MEDIUM,
    #     description="Chroma Features",
    #     extractor_class=ChromaExtractor,
    #     default_params={"n_chroma": 12},
    #     input_requirements={"sample_rate": 16000}
    # ))

    # Novos extratores implementados
    extractor_registry.register(ExtractorSpec(
        name="spectral_advanced",
        feature_type=FeatureType.SPECTRAL,
        complexity=ExtractorComplexity.HIGH,
        description="Advanced Spectral Features (flux, decrease, crest, "
                    "irregularity, roughness, inharmonicity)",
        extractor_class=SpectralFeatureExtractor,
        default_params={"sr": 22050, "frame_length": 2048, "hop_length": 512},
        input_requirements={"sample_rate": 22050}
    ))

    extractor_registry.register(ExtractorSpec(
        name="cepstral_advanced",
        feature_type=FeatureType.CEPSTRAL,
        complexity=ExtractorComplexity.HIGH,
        description="Advanced Cepstral Features (MFCC, Delta-MFCC, "
                    "Delta-Delta-MFCC, PLP, LPCC)",
        extractor_class=CepstralFeatureExtractor,
        default_params={"sr": 22050, "n_mfcc": 13, "n_mels": 128},
        input_requirements={"sample_rate": 22050}
    ))

    extractor_registry.register(ExtractorSpec(
        name="prosodic_advanced",
        feature_type=FeatureType.PROSODIC,
        complexity=ExtractorComplexity.HIGH,
        description="Advanced Prosodic Features (F0, jitter, shimmer, "
                    "HNR, voicing)",
        extractor_class=ProsodicFeatureExtractor,
        default_params={"sr": 22050, "f0_min": 80.0, "f0_max": 400.0},
        input_requirements={"sample_rate": 22050}
    ))

    # extractor_registry.register(ExtractorSpec(
    #     name="formant_advanced",
    #     feature_type=FeatureType.PROSODIC,
    #     complexity=ExtractorComplexity.VERY_HIGH,
    #     description="Advanced Formant Features (F1-F4, bandwidths, "
    #                 "trajectories, dispersion)",
    #     extractor_class=FormantFeatureExtractor,
    #     default_params={"sr": 22050, "n_formants": 4},
    #     input_requirements={"sample_rate": 22050}
    # ))

    # extractor_registry.register(ExtractorSpec(
    #     name="voice_quality_advanced",
    #     feature_type=FeatureType.PROSODIC,
    #     complexity=ExtractorComplexity.VERY_HIGH,
    #     description="Advanced Voice Quality Features (perturbation, noise, "
    #                 "breathiness, roughness)",
    #     extractor_class=VoiceQualityFeatureExtractor,
    #     default_params={"sr": 22050, "f0_min": 80.0, "f0_max": 400.0},
    #     input_requirements={"sample_rate": 22050}
    # ))

    logger.info(
        f"Registrados {len(extractor_registry._extractors)} extratores padrão"
    )


# Funções de conveniência
def get_extractor(name: str, **params) -> IFeatureExtractor:
    """Função de conveniência para obter um extrator.

    Args:
        name: Nome do extrator
        **params: Parâmetros para o extrator

    Returns:
        Instância do extrator
    """
    return extractor_registry.get_extractor(name, **params)


def list_extractors(
    feature_type: Optional[FeatureType] = None,
    complexity: Optional[ExtractorComplexity] = None
) -> List[str]:
    """Função de conveniência para listar extratores.

    Args:
        feature_type: Filtrar por tipo de feature
        complexity: Filtrar por complexidade

    Returns:
        Lista de nomes de extratores
    """
    return extractor_registry.list_extractors(feature_type, complexity)


def get_extractor_info(name: str) -> Dict[str, Any]:
    """Função de conveniência para obter informações de um extrator.

    Args:
        name: Nome do extrator

    Returns:
        Dicionário com informações do extrator
    """
    spec = extractor_registry.get_spec(name)
    return {
        "name": spec.name,
        "type": spec.feature_type.value,
        "complexity": spec.complexity.value,
        "description": spec.description,
        "default_params": spec.default_params,
        "input_requirements": spec.input_requirements,
        "output_shape": spec.output_shape,
        "dependencies": spec.dependencies,
        "version": spec.version,
        "author": spec.author
    }


# Inicializar extratores padrão quando o módulo for importado
if not extractor_registry._initialized:
    try:
        register_default_extractors()
        extractor_registry._initialized = True
        logger.info("Registry de extratores inicializado com sucesso")
    except Exception as e:
        logger.error(f"Erro ao inicializar registry de extratores: {e}")
