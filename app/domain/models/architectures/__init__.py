"""Módulo de arquiteturas de modelos."""

# Importar todas as arquiteturas disponíveis
from . import aasist
from . import rawgat_st
from . import efficientnet_lstm
from . import multiscale_cnn
from . import spectrogram_transformer
from . import conformer
from . import ensemble
from . import svm
from . import random_forest
from . import transforms
from . import registry
from . import safe_normalization
from . import architecture_patcher

# Importar registry functions para conveniência
from .registry import (
    architecture_registry,
    get_available_architectures,
    create_model_by_name,
    create_safe_model_by_name,
    get_architecture_info,
    validate_architecture_input
)

# Importar funções de normalização segura
from .safe_normalization import (
    SafeInstanceNormalization,
    SafeLayerNormalization,
    SafeGroupNormalization,
    get_safe_normalization_layer
)

# Importar funções de correção de arquiteturas
from .architecture_patcher import (
    ArchitecturePatcher,
    patch_architecture_for_safety,
    validate_model_safety
)

__all__ = [
    "aasist",
    "rawgat_st",
    "efficientnet_lstm",
    "multiscale_cnn",
    "spectrogram_transformer",
    "conformer",
    "ensemble",
    "svm",
    "random_forest",
    "transforms",
    "registry",
    "safe_normalization",
    "architecture_patcher",
    # Registry functions
    "architecture_registry",
    "get_available_architectures",
    "create_model_by_name",
    "create_safe_model_by_name",
    "get_architecture_info",
    "validate_architecture_input",
    # Safe normalization
    "SafeInstanceNormalization",
    "SafeLayerNormalization",
    "SafeGroupNormalization",
    "get_safe_normalization_layer",
    # Architecture patching
    "ArchitecturePatcher",
    "patch_architecture_for_safety",
    "validate_model_safety"
]
