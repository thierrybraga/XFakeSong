"""Módulo de arquiteturas de modelos."""

# Importar todas as arquiteturas disponíveis
from . import (
    aasist,
    architecture_patcher,
    conformer,
    efficientnet_lstm,
    ensemble,
    multiscale_cnn,
    random_forest,
    rawgat_st,
    registry,
    safe_normalization,
    spectrogram_transformer,
    svm,
    transforms,
)

# Importar funções de correção de arquiteturas
from .architecture_patcher import (
    ArchitecturePatcher,
    patch_architecture_for_safety,
    validate_model_safety,
)

# Importar registry functions para conveniência
from .registry import (
    architecture_registry,
    create_model_by_name,
    create_safe_model_by_name,
    get_architecture_info,
    get_available_architectures,
    validate_architecture_input,
)

# Importar funções de normalização segura
from .safe_normalization import (
    SafeGroupNormalization,
    SafeInstanceNormalization,
    SafeLayerNormalization,
    get_safe_normalization_layer,
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
