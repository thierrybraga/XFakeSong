from dataclasses import dataclass
from typing import List, Dict, Any, Optional
from app.core.interfaces.audio import FeatureType, AudioFeatures


@dataclass
class ExtractionConfig:
    """Configuração para extração de características"""
    feature_types: List[FeatureType]
    sample_rate: int = 16000
    normalize: bool = True
    include_deltas: bool = False
    include_delta_deltas: bool = False
    window_size: float = 0.025  # 25ms
    hop_length: float = 0.010   # 10ms
    aggregate_method: str = 'mean'  # 'mean', 'median', 'std', 'all'

    # Novas configurações para modularidade
    use_plugin_system: bool = True
    validate_pipeline: bool = True
    auto_optimize: bool = False
    fallback_extractors: List[str] = None

    def __post_init__(self):
        if self.fallback_extractors is None:
            self.fallback_extractors = []


@dataclass
class ExtractionResult:
    """Resultado da extração de características"""
    file_path: str
    features: AudioFeatures
    extraction_time: float
    feature_shape: tuple
    metadata: Dict[str, Any]
