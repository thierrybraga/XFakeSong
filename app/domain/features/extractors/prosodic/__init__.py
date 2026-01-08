"""Extratores de Características Prosódicas

Módulos para extração de características relacionadas à prosódia e qualidade vocal:
- prosodic_features: Características de pitch, F0, entonação
- formant_features: Características de formantes
- voice_quality_features: Jitter, shimmer, HNR, etc.
"""

from .prosodic_features import ProsodicFeatureExtractor

__all__ = [
    "ProsodicFeatureExtractor"
]
