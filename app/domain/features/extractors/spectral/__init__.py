"""Extratores de Características Espectrais

Módulos para extração de características baseadas no domínio da frequência:
- spectral_features: Características espectrais básicas (centroide, rolloff, etc.)
"""

from .spectral_features import SpectralFeatureExtractor

__all__ = ["SpectralFeatureExtractor"]
