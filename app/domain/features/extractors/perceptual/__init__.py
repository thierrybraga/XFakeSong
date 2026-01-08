"""Extratores de Características Perceptuais

Módulos para extração de características baseadas no sistema auditivo humano:
- perceptual_features: Loudness, sharpness, roughness, escalas Bark/ERB, etc.
"""

from .perceptual_features import PerceptualFeatureExtractor

__all__ = ["PerceptualFeatureExtractor"]
