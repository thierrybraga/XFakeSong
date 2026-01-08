"""Extratores de Características Cepstrais

Módulos para extração de características baseadas no domínio cepstral:
- cepstral_features: Características cepstrais principais (MFCC, LPCC)
- advanced_cepstral_features: Características avançadas (PLP, RASTA-PLP)
"""

from .cepstral_features import CepstralFeatureExtractor

__all__ = ["CepstralFeatureExtractor"]
