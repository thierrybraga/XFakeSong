"""Extratores de Características de Áudio

Pacote principal contendo todos os extratores de características organizados por categoria:
- spectral: Características espectrais e cepstrais
- temporal: Características temporais
- prosodic: Características prosódicas e de qualidade vocal
- perceptual: Características baseadas no sistema auditivo humano
- advanced: Características complexas e especializadas
"""

# Importar todos os extratores das subcategorias
from .spectral import SpectralFeatureExtractor
from .cepstral import CepstralFeatureExtractor
from .prosodic import ProsodicFeatureExtractor
from .temporal import TemporalFeatureExtractor
from .perceptual import PerceptualFeatureExtractor
from .complexity import ComplexityFeatureExtractor
from .formant import FormantFeatureExtractor
from .predictive import PredictiveFeatureExtractor
from .speech import SpeechFeatureExtractor
from .transform import TransformFeatureExtractor
from .timefreq import TimeFrequencyFeatureExtractor

# Consolidar todos os extratores disponíveis
__all__ = [
    # Spectral
    "SpectralFeatureExtractor",
    "CepstralFeatureExtractor",
    # Temporal
    "TemporalFeatureExtractor",
    # Prosodic
    "ProsodicFeatureExtractor",
    "FormantFeatureExtractor",
    # Perceptual
    "PerceptualFeatureExtractor",
    # Advanced / Others
    "ComplexityFeatureExtractor",
    "PredictiveFeatureExtractor",
    "SpeechFeatureExtractor",
    "TransformFeatureExtractor",
    "TimeFrequencyFeatureExtractor"
]
