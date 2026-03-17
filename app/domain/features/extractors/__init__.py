"""Extratores de Características de Áudio

Pacote principal contendo todos os extratores de características organizados por categoria:
- spectral: Características espectrais e cepstrais
- temporal: Características temporais
- prosodic: Características prosódicas e de qualidade vocal
- perceptual: Características baseadas no sistema auditivo humano
- advanced: Características complexas e especializadas
"""

# Importar todos os extratores das subcategorias
from .cepstral import CepstralFeatureExtractor
from .complexity import ComplexityFeatureExtractor
from .formant import FormantFeatureExtractor
from .perceptual import PerceptualFeatureExtractor
from .predictive import PredictiveFeatureExtractor
from .prosodic import ProsodicFeatureExtractor
from .spectral import SpectralFeatureExtractor
from .speech import SpeechFeatureExtractor
from .temporal import TemporalFeatureExtractor
from .timefreq import TimeFrequencyFeatureExtractor
from .transform import TransformFeatureExtractor
from .voice_quality import VoiceQualityFeatureExtractor

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
    "VoiceQualityFeatureExtractor",
    # Perceptual
    "PerceptualFeatureExtractor",
    # Advanced / Others
    "ComplexityFeatureExtractor",
    "PredictiveFeatureExtractor",
    "SpeechFeatureExtractor",
    "TransformFeatureExtractor",
    "TimeFrequencyFeatureExtractor"
]
