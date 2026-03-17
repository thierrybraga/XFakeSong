"""
Extração de Características Perceptuais de Áudio
===============================================

Este módulo implementa características baseadas no sistema auditivo humano,
incluindo escalas Bark, ERB, loudness, sharpness e outras medidas perceptuais.
"""

# Third-party imports
from typing import Dict

import numpy as np

from .components import (
    bark_to_hz,
    erb_to_hz,
    extract_bark_features,
    extract_erb_features,
    extract_fluctuation_features,
    extract_loudness_features,
    extract_masking_features,
    extract_roughness_features,
    extract_sharpness_features,
    extract_tonality_features,
    hz_to_bark,
    hz_to_erb,
)


class PerceptualFeatureExtractor:
    """
    Extrator de características perceptuais de áudio.
    """

    def __init__(self, sr: int = 22050, frame_length: int = 2048,
                 hop_length: int = 512):
        """
        Inicializa o extrator de características perceptuais.

        Args:
            sr: Taxa de amostragem
            frame_length: Comprimento da janela
            hop_length: Salto entre janelas
        """
        self.sr = sr
        self.frame_length = frame_length
        self.hop_length = hop_length

    def extract_features(self, y: np.ndarray) -> Dict:
        """
        Extrai todas as características perceptuais.

        Args:
            y: Sinal de áudio

        Returns:
            Dicionário com características perceptuais
        """
        features = {}

        try:
            # === CARACTERÍSTICAS BARK SCALE ===
            # print("        Extraindo características Bark...")
            bark_features = self._extract_bark_features(y)
            features.update(bark_features)
            # print(f"        Bark: {len(bark_features)} características")

            # === CARACTERÍSTICAS ERB SCALE ===
            # print("        Extraindo características ERB...")
            erb_features = self._extract_erb_features(y)
            features.update(erb_features)
            # print(f"        ERB: {len(erb_features)} características")

            # === CARACTERÍSTICAS DE LOUDNESS ===
            # print("        Extraindo características Loudness...")
            loudness_features = self._extract_loudness_features(y)
            features.update(loudness_features)
            # print(f"        Loudness: {len(loudness_features)} características")

            # === CARACTERÍSTICAS DE SHARPNESS ===
            # print("        Extraindo características Sharpness...")
            sharpness_features = self._extract_sharpness_features(y)
            features.update(sharpness_features)
            # print(f"        Sharpness: {len(sharpness_features)} características")

            # === CARACTERÍSTICAS DE ROUGHNESS ===
            # print("        Extraindo características Roughness...")
            roughness_features = self._extract_roughness_features(y)
            features.update(roughness_features)
            # print(f"        Roughness: {len(roughness_features)} características")

            # === CARACTERÍSTICAS DE FLUCTUATION STRENGTH ===
            # print("        Extraindo características Fluctuation...")
            fluctuation_features = self._extract_fluctuation_features(y)
            features.update(fluctuation_features)
            # print(f"        Fluctuation: {len(fluctuation_features)} características")

            # === CARACTERÍSTICAS DE TONALITY ===
            # print("        Extraindo características Tonality...")
            tonality_features = self._extract_tonality_features(y)
            features.update(tonality_features)
            # print(f"        Tonality: {len(tonality_features)} características")

            # === CARACTERÍSTICAS DE MASCARAMENTO ===
            # print("        Extraindo características Masking...")
            masking_features = self._extract_masking_features(y)
            features.update(masking_features)
            # print(f"        Masking: {len(masking_features)} características")

        except Exception as e:
            print(f"Erro na extração perceptual: {str(e)}")
            # Retornar características básicas em caso de erro
            features = {
                'bark_energy_mean': 0.0,
                'erb_energy_mean': 0.0,
                'loudness_mean': 0.0,
                'sharpness_mean': 0.0
            }

        return features

    def _hz_to_bark(self, freq_hz: np.ndarray) -> np.ndarray:
        """Converte frequência em Hz para escala Bark."""
        return hz_to_bark(freq_hz)

    def _bark_to_hz(self, bark: np.ndarray) -> np.ndarray:
        """Converte escala Bark para frequência em Hz (aproximação)."""
        return bark_to_hz(bark)

    def _hz_to_erb(self, freq_hz: np.ndarray) -> np.ndarray:
        """Converte frequência em Hz para escala ERB."""
        return hz_to_erb(freq_hz)

    def _erb_to_hz(self, erb: np.ndarray) -> np.ndarray:
        """Converte escala ERB para frequência em Hz."""
        return erb_to_hz(erb)

    def _extract_bark_features(self, y: np.ndarray) -> Dict:
        """Extrai características baseadas na escala Bark."""
        return extract_bark_features(
            y, self.sr, self.frame_length, self.hop_length)

    def _extract_erb_features(self, y: np.ndarray) -> Dict:
        """Extrai características baseadas na escala ERB."""
        return extract_erb_features(
            y, self.sr, self.frame_length, self.hop_length)

    def _extract_loudness_features(self, y: np.ndarray) -> Dict:
        """Extrai características de loudness (Stevens/Zwicker)."""
        return extract_loudness_features(
            y, self.sr, self.frame_length, self.hop_length)

    def _extract_sharpness_features(self, y: np.ndarray) -> Dict:
        """Extrai características de sharpness (agudeza)."""
        return extract_sharpness_features(
            y, self.sr, self.frame_length, self.hop_length)

    def _extract_roughness_features(self, y: np.ndarray) -> Dict:
        """Extrai características de roughness (rugosidade)."""
        return extract_roughness_features(
            y, self.sr, self.frame_length, self.hop_length)

    def _extract_fluctuation_features(self, y: np.ndarray) -> Dict:
        """Extrai características de fluctuation strength."""
        return extract_fluctuation_features(y, self.sr)

    def _extract_tonality_features(self, y: np.ndarray) -> Dict:
        """Extrai características de tonality."""
        return extract_tonality_features(
            y, self.sr, self.frame_length, self.hop_length)

    def _extract_masking_features(self, y: np.ndarray) -> Dict:
        """Extrai características de mascaramento auditivo (versão otimizada)."""
        return extract_masking_features(
            y, self.sr, self.frame_length, self.hop_length)


def test_perceptual_features():
    """Testa as características perceptuais."""
    # Gerar sinal de teste com características perceptuais variadas
    duration = 2.0
    sr = 22050
    t = np.linspace(0, duration, int(duration * sr))

    # Componentes tonais em diferentes bandas críticas
    f1, f2, f3 = 440, 880, 1760  # Harmônicos
    y = (np.sin(2 * np.pi * f1 * t) +
         0.7 * np.sin(2 * np.pi * f2 * t) +
         0.5 * np.sin(2 * np.pi * f3 * t))

    # Adicionar componente de alta frequência (sharpness)
    f_high = 4000
    y += 0.3 * np.sin(2 * np.pi * f_high * t)

    # Adicionar modulação em amplitude (fluctuation)
    mod_freq = 4  # Hz
    y = y * (1 + 0.5 * np.sin(2 * np.pi * mod_freq * t))

    # Adicionar rugosidade (batimentos)
    f_beat1, f_beat2 = 1000, 1070  # Diferença de 70 Hz
    y += 0.4 * (np.sin(2 * np.pi * f_beat1 * t) +
                np.sin(2 * np.pi * f_beat2 * t))

    # Adicionar ruído para reduzir tonality
    noise_level = 0.1
    y += noise_level * np.random.randn(len(y))

    # Aplicar envelope
    envelope = np.exp(-0.5 * t) + 0.4
    y = y * envelope

    # Extrair características
    extractor = PerceptualFeatureExtractor(sr=sr)
    features = extractor.extract_features(y)

    print("🎧 Características Perceptuais Extraídas:")
    for name, value in features.items():
        if isinstance(value, (int, float)):
            print(f"  {name}: {value:.3f}")
        else:
            print(f"  {name}: {value}")


if __name__ == "__main__":
    test_perceptual_features()
