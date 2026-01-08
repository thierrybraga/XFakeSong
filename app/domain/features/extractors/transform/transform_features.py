#!/usr/bin/env python3
"""
Extrator de Características de Transformadas

Este módulo implementa a extração de características baseadas em transformadas
especializadas para análise de áudio, incluindo:
- Transformadas Wavelet (DWT, CWT)
- Constant-Q Transform (CQT)
- Chromagram
- Tonnetz
- MDCT (Modified Discrete Cosine Transform)
- Características auditivas

Refatorado para arquitetura modular.

Autor: Sistema de Detecção de DeepFake
Data: 2024
"""

import numpy as np
import warnings
from typing import Dict, Any

# Imports dos componentes
from .components.wavelet import extract_wavelet_features
from .components.cqt import extract_cqt_features
from .components.chroma import extract_chroma_features
from .components.tonnetz import extract_tonnetz_features
from .components.mdct import extract_mdct_features
from .components.auditory import extract_auditory_features


class TransformFeatureExtractor:
    """
    Extrator de características baseadas em transformadas especializadas.

    Esta classe implementa métodos para extrair características avançadas
    usando diferentes tipos de transformadas matemáticas aplicadas ao
    sinal de áudio.
    """

    def __init__(self,
                 sr: int = 22050,
                 frame_length: int = 2048,
                 hop_length: int = 512,
                 wavelet: str = 'db4',
                 bins_per_octave: int = 12):
        """
        Inicializa o extrator de características de transformadas.

        Args:
            sr: Taxa de amostragem
            frame_length: Tamanho do frame para análise
            hop_length: Tamanho do salto entre frames
            wavelet: Tipo de wavelet para transformadas
            bins_per_octave: Bins por oitava para CQT
        """
        self.sr = sr
        self.frame_length = frame_length
        self.hop_length = hop_length
        self.wavelet = wavelet
        self.bins_per_octave = bins_per_octave

        # Validação de parâmetros
        if sr <= 0:
            raise ValueError("Taxa de amostragem deve ser positiva")
        if frame_length <= 0 or hop_length <= 0:
            raise ValueError("Tamanhos de frame e hop devem ser positivos")
        if bins_per_octave <= 0:
            raise ValueError("Bins por oitava deve ser positivo")

    def extract(self, audio_data, metadata=None) -> Dict[str, np.ndarray]:
        """
        Extrai características de transformadas do sinal de áudio.
        Método compatível com o sistema de extração segmentada.

        Args:
            audio_data: AudioData ou array numpy com sinal de áudio
            metadata: Metadados do áudio (opcional)

        Returns:
            Dicionário com características extraídas como arrays numpy
        """
        # Extrair array numpy do AudioData se necessário
        if hasattr(audio_data, 'samples'):
            y = audio_data.samples
        else:
            y = audio_data

        # Chamar método de extração existente
        features_dict = self.extract_features(y)

        # Converter valores para arrays numpy
        numpy_features = {}
        for key, value in features_dict.items():
            if isinstance(value, (list, tuple)):
                numpy_features[key] = np.array(value)
            elif isinstance(value, (int, float)):
                numpy_features[key] = np.array([value])
            else:
                numpy_features[key] = np.array(value)

        return numpy_features

    def extract_features(self, y: np.ndarray) -> Dict[str, Any]:
        """
        Extrai características de transformadas do sinal de áudio.

        Args:
            y: Sinal de áudio

        Returns:
            Dicionário com características extraídas
        """
        if len(y) == 0:
            return self._get_default_features()

        features = {}

        try:
            # === CARACTERÍSTICAS WAVELET ===
            try:
                wavelet_features = extract_wavelet_features(
                    y, self.sr, self.wavelet)
                features.update(wavelet_features)
            except Exception as e:
                warnings.warn(f"Erro na extração wavelet: {str(e)}")

            # === CARACTERÍSTICAS CONSTANT-Q ===
            try:
                cqt_features = extract_cqt_features(
                    y, self.sr, self.hop_length, self.bins_per_octave)
                features.update(cqt_features)
            except Exception as e:
                warnings.warn(f"Erro na extração CQT: {str(e)}")

            # === CARACTERÍSTICAS CHROMAGRAM ===
            try:
                chroma_features = extract_chroma_features(
                    y, self.sr, self.hop_length)
                features.update(chroma_features)
            except Exception as e:
                warnings.warn(f"Erro na extração chroma: {str(e)}")

            # === CARACTERÍSTICAS TONNETZ ===
            try:
                tonnetz_features = extract_tonnetz_features(y, self.sr)
                features.update(tonnetz_features)
            except Exception as e:
                warnings.warn(f"Erro na extração tonnetz: {str(e)}")

            # === CARACTERÍSTICAS MDCT ===
            try:
                mdct_features = extract_mdct_features(
                    y, self.frame_length, self.hop_length)
                features.update(mdct_features)
            except Exception as e:
                warnings.warn(f"Erro na extração MDCT: {str(e)}")

            # === CARACTERÍSTICAS AUDITIVAS ===
            try:
                auditory_features = extract_auditory_features(
                    y, self.sr, self.frame_length, self.hop_length)
                features.update(auditory_features)
            except Exception as e:
                warnings.warn(f"Erro na extração auditiva: {str(e)}")

        except Exception as e:
            warnings.warn(
                f"Erro na extração de características de transformadas: {
                    str(e)}")
            return self._get_default_features()

        return features

    def _get_default_features(self) -> Dict[str, Any]:
        """Retorna características padrão em caso de erro."""
        return {
            'dwt_energy': [0],
            'dwt_entropy': [0],
            'cwt_energy': np.array([0]),
            'cwt_centroid': np.array([0]),
            'cqt_centroid': np.array([0]),
            'cqt_rolloff': np.array([0]),
            'cqt_octave_energy': np.array([0]),
            'chroma_mean': np.zeros(12),
            'chroma_std': np.zeros(12),
            'chroma_centroid': np.array([0]),
            'chroma_flux': np.array([0]),
            'tonnetz_mean': np.zeros(6),
            'tonnetz_std': np.zeros(6),
            'tonnetz_centroid': np.array([0]),
            'mdct_centroid': np.array([0]),
            'mdct_rolloff': np.array([0]),
            'mdct_energy': np.array([0]),
            'auditory_centroid': np.array([0]),
            'auditory_rolloff': np.array([0]),
            'auditory_flux': np.array([0])
        }
