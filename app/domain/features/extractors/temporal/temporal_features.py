"""
Extração de Características Temporais de Áudio
=============================================

Este módulo implementa características baseadas no domínio temporal,
incluindo energia, taxa de cruzamento por zero e outras métricas temporais.
"""

import numpy as np
import warnings
from typing import Dict, Any

from .components.energy import (
    compute_rms_energy, compute_short_time_energy, compute_energy_entropy,
    compute_teager_energy, compute_log_energy, compute_frame_energy_variance
)
from .components.envelope import (
    compute_temporal_centroid, compute_temporal_rolloff, compute_temporal_flux,
    compute_roughness, compute_attack_time, compute_decay_time, compute_sustain_level
)
from .components.statistics import (
    compute_sign_change_rate, compute_mean_crossing_rate, compute_zcr,
    compute_zcr_variance, extract_signal_statistics
)
from .components.dynamics import (
    compute_amplitude_modulation, compute_tremolo_rate, extract_envelope_statistics
)
from app.domain.features.interfaces import IFeatureExtractor
from app.core.interfaces.audio import AudioData, FeatureType
from app.core.interfaces.base import ProcessingResult, ProcessingStatus


class TemporalFeatureExtractor(IFeatureExtractor):
    """
    Extrator de características temporais de áudio.
    """

    def __init__(self, sr: int = 22050, frame_length: int = 2048,
                 hop_length: int = 512, window: str = 'hann'):
        """
        Inicializa o extrator de características temporais.

        Args:
            sr: Taxa de amostragem
            frame_length: Comprimento da janela
            hop_length: Salto entre janelas
            window: Tipo de janela
        """
        # Validação de parâmetros
        if sr <= 0:
            raise ValueError("Taxa de amostragem deve ser positiva")
        if frame_length <= 0 or hop_length <= 0:
            raise ValueError("Comprimentos devem ser positivos")
        if hop_length > frame_length:
            warnings.warn(
                "hop_length maior que frame_length pode causar problemas")

        self.sr = sr
        self.frame_length = frame_length
        self.hop_length = hop_length
        self.window = window

    def extract(self, audio_data: AudioData) -> ProcessingResult:
        """
        Extrai características temporais do áudio.

        Args:
            audio_data: Dados de áudio

        Returns:
            ProcessingResult com características temporais
        """
        try:
            if audio_data.samples is None or len(audio_data.samples) == 0:
                return ProcessingResult(
                    status=ProcessingStatus.ERROR,
                    errors=["Dados de áudio vazios"]
                )

            features = self.extract_features(audio_data.samples)
            
            return ProcessingResult(
                status=ProcessingStatus.SUCCESS,
                data=features
            )
        except Exception as e:
            return ProcessingResult(
                status=ProcessingStatus.ERROR,
                errors=[str(e)]
            )

    def get_feature_type(self) -> FeatureType:
        return FeatureType.TEMPORAL

    def get_feature_names(self) -> list[str]:
        # Lista aproximada de features retornadas
        return [
            "rms_energy", "zero_crossing_rate", "temporal_centroid",
            "attack_time", "decay_time", "sustain_level"
        ]

    def get_extraction_params(self) -> Dict[str, Any]:
        return {
            "sr": self.sr,
            "frame_length": self.frame_length,
            "hop_length": self.hop_length,
            "window": self.window
        }

    def extract_features(self, y: np.ndarray, context: Any = None) -> Dict:
        """
        Extrai todas as características temporais.

        Args:
            y: Sinal de áudio
            context: Contexto de processamento (opcional)

        Returns:
            Dicionário com características temporais
        """
        # Validação de entrada
        if not isinstance(y, np.ndarray):
            raise TypeError("Entrada deve ser um array numpy")

        if len(y) == 0:
            raise ValueError("Sinal de áudio não pode estar vazio")

        if len(y) < self.frame_length:
            warnings.warn(
                f"Sinal muito curto ({
                    len(y)} samples), padding será aplicado")

        # Normalizar sinal
        y_norm = y / (np.max(np.abs(y)) + 1e-10)

        features = {}

        try:
            # === CARACTERÍSTICAS BÁSICAS ===
            basic_features = self._extract_basic_features(y_norm)
            features.update(basic_features)

            # === CARACTERÍSTICAS DE ENVELOPE ===
            envelope_features = self._extract_envelope_features(y_norm)
            features.update(envelope_features)

            # === CARACTERÍSTICAS ESTATÍSTICAS TEMPORAIS ===
            statistical_features = self._extract_statistical_features(y_norm)
            features.update(statistical_features)

            # === CARACTERÍSTICAS DE DINÂMICA TEMPORAL ===
            dynamics_features = self._extract_dynamics_features(y_norm)
            features.update(dynamics_features)

        except Exception as e:
            warnings.warn(
                f"Erro na extração de características temporais: {
                    str(e)}")
            features = self._get_default_temporal_features()

        return features

    def _extract_basic_features(self, y: np.ndarray) -> Dict:
        """Extrai características temporais básicas."""
        features = {}

        # Características básicas de energia
        features['rms_energy'] = compute_rms_energy(
            y, self.frame_length, self.hop_length)
        features['short_time_energy'] = compute_short_time_energy(
            y, self.frame_length, self.hop_length)
        features['energy_entropy'] = compute_energy_entropy(
            y, self.frame_length, self.hop_length)
        features['teager_energy'] = compute_teager_energy(
            y, self.frame_length, self.hop_length)

        # Taxa de cruzamento por zero
        features['zero_crossing_rate'] = compute_zcr(
            y, self.frame_length, self.hop_length)
        features['zcr_variance'] = compute_zcr_variance(
            y, self.frame_length, self.hop_length)
        features['sign_change_rate'] = compute_sign_change_rate(y)
        features['mean_crossing_rate'] = compute_mean_crossing_rate(y)

        # Características de energia avançadas
        features['log_energy'] = compute_log_energy(
            y, self.frame_length, self.hop_length)
        features['frame_energy_variance'] = compute_frame_energy_variance(
            y, self.frame_length, self.hop_length)

        # Características de centroide temporal
        features['temporal_centroid'] = compute_temporal_centroid(
            y, self.sr, self.frame_length, self.hop_length)
        features['temporal_rolloff'] = compute_temporal_rolloff(
            y, self.sr, self.frame_length, self.hop_length)
        features['temporal_flux'] = compute_temporal_flux(
            y, self.frame_length, self.hop_length)
        features['roughness'] = compute_roughness(y)

        return features

    def _extract_envelope_features(self, y: np.ndarray) -> Dict:
        """Extrai características de envelope."""
        features = {}

        # Características de envelope
        features['attack_time'] = compute_attack_time(
            y, self.sr, self.frame_length, self.hop_length)
        features['decay_time'] = compute_decay_time(
            y, self.sr, self.frame_length, self.hop_length)
        features['sustain_level'] = compute_sustain_level(
            y, self.frame_length, self.hop_length)

        return features

    def _extract_statistical_features(self, y: np.ndarray) -> Dict:
        """Extrai características estatísticas temporais."""
        return extract_signal_statistics(y)

    def _extract_dynamics_features(self, y: np.ndarray) -> Dict:
        """Extrai características de dinâmica temporal."""
        return extract_envelope_statistics(y, self.sr)

    def _get_default_temporal_features(self) -> Dict:
        """Retorna características temporais padrão em caso de erro."""
        return {
            'rms_energy': np.array([0.0]),
            'short_time_energy': np.array([0.0]),
            'energy_entropy': np.array([0.0]),
            'teager_energy': np.array([0.0]),
            'zero_crossing_rate': np.array([0.0]),
            'zcr_variance': 0.0,
            'temporal_centroid': 0.0,
            'temporal_rolloff': 0.0,
            'temporal_flux': 0.0,
            'attack_time': 0.0,
            'decay_time': 0.0,
            'sustain_level': 0.0,
            'signal_mean': 0.0,
            'signal_std': 0.0,
            'signal_variance': 0.0,
            'signal_skewness': 0.0,
            'signal_kurtosis': 0.0,
            'peak_amplitude': 0.0,
            'rms_amplitude': 0.0,
            'crest_factor': 0.0,
            'dynamic_range': 0.0,
            'signal_energy': 0.0,
            'envelope_mean': 0.0,
            'envelope_std': 0.0,
            'envelope_max': 0.0,
            'envelope_variation': 0.0,
            'envelope_slope': 0.0,
            'amplitude_modulation': 0.0,
            'tremolo_rate': 0.0
        }

    # Backward compatibility methods (proxies to components)
    def _compute_sign_change_rate(self, y: np.ndarray) -> float:
        return compute_sign_change_rate(y)

    def _compute_mean_crossing_rate(self, y: np.ndarray) -> float:
        return compute_mean_crossing_rate(y)

    def _compute_log_energy(self, y: np.ndarray) -> np.ndarray:
        return compute_log_energy(y, self.frame_length, self.hop_length)

    def _compute_frame_energy_variance(self, y: np.ndarray) -> float:
        return compute_frame_energy_variance(
            y, self.frame_length, self.hop_length)

    def _compute_roughness(self, y: np.ndarray) -> float:
        return compute_roughness(y)

    def _compute_rms_energy(self, y: np.ndarray) -> np.ndarray:
        return compute_rms_energy(y, self.frame_length, self.hop_length)

    def _compute_short_time_energy(self, y: np.ndarray) -> np.ndarray:
        return compute_short_time_energy(y, self.frame_length, self.hop_length)

    def _compute_energy_entropy(self, y: np.ndarray) -> float:
        return compute_energy_entropy(y, self.frame_length, self.hop_length)

    def _compute_teager_energy(self, y: np.ndarray) -> np.ndarray:
        return compute_teager_energy(y, self.frame_length, self.hop_length)

    def _compute_zcr(self, y: np.ndarray) -> np.ndarray:
        return compute_zcr(y, self.frame_length, self.hop_length)

    def _compute_zcr_variance(self, y: np.ndarray) -> float:
        return compute_zcr_variance(y, self.frame_length, self.hop_length)

    def _compute_temporal_centroid(self, y: np.ndarray) -> float:
        return compute_temporal_centroid(
            y, self.sr, self.frame_length, self.hop_length)

    def _compute_temporal_rolloff(
            self, y: np.ndarray, rolloff_percent: float = 0.85) -> float:
        return compute_temporal_rolloff(
            y, self.sr, self.frame_length, self.hop_length, rolloff_percent)

    def _compute_temporal_flux(self, y: np.ndarray) -> float:
        return compute_temporal_flux(y, self.frame_length, self.hop_length)

    def _compute_attack_time(self, y: np.ndarray) -> float:
        return compute_attack_time(
            y, self.sr, self.frame_length, self.hop_length)

    def _compute_decay_time(self, y: np.ndarray) -> float:
        return compute_decay_time(
            y, self.sr, self.frame_length, self.hop_length)

    def _compute_sustain_level(self, y: np.ndarray) -> float:
        return compute_sustain_level(y, self.frame_length, self.hop_length)

    def _compute_amplitude_modulation(self, y: np.ndarray) -> float:
        return compute_amplitude_modulation(y)

    def _compute_tremolo_rate(self, envelope: np.ndarray) -> float:
        return compute_tremolo_rate(envelope, self.sr)
