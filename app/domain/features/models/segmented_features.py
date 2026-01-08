#!/usr/bin/env python3
"""
Modelos de Features Segmentadas
==============================

Define as estruturas de dados para features extraídas de segmentos de áudio.
"""

from dataclasses import dataclass
from typing import List, Dict, Any, Optional
import numpy as np


@dataclass
class SegmentedFeatures:
    """
    Representa features extraídas de um segmento de áudio.

    Attributes:
        segment_index: Índice do segmento no arquivo original
        start_time: Tempo de início do segmento em segundos
        end_time: Tempo de fim do segmento em segundos
        spectral_features: Features espectrais extraídas
        temporal_features: Features temporais extraídas
        combined_features: Todas as features combinadas em um array
        feature_names: Nomes das features na ordem do combined_features
        metadata: Metadados adicionais do segmento
    """
    segment_index: int
    start_time: float
    end_time: float
    spectral_features: Dict[str, float]
    temporal_features: Dict[str, float]
    cepstral_features: Dict[str, float]
    prosodic_features: Dict[str, float]
    perceptual_features: Dict[str, float]
    formant_features: Dict[str, float]
    speech_features: Dict[str, float]
    combined_features: np.ndarray
    feature_names: List[str]
    metadata: Dict[str, Any]

    def __post_init__(self):
        """Validação pós-inicialização."""
        if len(self.combined_features) != len(self.feature_names):
            raise ValueError(
                f"Número de features ({len(self.combined_features)}) "
                f"não corresponde ao número de nomes ({
                    len(
                        self.feature_names)})"
            )

    @property
    def duration(self) -> float:
        """Duração do segmento em segundos."""
        return self.end_time - self.start_time

    @property
    def feature_count(self) -> int:
        """Número total de features."""
        return len(self.combined_features)

    def get_feature_by_name(self, name: str) -> Optional[float]:
        """Obtém o valor de uma feature pelo nome."""
        try:
            index = self.feature_names.index(name)
            return float(self.combined_features[index])
        except ValueError:
            return None

    def to_dict(self) -> Dict[str, Any]:
        """Converte para dicionário."""
        return {
            'segment_index': self.segment_index,
            'start_time': self.start_time,
            'end_time': self.end_time,
            'duration': self.duration,
            'spectral_features': self.spectral_features,
            'temporal_features': self.temporal_features,
            'combined_features': self.combined_features.tolist(),
            'feature_names': self.feature_names,
            'feature_count': self.feature_count,
            'metadata': self.metadata
        }
