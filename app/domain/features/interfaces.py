"""Interfaces para extratores de características."""

from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional
import numpy as np

from app.domain.features.types import FeatureType, ProcessingResult


class IFeatureExtractor(ABC):
    """Interface base para extratores de características de áudio."""

    @abstractmethod
    def extract(self, audio_data: np.ndarray, sample_rate: int,
                **kwargs) -> ProcessingResult:
        """Extrai características do áudio.

        Args:
            audio_data: Dados de áudio como array numpy
            sample_rate: Taxa de amostragem do áudio
            **kwargs: Parâmetros adicionais específicos do extrator

        Returns:
            ProcessingResult com as características extraídas
        """
        pass

    @abstractmethod
    def get_feature_type(self) -> FeatureType:
        """Retorna o tipo de característica extraída."""
        pass

    @abstractmethod
    def get_feature_names(self) -> List[str]:
        """Retorna os nomes das características extraídas."""
        pass

    @abstractmethod
    def get_extraction_params(self) -> Dict[str, Any]:
        """Retorna os parâmetros de extração atuais."""
        pass
