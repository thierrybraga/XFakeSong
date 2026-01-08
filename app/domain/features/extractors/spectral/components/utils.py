"""
Funções utilitárias para extração de características espectrais.
"""
import numpy as np
from typing import Dict, Any


def apply_preemphasis(y: np.ndarray, preemphasis: float = 0.97) -> np.ndarray:
    """
    Aplica pré-ênfase ao sinal.

    Args:
        y: Sinal de áudio
        preemphasis: Coeficiente de pré-ênfase

    Returns:
        Sinal com pré-ênfase aplicada
    """
    if preemphasis == 0:
        return y

    preemphasized = np.zeros_like(y)
    preemphasized[0] = y[0]
    for i in range(1, len(y)):
        preemphasized[i] = y[i] - preemphasis * y[i - 1]

    return preemphasized


def get_default_spectral_features() -> Dict[str, Any]:
    """
    Retorna características padrão em caso de erro.

    Returns:
        Dicionário com valores padrão
    """
    return {
        'spectral_centroid': np.array([0.0]),
        'spectral_rolloff_85': np.array([0.0]),
        'spectral_rolloff_95': np.array([0.0]),
        'spectral_bandwidth': np.array([0.0]),
        'spectral_flatness': np.array([0.0]),
        'spectral_slope': np.array([0.0]),
        'spectral_kurtosis': np.array([0.0]),
        'spectral_skewness': np.array([0.0]),
        'spectral_spread': np.array([0.0]),
        'spectral_entropy': np.array([0.0]),
        'spectral_contrast': np.zeros((7, 1)),
        'high_freq_content': np.array([0.0]),
        'low_freq_ratio': 0.0,
        'mid_freq_ratio': 0.0,
        'high_freq_ratio': 0.0,
        'spectral_flux': np.array([0.0]),
        'spectral_decrease': np.array([0.0]),
        'spectral_crest': np.array([0.0]),
        'spectral_irregularity': np.array([0.0]),
        'spectral_roughness': np.array([0.0]),
        'spectral_inharmonicity': np.array([0.0])
    }
