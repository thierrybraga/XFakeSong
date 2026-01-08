"""
Funções utilitárias para conversão de escalas perceptuais.
"""
import numpy as np


def hz_to_bark(freq_hz: np.ndarray) -> np.ndarray:
    """Converte frequência em Hz para escala Bark."""
    return 13 * np.arctan(0.00076 * freq_hz) + 3.5 * \
        np.arctan((freq_hz / 7500) ** 2)


def bark_to_hz(bark: np.ndarray) -> np.ndarray:
    """Converte escala Bark para frequência em Hz (aproximação)."""
    # Aproximação inversa
    return 600 * np.sinh(bark / 4)


def hz_to_erb(freq_hz: np.ndarray) -> np.ndarray:
    """Converte frequência em Hz para escala ERB."""
    return 21.4 * np.log10(1 + 0.00437 * freq_hz)


def erb_to_hz(erb: np.ndarray) -> np.ndarray:
    """Converte escala ERB para frequência em Hz."""
    return (10 ** (erb / 21.4) - 1) / 0.00437
