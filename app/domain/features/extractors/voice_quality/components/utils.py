"""
Funções utilitárias para extração de características de qualidade vocal.
"""
from typing import Dict

import numpy as np


def get_default_voice_quality_features() -> Dict[str, float]:
    """Retorna dicionário com valores padrão (zeros) para todas as características."""
    return {
        # Perturbation
        'rap': 0.0, 'ppq': 0.0, 'apq': 0.0, 'vf0': 0.0, 'shimmer_db': 0.0,
        # Noise
        'nhr': 0.0, 'vti': 0.0, 'spi': 0.0, 'dfa_alpha': 0.0,
        # Quality
        'spectral_tilt': 0.0, 'breathiness_index': 0.0,
        'roughness_index': 0.0, 'voice_breaks': 0.0,
        # Mapped
        'shdb': 0.0, 'breathiness': 0.0, 'roughness': 0.0, 'hoarseness': 0.0
    }


def extract_pitch_periods(f0: np.ndarray, sr: int) -> np.ndarray:
    """
    Converte contorno de F0 em períodos de pitch.

    Args:
        f0: Array de frequências fundamentais
        sr: Taxa de amostragem

    Returns:
        Array de períodos de pitch em segundos
    """
    # Filtrar valores não vozeados (<= 0 ou NaN)
    voiced_mask = (f0 > 0) & (~np.isnan(f0))
    if np.sum(voiced_mask) == 0:
        return np.array([])

    # Converter F0 para períodos (T = 1/F)
    periods = 1.0 / f0[voiced_mask]
    return periods


def voiced_runs(f0: np.ndarray, sr: int) -> list[np.ndarray]:
    """Períodos de pitch agrupados por trecho vozeado CONTÍGUO.

    `extract_pitch_periods` remove os quadros não vozeados e concatena o que
    sobra. Isso quebra a premissa de RAP, PPQ e shimmer, que são medidas
    ciclo-a-ciclo: dois períodos vizinhos NO ARRAY podem estar separados por
    centenas de milissegundos de silêncio no tempo, e a junção artificial vira
    "perturbação" que não existe.

    Medido: um sinal com F0 constante de 200 Hz, uma pausa, e depois F0
    constante de 120 Hz — jitter real ZERO — produzia RAP = 0,0088. Jitter
    saudável fica em torno de 0,005, então o artefato sozinho ultrapassava o
    limiar patológico.

    Devolve uma lista de arrays, um por trecho vozeado contíguo, para que as
    métricas sejam calculadas dentro de cada trecho e agregadas depois — que é
    como o Praat trata descontinuidade de vozeamento.
    """
    f0 = np.asarray(f0, dtype=float)
    vozeado = (f0 > 0) & (~np.isnan(f0))
    if not vozeado.any():
        return []

    trechos: list[np.ndarray] = []
    inicio = None
    for i, ativo in enumerate(vozeado):
        if ativo and inicio is None:
            inicio = i
        elif not ativo and inicio is not None:
            trechos.append(1.0 / f0[inicio:i])
            inicio = None
    if inicio is not None:
        trechos.append(1.0 / f0[inicio:])
    return trechos


def compute_amplitude_envelope(
        y: np.ndarray, frame_length: int = 2048,
        hop_length: int = 512) -> np.ndarray:
    """Calcula envelope de amplitude do sinal."""
    # Usar RMS por frame como aproximação da amplitude
    import librosa
    rms = librosa.feature.rms(
        y=y,
        frame_length=frame_length,
        hop_length=hop_length)[0]
    return rms
