"""
Extração de Características de Análise Tempo-Frequência
======================================================

Este módulo implementa características baseadas em representações tempo-frequência,
incluindo espectrograma, transformadas reassigned, EMD, VMD e outras técnicas avançadas.
"""

from typing import Dict, Optional

import numpy as np
import scipy.signal

from .components import (
    extract_emd_features,
    extract_instantaneous_features,
    extract_phase_features,
    extract_reassigned_features,
    extract_spectrogram_features,
    extract_synchrosqueezing_features,
    extract_vmd_features,
)


class TimeFrequencyFeatureExtractor:
    """
    Extrator de características de análise tempo-frequência.
    """

    def __init__(self, sr: int = 22050, frame_length: int = 2048,
                 hop_length: int = 512, n_fft: int = 2048):
        """
        Inicializa o extrator de características tempo-frequência.

        Args:
            sr: Taxa de amostragem
            frame_length: Comprimento da janela
            hop_length: Salto entre janelas
            n_fft: Tamanho da FFT
        """
        self.sr = sr
        self.frame_length = frame_length
        self.hop_length = hop_length
        self.n_fft = n_fft

    def extract_features(self, y: np.ndarray) -> Dict:
        """
        Extrai todas as características tempo-frequência.

        Args:
            y: Sinal de áudio

        Returns:
            Dicionário com características tempo-frequência
        """
        features = {}

        # === CARACTERÍSTICAS DO ESPECTROGRAMA ===
        spectrogram_features = extract_spectrogram_features(
            y, self.sr, self.frame_length, self.hop_length)
        features.update(spectrogram_features)

        # === CARACTERÍSTICAS DE FASE ===
        phase_features = extract_phase_features(
            y, self.sr, self.frame_length, self.hop_length)
        features.update(phase_features)

        # === TRANSFORMADA REASSIGNED ===
        reassigned_features = extract_reassigned_features(
            y, self.sr, self.frame_length, self.hop_length)
        features.update(reassigned_features)

        # === SYNCHROSQUEEZING ===
        synchro_features = extract_synchrosqueezing_features(
            y, self.sr, self.frame_length, self.hop_length)
        features.update(synchro_features)

        # === EMPIRICAL MODE DECOMPOSITION ===
        emd_features = extract_emd_features(y, self.sr)
        features.update(emd_features)

        # === VARIATIONAL MODE DECOMPOSITION ===
        vmd_features = extract_vmd_features(y, self.sr)
        features.update(vmd_features)

        # === CARACTERÍSTICAS INSTANTÂNEAS ===
        instantaneous_features = extract_instantaneous_features(y, self.sr)
        features.update(instantaneous_features)

        return features

    # Métodos auxiliares removidos em favor de componentes especializados

    def _simple_emd(self, y: np.ndarray, max_imfs: int = 5) -> list:
        """Implementação simplificada de EMD."""
        imfs = []
        residue = y.copy()

        for _ in range(max_imfs):
            # Critério de parada simples
            if len(residue) < 10 or np.std(residue) < 1e-6:
                break

            # Processo de peneiramento simplificado
            imf = self._sifting_process(residue)

            if imf is not None and len(imf) > 0:
                imfs.append(imf)
                residue = residue - imf
            else:
                break

        return imfs

    def _sifting_process(self, signal: np.ndarray,
                         max_iterations: int = 10) -> Optional[np.ndarray]:
        """Processo de peneiramento para EMD."""
        h = signal.copy()

        for _ in range(max_iterations):
            # Encontrar máximos e mínimos locais
            maxima_indices = scipy.signal.find_peaks(h)[0]
            minima_indices = scipy.signal.find_peaks(-h)[0]

            if len(maxima_indices) < 2 or len(minima_indices) < 2:
                break

            # Interpolar envelopes (simplificado)
            try:
                # Envelope superior
                x = np.arange(len(h))
                upper_envelope = np.interp(
                    x, maxima_indices, h[maxima_indices])

                # Envelope inferior
                lower_envelope = np.interp(
                    x, minima_indices, h[minima_indices])

                # Média dos envelopes
                mean_envelope = (upper_envelope + lower_envelope) / 2

                # Atualizar h
                h_new = h - mean_envelope

                # Critério de parada
                if np.sum((h - h_new) ** 2) / np.sum(h ** 2) < 0.01:
                    break

                h = h_new

            except BaseException:
                break

        return h


def test_timefreq_features():
    """Testa as características tempo-frequência."""
    # Gerar sinal de teste com características tempo-frequência variadas
    duration = 2.0
    sr = 22050
    t = np.linspace(0, duration, int(duration * sr))

    # Sinal com frequência variável (chirp)
    f_start, f_end = 440, 1760
    y_chirp = scipy.signal.chirp(t, f_start, duration, f_end, method='linear')

    # Adicionar componente harmônica
    y_harmonic = 0.5 * np.sin(2 * np.pi * 880 * t)

    # Adicionar modulação em amplitude
    mod_freq = 5  # Hz
    y_modulated = y_chirp * (1 + 0.3 * np.sin(2 * np.pi * mod_freq * t))

    # Sinal final
    y = y_modulated + y_harmonic

    # Adicionar envelope
    envelope = np.exp(-0.5 * t) + 0.3
    y = y * envelope

    # Extrair características
    extractor = TimeFrequencyFeatureExtractor(sr=sr)
    features = extractor.extract_features(y)

    print("⏰ Características Tempo-Frequência Extraídas:")

    # Agrupar características por categoria
    categories = {
        'Espectrograma': [k for k in features.keys() if 'spectral' in k or 'temporal' in k or 'timefreq' in k],
        'Fase': [k for k in features.keys() if 'phase' in k],
        'Reassigned': [k for k in features.keys() if 'reassigned' in k],
        'Synchrosqueezing': [k for k in features.keys() if 'synchro' in k],
        'EMD': [k for k in features.keys() if 'emd' in k],
        'VMD': [k for k in features.keys() if 'vmd' in k],
        'Instantâneas': [k for k in features.keys() if 'instantaneous' in k]
    }

    for category, feature_names in categories.items():
        if feature_names:
            print(f"\n📊 {category}:")
            for name in feature_names[:5]:  # Primeiras 5 de cada categoria
                if name in features:
                    value = features[name]
                    if isinstance(value, (int, float)):
                        print(f"  {name}: {value:.3f}")

    print(f"\n📈 Total de características extraídas: {len(features)}")


if __name__ == "__main__":
    test_timefreq_features()
