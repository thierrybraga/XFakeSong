"""Extração de Características Cepstrais Avançadas
=====================================================

Implementa características cepstrais avançadas incluindo PLP, RASTA-PLP,
LPCC, Gammatone Cepstral Coefficients e outras variações especializadas.
"""

import numpy as np
import warnings
from typing import Dict

from .components.lpc_utils import (
    solve_yule_walker, lpc_to_cepstral, compute_lpc, apply_preemphasis
)
from .components.plp import (
    extract_plp_features, extract_rasta_plp_features,
    apply_bark_scale, apply_equal_loudness_curve, apply_rasta_filter
)
from .components.lpcc import extract_lpcc_features
from .components.gammatone import (
    extract_gtcc_features, apply_gammatone_filterbank, hz_to_erb, erb_to_hz
)
from .components.pncc import (
    extract_pncc_features, extract_mhec_features,
    power_normalize, apply_harmonic_emphasis
)


class AdvancedCepstralFeatureExtractor:
    """Extrator de características cepstrais avançadas."""

    def __init__(self, sr: int = 22050, frame_length: int = 2048,
                 hop_length: int = 512, n_mels: int = 40, n_mfcc: int = 13,
                 n_plp: int = 13, n_lpcc: int = 13, preemphasis: float = 0.97,
                 fmin: float = 80.0, fmax: float = 8000.0):
        """
        Inicializa o extrator de características cepstrais avançadas.

        Args:
            sr: Taxa de amostragem
            frame_length: Comprimento da janela
            hop_length: Salto entre janelas
            n_mels: Número de filtros mel
            n_mfcc: Número de coeficientes MFCC
            n_plp: Número de coeficientes PLP
            n_lpcc: Número de coeficientes LPCC
            preemphasis: Coeficiente de pré-ênfase
            fmin: Frequência mínima
            fmax: Frequência máxima
        """
        self.sr = sr
        self.frame_length = frame_length
        self.hop_length = hop_length
        self.n_mels = n_mels
        self.n_mfcc = n_mfcc
        self.n_plp = n_plp
        self.n_lpcc = n_lpcc
        self.preemphasis = preemphasis
        self.fmin = fmin
        self.fmax = fmax

        # Parâmetros para filtros Gammatone
        self.n_gammatone = 64
        self.gammatone_order = 4

    def extract_features(self, y: np.ndarray) -> Dict:
        """
        Extrai todas as características cepstrais avançadas.

        Args:
            y: Sinal de áudio

        Returns:
            Dicionário com características cepstrais avançadas
        """
        if len(y) == 0:
            return self._get_default_features()

        # Aplicar pré-ênfase
        y = self._apply_preemphasis(y)

        # Normalizar
        y = y / (np.max(np.abs(y)) + 1e-10)

        features = {}

        try:
            # === PLP (Perceptual Linear Prediction) ===
            plp_features = self._extract_plp_features(y)
            features.update(plp_features)

            # === RASTA-PLP ===
            rasta_plp_features = self._extract_rasta_plp_features(y)
            features.update(rasta_plp_features)

            # === LPCC (Linear Predictive Cepstral Coefficients) ===
            lpcc_features = self._extract_lpcc_features(y)
            features.update(lpcc_features)

            # === Gammatone Cepstral Coefficients ===
            gtcc_features = self._extract_gtcc_features(y)
            features.update(gtcc_features)

            # === Power-Normalized Cepstral Coefficients (PNCC) ===
            pncc_features = self._extract_pncc_features(y)
            features.update(pncc_features)

            # === Mel-scale Harmonic Emphasis Cepstral (MHEC) ===
            mhec_features = self._extract_mhec_features(y)
            features.update(mhec_features)

        except Exception as e:
            warnings.warn(
                f"Erro na extração de características cepstrais avançadas: "
                f"{str(e)}"
            )
            features = self._get_default_features()

        return features

    def _apply_preemphasis(self, y: np.ndarray) -> np.ndarray:
        """Aplica pré-ênfase ao sinal."""
        return apply_preemphasis(y, self.preemphasis)

    def _extract_plp_features(self, y: np.ndarray) -> Dict:
        """Extrai características PLP (Perceptual Linear Prediction)."""
        return extract_plp_features(
            y, self.sr, self.frame_length, self.hop_length, self.n_plp
        )

    def _extract_rasta_plp_features(self, y: np.ndarray) -> Dict:
        """Extrai características RASTA-PLP."""
        return extract_rasta_plp_features(
            y, self.sr, self.frame_length, self.hop_length, self.n_plp
        )

    def _extract_lpcc_features(self, y: np.ndarray) -> Dict:
        """Extrai LPCC (Linear Predictive Cepstral Coefficients)."""
        return extract_lpcc_features(
            y, self.frame_length, self.hop_length, self.n_lpcc
        )

    def _extract_gtcc_features(self, y: np.ndarray) -> Dict:
        """Extrai Gammatone Cepstral Coefficients."""
        return extract_gtcc_features(
            y, self.sr, self.frame_length, self.hop_length,
            self.n_mfcc, self.fmin, self.fmax
        )

    def _extract_pncc_features(self, y: np.ndarray) -> Dict:
        """Extrai Power-Normalized Cepstral Coefficients."""
        return extract_pncc_features(
            y, self.sr, self.frame_length, self.hop_length,
            self.n_mfcc, self.n_mels, self.fmin, self.fmax
        )

    def _extract_mhec_features(self, y: np.ndarray) -> Dict:
        """Extrai Mel-scale Harmonic Emphasis Cepstral coefficients."""
        return extract_mhec_features(
            y, self.sr, self.frame_length, self.hop_length,
            self.n_mfcc, self.n_mels, self.fmin, self.fmax
        )

    def _get_default_features(self) -> Dict:
        """Retorna características padrão preenchidas com zeros."""
        features = {}

        # PLP e RASTA-PLP
        for name in ['plp', 'rasta_plp']:
            features[name] = np.zeros((self.n_plp, 1))
            features[f'{name}_delta'] = np.zeros((self.n_plp, 1))
            features[f'{name}_delta2'] = np.zeros((self.n_plp, 1))

        # LPCC
        features['lpcc'] = np.zeros((self.n_lpcc, 1))
        features['lpcc_delta'] = np.zeros((self.n_lpcc, 1))
        features['lpcc_delta2'] = np.zeros((self.n_lpcc, 1))

        # GTCC, PNCC, MHEC
        for name in ['gtcc', 'pncc', 'mhec']:
            features[name] = np.zeros((self.n_mfcc, 1))
            features[f'{name}_delta'] = np.zeros((self.n_mfcc, 1))
            features[f'{name}_delta2'] = np.zeros((self.n_mfcc, 1))

        return features

    # Backward compatibility methods (proxies to components)
    def _apply_bark_scale(self, S: np.ndarray) -> np.ndarray:
        return apply_bark_scale(S, self.sr, self.frame_length)

    def _apply_equal_loudness_curve(self, spectrum: np.ndarray) -> np.ndarray:
        return apply_equal_loudness_curve(spectrum)

    def _solve_yule_walker(self, r: np.ndarray, order: int) -> np.ndarray:
        return solve_yule_walker(r, order)

    def _lpc_to_cepstral(self, lpc_coeffs: np.ndarray) -> np.ndarray:
        return lpc_to_cepstral(lpc_coeffs, self.n_mfcc)

    def _compute_lpc(self, frame: np.ndarray, order: int) -> np.ndarray:
        return compute_lpc(frame, order)

    def _apply_rasta_filter(self, spectrum: np.ndarray) -> np.ndarray:
        return apply_rasta_filter(spectrum)

    def _apply_gammatone_filterbank(self, y: np.ndarray) -> np.ndarray:
        return apply_gammatone_filterbank(
            y, self.sr, self.frame_length, self.hop_length,
            self.fmin, self.fmax, self.n_gammatone, self.gammatone_order
        )

    def _hz_to_erb(self, hz: float) -> float:
        return hz_to_erb(hz)

    def _erb_to_hz(self, erb: float) -> float:
        return erb_to_hz(erb)

    def _power_normalize(self, spectrum: np.ndarray) -> np.ndarray:
        return power_normalize(spectrum)

    def _apply_harmonic_emphasis(self, S: np.ndarray) -> np.ndarray:
        return apply_harmonic_emphasis(S)
