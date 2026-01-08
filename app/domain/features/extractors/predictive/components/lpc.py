"""
Extração de características LPC e LSF.
"""

import numpy as np
from typing import Dict
from .utils import (
    frame_signal, compute_lpc, lpc_to_lsf,
    lpc_to_frequency_response, compute_spectral_tilt, compute_spectral_flatness
)


def extract_lpc_features(y: np.ndarray, sr: int, frame_length: int,
                         hop_length: int, lpc_order: int) -> Dict:
    """Extrai características dos coeficientes LPC."""
    features = {}

    # Dividir sinal em frames
    frames = frame_signal(y, frame_length, hop_length)

    all_lpc_coeffs = []
    all_gains = []

    for frame in frames:
        if len(frame) > lpc_order + 1:
            # Calcular coeficientes LPC
            lpc_coeffs, gain = compute_lpc(frame, lpc_order)

            if lpc_coeffs is not None:
                all_lpc_coeffs.append(lpc_coeffs)
                all_gains.append(gain)

    if all_lpc_coeffs:
        all_lpc_coeffs = np.array(all_lpc_coeffs)
        all_gains = np.array(all_gains)

        # === CARACTERÍSTICAS DOS COEFICIENTES LPC ===

        # Estatísticas dos coeficientes individuais
        for i in range(min(lpc_order, all_lpc_coeffs.shape[1])):
            coeff_values = all_lpc_coeffs[:, i]
            features[f'lpc_coeff_{i + 1}_mean'] = np.mean(coeff_values)
            features[f'lpc_coeff_{i + 1}_std'] = np.std(coeff_values)

        # Estatísticas globais dos coeficientes
        features['lpc_coeffs_mean'] = np.mean(all_lpc_coeffs)
        features['lpc_coeffs_std'] = np.std(all_lpc_coeffs)
        features['lpc_coeffs_range'] = np.max(
            all_lpc_coeffs) - np.min(all_lpc_coeffs)

        # Variação temporal dos coeficientes
        if len(all_lpc_coeffs) > 1:
            lpc_variation = np.diff(all_lpc_coeffs, axis=0)
            features['lpc_temporal_variation'] = np.mean(
                np.std(lpc_variation, axis=0))

        # === CARACTERÍSTICAS DO GANHO ===
        features['lpc_gain_mean'] = np.mean(all_gains)
        features['lpc_gain_std'] = np.std(all_gains)
        features['lpc_gain_max'] = np.max(all_gains)
        features['lpc_gain_min'] = np.min(all_gains)

        # Variação temporal do ganho
        if len(all_gains) > 1:
            gain_variation = np.diff(all_gains)
            features['lpc_gain_variation'] = np.std(gain_variation)

        # === CARACTERÍSTICAS ESPECTRAIS DERIVADAS ===

        # Calcular resposta em frequência média
        freq_response = lpc_to_frequency_response(
            np.mean(all_lpc_coeffs, axis=0), sr)

        if freq_response is not None:
            features['lpc_spectral_tilt'] = compute_spectral_tilt(
                freq_response, sr)
            features['lpc_spectral_flatness'] = compute_spectral_flatness(
                freq_response)

    return features


def extract_lsf_features(y: np.ndarray, sr: int, frame_length: int,
                         hop_length: int, lpc_order: int) -> Dict:
    """Extrai características das Line Spectral Frequencies."""
    features = {}

    # Dividir sinal em frames
    frames = frame_signal(y, frame_length, hop_length)

    all_lsf = []

    for frame in frames:
        if len(frame) > lpc_order + 1:
            # Calcular coeficientes LPC
            lpc_coeffs, _ = compute_lpc(frame, lpc_order)

            if lpc_coeffs is not None:
                # Converter LPC para LSF
                lsf = lpc_to_lsf(lpc_coeffs, sr)

                if lsf is not None:
                    all_lsf.append(lsf)

    if all_lsf:
        all_lsf = np.array(all_lsf)

        # === CARACTERÍSTICAS DAS LSF ===

        # Estatísticas das LSF individuais
        for i in range(min(lpc_order, all_lsf.shape[1])):
            lsf_values = all_lsf[:, i]
            features[f'lsf_{i + 1}_mean'] = np.mean(lsf_values)
            features[f'lsf_{i + 1}_std'] = np.std(lsf_values)

        # Estatísticas globais das LSF
        features['lsf_mean'] = np.mean(all_lsf)
        features['lsf_std'] = np.std(all_lsf)
        features['lsf_range'] = np.max(all_lsf) - np.min(all_lsf)

        # Separação entre LSF adjacentes
        lsf_separations = []

        for lsf_frame in all_lsf:
            separations = np.diff(lsf_frame)
            lsf_separations.extend(separations)

        if lsf_separations:
            features['lsf_separation_mean'] = np.mean(lsf_separations)
            features['lsf_separation_std'] = np.std(lsf_separations)
            features['lsf_separation_min'] = np.min(lsf_separations)

        # Variação temporal das LSF
        if len(all_lsf) > 1:
            lsf_variation = np.diff(all_lsf, axis=0)
            features['lsf_temporal_variation'] = np.mean(
                np.std(lsf_variation, axis=0))

    return features
