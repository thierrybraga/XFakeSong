"""
Extração de coeficientes de reflexão e PARCOR.
"""

import numpy as np
from typing import Dict
from .utils import frame_signal, compute_reflection_coefficients


def extract_reflection_features(
        y: np.ndarray, frame_length: int, hop_length: int,
        lpc_order: int) -> Dict:
    """Extrai características dos coeficientes de reflexão."""
    features = {}

    # Dividir sinal em frames
    frames = frame_signal(y, frame_length, hop_length)

    all_reflection_coeffs = []

    for frame in frames:
        if len(frame) > lpc_order + 1:
            # Calcular coeficientes de reflexão usando algoritmo de
            # Levinson-Durbin
            reflection_coeffs = compute_reflection_coefficients(
                frame, lpc_order)

            if reflection_coeffs is not None:
                all_reflection_coeffs.append(reflection_coeffs)

    if all_reflection_coeffs:
        all_reflection_coeffs = np.array(all_reflection_coeffs)

        # === CARACTERÍSTICAS DOS COEFICIENTES DE REFLEXÃO ===

        # Estatísticas dos coeficientes individuais
        for i in range(min(lpc_order, all_reflection_coeffs.shape[1])):
            refl_values = all_reflection_coeffs[:, i]
            features[f'reflection_coeff_{i + 1}_mean'] = np.mean(refl_values)
            features[f'reflection_coeff_{i + 1}_std'] = np.std(refl_values)

        # Estatísticas globais
        features['reflection_coeffs_mean'] = np.mean(all_reflection_coeffs)
        features['reflection_coeffs_std'] = np.std(all_reflection_coeffs)
        features['reflection_coeffs_range'] = np.max(
            all_reflection_coeffs) - np.min(all_reflection_coeffs)

        # Estabilidade (coeficientes de reflexão devem estar em [-1, 1])
        stability_violations = np.sum(np.abs(all_reflection_coeffs) >= 1.0)
        total_coeffs = all_reflection_coeffs.size
        features['reflection_stability_ratio'] = 1 - \
            (stability_violations / total_coeffs)

        # Variação temporal
        if len(all_reflection_coeffs) > 1:
            refl_variation = np.diff(all_reflection_coeffs, axis=0)
            features['reflection_temporal_variation'] = np.mean(
                np.std(refl_variation, axis=0))

    return features


def extract_parcor_features(
        y: np.ndarray, frame_length: int, hop_length: int,
        lpc_order: int) -> Dict:
    """Extrai características dos coeficientes PARCOR (Partial Correlation)."""
    features = {}

    # PARCOR são essencialmente os coeficientes de reflexão
    # mas com interpretação de correlação parcial

    # Dividir sinal em frames
    frames = frame_signal(y, frame_length, hop_length)

    all_parcor_coeffs = []

    for frame in frames:
        if len(frame) > lpc_order + 1:
            # Calcular PARCOR (mesmo que coeficientes de reflexão)
            parcor_coeffs = compute_reflection_coefficients(frame, lpc_order)

            if parcor_coeffs is not None:
                all_parcor_coeffs.append(parcor_coeffs)

    if all_parcor_coeffs:
        all_parcor_coeffs = np.array(all_parcor_coeffs)

        # === CARACTERÍSTICAS PARCOR ===

        # Estatísticas dos coeficientes individuais
        for i in range(min(lpc_order, all_parcor_coeffs.shape[1])):
            parcor_values = all_parcor_coeffs[:, i]
            features[f'parcor_coeff_{i + 1}_mean'] = np.mean(parcor_values)
            features[f'parcor_coeff_{i + 1}_std'] = np.std(parcor_values)

        # Estatísticas globais
        features['parcor_coeffs_mean'] = np.mean(all_parcor_coeffs)
        features['parcor_coeffs_std'] = np.std(all_parcor_coeffs)

        # Correlação parcial dominante
        parcor_magnitudes = np.abs(all_parcor_coeffs)
        features['parcor_max_magnitude'] = np.max(parcor_magnitudes)
        features['parcor_mean_magnitude'] = np.mean(parcor_magnitudes)

        # Ordem efetiva (último coeficiente significativo)
        effective_orders = []

        for parcor_frame in all_parcor_coeffs:
            # Encontrar último coeficiente com magnitude > threshold
            threshold = 0.1
            significant_indices = np.where(np.abs(parcor_frame) > threshold)[0]

            if len(significant_indices) > 0:
                effective_order = significant_indices[-1] + 1
            else:
                effective_order = 0

            effective_orders.append(effective_order)

        if effective_orders:
            features['parcor_effective_order_mean'] = np.mean(effective_orders)
            features['parcor_effective_order_std'] = np.std(effective_orders)

    return features
