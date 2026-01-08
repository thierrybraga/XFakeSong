"""
Extração de características de estabilidade.
"""

import numpy as np
from typing import Dict
from .utils import frame_signal, compute_lpc, check_stability


def extract_stability_features(
        y: np.ndarray, frame_length: int, hop_length: int,
        lpc_order: int) -> Dict:
    """Extrai características de estabilidade do modelo."""
    features = {}

    # Dividir sinal em frames
    frames = frame_signal(y, frame_length, hop_length)

    stable_frames = 0
    total_frames = 0
    all_pole_magnitudes = []

    for frame in frames:
        if len(frame) > lpc_order + 1:
            total_frames += 1

            # Calcular coeficientes LPC
            lpc_coeffs, _ = compute_lpc(frame, lpc_order)

            if lpc_coeffs is not None:
                # Verificar estabilidade através dos polos
                is_stable, pole_magnitudes = check_stability(lpc_coeffs)

                if is_stable:
                    stable_frames += 1

                if pole_magnitudes is not None:
                    all_pole_magnitudes.extend(pole_magnitudes)

    # === CARACTERÍSTICAS DE ESTABILIDADE ===

    if total_frames > 0:
        features['stability_ratio'] = stable_frames / total_frames
    else:
        features['stability_ratio'] = 0

    if all_pole_magnitudes:
        all_pole_magnitudes = np.array(all_pole_magnitudes)

        # Estatísticas das magnitudes dos polos
        features['pole_magnitude_mean'] = np.mean(all_pole_magnitudes)
        features['pole_magnitude_std'] = np.std(all_pole_magnitudes)
        features['pole_magnitude_max'] = np.max(all_pole_magnitudes)

        # Margem de estabilidade (distância do círculo unitário)
        stability_margin = 1 - np.max(all_pole_magnitudes)
        features['stability_margin'] = stability_margin

        # Número de polos próximos ao círculo unitário
        near_unstable = np.sum(all_pole_magnitudes > 0.95)
        features['near_unstable_poles'] = near_unstable / \
            len(all_pole_magnitudes)

    return features
