import numpy as np
from typing import Dict


def compute_formant_trajectories(formants_data: Dict, n_formants: int) -> Dict:
    """Computa trajetórias dos formantes (velocidades de mudança)."""
    trajectories = {}

    for i in range(n_formants):
        formant_name = f'F{i + 1}'
        freq_key = f'{formant_name}_freq'

        if freq_key in formants_data:
            freqs = formants_data[freq_key]

            # Calcular primeira derivada (velocidade)
            valid_mask = ~np.isnan(freqs)
            if np.sum(valid_mask) > 1:
                # Interpolação linear para lidar com NaN
                valid_indices = np.where(valid_mask)[0]
                valid_freqs = freqs[valid_mask]

                if len(valid_freqs) > 1:
                    # Calcular derivada
                    interp_freqs = np.interp(
                        range(len(freqs)), valid_indices, valid_freqs)
                    trajectory = np.gradient(interp_freqs)
                    trajectories[f'{formant_name}_trajectory_mean'] = np.mean(
                        trajectory)
                    trajectories[f'{formant_name}_trajectory_std'] = np.std(
                        trajectory)
                else:
                    trajectories[f'{formant_name}_trajectory_mean'] = 0
                    trajectories[f'{formant_name}_trajectory_std'] = 0
            else:
                trajectories[f'{formant_name}_trajectory_mean'] = 0
                trajectories[f'{formant_name}_trajectory_std'] = 0

    return trajectories


def compute_vowel_space_area(f1: np.ndarray, f2: np.ndarray) -> float:
    """Computa área do espaço vocálico (triângulo F1-F2)."""
    # Remover NaN
    valid_mask = ~np.isnan(f1) & ~np.isnan(f2)

    if np.sum(valid_mask) < 3:
        return 0

    f1_valid = f1[valid_mask]
    f2_valid = f2[valid_mask]

    # Encontrar pontos extremos para formar triângulo
    # Aproximação: usar percentis para representar espaço vocálico
    f1_low = np.percentile(f1_valid, 10)   # Vogal baixa
    f1_high = np.percentile(f1_valid, 90)  # Vogal alta
    f2_front = np.percentile(f2_valid, 90)  # Vogal frontal
    f2_back = np.percentile(f2_valid, 10)  # Vogal posterior

    # Calcular área aproximada do espaço vocálico
    # Usando fórmula de área de triângulo
    vertices = np.array([
        [f1_low, f2_front],   # Vogal baixa frontal (ex: /æ/)
        [f1_low, f2_back],    # Vogal baixa posterior (ex: /ɑ/)
        [f1_high, f2_front]   # Vogal alta frontal (ex: /i/)
    ])

    # Fórmula da área usando produto cruzado
    area = 0.5 * abs(
        (vertices[1][0] - vertices[0][0]) * (vertices[2][1] - vertices[0][1]) -
        (vertices[2][0] - vertices[0][0]) * (vertices[1][1] - vertices[0][1])
    )

    return area


def compute_formant_dispersion(formants_data: Dict, n_formants: int) -> float:
    """Computa dispersão dos formantes."""
    # Coletar todas as frequências válidas
    all_freqs = []

    for i in range(n_formants):
        freq_key = f'F{i + 1}_freq'
        if freq_key in formants_data:
            freqs = formants_data[freq_key]
            valid_freqs = freqs[~np.isnan(freqs)]
            all_freqs.extend(valid_freqs)

    if len(all_freqs) < 2:
        return 0

    # Dispersão como desvio padrão das frequências
    dispersion = np.std(all_freqs)
    return dispersion


def compute_effective_f2(f1: np.ndarray, f2: np.ndarray,
                         f3: np.ndarray) -> np.ndarray:
    """Computa F2' efetivo (formante ajustado)."""
    # F2' leva em conta a influência de F3 em F2
    # Fórmula aproximada: F2' = F2 + k * (F3 - F2) onde k é um fator

    k = 0.1  # Fator de ajuste

    valid_mask = ~np.isnan(f1) & ~np.isnan(f2) & ~np.isnan(f3)

    f2_effective = f2.copy()
    f2_effective[valid_mask] = (f2[valid_mask] +
                                k * (f3[valid_mask] - f2[valid_mask]))

    return f2_effective


def compute_formant_ratios(formants_data: Dict) -> Dict:
    """Computa razões entre formantes."""
    ratios = {}

    # F2/F1 ratio (indicador de frontalidade)
    f1 = formants_data['F1_freq']
    f2 = formants_data['F2_freq']
    f3 = formants_data['F3_freq']

    valid_mask_12 = ~np.isnan(f1) & ~np.isnan(f2) & (f1 > 0)
    if np.any(valid_mask_12):
        ratio_f2_f1 = f2[valid_mask_12] / f1[valid_mask_12]
        ratios['F2_F1_ratio_mean'] = np.mean(ratio_f2_f1)
        ratios['F2_F1_ratio_std'] = np.std(ratio_f2_f1)
    else:
        ratios['F2_F1_ratio_mean'] = 0
        ratios['F2_F1_ratio_std'] = 0

    # F3/F2 ratio
    valid_mask_32 = ~np.isnan(f2) & ~np.isnan(f3) & (f2 > 0)
    if np.any(valid_mask_32):
        ratio_f3_f2 = f3[valid_mask_32] / f2[valid_mask_32]
        ratios['F3_F2_ratio_mean'] = np.mean(ratio_f3_f2)
        ratios['F3_F2_ratio_std'] = np.std(ratio_f3_f2)
    else:
        ratios['F3_F2_ratio_mean'] = 0
        ratios['F3_F2_ratio_std'] = 0

    return ratios
