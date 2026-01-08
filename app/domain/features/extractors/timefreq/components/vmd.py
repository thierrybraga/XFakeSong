"""
Extração de características de Variational Mode Decomposition (VMD) simplificada.
"""
import numpy as np
import scipy.signal
from typing import Dict


def extract_vmd_features(y: np.ndarray, sr: int) -> Dict:
    features = {}
    if len(y) > 3000:
        y_vmd = y[:3000]
    else:
        y_vmd = y.copy()
    nyquist = sr / 2
    bands = [
        (0, nyquist * 0.1),
        (nyquist * 0.1, nyquist * 0.3),
        (nyquist * 0.3, nyquist * 0.6),
        (nyquist * 0.6, nyquist * 0.9),
    ]
    modes = []
    for low_freq, high_freq in bands:
        if low_freq > 0:
            sos = scipy.signal.butter(
                4, [low_freq, high_freq], btype='band', fs=sr, output='sos')
        else:
            sos = scipy.signal.butter(
                4, high_freq, btype='low', fs=sr, output='sos')
        try:
            filtered = scipy.signal.sosfilt(sos, y_vmd)
            modes.append(filtered)
        except BaseException:
            modes.append(np.zeros_like(y_vmd))
    if len(modes) > 0:
        mode_energies = [np.sum(mode ** 2) for mode in modes]
        total_energy = sum(mode_energies)
        if total_energy > 0:
            energy_ratios = [energy / total_energy for energy in mode_energies]
            features['vmd_energy_ratio_mode1'] = energy_ratios[0] if len(
                energy_ratios) > 0 else 0
            features['vmd_energy_ratio_mode2'] = energy_ratios[1] if len(
                energy_ratios) > 1 else 0
            features['vmd_energy_ratio_mode3'] = energy_ratios[2] if len(
                energy_ratios) > 2 else 0
            features['vmd_energy_ratio_mode4'] = energy_ratios[3] if len(
                energy_ratios) > 3 else 0
            energy_balance = np.std(energy_ratios)
            features['vmd_energy_balance'] = energy_balance
        mode_correlations = []
        for i in range(len(modes)):
            for j in range(i + 1, len(modes)):
                if len(modes[i]) > 1 and len(modes[j]) > 1:
                    correlation = np.corrcoef(modes[i], modes[j])[0, 1]
                    if not np.isnan(correlation):
                        mode_correlations.append(abs(correlation))
        if mode_correlations:
            features['vmd_mode_correlation_mean'] = np.mean(mode_correlations)
            features['vmd_mode_correlation_max'] = np.max(mode_correlations)
    return features
