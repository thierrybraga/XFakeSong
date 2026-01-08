"""
Extração de características de Empirical Mode Decomposition (EMD) simplificada.
"""
import numpy as np
import scipy.signal
from typing import Dict, Optional, List


def simple_emd(y: np.ndarray, max_imfs: int = 5) -> List[np.ndarray]:
    imfs = []
    residue = y.copy()
    for _ in range(max_imfs):
        if len(residue) < 10 or np.std(residue) < 1e-6:
            break
        imf = _sifting_process(residue)
        if imf is not None and len(imf) > 0:
            imfs.append(imf)
            residue = residue - imf
        else:
            break
    return imfs


def _sifting_process(signal: np.ndarray,
                     max_iterations: int = 10) -> Optional[np.ndarray]:
    h = signal.copy()
    for _ in range(max_iterations):
        maxima_indices = scipy.signal.find_peaks(h)[0]
        minima_indices = scipy.signal.find_peaks(-h)[0]
        if len(maxima_indices) < 2 or len(minima_indices) < 2:
            break
        try:
            x = np.arange(len(h))
            upper_envelope = np.interp(x, maxima_indices, h[maxima_indices])
            lower_envelope = np.interp(x, minima_indices, h[minima_indices])
            mean_envelope = (upper_envelope + lower_envelope) / 2
            h_new = h - mean_envelope
            if np.sum((h - h_new) ** 2) / np.sum(h ** 2) < 0.01:
                break
            h = h_new
        except BaseException:
            break
    return h


def extract_emd_features(y: np.ndarray, sr: int) -> Dict:
    features = {}
    if len(y) > 5000:
        y_emd = y[:5000]
    else:
        y_emd = y.copy()
    imfs = simple_emd(y_emd, max_imfs=5)
    if len(imfs) > 0:
        imf_energies = [np.sum(imf ** 2) for imf in imfs]
        total_energy = sum(imf_energies)
        if total_energy > 0:
            energy_ratios = [energy / total_energy for energy in imf_energies]
            features['emd_energy_ratio_imf1'] = energy_ratios[0] if len(
                energy_ratios) > 0 else 0
            features['emd_energy_ratio_imf2'] = energy_ratios[1] if len(
                energy_ratios) > 1 else 0
            features['emd_energy_ratio_imf3'] = energy_ratios[2] if len(
                energy_ratios) > 2 else 0
            entropy = -sum([ratio * np.log(ratio + 1e-12)
                           for ratio in energy_ratios if ratio > 0])
            features['emd_energy_entropy'] = entropy
        imf_frequencies = []
        for imf in imfs[:3]:
            if len(imf) > 10:
                zero_crossings = np.where(np.diff(np.signbit(imf)))[0]
                if len(zero_crossings) > 1:
                    freq = len(zero_crossings) / (2 * len(imf) / sr)
                    imf_frequencies.append(freq)
        if imf_frequencies:
            features['emd_freq_imf1'] = imf_frequencies[0] if len(
                imf_frequencies) > 0 else 0
            features['emd_freq_imf2'] = imf_frequencies[1] if len(
                imf_frequencies) > 1 else 0
            features['emd_freq_imf3'] = imf_frequencies[2] if len(
                imf_frequencies) > 2 else 0
            if len(imf_frequencies) > 1:
                freq_separation = np.mean(
                    np.diff(
                        sorted(
                            imf_frequencies,
                            reverse=True)))
                features['emd_frequency_separation'] = freq_separation
    return features
