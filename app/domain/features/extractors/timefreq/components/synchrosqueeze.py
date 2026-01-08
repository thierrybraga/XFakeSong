"""
Extração de características de synchrosqueezing (aproximação).
"""
import numpy as np
import scipy.signal
from typing import Dict


def extract_synchrosqueezing_features(
        y: np.ndarray, sr: int, frame_length: int, hop_length: int) -> Dict:
    features = {}
    f, t, Zxx = scipy.signal.stft(y, fs=sr,
                                  nperseg=frame_length,
                                  noverlap=frame_length - hop_length)
    magnitude = np.abs(Zxx)
    phase = np.angle(Zxx)
    if magnitude.shape[1] > 2:
        phase_unwrapped = np.unwrap(phase, axis=1)
        instantaneous_freq = np.diff(
            phase_unwrapped, axis=1) / (2 * np.pi * (t[1] - t[0]))
        synchro_concentration = []
        for i in range(instantaneous_freq.shape[1]):
            frame_magnitude = magnitude[:, i + 1]
            peaks, _ = scipy.signal.find_peaks(
                frame_magnitude, height=np.max(frame_magnitude) * 0.1)
            if len(peaks) > 0:
                peak_energy = np.sum(frame_magnitude[peaks])
                total_energy = np.sum(frame_magnitude)
                if total_energy > 0:
                    concentration = peak_energy / total_energy
                    synchro_concentration.append(concentration)
        if synchro_concentration:
            features['synchro_concentration_mean'] = np.mean(
                synchro_concentration)
            features['synchro_concentration_std'] = np.std(
                synchro_concentration)
            if len(synchro_concentration) > 1:
                concentration_variation = np.diff(synchro_concentration)
                features['synchro_stability'] = 1 / \
                    (1 + np.std(concentration_variation))
    return features
