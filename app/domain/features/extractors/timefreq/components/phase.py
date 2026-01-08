"""
Extração de características de fase.
"""
import numpy as np
import scipy.signal
from typing import Dict


def extract_phase_features(y: np.ndarray, sr: int,
                           frame_length: int, hop_length: int) -> Dict:
    features = {}
    f, t, Zxx = scipy.signal.stft(y, fs=sr,
                                  nperseg=frame_length,
                                  noverlap=frame_length - hop_length)
    phase = np.angle(Zxx)
    if phase.shape[1] > 1:
        phase_unwrapped = np.unwrap(phase, axis=1)
        phase_derivative_unwrapped = np.diff(phase_unwrapped, axis=1)
        instantaneous_freq = phase_derivative_unwrapped / \
            (2 * np.pi * (t[1] - t[0]))
        features['instantaneous_freq_mean'] = np.mean(instantaneous_freq)
        features['instantaneous_freq_std'] = np.std(instantaneous_freq)
        features['instantaneous_freq_range'] = np.max(
            instantaneous_freq) - np.min(instantaneous_freq)
        if instantaneous_freq.shape[1] > 1:
            freq_variation = np.diff(instantaneous_freq, axis=1)
            features['instantaneous_freq_variation'] = np.std(freq_variation)
    phase_coherence = []
    for i in range(phase.shape[0] - 1):
        phase_diff = phase[i + 1, :] - phase[i, :]
        coherence = np.abs(np.mean(np.exp(1j * phase_diff)))
        phase_coherence.append(coherence)
    if phase_coherence:
        features['phase_coherence_mean'] = np.mean(phase_coherence)
        features['phase_coherence_std'] = np.std(phase_coherence)
    phase_dispersion = []
    for frame_phase in phase.T:
        complex_phase = np.exp(1j * frame_phase)
        mean_direction = np.mean(complex_phase)
        dispersion = 1 - np.abs(mean_direction)
        phase_dispersion.append(dispersion)
    if phase_dispersion:
        features['phase_dispersion_mean'] = np.mean(phase_dispersion)
        features['phase_dispersion_std'] = np.std(phase_dispersion)
    return features
