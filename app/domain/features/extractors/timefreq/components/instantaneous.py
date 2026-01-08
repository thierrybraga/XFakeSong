"""
Extração de características instantâneas via transformada de Hilbert.
"""
import numpy as np
from typing import Dict
from scipy.signal import hilbert


def extract_instantaneous_features(y: np.ndarray, sr: int) -> Dict:
    features = {}
    analytic_signal = hilbert(y)
    instantaneous_amplitude = np.abs(analytic_signal)
    instantaneous_phase = np.angle(analytic_signal)
    instantaneous_phase_unwrapped = np.unwrap(instantaneous_phase)
    if len(instantaneous_phase_unwrapped) > 1:
        instantaneous_frequency = np.diff(
            instantaneous_phase_unwrapped) / (2 * np.pi / sr)
    else:
        instantaneous_frequency = np.array([0])
    features['instantaneous_amplitude_mean'] = np.mean(instantaneous_amplitude)
    features['instantaneous_amplitude_std'] = np.std(instantaneous_amplitude)
    features['instantaneous_amplitude_max'] = np.max(instantaneous_amplitude)
    if len(instantaneous_amplitude) > 1:
        amplitude_variation = np.diff(instantaneous_amplitude)
        features['instantaneous_amplitude_variation'] = np.std(
            amplitude_variation)
    if len(instantaneous_frequency) > 0:
        valid_freqs = instantaneous_frequency[(
            instantaneous_frequency > 0) & (instantaneous_frequency < sr / 2)]
        if len(valid_freqs) > 0:
            features['instantaneous_frequency_mean'] = np.mean(valid_freqs)
            features['instantaneous_frequency_std'] = np.std(valid_freqs)
            features['instantaneous_frequency_range'] = np.max(
                valid_freqs) - np.min(valid_freqs)
            if len(valid_freqs) > 1:
                freq_variation = np.diff(valid_freqs)
                features['instantaneous_frequency_variation'] = np.std(
                    freq_variation)
    phase_dispersion = 1 - np.abs(np.mean(np.exp(1j * instantaneous_phase)))
    features['instantaneous_phase_dispersion'] = phase_dispersion
    if len(instantaneous_phase) > 1:
        phase_variation = np.diff(instantaneous_phase_unwrapped)
        features['instantaneous_phase_variation'] = np.std(phase_variation)
    return features
