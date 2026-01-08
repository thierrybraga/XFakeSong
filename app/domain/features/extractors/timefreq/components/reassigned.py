"""
Extração de características da transformada reassigned (aproximação).
"""
import numpy as np
import scipy.signal
from typing import Dict


def extract_reassigned_features(
        y: np.ndarray, sr: int, frame_length: int, hop_length: int) -> Dict:
    features = {}
    window1 = scipy.signal.windows.hann(frame_length)
    window2 = scipy.signal.windows.hann(frame_length) * np.arange(frame_length)
    f, t, Zxx1 = scipy.signal.stft(y, fs=sr, window=window1,
                                   nperseg=frame_length,
                                   noverlap=frame_length - hop_length)
    f, t, Zxx2 = scipy.signal.stft(y, fs=sr, window=window2,
                                   nperseg=frame_length,
                                   noverlap=frame_length - hop_length)
    magnitude1 = np.abs(Zxx1)
    magnitude2 = np.abs(Zxx2)
    if np.sum(magnitude1) > 0 and np.sum(magnitude2) > 0:
        concentration_ratio = np.sum(magnitude2 ** 2) / np.sum(magnitude1 ** 2)
        features['reassigned_concentration'] = concentration_ratio
        spectral_sharpness = []
        for i in range(magnitude1.shape[1]):
            frame1 = magnitude1[:, i]
            if np.sum(frame1) > 0:
                sharpness = np.sum(frame1 ** 2) / (np.sum(frame1) ** 2)
                spectral_sharpness.append(sharpness)
        if spectral_sharpness:
            features['reassigned_sharpness_mean'] = np.mean(spectral_sharpness)
            features['reassigned_sharpness_std'] = np.std(spectral_sharpness)
    return features
