import numpy as np
import librosa
import scipy.signal
import scipy.fftpack
import warnings
from typing import Dict


def power_normalize(spectrum: np.ndarray) -> np.ndarray:
    """Aplica normalização de potência."""
    # Normalização por frame
    normalized = np.zeros_like(spectrum)

    for i in range(spectrum.shape[1]):
        frame = spectrum[:, i]
        power = np.sum(frame)

        if power > 0:
            normalized[:, i] = frame / power
        else:
            normalized[:, i] = frame

    return normalized


def apply_harmonic_emphasis(S: np.ndarray) -> np.ndarray:
    """Aplica ênfase harmônica ao espectrograma."""
    # Detectar picos harmônicos e enfatizá-los
    emphasized = S.copy()

    for i in range(S.shape[1]):
        frame = S[:, i]

        # Encontrar picos
        peaks, _ = scipy.signal.find_peaks(frame, height=np.max(frame) * 0.1)

        # Enfatizar picos
        for peak in peaks:
            emphasized[peak, i] *= 1.2

    return emphasized


def extract_pncc_features(y: np.ndarray, sr: int, frame_length: int, hop_length: int,
                          n_mfcc: int, n_mels: int, fmin: float, fmax: float) -> Dict:
    """Extrai Power-Normalized Cepstral Coefficients."""
    features = {}

    try:
        # Calcular espectrograma de potência
        S = np.abs(librosa.stft(y, n_fft=frame_length,
                                hop_length=hop_length))**2

        # Aplicar banco de filtros mel
        mel_spectrum = librosa.feature.melspectrogram(
            S=S, sr=sr, n_mels=n_mels,
            fmin=fmin, fmax=fmax)

        # Normalização de potência
        power_normalized = power_normalize(mel_spectrum)

        # Aplicar logaritmo
        log_spectrum = np.log(power_normalized + 1e-10)

        # Aplicar DCT
        pncc = scipy.fftpack.dct(log_spectrum, axis=0, norm='ortho')[:n_mfcc]

        features['pncc'] = pncc
        features['pncc_delta'] = librosa.feature.delta(pncc)
        features['pncc_delta2'] = librosa.feature.delta(pncc, order=2)

    except Exception as e:
        warnings.warn(f"Erro na extração PNCC: {str(e)}")
        features['pncc'] = np.zeros((n_mfcc, 1))
        features['pncc_delta'] = np.zeros((n_mfcc, 1))
        features['pncc_delta2'] = np.zeros((n_mfcc, 1))

    return features


def extract_mhec_features(y: np.ndarray, sr: int, frame_length: int, hop_length: int,
                          n_mfcc: int, n_mels: int, fmin: float, fmax: float) -> Dict:
    """Extrai Mel-scale Harmonic Emphasis Cepstral coefficients."""
    features = {}

    try:
        # Calcular espectrograma
        S = np.abs(librosa.stft(y, n_fft=frame_length,
                                hop_length=hop_length))

        # Aplicar ênfase harmônica
        harmonic_emphasized = apply_harmonic_emphasis(S)

        # Aplicar banco de filtros mel
        mel_spectrum = librosa.feature.melspectrogram(
            S=harmonic_emphasized, sr=sr, n_mels=n_mels,
            fmin=fmin, fmax=fmax)

        # Aplicar logaritmo
        log_spectrum = np.log(mel_spectrum + 1e-10)

        # Aplicar DCT
        mhec = scipy.fftpack.dct(log_spectrum, axis=0, norm='ortho')[:n_mfcc]

        features['mhec'] = mhec
        features['mhec_delta'] = librosa.feature.delta(mhec)
        features['mhec_delta2'] = librosa.feature.delta(mhec, order=2)

    except Exception as e:
        warnings.warn(f"Erro na extração MHEC: {str(e)}")
        features['mhec'] = np.zeros((n_mfcc, 1))
        features['mhec_delta'] = np.zeros((n_mfcc, 1))
        features['mhec_delta2'] = np.zeros((n_mfcc, 1))

    return features
