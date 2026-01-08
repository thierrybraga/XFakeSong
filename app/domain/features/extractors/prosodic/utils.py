import numpy as np
import librosa


def extract_periods(y: np.ndarray, f0: np.ndarray, sr: int) -> np.ndarray:
    """Extrai períodos do sinal baseado em F0."""
    periods = []

    # Converter F0 para períodos em amostras
    for f0_val in f0:
        if not np.isnan(f0_val) and f0_val > 0:
            period = sr / f0_val
            periods.append(period)

    return np.array(periods)


def extract_peak_amplitudes(
        y: np.ndarray, f0: np.ndarray, frame_length: int, hop_length: int) -> np.ndarray:
    """Extrai amplitudes de picos baseado em F0."""
    amplitudes = []

    frames = librosa.util.frame(y, frame_length=frame_length,
                                hop_length=hop_length)

    for i, frame in enumerate(frames.T):
        if i < len(f0) and not np.isnan(f0[i]) and f0[i] > 0:
            # Encontrar pico no frame
            max_amplitude = np.max(np.abs(frame))
            amplitudes.append(max_amplitude)

    return np.array(amplitudes)


def find_peaks(signal: np.ndarray, min_distance: int = 3) -> np.ndarray:
    """Encontra picos em um sinal."""
    if len(signal) < 3:
        return np.array([])

    peaks = []
    for i in range(1, len(signal) - 1):
        if (signal[i] > signal[i - 1] and signal[i] > signal[i + 1]):
            # Verificar distância mínima
            if not peaks or (i - peaks[-1]) >= min_distance:
                peaks.append(i)

    return np.array(peaks)
