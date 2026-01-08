"""
Extração de características de qualidade perceptual (roughness, fluctuation, tonality).
"""
import numpy as np
import librosa
import scipy.signal
from typing import Dict


def extract_roughness_features(
        y: np.ndarray, sr: int, frame_length: int,
        hop_length: int) -> Dict[str, float]:
    """Extrai características de roughness (rugosidade)."""
    features = {}

    # Calcular espectrograma
    stft = librosa.stft(y, n_fft=frame_length, hop_length=hop_length)
    magnitude = np.abs(stft)

    # Frequências dos bins
    freqs = librosa.fft_frequencies(sr=sr, n_fft=frame_length)

    roughness_frames = []

    for frame in magnitude.T:
        # Modelo de roughness baseado em modulação de amplitude
        # Roughness é máxima para modulações de ~70 Hz

        roughness_total = 0

        # Analisar pares de componentes espectrais
        for i in range(len(freqs) - 1):
            for j in range(i + 1, len(freqs)):
                # Contribuição de par de frequências
                f1 = freqs[i]
                s1, s2 = frame[i], frame[j]
                freq_diff = freqs[j] - freqs[i]

                if freq_diff > 0 and s1 > 0 and s2 > 0:
                    # Função de roughness (Vassilakis, 2001)

                    # Dissonância sensorial
                    x = 0.24 / (0.0207 * f1 + 18.96)
                    y_val = (3.5 * s1 * s2) / (s1 + s2)
                    z = np.exp(-3.5 * x * freq_diff) - \
                        np.exp(-5.75 * x * freq_diff)

                    roughness_component = y_val * z
                    roughness_total += roughness_component

        roughness_frames.append(roughness_total)

    roughness_frames = np.array(roughness_frames)

    # Características de roughness
    features['roughness_mean'] = np.mean(roughness_frames)
    features['roughness_std'] = np.std(roughness_frames)
    features['roughness_max'] = np.max(roughness_frames)

    return features


def extract_fluctuation_features(y: np.ndarray, sr: int) -> Dict:
    """Extrai características de fluctuation strength."""
    features = {}

    # Calcular envelope do sinal
    analytic_signal = scipy.signal.hilbert(y)
    envelope = np.abs(analytic_signal)

    # Suavizar envelope
    window_size = int(0.01 * sr)  # 10 ms
    if window_size > 0:
        envelope = scipy.signal.convolve(
            envelope, np.ones(window_size) / window_size, mode='same')

    # Calcular modulações do envelope
    if len(envelope) > 1:
        # FFT do envelope para encontrar modulações
        envelope_fft = np.fft.fft(envelope)
        envelope_freqs = np.fft.fftfreq(len(envelope), 1 / sr)
        envelope_magnitude = np.abs(envelope_fft)

        # Fluctuation strength é máxima para modulações de ~4 Hz
        target_mod_freq = 4.0  # Hz

        # Encontrar energia próxima à frequência de modulação alvo
        mod_freq_mask = (
            envelope_freqs > 0.5) & (
            envelope_freqs < 20)  # 0.5-20 Hz

        if np.any(mod_freq_mask):
            mod_energies = envelope_magnitude[mod_freq_mask]
            mod_freqs_valid = envelope_freqs[mod_freq_mask]

            # Ponderação baseada na proximidade com 4 Hz
            weights = np.exp(-0.5 *
                             ((mod_freqs_valid - target_mod_freq) / 2) ** 2)

            fluctuation_strength = np.sum(mod_energies * weights)
            features['fluctuation_strength'] = fluctuation_strength

            # Frequência de modulação dominante
            dominant_idx = np.argmax(mod_energies)
            features['dominant_modulation_freq'] = mod_freqs_valid[dominant_idx]
        else:
            features['fluctuation_strength'] = 0
            features['dominant_modulation_freq'] = 0
    else:
        features['fluctuation_strength'] = 0
        features['dominant_modulation_freq'] = 0

    return features


def extract_tonality_features(
        y: np.ndarray, sr: int, frame_length: int, hop_length: int) -> Dict:
    """Extrai características de tonality."""
    features = {}

    # Calcular espectrograma
    stft = librosa.stft(y, n_fft=frame_length, hop_length=hop_length)
    magnitude = np.abs(stft)

    tonality_frames = []

    for frame in magnitude.T:
        # Detectar picos espectrais (componentes tonais)
        peaks, _ = scipy.signal.find_peaks(frame, height=np.max(frame) * 0.1)

        if len(peaks) > 0:
            # Energia dos picos (componentes tonais)
            tonal_energy = np.sum(frame[peaks] ** 2)

            # Energia total
            total_energy = np.sum(frame ** 2)

            # Tonality como razão entre energia tonal e total
            if total_energy > 0:
                tonality = tonal_energy / total_energy
            else:
                tonality = 0
        else:
            tonality = 0

        tonality_frames.append(tonality)

    tonality_frames = np.array(tonality_frames)

    # Características de tonality
    features['tonality_mean'] = np.mean(tonality_frames)
    features['tonality_std'] = np.std(tonality_frames)
    features['tonality_max'] = np.max(tonality_frames)

    return features
