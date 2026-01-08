import numpy as np
import librosa
from .utils import extract_periods, extract_peak_amplitudes


def compute_jitter(y: np.ndarray, f0: np.ndarray, sr: int) -> float:
    """Computa jitter (variabilidade do período)."""
    # Detectar períodos
    periods = extract_periods(y, f0, sr)

    if len(periods) < 2:
        return 0

    # Calcular jitter como variabilidade relativa dos períodos
    period_diffs = np.abs(np.diff(periods))
    mean_period = np.mean(periods)

    if mean_period > 0:
        jitter = np.mean(period_diffs) / mean_period
    else:
        jitter = 0

    return jitter


def compute_shimmer(y: np.ndarray, f0: np.ndarray,
                    frame_length: int, hop_length: int) -> float:
    """Computa shimmer (variabilidade da amplitude)."""
    # Extrair amplitudes de picos
    amplitudes = extract_peak_amplitudes(y, f0, frame_length, hop_length)

    if len(amplitudes) < 2:
        return 0

    # Calcular shimmer como variabilidade relativa das amplitudes
    amplitude_diffs = np.abs(np.diff(amplitudes))
    mean_amplitude = np.mean(amplitudes)

    if mean_amplitude > 0:
        shimmer = np.mean(amplitude_diffs) / mean_amplitude
    else:
        shimmer = 0

    return shimmer


def compute_hnr(y: np.ndarray, f0: np.ndarray, sr: int,
                frame_length: int, hop_length: int) -> float:
    """Computa Harmonic-to-Noise Ratio."""
    # Esta é uma implementação simplificada
    # HNR real requer análise mais sofisticada

    # Calcular espectrograma
    S = np.abs(librosa.stft(y, n_fft=frame_length,
                            hop_length=hop_length))
    # freqs = librosa.fft_frequencies(sr=sr, n_fft=frame_length) # Not used in
    # simplified calculation logic below but kept for reference

    harmonic_energy = 0
    total_energy = 0

    for i, f0_val in enumerate(f0):
        if not np.isnan(f0_val) and f0_val > 0 and i < S.shape[1]:
            frame_spectrum = S[:, i]
            total_energy += np.sum(frame_spectrum ** 2)

            # Somar energia dos primeiros 5 harmônicos
            for harmonic in range(1, 6):
                harmonic_freq = harmonic * f0_val
                if harmonic_freq < sr / 2:
                    # Encontrar bin mais próximo
                    bin_idx = int(harmonic_freq * frame_length / sr)
                    if bin_idx < len(frame_spectrum):
                        # Somar energia em uma pequena janela ao redor do
                        # harmônico
                        window = 2
                        start_bin = max(0, bin_idx - window)
                        end_bin = min(
                            len(frame_spectrum), bin_idx + window + 1)
                        harmonic_energy += np.sum(
                            frame_spectrum[start_bin:end_bin] ** 2)

    # Calcular HNR
    if total_energy > harmonic_energy and harmonic_energy > 0:
        noise_energy = total_energy - harmonic_energy
        hnr = 10 * np.log10(harmonic_energy / noise_energy)
    else:
        hnr = -np.inf

    return hnr if not np.isinf(hnr) else 0


def compute_snr(y: np.ndarray, frame_length: int, hop_length: int) -> float:
    """Computa Signal-to-Noise Ratio aproximado."""
    # Estimar ruído usando partes de baixa energia
    frame_energy = []
    frames = librosa.util.frame(y, frame_length=frame_length,
                                hop_length=hop_length)

    for frame in frames.T:
        energy = np.sum(frame ** 2)
        frame_energy.append(energy)

    frame_energy = np.array(frame_energy)

    if len(frame_energy) == 0:
        return 0

    # Usar percentil baixo como estimativa do ruído
    noise_level = np.percentile(frame_energy, 10)
    signal_level = np.mean(frame_energy)

    if noise_level > 0:
        snr = 10 * np.log10(signal_level / noise_level)
    else:
        snr = np.inf

    return snr if not np.isinf(snr) else 100


def compute_cpp(y: np.ndarray, sr: int, frame_length: int,
                hop_length: int) -> float:
    """Computa Cepstral Peak Prominence."""
    # Calcular cepstrum
    frames = librosa.util.frame(y, frame_length=frame_length,
                                hop_length=hop_length)

    cpp_values = []

    for frame in frames.T:
        # Aplicar janela
        windowed = frame * np.hamming(len(frame))

        # Calcular FFT
        spectrum = np.fft.fft(windowed)
        log_spectrum = np.log(np.abs(spectrum) + 1e-10)

        # Calcular cepstrum
        cepstrum = np.fft.ifft(log_spectrum).real

        # Encontrar pico cepstral (quefrency correspondente a pitch)
        # Região de interesse para pitch humano (80-400 Hz)
        min_quefrency = int(sr / 400)  # 400 Hz
        max_quefrency = int(sr / 80)   # 80 Hz

        if max_quefrency < len(cepstrum):
            search_region = cepstrum[min_quefrency:max_quefrency]
            if len(search_region) > 0:
                peak_value = np.max(search_region)
                baseline = np.mean(cepstrum[:min_quefrency])
                cpp = peak_value - baseline
            else:
                cpp = 0
        else:
            cpp = 0

        cpp_values.append(cpp)

    return np.mean(cpp_values) if cpp_values else 0
