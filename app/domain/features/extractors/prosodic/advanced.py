import numpy as np
import librosa
from .utils import extract_periods, extract_peak_amplitudes


def compute_rap(y: np.ndarray, f0: np.ndarray, sr: int) -> float:
    """Calcula Relative Average Perturbation (RAP)."""
    periods = extract_periods(y, f0, sr)

    if len(periods) < 3:
        return 0.0

    perturbations = []
    for i in range(1, len(periods) - 1):
        # RAP usa média de 3 períodos consecutivos
        local_mean = (periods[i - 1] + periods[i] + periods[i + 1]) / 3
        if local_mean > 0:
            perturbation = abs(periods[i] - local_mean) / local_mean
            perturbations.append(perturbation)

    return np.mean(perturbations) if perturbations else 0.0


def compute_ppq(y: np.ndarray, f0: np.ndarray, sr: int) -> float:
    """Calcula Pitch Period Perturbation Quotient (PPQ)."""
    periods = extract_periods(y, f0, sr)

    if len(periods) < 5:
        return 0.0

    perturbations = []
    for i in range(2, len(periods) - 2):
        # PPQ usa média de 5 períodos consecutivos
        local_mean = np.mean(periods[i - 2:i + 3])
        if local_mean > 0:
            perturbation = abs(periods[i] - local_mean) / local_mean
            perturbations.append(perturbation)

    return np.mean(perturbations) if perturbations else 0.0


def compute_apq(y: np.ndarray, f0: np.ndarray,
                frame_length: int, hop_length: int) -> float:
    """Calcula Amplitude Perturbation Quotient (APQ)."""
    amplitudes = extract_peak_amplitudes(y, f0, frame_length, hop_length)

    if len(amplitudes) < 11:
        return 0.0

    perturbations = []
    for i in range(5, len(amplitudes) - 5):
        # APQ usa média de 11 amplitudes consecutivas
        local_mean = np.mean(amplitudes[i - 5:i + 6])
        if local_mean > 0:
            perturbation = abs(amplitudes[i] - local_mean) / local_mean
            perturbations.append(perturbation)

    return np.mean(perturbations) if perturbations else 0.0


def compute_vf0(f0: np.ndarray) -> float:
    """Calcula Fundamental Frequency Variation (vF0)."""
    valid_f0 = f0[~np.isnan(f0) & (f0 > 0)]

    if len(valid_f0) < 2:
        return 0.0

    mean_f0 = np.mean(valid_f0)
    if mean_f0 > 0:
        return np.std(valid_f0) / mean_f0 * 100  # Em porcentagem
    else:
        return 0.0


def compute_shdb(y: np.ndarray, f0: np.ndarray,
                 frame_length: int, hop_length: int) -> float:
    """Calcula Shimmer in dB (ShdB)."""
    amplitudes = extract_peak_amplitudes(y, f0, frame_length, hop_length)

    if len(amplitudes) < 2:
        return 0.0

    # Converter para dB
    amplitudes_db = 20 * np.log10(amplitudes + 1e-10)

    # Calcular diferenças consecutivas
    differences = np.abs(np.diff(amplitudes_db))

    return np.mean(differences)


def compute_nhr(y: np.ndarray, f0: np.ndarray, sr: int,
                frame_length: int, hop_length: int) -> float:
    """Calcula Noise-to-Harmonics Ratio (NHR)."""
    # Implementação simplificada baseada na relação entre
    # energia harmônica e energia de ruído

    # Calcular espectrograma
    D = librosa.stft(y, n_fft=frame_length, hop_length=hop_length)
    magnitude = np.abs(D)

    # Estimar componentes harmônicas e de ruído
    harmonic_energy = 0
    noise_energy = 0

    valid_f0 = f0[~np.isnan(f0) & (f0 > 0)]

    if len(valid_f0) == 0:
        return float('inf')

    mean_f0 = np.mean(valid_f0)

    # Frequências dos bins
    freqs = librosa.fft_frequencies(sr=sr, n_fft=frame_length)

    # Identificar bins harmônicos (aproximação)
    for harmonic in range(1, 6):  # Primeiros 5 harmônicos
        target_freq = harmonic * mean_f0
        if target_freq < sr / 2:
            # Encontrar bin mais próximo
            bin_idx = np.argmin(np.abs(freqs - target_freq))

            # Energia harmônica (bin central + vizinhos)
            start_bin = max(0, bin_idx - 2)
            end_bin = min(len(freqs), bin_idx + 3)
            harmonic_energy += np.mean(magnitude[start_bin:end_bin, :] ** 2)

    # Energia total
    total_energy = np.mean(magnitude ** 2)

    # Energia de ruído (aproximação)
    noise_energy = total_energy - harmonic_energy

    if harmonic_energy > 0:
        return noise_energy / harmonic_energy
    else:
        return float('inf')


def compute_vti(y: np.ndarray, frame_length: int, hop_length: int) -> float:
    """Calcula Voice Turbulence Index (VTI)."""
    # Implementação simplificada baseada na variabilidade espectral

    # Calcular espectrograma
    D = librosa.stft(y, n_fft=frame_length, hop_length=hop_length)
    magnitude = np.abs(D)

    # Calcular variabilidade temporal do espectro
    spectral_variance = np.var(magnitude, axis=1)

    # VTI como média da variabilidade espectral normalizada
    mean_magnitude = np.mean(magnitude, axis=1)

    vti = np.mean(spectral_variance / (mean_magnitude + 1e-10))

    return vti


def compute_spi(y: np.ndarray, sr: int, frame_length: int,
                hop_length: int) -> float:
    """Calcula Soft Phonation Index (SPI)."""
    # Implementação simplificada baseada na relação entre
    # energia de baixa e alta frequência

    # Calcular espectrograma
    D = librosa.stft(y, n_fft=frame_length, hop_length=hop_length)
    magnitude = np.abs(D) ** 2

    # Frequências dos bins
    freqs = librosa.fft_frequencies(sr=sr, n_fft=frame_length)

    # Dividir em bandas de baixa (< 2kHz) e alta (> 2kHz) frequência
    low_freq_mask = freqs < 2000
    high_freq_mask = freqs >= 2000

    # Energia em cada banda
    low_freq_energy = np.mean(magnitude[low_freq_mask, :])
    high_freq_energy = np.mean(magnitude[high_freq_mask, :])

    # SPI como razão (em dB)
    if high_freq_energy > 0:
        spi = 10 * np.log10(low_freq_energy / high_freq_energy)
    else:
        spi = float('inf')

    return spi


def compute_dfa(y: np.ndarray) -> float:
    """Calcula Detrended Fluctuation Analysis (DFA) exponent."""
    N = len(y)
    if N < 16:
        return 0.0

    # Integrar série (perfil)
    y_integrated = np.cumsum(y - np.mean(y))

    # Diferentes tamanhos de janela
    scales = np.unique(np.logspace(1, np.log10(N // 4), 10, dtype=int))
    fluctuations = []

    for scale in scales:
        if scale >= N:
            continue

        # Dividir em segmentos
        n_segments = N // scale
        segment_fluctuations = []

        for i in range(n_segments):
            segment = y_integrated[i * scale:(i + 1) * scale]

            # Ajustar tendência linear
            x = np.arange(len(segment))
            coeffs = np.polyfit(x, segment, 1)
            trend = np.polyval(coeffs, x)

            # Calcular flutuação
            detrended = segment - trend
            fluctuation = np.sqrt(np.mean(detrended ** 2))
            segment_fluctuations.append(fluctuation)

        if segment_fluctuations:
            fluctuations.append(np.mean(segment_fluctuations))

    # Ajuste linear em escala log-log para obter expoente DFA
    if len(fluctuations) >= 3:
        valid_scales = scales[:len(fluctuations)]
        valid_fluctuations = np.array(fluctuations)

        # Remover valores inválidos
        valid_mask = (valid_fluctuations > 0) & np.isfinite(valid_fluctuations)

        if np.sum(valid_mask) >= 3:
            log_scales = np.log(valid_scales[valid_mask])
            log_fluctuations = np.log(valid_fluctuations[valid_mask])

            dfa_exponent = np.polyfit(log_scales, log_fluctuations, 1)[0]
            return dfa_exponent

    return 0.0
