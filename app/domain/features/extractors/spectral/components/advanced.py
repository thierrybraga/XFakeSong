"""
Extração de características espectrais avançadas.
"""
import numpy as np
import scipy.signal


def compute_spectral_flux(S: np.ndarray) -> np.ndarray:
    """
    Computa o fluxo espectral (mudança espectral entre frames).

    Args:
        S: Espectrograma de magnitude

    Returns:
        Array com valores de fluxo espectral
    """
    if S.shape[1] < 2:
        return np.array([0.0])

    flux = []
    for i in range(1, S.shape[1]):
        # Diferença entre frames consecutivos
        diff = S[:, i] - S[:, i - 1]
        # Considerar apenas aumentos de energia
        positive_diff = np.maximum(diff, 0)
        # Somar todas as diferenças
        frame_flux = np.sum(positive_diff)
        flux.append(frame_flux)

    return np.array(flux)


def compute_spectral_decrease(S: np.ndarray, freqs: np.ndarray) -> np.ndarray:
    """
    Computa o decréscimo espectral.

    Args:
        S: Espectrograma de magnitude
        freqs: Frequências correspondentes

    Returns:
        Array com valores de decréscimo espectral
    """
    decrease = []

    for frame in S.T:
        if len(frame) > 1 and frame[0] > 0:
            # Decréscimo = (sum(k * (X[k] - X[0])) / sum(k)) / X[0]
            k_indices = np.arange(1, len(frame))
            numerator = np.sum(k_indices * (frame[1:] - frame[0]))
            denominator = np.sum(k_indices) * frame[0]

            if denominator > 0:
                dec = numerator / denominator
            else:
                dec = 0
        else:
            dec = 0

        decrease.append(dec)

    return np.array(decrease)


def compute_spectral_crest(S: np.ndarray) -> np.ndarray:
    """
    Computa o fator de crista espectral.

    Args:
        S: Espectrograma de magnitude

    Returns:
        Array com valores de fator de crista
    """
    crest = []

    for frame in S.T:
        if len(frame) > 0:
            max_val = np.max(frame)
            mean_val = np.mean(frame)

            if mean_val > 0:
                crest_val = max_val / mean_val
            else:
                crest_val = 0
        else:
            crest_val = 0

        crest.append(crest_val)

    return np.array(crest)


def compute_spectral_irregularity(S: np.ndarray) -> np.ndarray:
    """
    Computa a irregularidade espectral.

    Args:
        S: Espectrograma de magnitude

    Returns:
        Array com valores de irregularidade
    """
    irregularity = []

    for frame in S.T:
        if len(frame) > 2:
            # Irregularidade = sum(|X[k] - X[k-1]|) / sum(X[k])
            diff_sum = np.sum(np.abs(np.diff(frame)))
            total_sum = np.sum(frame)

            if total_sum > 0:
                irreg = diff_sum / total_sum
            else:
                irreg = 0
        else:
            irreg = 0

        irregularity.append(irreg)

    return np.array(irregularity)


def compute_spectral_roughness(S: np.ndarray, freqs: np.ndarray) -> np.ndarray:
    """
    Computa a rugosidade espectral.

    Args:
        S: Espectrograma de magnitude
        freqs: Frequências correspondentes

    Returns:
        Array com valores de rugosidade
    """
    roughness = []

    for frame in S.T:
        rough_val = 0

        # Analisar pares de componentes espectrais próximas
        for i in range(len(frame) - 1):
            for j in range(i + 1, min(i + 5, len(frame))):
                if frame[i] > 0 and frame[j] > 0:
                    freq_diff = freqs[j] - freqs[i]
                    amp_product = frame[i] * frame[j]

                    # Modelo simplificado de rugosidade
                    if freq_diff > 0:
                        rough_component = amp_product * \
                            np.exp(-freq_diff / 100)
                        rough_val += rough_component

        roughness.append(rough_val)

    return np.array(roughness)


def compute_spectral_inharmonicity(
        S: np.ndarray, freqs: np.ndarray) -> np.ndarray:
    """
    Computa a inarmonicidade espectral.

    Args:
        S: Espectrograma de magnitude
        freqs: Frequências correspondentes

    Returns:
        Array com valores de inarmonicidade
    """
    inharmonicity = []

    for frame in S.T:
        # Encontrar picos espectrais
        peaks, _ = scipy.signal.find_peaks(frame, height=np.max(frame) * 0.1)

        if len(peaks) > 1:
            # Calcular inarmonicidade baseada na relação entre picos
            peak_freqs = freqs[peaks]
            peak_amps = frame[peaks]

            # Encontrar frequência fundamental (menor pico significativo)
            f0_idx = np.argmax(peak_amps)
            f0 = peak_freqs[f0_idx]

            if f0 > 0:
                inharm_sum = 0
                total_energy = 0

                for i, (freq, amp) in enumerate(zip(peak_freqs, peak_amps)):
                    if freq > f0:
                        # Calcular desvio da relação harmônica
                        harmonic_ratio = freq / f0
                        nearest_harmonic = round(harmonic_ratio)

                        if nearest_harmonic > 0:
                            deviation = abs(
                                harmonic_ratio - nearest_harmonic) / nearest_harmonic
                            inharm_sum += deviation * amp
                            total_energy += amp

                if total_energy > 0:
                    inharm_val = inharm_sum / total_energy
                else:
                    inharm_val = 0
            else:
                inharm_val = 0
        else:
            inharm_val = 0

        inharmonicity.append(inharm_val)

    return np.array(inharmonicity)
