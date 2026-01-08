import numpy as np


def compute_spectral_centroid(spectrum: np.ndarray) -> np.ndarray:
    """Computa centroide espectral."""
    freq_indices = np.arange(spectrum.shape[0])
    centroids = []

    for frame in range(spectrum.shape[1]):
        spectrum_frame = spectrum[:, frame]
        if np.sum(spectrum_frame) > 0:
            centroid = np.sum(freq_indices * spectrum_frame) / \
                np.sum(spectrum_frame)
        else:
            centroid = 0.0
        centroids.append(centroid)

    return np.array(centroids)


def compute_spectral_rolloff(spectrum: np.ndarray,
                             rolloff_percent: float = 0.85) -> np.ndarray:
    """Computa rolloff espectral."""
    rolloffs = []

    for frame in range(spectrum.shape[1]):
        spectrum_frame = spectrum[:, frame]
        total_energy = np.sum(spectrum_frame)

        if total_energy > 0:
            cumulative_energy = np.cumsum(spectrum_frame)
            rolloff_threshold = rolloff_percent * total_energy
            rolloff_idx = np.where(cumulative_energy >= rolloff_threshold)[0]

            if len(rolloff_idx) > 0:
                rolloff = rolloff_idx[0]
            else:
                rolloff = len(spectrum_frame) - 1
        else:
            rolloff = 0.0

        rolloffs.append(rolloff)

    return np.array(rolloffs)


def compute_spectral_flux(spectrum: np.ndarray) -> np.ndarray:
    """Computa flux espectral."""
    if spectrum.shape[1] < 2:
        return np.array([0.0])

    flux = np.sum(np.diff(spectrum, axis=1)**2, axis=0)
    return np.concatenate([[0.0], flux])


def compute_octave_energy(cqt_mag: np.ndarray,
                          bins_per_octave: int) -> np.ndarray:
    """Computa energia por oitava no CQT."""
    octave_energies = []

    for octave in range(cqt_mag.shape[0] // bins_per_octave):
        start_bin = octave * bins_per_octave
        end_bin = (octave + 1) * bins_per_octave
        octave_energy = np.sum(cqt_mag[start_bin:end_bin, :], axis=0)
        octave_energies.append(np.mean(octave_energy))

    return np.array(octave_energies)
