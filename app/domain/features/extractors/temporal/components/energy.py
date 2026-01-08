import numpy as np
import librosa


def compute_rms_energy(y: np.ndarray, frame_length: int,
                       hop_length: int) -> np.ndarray:
    """Computa energia RMS por frame."""
    return librosa.feature.rms(y=y, frame_length=frame_length,
                               hop_length=hop_length)[0]


def compute_short_time_energy(
        y: np.ndarray, frame_length: int, hop_length: int) -> np.ndarray:
    """Computa energia de curto prazo."""
    # Dividir em frames
    frames = librosa.util.frame(y, frame_length=frame_length,
                                hop_length=hop_length)

    # Calcular energia por frame
    energy = np.sum(frames ** 2, axis=0)
    return energy


def compute_energy_entropy(
        y: np.ndarray, frame_length: int, hop_length: int) -> float:
    """Computa entropia da energia."""
    # Calcular energia por frame
    energy = compute_short_time_energy(y, frame_length, hop_length)

    # Normalizar para formar distribuição de probabilidade
    total_energy = np.sum(energy)
    if total_energy == 0:
        return 0.0

    prob_energy = energy / total_energy

    # Calcular entropia
    prob_energy = prob_energy[prob_energy > 0]  # Evitar log(0)
    entropy = -np.sum(prob_energy * np.log2(prob_energy))

    return float(entropy)


def compute_teager_energy(y: np.ndarray, frame_length: int,
                          hop_length: int) -> np.ndarray:
    """Computa operador de energia de Teager-Kaiser."""
    if len(y) < 3:
        return np.array([0.0])

    # TEO: x[n]^2 - x[n-1]*x[n+1]
    teager = np.zeros(len(y))

    for i in range(1, len(y) - 1):
        teager[i] = y[i] ** 2 - y[i - 1] * y[i + 1]

    # Dividir em frames
    frames = librosa.util.frame(teager, frame_length=frame_length,
                                hop_length=hop_length)

    # Média por frame
    return np.mean(frames, axis=0)


def compute_log_energy(y: np.ndarray, frame_length: int,
                       hop_length: int) -> np.ndarray:
    """Calcula a energia logarítmica por frame."""
    frames = librosa.util.frame(y, frame_length=frame_length,
                                hop_length=hop_length)

    log_energies = []
    for frame in frames.T:
        energy = np.sum(frame ** 2)
        log_energy = np.log(energy + 1e-10)  # Evitar log(0)
        log_energies.append(log_energy)

    return np.array(log_energies)


def compute_frame_energy_variance(
        y: np.ndarray, frame_length: int, hop_length: int) -> float:
    """Calcula a variância da energia entre frames."""
    rms_energy = compute_rms_energy(y, frame_length, hop_length)
    return float(np.var(rms_energy))
