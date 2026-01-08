import numpy as np
import librosa
from .energy import compute_short_time_energy


def compute_temporal_centroid(
        y: np.ndarray, sr: int, frame_length: int, hop_length: int) -> float:
    """Computa centroide temporal."""
    # Envelope de energia
    energy = compute_short_time_energy(y, frame_length, hop_length)

    if np.sum(energy) == 0:
        return 0.0

    # Tempos dos frames
    times = librosa.frames_to_time(np.arange(len(energy)),
                                   sr=sr, hop_length=hop_length)

    # Centroide temporal
    centroid = np.sum(times * energy) / np.sum(energy)
    return float(centroid)


def compute_temporal_rolloff(y: np.ndarray, sr: int, frame_length: int,
                             hop_length: int, rolloff_percent: float = 0.85) -> float:
    """Computa rolloff temporal."""
    energy = compute_short_time_energy(y, frame_length, hop_length)

    if np.sum(energy) == 0:
        return 0.0

    # Energia cumulativa
    cumulative_energy = np.cumsum(energy)
    total_energy = cumulative_energy[-1]

    # Encontrar ponto de rolloff
    rolloff_threshold = rolloff_percent * total_energy
    rolloff_idx = np.where(cumulative_energy >= rolloff_threshold)[0]

    if len(rolloff_idx) == 0:
        # Fallback se não encontrar
        return float(librosa.frames_to_time(
            len(energy) - 1, sr=sr, hop_length=hop_length))

    # Converter para tempo
    rolloff_time = librosa.frames_to_time(rolloff_idx[0],
                                          sr=sr, hop_length=hop_length)
    return float(rolloff_time)


def compute_temporal_flux(
        y: np.ndarray, frame_length: int, hop_length: int) -> float:
    """Computa fluxo temporal (variação da energia)."""
    energy = compute_short_time_energy(y, frame_length, hop_length)

    if len(energy) < 2:
        return 0.0

    # Diferenças consecutivas
    flux = np.diff(energy)

    # Média das diferenças positivas
    positive_flux = flux[flux > 0]
    return float(np.mean(positive_flux)) if len(positive_flux) > 0 else 0.0


def compute_roughness(y: np.ndarray) -> float:
    """Calcula a rugosidade temporal do sinal."""
    if len(y) < 3:
        return 0.0

    # Calcular segunda derivada como medida de rugosidade
    second_derivative = np.diff(y, n=2)
    roughness = np.mean(np.abs(second_derivative))

    return float(roughness)


def compute_attack_time(y: np.ndarray, sr: int,
                        frame_length: int, hop_length: int) -> float:
    """Computa tempo de ataque (onset to peak)."""
    # Detectar onset
    onset_frames = librosa.onset.onset_detect(y=y, sr=sr,
                                              hop_length=hop_length)

    if len(onset_frames) == 0:
        return 0.0

    # Usar primeiro onset
    onset_time = librosa.frames_to_time(onset_frames[0],
                                        sr=sr, hop_length=hop_length)

    # Encontrar pico de energia após onset
    energy = compute_short_time_energy(y, frame_length, hop_length)
    onset_frame = onset_frames[0]

    if onset_frame >= len(energy):
        return 0.0

    # Procurar pico nos próximos frames
    search_window = min(
        len(energy) - onset_frame,
        sr // hop_length)  # 1 segundo
    energy_segment = energy[onset_frame:onset_frame + search_window]

    if len(energy_segment) == 0:
        return 0.0

    peak_idx = np.argmax(energy_segment)
    peak_time = librosa.frames_to_time(onset_frame + peak_idx,
                                       sr=sr, hop_length=hop_length)

    attack_time = peak_time - onset_time
    return float(max(0, attack_time))


def compute_decay_time(y: np.ndarray, sr: int,
                       frame_length: int, hop_length: int) -> float:
    """Computa tempo de decaimento."""
    energy = compute_short_time_energy(y, frame_length, hop_length)

    if len(energy) == 0:
        return 0.0

    # Encontrar pico global
    peak_idx = np.argmax(energy)
    peak_energy = energy[peak_idx]

    if peak_energy == 0:
        return 0.0

    # Procurar ponto onde energia cai para 10% do pico
    decay_threshold = 0.1 * peak_energy

    # Procurar após o pico
    post_peak_energy = energy[peak_idx:]
    decay_indices = np.where(post_peak_energy <= decay_threshold)[0]

    if len(decay_indices) == 0:
        return 0.0

    decay_idx = peak_idx + decay_indices[0]

    # Converter para tempo
    peak_time = librosa.frames_to_time(peak_idx, sr=sr, hop_length=hop_length)
    decay_time_val = librosa.frames_to_time(
        decay_idx, sr=sr, hop_length=hop_length)

    return float(decay_time_val - peak_time)


def compute_sustain_level(
        y: np.ndarray, frame_length: int, hop_length: int) -> float:
    """Computa nível de sustentação."""
    energy = compute_short_time_energy(y, frame_length, hop_length)

    if len(energy) == 0:
        return 0.0

    # Usar porção média do sinal (evitar ataque e decaimento)
    start_idx = len(energy) // 4
    end_idx = 3 * len(energy) // 4

    if start_idx >= end_idx:
        return float(np.mean(energy))

    sustain_energy = energy[start_idx:end_idx]
    return float(np.mean(sustain_energy))
