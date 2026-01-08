import numpy as np
import scipy.signal
import scipy.stats
from typing import List, Tuple, Optional


def frame_signal(y: np.ndarray, frame_length: int,
                 hop_length: int) -> List[np.ndarray]:
    """Divide o sinal em frames."""
    frames = []

    for i in range(0, len(y) - frame_length + 1, hop_length):
        frame = y[i:i + frame_length]

        # Aplicar janela
        window = scipy.signal.windows.hann(len(frame))
        windowed_frame = frame * window

        frames.append(windowed_frame)

    return frames


def compute_energy_envelope(
        y: np.ndarray, frame_length: int, hop_length: int) -> np.ndarray:
    """Calcula envelope de energia."""
    energy_envelope = []

    for i in range(0, len(y) - frame_length + 1, hop_length):
        frame = y[i:i + frame_length]
        energy = np.sum(frame ** 2)
        energy_envelope.append(energy)

    return np.array(energy_envelope)


def frames_to_segments(
        frame_indices: List[int], hop_length: int) -> List[Tuple[int, int]]:
    """Converte índices de frames para segmentos temporais."""
    if not frame_indices:
        return []

    segments = []
    start = frame_indices[0]
    prev = start

    for i in range(1, len(frame_indices)):
        current = frame_indices[i]

        # Se há gap entre frames, finalizar segmento atual
        if current - prev > 1:
            end = prev
            segments.append((start * hop_length, (end + 1) * hop_length))
            start = current

        prev = current

    # Adicionar último segmento
    segments.append((start * hop_length, (prev + 1) * hop_length))

    return segments


def detect_speech_segments(y: np.ndarray, sr: int, frame_length: int,
                           hop_length: int,
                           threshold: float = 0.01) -> List[Tuple[int, int]]:
    """Detecta segmentos de fala baseado em energia."""
    # Calcular energia por frame
    frame_energy = []

    for i in range(0, len(y) - frame_length + 1, hop_length):
        frame = y[i:i + frame_length]
        energy = np.sum(frame ** 2) / len(frame)
        frame_energy.append(energy)

    # Detectar segmentos acima do threshold
    is_speech = np.array(frame_energy) > threshold

    # Encontrar início e fim dos segmentos
    segments = []
    in_segment = False
    start = 0

    for i, speech in enumerate(is_speech):
        if speech and not in_segment:
            start = i * hop_length
            in_segment = True
        elif not speech and in_segment:
            end = i * hop_length
            segments.append((start, end))
            in_segment = False

    # Fechar último segmento se necessário
    if in_segment:
        segments.append((start, len(y)))

    return segments


def detect_vowel_segments(y: np.ndarray, sr: int, frame_length: int,
                          hop_length: int) -> List[Tuple[int, int]]:
    """Detecta segmentos vocálicos baseado em características espectrais."""
    frames = frame_signal(y, frame_length, hop_length)

    vowel_frames = []

    for i, frame in enumerate(frames):
        # Características indicativas de vogais:
        # 1. Energia relativamente alta
        # 2. Estrutura harmônica clara
        # 3. Formantes bem definidos

        energy = np.sum(frame ** 2)

        if energy > 0.001:  # Threshold mínimo de energia
            # Calcular autocorrelação para detectar periodicidade
            autocorr = np.correlate(frame, frame, mode='full')
            autocorr = autocorr[len(autocorr) // 2:]

            # Procurar pico de autocorrelação (indicativo de periodicidade)
            if len(autocorr) > 1:
                max_autocorr = np.max(autocorr[1:]) / autocorr[0]

                if max_autocorr > 0.3:  # Threshold de periodicidade
                    vowel_frames.append(i)

    # Converter frames para segmentos temporais
    segments = frames_to_segments(vowel_frames, hop_length)

    return segments


def detect_consonant_segments(
        y: np.ndarray, sr: int, frame_length: int, hop_length: int) -> List[Tuple[int, int]]:
    """Detecta segmentos consonantais."""
    # Detectar todos os segmentos de fala
    speech_segments = detect_speech_segments(y, sr, frame_length, hop_length)

    # Detectar segmentos vocálicos
    vowel_segments = detect_vowel_segments(y, sr, frame_length, hop_length)

    # Consonantes são segmentos de fala que não são vogais
    consonant_segments = []

    for speech_start, speech_end in speech_segments:
        current_pos = speech_start

        for vowel_start, vowel_end in vowel_segments:
            # Verificar se há consonante antes da vogal
            if vowel_start > current_pos and vowel_start <= speech_end:
                consonant_segments.append((current_pos, vowel_start))
                current_pos = vowel_end

        # Verificar se há consonante após a última vogal
        if current_pos < speech_end:
            consonant_segments.append((current_pos, speech_end))

    return consonant_segments


def estimate_syllable_count(y: np.ndarray, sr: int,
                            frame_length: int, hop_length: int) -> int:
    """Estima número de sílabas baseado em picos de energia."""
    # Calcular envelope de energia
    energy_envelope = compute_energy_envelope(y, frame_length, hop_length)

    # Verificar se o envelope tem tamanho suficiente
    if len(energy_envelope) < 5:
        return 1  # Retorna 1 sílaba para segmentos muito pequenos

    # Calcular window_length válido para savgol_filter
    max_window = len(energy_envelope)
    if max_window % 2 == 0:
        max_window -= 1  # Deve ser ímpar
    window_length = min(21, max_window)
    if window_length < 3:
        window_length = 3  # Mínimo para polyorder=3

    # Ajustar polyorder se necessário
    polyorder = min(3, window_length - 1)

    # Suavizar envelope
    smoothed_envelope = scipy.signal.savgol_filter(energy_envelope,
                                                   window_length=window_length,
                                                   polyorder=polyorder)

    # Detectar picos
    peaks, _ = scipy.signal.find_peaks(smoothed_envelope,
                                       height=np.mean(smoothed_envelope),
                                       distance=int(0.1 * sr / hop_length))

    return max(1, len(peaks))  # Retorna pelo menos 1 sílaba


def estimate_vot(segment: np.ndarray, sr: int) -> Optional[float]:
    """Estimativa simplificada de VOT."""
    # Implementação simplificada baseada em mudança de energia
    step = int(0.005 * sr)
    window = int(0.01 * sr)
    if len(segment) < window:
        return None

    energy = np.array([np.sum(segment[i:i + window]**2)
                      for i in range(0, len(segment) - window, step)])

    if len(energy) < 2:
        return None

    # Encontrar ponto de maior mudança de energia
    energy_diff = np.diff(energy)
    max_change_idx = np.argmax(energy_diff)

    vot_estimate = max_change_idx * 0.005  # Converter para segundos

    return vot_estimate


def extract_formant_frequencies(
        y: np.ndarray, sr: int, frame_length: int, hop_length: int) -> List[float]:
    """Extrai frequências dos formantes (implementação simplificada)."""
    # Esta é uma implementação muito simplificada
    # Uma implementação real usaria LPC ou outros métodos mais sofisticados

    # Calcular espectro médio
    frames = frame_signal(y, frame_length, hop_length)

    if not frames:
        return []

    # Calcular espectro médio
    avg_spectrum = np.zeros(frame_length // 2)

    for frame in frames:
        spectrum = np.abs(np.fft.fft(frame))[:frame_length // 2]
        avg_spectrum += spectrum

    avg_spectrum /= len(frames)

    # Encontrar picos (formantes aproximados)
    freqs = np.fft.fftfreq(frame_length, 1 / sr)[:frame_length // 2]

    # Procurar picos na faixa de formantes (200-3000 Hz)
    formant_mask = (freqs >= 200) & (freqs <= 3000)
    formant_freqs = freqs[formant_mask]
    formant_spectrum = avg_spectrum[formant_mask]

    # Encontrar picos
    peaks, _ = scipy.signal.find_peaks(formant_spectrum,
                                       height=np.mean(formant_spectrum),
                                       distance=int(200 / (sr / frame_length)))

    formant_frequencies = formant_freqs[peaks][:4]  # Primeiros 4 formantes

    return formant_frequencies.tolist()


def extract_formant_bandwidths(y: np.ndarray) -> List[float]:
    """Extrai larguras de banda dos formantes (implementação simplificada)."""
    # Implementação muito simplificada
    # Retorna valores típicos como placeholder
    return [50, 70, 90, 110]  # Hz


def estimate_glottal_signal(y: np.ndarray, sr: int) -> Optional[np.ndarray]:
    """Estima sinal glotal (implementação simplificada)."""
    # Implementação muito simplificada usando filtro passa-baixas
    # Uma implementação real usaria predição linear inversa

    try:
        # Filtro passa-baixas para simular remoção de formantes
        sos = scipy.signal.butter(4, 1000, btype='low', fs=sr, output='sos')
        glottal_estimate = scipy.signal.sosfilt(sos, y)

        return glottal_estimate

    except BaseException:
        return None


def detect_pause_segments(y: np.ndarray, sr: int, frame_length: int,
                          hop_length: int) -> List[Tuple[int, int]]:
    """Detecta segmentos de pausa."""
    speech_segments = detect_speech_segments(y, sr, frame_length, hop_length)

    pause_segments = []

    if not speech_segments:
        return [(0, len(y))]

    # Pausas antes do primeiro segmento
    if speech_segments[0][0] > 0:
        pause_segments.append((0, speech_segments[0][0]))

    # Pausas entre segmentos
    for i in range(len(speech_segments) - 1):
        pause_start = speech_segments[i][1]
        pause_end = speech_segments[i + 1][0]

        if pause_end > pause_start:
            pause_segments.append((pause_start, pause_end))

    # Pausa após último segmento
    if speech_segments[-1][1] < len(y):
        pause_segments.append((speech_segments[-1][1], len(y)))

    return pause_segments


def detect_repetitions(y: np.ndarray) -> List[Tuple[int, int]]:
    """Detecta repetições (implementação simplificada)."""
    # Placeholder - implementação real requereria análise de similaridade
    return []


def detect_prolongations(y: np.ndarray) -> List[Tuple[int, int]]:
    """Detecta prolongamentos (implementação simplificada)."""
    # Placeholder - implementação real analisaria estabilidade espectral
    return []


def detect_blocks(y: np.ndarray) -> List[Tuple[int, int]]:
    """Detecta bloqueios (implementação simplificada)."""
    # Placeholder - implementação real analisaria pausas anômalas
    return []


def detect_filled_pauses(y: np.ndarray) -> List[Tuple[int, int]]:
    """Detecta pausas preenchidas (implementação simplificada)."""
    # Placeholder - implementação real usaria reconhecimento de padrões
    return []


def detect_pitch_hesitations(y: np.ndarray) -> List[Tuple[int, int]]:
    """Detecta hesitações de pitch (implementação simplificada)."""
    # Placeholder - implementação real analisaria variações de F0
    return []


def detect_false_starts(y: np.ndarray) -> List[Tuple[int, int]]:
    """Detecta falsos inícios (implementação simplificada)."""
    # Placeholder - implementação real analisaria padrões de início
    return []


def detect_continuous_speech_segments(
        y: np.ndarray, sr: int, frame_length: int, hop_length: int) -> List[Tuple[int, int]]:
    """Detecta segmentos de fala contínua."""
    # Usar segmentos de fala existentes e filtrar por duração mínima
    speech_segments = detect_speech_segments(y, sr, frame_length, hop_length)

    # Filtrar segmentos muito curtos
    min_duration = 0.5 * sr  # 0.5 segundos
    continuous_segments = [(start, end) for start, end in speech_segments
                           if end - start >= min_duration]

    return continuous_segments


def detect_breathing_pauses(
        y: np.ndarray, sr: int, frame_length: int, hop_length: int) -> List[Tuple[int, int]]:
    """Detecta pausas respiratórias."""
    pause_segments = detect_pause_segments(y, sr, frame_length, hop_length)

    # Filtrar pausas por duração (pausas respiratórias são tipicamente
    # 0.2-2.0s)
    breathing_pauses = [(start, end) for start, end in pause_segments
                        if 0.2 <= (end - start) / sr <= 2.0]

    return breathing_pauses


def estimate_speech_duration(
        y: np.ndarray, sr: int, frame_length: int, hop_length: int) -> float:
    """Estima duração total de fala."""
    speech_segments = detect_speech_segments(y, sr, frame_length, hop_length)

    total_duration = sum((end - start) / sr for start, end in speech_segments)

    return total_duration
