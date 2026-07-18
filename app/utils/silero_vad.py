"""Silero VAD — Voice Activity Detection e AGC.

Implementacao conforme especificado no TCC (Secao 5.1):
  "Silero VAD (via Torch Hub) oferece excelente performance pre-treinada
   com execucao ultrarapida. Para AGC e reamostragem,
   torchaudio.transforms provê operacoes vetorizadas em GPU."

VAD (Eq. 6-7 do TCC):
  E[m] = 10 log10( (1/N) sum |x[mH+n]|^2 )
  Voice[m] = 1  se E[m] > -25dB e D[m] > 0.3s

AGC (Eq. 5 do TCC):
  x_norm[n] = x[n] * L_target / sqrt( (1/N) sum x[i]^2 )
  L_target = -23 LUFS (equivalente a normalizacao RMS)
"""

import logging
from typing import Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# Constantes alinhadas com o TCC (Secao 3.2)
TARGET_SR = 16_000
VAD_THRESHOLD_DB = -25.0      # theta_VAD
MIN_SEGMENT_DURATION = 0.3    # D_min em segundos
TARGET_LUFS = -23.0           # L_target (AGC)
AGC_HEADROOM = 0.95           # margem de pico


# ---------------------------------------------------------------------------
# Silero VAD (torch.hub) — wrapper com fallback para VAD por energia
# ---------------------------------------------------------------------------
_silero_model = None
_silero_utils = None
_silero_failed = False  # Set to True after first failed attempt to avoid retrying


def _load_silero():
    """Carrega o modelo Silero VAD via torch.hub (lazy loading)."""
    global _silero_model, _silero_utils, _silero_failed
    if _silero_model is not None:
        return True
    if _silero_failed:
        return False
    try:
        import torch
        _silero_model, _silero_utils = torch.hub.load(
            repo_or_dir="snakers4/silero-vad",
            model="silero_vad",
            force_reload=False,
            verbose=False,
            trust_repo=True,
        )
        logger.info("Silero VAD carregado via torch.hub")
        return True
    except Exception as e:
        logger.warning(f"Silero VAD indisponivel ({e}). Usando VAD por energia (fallback).")
        _silero_failed = True
        return False


def apply_silero_vad(
    audio: np.ndarray,
    sample_rate: int = TARGET_SR,
    threshold: float = 0.5,
    min_silence_duration_ms: int = 300,
    min_speech_duration_ms: int = 300,
) -> np.ndarray:
    """
    Remove segmentos de silencio usando Silero VAD.

    Args:
        audio: array 1D float32, sample_rate Hz
        sample_rate: deve ser 8000 ou 16000 (requisito Silero)
        threshold: limiar de probabilidade de voz (0-1)
        min_silence_duration_ms: silencio minimo para corte (ms)
        min_speech_duration_ms: fala minima para manter (ms)

    Returns:
        array com apenas os segmentos de voz concatenados
    """
    if _load_silero():
        try:
            import torch
            wav = torch.FloatTensor(audio)

            get_speech_timestamps = _silero_utils[0]
            collect_chunks = _silero_utils[3]

            timestamps = get_speech_timestamps(
                wav,
                _silero_model,
                sampling_rate=sample_rate,
                threshold=threshold,
                min_silence_duration_ms=min_silence_duration_ms,
                min_speech_duration_ms=min_speech_duration_ms,
            )

            if not timestamps:
                logger.debug("Silero VAD: nenhum segmento de voz detectado. Retornando audio original.")
                return audio

            speech = collect_chunks(timestamps, wav)
            return speech.numpy()

        except Exception as e:
            logger.warning(f"Silero VAD falhou ({e}). Usando fallback por energia.")

    # Fallback: VAD por energia (Eq. 6-7 do TCC)
    return _energy_vad(audio, sample_rate)


def _energy_vad(
    audio: np.ndarray,
    sample_rate: int = TARGET_SR,
    frame_length_ms: float = 25.0,
    hop_length_ms: float = 10.0,
    threshold_db: float = VAD_THRESHOLD_DB,
    min_duration_s: float = MIN_SEGMENT_DURATION,
) -> np.ndarray:
    """
    VAD por energia de quadros (fallback sem torch).
    Implementa diretamente Eq. 6-7 do TCC.
    """
    frame_length = int(sample_rate * frame_length_ms / 1000)
    hop_length = int(sample_rate * hop_length_ms / 1000)
    min_frames = int(min_duration_s * sample_rate / hop_length)

    # Calcular energia por quadro (Eq. 6)
    n_frames = max(1, (len(audio) - frame_length) // hop_length + 1)
    energy_db = np.full(n_frames, -100.0)
    for m in range(n_frames):
        start = m * hop_length
        frame = audio[start: start + frame_length]
        rms_sq = np.mean(frame ** 2) + 1e-10
        energy_db[m] = 10.0 * np.log10(rms_sq)

    # Mascara de voz (Eq. 7)
    voice_mask = energy_db > threshold_db

    # Coletar segmentos validos
    speech_segments = []
    in_speech = False
    seg_start = 0
    seg_frames = 0

    for m, is_voice in enumerate(voice_mask):
        if is_voice and not in_speech:
            in_speech = True
            seg_start = m
            seg_frames = 1
        elif is_voice and in_speech:
            seg_frames += 1
        elif not is_voice and in_speech:
            if seg_frames >= min_frames:
                start_sample = seg_start * hop_length
                end_sample = min((seg_start + seg_frames) * hop_length + frame_length, len(audio))
                speech_segments.append(audio[start_sample:end_sample])
            in_speech = False
            seg_frames = 0

    # Ultimo segmento
    if in_speech and seg_frames >= min_frames:
        start_sample = seg_start * hop_length
        speech_segments.append(audio[start_sample:])

    if not speech_segments:
        return audio  # sem corte se nao encontrou segmentos

    return np.concatenate(speech_segments)


# ---------------------------------------------------------------------------
# AGC — Automatic Gain Control (Eq. 5 do TCC)
# ---------------------------------------------------------------------------
def apply_agc(
    audio: np.ndarray,
    target_lufs: float = TARGET_LUFS,
    headroom: float = AGC_HEADROOM,
) -> np.ndarray:
    """
    Normalizacao AGC alinhada com Eq. 5 do TCC.

    x_norm[n] = x[n] * L_target / sqrt( (1/N) * sum x[i]^2 )

    L_target = -23 LUFS (aproximado como nivel RMS alvo).
    A conversao exata LUFS requereria ponderacao K-weighting;
    aqui usamos RMS como aproximacao pratica (comum em anti-spoofing).

    Args:
        audio: array float32
        target_lufs: nivel alvo em LUFS (default: -23.0)
        headroom: fator de pico maximo (default: 0.95)

    Returns:
        audio normalizado
    """
    rms = np.sqrt(np.mean(audio.astype(np.float64) ** 2) + 1e-10)

    # Converter LUFS para nivel linear (aproximacao RMS)
    target_rms = 10.0 ** (target_lufs / 20.0)

    gain = target_rms / rms
    normalized = audio * gain

    # Prevenir clipping
    peak = np.max(np.abs(normalized))
    if peak > headroom:
        normalized = normalized * (headroom / peak)

    return normalized.astype(np.float32)


# ---------------------------------------------------------------------------
# Pipeline completo de pre-processamento (VAD + AGC)
# ---------------------------------------------------------------------------
def preprocess_audio(
    audio: np.ndarray,
    sample_rate: int = TARGET_SR,
    apply_vad: bool = True,
    apply_gain: bool = True,
    vad_threshold: float = 0.5,
) -> Tuple[np.ndarray, dict]:
    """
    Pipeline completo de pre-processamento conforme TCC Secao 3.2.

    Ordem: AGC -> VAD (remover silencio) -> retornar com metadados.

    Args:
        audio: array float32, 16kHz mono
        sample_rate: taxa de amostragem (deve ser 16000)
        apply_vad: aplicar remocao de silencio
        apply_gain: aplicar normalizacao AGC
        vad_threshold: limiar Silero VAD (0-1)

    Returns:
        (audio_processado, metadata_dict)
    """
    meta = {
        "original_duration_s": len(audio) / sample_rate,
        "agc_applied": False,
        "vad_applied": False,
        "final_duration_s": len(audio) / sample_rate,
        "samples_removed_by_vad": 0,
    }

    # 1. AGC primeiro (normalizar antes do VAD melhora deteccao de silencio)
    if apply_gain and len(audio) > 0:
        audio = apply_agc(audio)
        meta["agc_applied"] = True

    # 2. VAD (remover silencio)
    if apply_vad and len(audio) > 0:
        original_len = len(audio)
        audio = apply_silero_vad(audio, sample_rate=sample_rate, threshold=vad_threshold)
        meta["vad_applied"] = True
        meta["samples_removed_by_vad"] = max(0, original_len - len(audio))

    meta["final_duration_s"] = len(audio) / sample_rate
    return audio, meta
