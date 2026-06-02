"""Pré-processamento de áudio unificado para treino e inferência.

**Por quê este módulo?** O training_wizard usa `tf.signal.stft` +
`linear_to_mel_weight_matrix` para gerar log-mel on-the-fly. A
inferência historicamente usava `librosa` via `MelSpectrogramExtractor`,
o que produz valores numericamente diferentes (janela, normalização,
escala mel). Resultado: train/test mismatch.

Este módulo centraliza a função de áudio→log-mel para que **ambas** as
pipelines usem exatamente os mesmos parâmetros e a mesma biblioteca.

Parâmetros padrão (alinhados com `training_wizard.py`):
    SAMPLE_RATE = 16000
    N_FFT       = 512
    HOP         = 128
    N_MELS      = 80
    FMIN        = 0.0
    FMAX        = SAMPLE_RATE / 2

Saída: log-mel `(T_frames, n_mels, 1)` em float32.
"""

from __future__ import annotations

import logging
from typing import Optional, Tuple

import numpy as np
import tensorflow as tf

logger = logging.getLogger(__name__)

# Defaults compartilhados (devem casar com training_wizard.py)
DEFAULT_SAMPLE_RATE = 16000
DEFAULT_N_FFT = 512
DEFAULT_HOP = 128
DEFAULT_N_MELS = 80
DEFAULT_DURATION_SEC = 3  # janela típica usada no treino do wizard

# Front-end de espectrograma. "lfcc" é o default para NOVOS treinos (literatura
# anti-spoofing: LFCC > mel em altas frequências, onde ficam artefatos de
# vocoder). Modelos legados sem `feature_frontend` no contrato continuam usando
# "logmel" para preservar a paridade train↔inference original.
DEFAULT_FRONTEND = "lfcc"
DEFAULT_N_LFCC = 80          # nº de coeficientes mantidos (preserva shape (T,80,1))
DEFAULT_N_LIN_FILTERS = 80   # nº de filtros triangulares lineares
LFCC_CLIP = 50.0             # bound numérico (NaN-safety)


def linear_filterbank_matrix(
    n_freq: int,
    n_filters: int,
    sample_rate: int,
    fmin: float = 0.0,
    fmax: Optional[float] = None,
) -> np.ndarray:
    """Banco de filtros triangulares **linearmente** espaçados (base do LFCC).

    Diferença-chave para mel: espaçamento linear preserva resolução em altas
    frequências. Retorna matriz (n_freq, n_filters) para multiplicar pelo
    espectro de potência.
    """
    if fmax is None:
        fmax = sample_rate / 2.0
    f_pts = np.linspace(fmin, fmax, n_filters + 2).astype(np.float32)
    bin_freqs = np.linspace(0.0, sample_rate / 2.0, n_freq).astype(np.float32)
    fb = np.zeros((n_freq, n_filters), dtype=np.float32)
    for m in range(1, n_filters + 1):
        left, center, right = f_pts[m - 1], f_pts[m], f_pts[m + 1]
        up = (bin_freqs - left) / max(center - left, 1e-6)
        down = (right - bin_freqs) / max(right - center, 1e-6)
        fb[:, m - 1] = np.maximum(0.0, np.minimum(up, down))
    return fb


def lfcc_from_waveform(
    audio_t: "tf.Tensor",
    *,
    sample_rate: int = DEFAULT_SAMPLE_RATE,
    n_fft: int = DEFAULT_N_FFT,
    hop_length: int = DEFAULT_HOP,
    n_filters: int = DEFAULT_N_LIN_FILTERS,
    n_lfcc: int = DEFAULT_N_LFCC,
    fmin: float = 0.0,
    fmax: Optional[float] = None,
    log_eps: float = 1e-6,
    clip: Optional[float] = LFCC_CLIP,
) -> "tf.Tensor":
    """Núcleo COMPARTILHADO áudio→LFCC (tensor, funciona em grafo e eager).

    Usado tanto pelo treino (`training_wizard._to_lfcc`, grafo `tf.data`) quanto
    pela inferência (`audio_to_lfcc`), garantindo **paridade bit-a-bit** entre
    treino e teste (mesma função, mesmos parâmetros).

    LFCC = DCT-II( log( banco_linear( |STFT|² ) ) )[:n_lfcc]

    Args:
        audio_t: tensor (..., T) — aceita dim de batch à esquerda.
    Returns:
        Tensor (..., T_frames, n_lfcc).
    """
    if fmax is None:
        fmax = sample_rate / 2.0
    stft = tf.signal.stft(
        audio_t,
        frame_length=n_fft,
        frame_step=hop_length,
        fft_length=n_fft,
        window_fn=tf.signal.hann_window,
        pad_end=True,
    )
    power = tf.square(tf.abs(stft))  # (..., T_frames, n_freq)
    n_freq = n_fft // 2 + 1
    fb = tf.constant(
        linear_filterbank_matrix(n_freq, n_filters, sample_rate, fmin, fmax)
    )  # (n_freq, n_filters)
    filt = tf.tensordot(power, fb, axes=[[-1], [0]])  # (..., T_frames, n_filters)
    log_fb = tf.math.log(filt + log_eps)
    lfcc = tf.signal.dct(log_fb, type=2, norm="ortho")  # DCT no eixo dos filtros
    lfcc = lfcc[..., :n_lfcc]
    if clip is not None:
        lfcc = tf.where(tf.math.is_finite(lfcc), lfcc, tf.zeros_like(lfcc))
        lfcc = tf.clip_by_value(lfcc, -clip, clip)
    return lfcc


def audio_to_lfcc(
    audio: np.ndarray,
    *,
    sample_rate: int = DEFAULT_SAMPLE_RATE,
    n_fft: int = DEFAULT_N_FFT,
    hop_length: int = DEFAULT_HOP,
    n_filters: int = DEFAULT_N_LIN_FILTERS,
    n_lfcc: int = DEFAULT_N_LFCC,
    fmin: float = 0.0,
    fmax: Optional[float] = None,
    add_channel_dim: bool = True,
    log_eps: float = 1e-6,
) -> np.ndarray:
    """Versão eager (numpy in/out) do front-end LFCC para inferência.

    Espelha `audio_to_log_mel`, mas usa o núcleo `lfcc_from_waveform` para
    garantir igualdade com o treino.
    """
    audio = np.asarray(audio, dtype=np.float32)
    if audio.ndim == 2:
        audio = audio.squeeze(-1) if audio.shape[-1] == 1 else audio.mean(axis=-1)
    if audio.ndim != 1:
        raise ValueError(f"audio deve ser 1D ou 2D, recebido shape={audio.shape}")

    audio_t = tf.constant(audio[np.newaxis, :], dtype=tf.float32)  # (1, T)
    lfcc = lfcc_from_waveform(
        audio_t,
        sample_rate=sample_rate,
        n_fft=n_fft,
        hop_length=hop_length,
        n_filters=n_filters,
        n_lfcc=n_lfcc,
        fmin=fmin,
        fmax=fmax,
        log_eps=log_eps,
    )
    out = lfcc.numpy().squeeze(0)  # (T_frames, n_lfcc)
    if add_channel_dim:
        out = out[..., np.newaxis]
    return out.astype(np.float32)


def normalize_audio(samples: np.ndarray) -> np.ndarray:
    """Normalização peak (mesmo método do wizard)."""
    samples = np.asarray(samples, dtype=np.float32)
    if samples.ndim > 1:
        samples = samples.mean(axis=-1)  # downmix para mono
    max_abs = float(np.max(np.abs(samples)))
    if max_abs > 0:
        samples = samples / max_abs
    return samples


def pad_or_truncate_audio(
    samples: np.ndarray,
    target_length: int,
    pad_mode: str = "constant",
) -> np.ndarray:
    """Ajusta o áudio para `target_length` amostras (center-crop ou pad zero)."""
    n = samples.shape[0]
    if n == target_length:
        return samples
    if n > target_length:
        # Center crop
        mid = n // 2
        start = max(0, mid - target_length // 2)
        return samples[start:start + target_length]
    # Pad
    pad = target_length - n
    return np.pad(samples, (0, pad), mode=pad_mode)


def audio_to_log_mel(
    audio: np.ndarray,
    *,
    sample_rate: int = DEFAULT_SAMPLE_RATE,
    n_fft: int = DEFAULT_N_FFT,
    hop_length: int = DEFAULT_HOP,
    n_mels: int = DEFAULT_N_MELS,
    fmin: float = 0.0,
    fmax: Optional[float] = None,
    add_channel_dim: bool = True,
    log_eps: float = 1e-6,
) -> np.ndarray:
    """Converte áudio bruto em log-mel-spectrogram via tf.signal.

    **Mesma fórmula** usada em `training_wizard.py` — garante que features
    geradas no treino e na inferência sejam **bit-equivalentes**.

    Args:
        audio: (T,) ou (T, 1) float32, mono. Será achatado para 1D se 2D.
        sample_rate: SR original do áudio (default 16000)
        n_fft, hop_length: parâmetros STFT
        n_mels: número de bandas mel (default 80)
        fmin, fmax: faixa de frequências (Hz). `fmax=None` ⇒ sr/2
        add_channel_dim: se True, retorna (T_frames, n_mels, 1); senão (T_frames, n_mels)
        log_eps: epsilon para evitar log(0)

    Returns:
        Array float32 com shape (T_frames, n_mels) ou (T_frames, n_mels, 1).
    """
    audio = np.asarray(audio, dtype=np.float32)
    if audio.ndim == 2:
        # (T, 1) ou (T, C) → 1D
        audio = audio.squeeze(-1) if audio.shape[-1] == 1 else audio.mean(axis=-1)
    if audio.ndim != 1:
        raise ValueError(
            f"audio deve ser 1D ou 2D, recebido shape={audio.shape}"
        )

    if fmax is None:
        fmax = sample_rate / 2.0

    audio_t = tf.constant(audio[np.newaxis, :], dtype=tf.float32)  # (1, T)
    stft = tf.signal.stft(
        audio_t,
        frame_length=n_fft,
        frame_step=hop_length,
        fft_length=n_fft,
        window_fn=tf.signal.hann_window,
        pad_end=True,
    )  # (1, T_frames, n_freq)
    mag = tf.abs(stft)

    n_freq = n_fft // 2 + 1
    mel_w = tf.signal.linear_to_mel_weight_matrix(
        num_mel_bins=n_mels,
        num_spectrogram_bins=n_freq,
        sample_rate=sample_rate,
        lower_edge_hertz=fmin,
        upper_edge_hertz=fmax,
    )
    mel = tf.tensordot(mag, mel_w, axes=1)  # (1, T_frames, n_mels)
    log_mel = tf.math.log(mel + log_eps)
    out = log_mel.numpy().squeeze(0)  # (T_frames, n_mels)

    if add_channel_dim:
        out = out[..., np.newaxis]  # (T_frames, n_mels, 1)
    return out.astype(np.float32)


def prepare_audio_for_model(
    samples: np.ndarray,
    *,
    input_type: str,
    input_shape: Tuple[int, ...],
    sample_rate: int = DEFAULT_SAMPLE_RATE,
    normalize: bool = True,
    n_fft: int = DEFAULT_N_FFT,
    hop_length: int = DEFAULT_HOP,
    n_mels: int = DEFAULT_N_MELS,
    feature_frontend: str = "logmel",
    n_lfcc: int = DEFAULT_N_LFCC,
) -> np.ndarray:
    """Pipeline unificado: áudio → tensor pronto para `model.predict()`.

    Despacha entre `raw_audio` e `spectrogram` conforme o `input_type` do
    spec da arquitetura.

    Args:
        samples: áudio bruto (T,) ou (T, C)
        input_type: "raw_audio" ou "spectrogram"
        input_shape: input_shape do modelo (sem dim batch); usado para
            calcular target_length em raw e para validar shape de saída.
        sample_rate: SR do áudio fornecido (não converte — assume que já é o esperado)
        normalize: aplica normalização peak antes do processamento
        n_fft, hop_length, n_mels: parâmetros STFT (apenas se spectrogram)

    Returns:
        Array float32 pronto para `model.predict(arr[np.newaxis, ...])`.
        - raw_audio: shape (T, 1)
        - spectrogram: shape (T_frames, n_mels, 1) ou (T_frames, n_mels)
    """
    samples = np.asarray(samples, dtype=np.float32)
    if samples.ndim > 1:
        samples = samples.mean(axis=-1) if samples.shape[-1] > 1 else samples.squeeze(-1)

    if normalize:
        samples = normalize_audio(samples)

    if input_type == "raw_audio":
        # Modelo opera em (T, 1) ou (T,). Calcula T pelo input_shape.
        target_T = input_shape[0] if input_shape and input_shape[0] else len(samples)
        samples = pad_or_truncate_audio(samples, int(target_T))
        # Output shape: (T, 1) ou (T,) — espelha input_shape do modelo
        if len(input_shape) >= 2 and input_shape[-1] == 1:
            return samples.reshape(int(target_T), 1).astype(np.float32)
        return samples.astype(np.float32)

    if input_type == "spectrogram":
        # Decide se modelo aceita (T, F) ou (T, F, 1)
        add_channel = bool(
            len(input_shape) == 3 and input_shape[-1] == 1
        )

        # Despacha entre LFCC (default p/ novos treinos) e log-mel (legado).
        # O `feature_frontend` vem do input_contract gravado no treino, então
        # a inferência usa SEMPRE o mesmo front-end com que o modelo foi treinado.
        if feature_frontend == "lfcc":
            spec = audio_to_lfcc(
                samples,
                sample_rate=sample_rate,
                n_fft=n_fft,
                hop_length=hop_length,
                n_filters=n_lfcc,
                n_lfcc=n_lfcc,
                add_channel_dim=add_channel,
            )
        else:
            spec = audio_to_log_mel(
                samples,
                sample_rate=sample_rate,
                n_fft=n_fft,
                hop_length=hop_length,
                n_mels=n_mels,
                add_channel_dim=add_channel,
            )

        # Ajusta T_frames ao input_shape do modelo (pad/truncate)
        if input_shape and input_shape[0]:
            target_T = int(input_shape[0])
            cur_T = spec.shape[0]
            if cur_T > target_T:
                spec = spec[:target_T]
            elif cur_T < target_T:
                pad = target_T - cur_T
                # Pad nas T_frames (eixo 0)
                pad_widths = [(0, pad)] + [(0, 0)] * (spec.ndim - 1)
                spec = np.pad(spec, pad_widths, mode="constant")
        return spec.astype(np.float32)

    raise ValueError(
        f"input_type '{input_type}' desconhecido. Use 'raw_audio' ou 'spectrogram'."
    )
