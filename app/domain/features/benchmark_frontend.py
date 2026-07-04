"""Front-end de características do BENCHMARK — fonte única treino↔inferência.

Este módulo contém, verbatim, as transformações de entrada usadas pelo
benchmark (``benchmarks/data.py``) para treinar os modelos promovidos em
``app/models/benchmark_final``:

- **raw**: janela center-crop (ou repetição p/ clipes curtos) + z-score por
  amostra + canal — consumida por AASIST/RawGAT-ST/RawNet2 (alvo 16000 ≙ 1 s);
- **log-mel**: librosa ``melspectrogram`` (n_fft=512, hop dinâmico p/ fixar
  ``time_steps`` quadros), ``power_to_db(ref=max)`` POR AMOSTRA e z-score por
  amostra — mapa ``(time_steps, feature_dim)`` = (100, 80) — consumida por
  Conformer/Res2Net/AST/CCT;
- **tabular-63**: 11 estatísticas temporais + 26 MFCC + 26 RASTA-PLP —
  consumida por SVM/Random Forest (nomes canônicos em
  ``app/core/xai/tabular.py``).

Por que aqui: a inferência do app usava um front-end próprio
(``audio_preprocessing.py``: log-magnitude-mel, hop 128, sem z-score) que NÃO
reproduz o do benchmark — sem paridade, as métricas reportadas não se
transferem para a predição em produção. ``benchmarks/data.py`` delega para
este módulo (mesma função ⇒ paridade por construção), e o
``FeaturePreparer`` roteia para cá quando o ``input_contract`` do modelo
declara um dos ``feature_frontend`` abaixo.
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

# Identificadores gravados no input_contract dos modelos do benchmark.
FRONTEND_RAW = "benchmark_raw_v1"
FRONTEND_LOGMEL = "benchmark_logmel_v1"
FRONTEND_TABULAR = "benchmark_tabular_v1"
BENCHMARK_FRONTENDS = (FRONTEND_RAW, FRONTEND_LOGMEL, FRONTEND_TABULAR)

#: Janela-fonte canônica do benchmark: 5 s @ 16 kHz.
DEFAULT_SOURCE_SAMPLES = 80000
DEFAULT_SAMPLE_RATE = 16000
DEFAULT_FEATURE_DIM = 80
DEFAULT_TIME_STEPS = 100
DEFAULT_RAW_TARGET = 16000  # recorte central de 1 s (AASIST/RawGAT/RawNet2)

N_TABULAR_FEATURES = 63


def fit_length_tile(flat: np.ndarray, target_len: int) -> np.ndarray:
    """Ajusta ``(N, T)`` para ``target_len``: center-crop ou repetição (tile).

    Semântica idêntica a ``benchmarks/data.py::_fit_length`` — clipes curtos
    são REPETIDOS (não zero-preenchidos) até o alvo.
    """
    flat = np.asarray(flat, dtype="float32")
    if flat.ndim == 1:
        flat = flat[np.newaxis, :]
    if flat.shape[1] == target_len:
        return flat
    if flat.shape[1] > target_len:
        start = max(0, (flat.shape[1] - target_len) // 2)
        return flat[:, start : start + target_len]
    repeats = int(np.ceil(target_len / max(1, flat.shape[1])))
    return np.tile(flat, (1, repeats))[:, :target_len]


def normalize_per_sample(X: np.ndarray) -> np.ndarray:
    """Z-score por amostra sobre TODOS os eixos não-batch (idem benchmark)."""
    X = np.asarray(X)
    flat = X.reshape(len(X), -1)
    mean = flat.mean(axis=1, keepdims=True)
    std = flat.std(axis=1, keepdims=True)
    return ((flat - mean) / np.maximum(std, 1e-6)).reshape(X.shape).astype("float32")


def _resize_time_axis(X: np.ndarray, target: int) -> np.ndarray:
    """Trunca ou edge-preenche o eixo 1 (tempo) — idem ``_resize_axis``."""
    current = X.shape[1]
    if current == target:
        return X
    if current > target:
        return X[:, :target]
    pad_width = [(0, 0)] * X.ndim
    pad_width[1] = (0, target - current)
    return np.pad(X, pad_width, mode="edge")


def raw_audio_batch(X: np.ndarray, target_len: int = DEFAULT_RAW_TARGET) -> np.ndarray:
    """Janela raw do benchmark: ``(N, ·)`` → ``(N, target_len, 1)`` z-scored."""
    flat = np.asarray(X, dtype="float32").reshape(len(X), -1)
    raw = fit_length_tile(flat, max(1, int(target_len)))
    raw = normalize_per_sample(raw)
    return raw[..., np.newaxis]


def raw_audio_single(y: np.ndarray, target_len: int = DEFAULT_RAW_TARGET) -> np.ndarray:
    """Versão single-sample de :func:`raw_audio_batch` → ``(target_len, 1)``."""
    return raw_audio_batch(np.asarray(y, dtype="float32")[np.newaxis, :], target_len)[0]


def log_mel_batch(
    X: np.ndarray,
    sample_rate: int = DEFAULT_SAMPLE_RATE,
    feature_dim: int = DEFAULT_FEATURE_DIM,
    time_steps: int = DEFAULT_TIME_STEPS,
) -> np.ndarray:
    """Log-mel do benchmark: ``(N, T)`` raw → ``(N, time_steps, feature_dim)``.

    Reprodução exata de ``benchmarks/data.py::_raw_audio_to_logmel``:
    n_fft=512, hop dinâmico ``ceil(T/time_steps)`` (≥64), potência, dB com
    ``ref=max`` POR AMOSTRA, ajuste do eixo temporal e z-score por amostra.
    """
    import librosa

    flat = np.asarray(X, dtype="float32").reshape(len(X), -1)
    sample_rate = int(sample_rate or DEFAULT_SAMPLE_RATE)
    feature_dim = int(feature_dim or DEFAULT_FEATURE_DIM)
    time_steps = int(time_steps or DEFAULT_TIME_STEPS)
    hop_length = max(64, int(np.ceil(flat.shape[1] / max(time_steps, 1))))

    specs = []
    for y in flat:
        mel = librosa.feature.melspectrogram(
            y=y.astype("float32"),
            sr=sample_rate,
            n_fft=512,
            hop_length=hop_length,
            n_mels=feature_dim,
            power=2.0,
        )
        mel_db = librosa.power_to_db(mel + 1e-10, ref=np.max).T
        mel_db = _resize_time_axis(mel_db[np.newaxis, ...], max(1, time_steps))[0]
        specs.append(mel_db[:, :feature_dim])
    return normalize_per_sample(np.asarray(specs, dtype="float32"))


def log_mel_single(
    y: np.ndarray,
    sample_rate: int = DEFAULT_SAMPLE_RATE,
    feature_dim: int = DEFAULT_FEATURE_DIM,
    time_steps: int = DEFAULT_TIME_STEPS,
    source_samples: int = DEFAULT_SOURCE_SAMPLES,
) -> np.ndarray:
    """Versão single-sample de :func:`log_mel_batch` → ``(time_steps, F)``.

    Primeiro ajusta o clipe à janela-fonte do benchmark (``source_samples``,
    5 s por padrão) para que o hop dinâmico seja o MESMO do treino.
    """
    flat = fit_length_tile(
        np.asarray(y, dtype="float32")[np.newaxis, :], int(source_samples)
    )
    return log_mel_batch(
        flat, sample_rate=sample_rate, feature_dim=feature_dim, time_steps=time_steps
    )[0]


def _rasta_plp_stats(flat: np.ndarray, n_plp: int = 13) -> np.ndarray:
    """Média/desvio por coeficiente RASTA-PLP — idem benchmark (zeros em falha)."""
    try:
        from app.domain.features.extractors.cepstral.components.plp import (
            extract_rasta_plp_features,
        )
    except Exception:  # noqa: BLE001 - extrator opcional
        return np.zeros((2 * n_plp, len(flat)), dtype="float32")

    rows = []
    for y in flat:
        try:
            feats = extract_rasta_plp_features(
                y.astype("float32"), sr=16000, frame_length=512,
                hop_length=256, n_plp=n_plp,
            )
            rp = np.asarray(feats.get("rasta_plp"))
            if rp.ndim != 2 or rp.shape[0] != n_plp:
                raise ValueError("forma RASTA-PLP inesperada")
            rows.append(np.concatenate([rp.mean(axis=1), rp.std(axis=1)]))
        except Exception:  # noqa: BLE001 - degrada p/ zeros por amostra
            rows.append(np.zeros(2 * n_plp, dtype="float32"))
    arr = np.nan_to_num(
        np.asarray(rows, dtype="float32"), nan=0.0, posinf=0.0, neginf=0.0
    )
    return arr.T


def tabular_features_batch(X: np.ndarray) -> np.ndarray:
    """Vetor tabular de 63 descritores do benchmark: ``(N, ·)`` → ``(N, 63)``.

    Ordem canônica (nomes em ``app/core/xai/tabular.py``): 11 estatísticas
    temporais, 26 MFCC (média+desvio de 13), 26 RASTA-PLP (média+desvio de 13).
    """
    flat = np.asarray(X, dtype="float32").reshape(len(X), -1)
    feats = [
        flat.mean(axis=1),
        flat.std(axis=1),
        np.mean(np.abs(flat), axis=1),
        np.sqrt(np.mean(flat ** 2, axis=1)),
        flat.min(axis=1),
        flat.max(axis=1),
        np.percentile(flat, 25, axis=1),
        np.percentile(flat, 50, axis=1),
        np.percentile(flat, 75, axis=1),
        np.mean(np.diff(flat, axis=1) ** 2, axis=1),
        np.mean(np.signbit(flat[:, 1:]) != np.signbit(flat[:, :-1]), axis=1),
    ]
    try:
        import librosa

        mfcc_stats = []
        for y in flat:
            mfcc = librosa.feature.mfcc(y=y, sr=16000, n_mfcc=13)
            mfcc_stats.append(
                np.concatenate([mfcc.mean(axis=1), mfcc.std(axis=1)])
            )
        feats.append(np.asarray(mfcc_stats, dtype="float32").T)
    except Exception:  # noqa: BLE001 - librosa opcional (idem benchmark)
        pass

    feats.append(_rasta_plp_stats(flat))
    return np.vstack(feats).T.astype("float32")


def tabular_features_single(
    y: np.ndarray, source_samples: int = DEFAULT_SOURCE_SAMPLES
) -> np.ndarray:
    """Versão single-sample de :func:`tabular_features_batch` → ``(63,)``.

    Ajusta o clipe à janela-fonte de 5 s antes da extração (as estatísticas
    do treino foram computadas sobre essa janela).
    """
    flat = fit_length_tile(
        np.asarray(y, dtype="float32")[np.newaxis, :], int(source_samples)
    )
    return tabular_features_batch(flat)[0]


def prepare_single(
    y: np.ndarray,
    feature_frontend: str,
    *,
    sample_rate: int = DEFAULT_SAMPLE_RATE,
    feature_dim: int = DEFAULT_FEATURE_DIM,
    time_steps: int = DEFAULT_TIME_STEPS,
    target_sequence_length: int = DEFAULT_RAW_TARGET,
    source_samples: int = DEFAULT_SOURCE_SAMPLES,
    add_channel_dim: Optional[bool] = None,
) -> np.ndarray:
    """Roteia uma amostra pelo front-end do benchmark declarado no contrato.

    Args:
        y: áudio mono (T,) float32 já na taxa do contrato.
        feature_frontend: um de :data:`BENCHMARK_FRONTENDS`.
        add_channel_dim: para log-mel, acrescenta eixo de canal quando o
            modelo espera ``(T, F, 1)``; ``None`` mantém ``(T, F)``.
    """
    if feature_frontend == FRONTEND_RAW:
        return raw_audio_single(y, target_len=target_sequence_length)
    if feature_frontend == FRONTEND_LOGMEL:
        spec = log_mel_single(
            y,
            sample_rate=sample_rate,
            feature_dim=feature_dim,
            time_steps=time_steps,
            source_samples=source_samples,
        )
        if add_channel_dim:
            spec = spec[..., np.newaxis]
        return spec
    if feature_frontend == FRONTEND_TABULAR:
        return tabular_features_single(y, source_samples=source_samples)
    raise ValueError(
        f"feature_frontend desconhecido: {feature_frontend!r} "
        f"(esperado um de {BENCHMARK_FRONTENDS})"
    )
