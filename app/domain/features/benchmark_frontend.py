"""Front-end de características do BENCHMARK — fonte única treino↔inferência.

Este módulo contém, verbatim, as transformações de entrada usadas pelo
benchmark (``benchmarks/data.py``) para treinar os modelos promovidos em
``data/models/benchmark_final``:

- **raw**: janela center-crop (ou repetição p/ clipes curtos) + z-score por
  amostra + canal — AASIST/RawGAT-ST usam a janela canonica e multicrop;
- **log-mel**: librosa ``melspectrogram`` (n_fft=512, hop dinâmico p/ fixar
  ``time_steps`` quadros), ``power_to_db(ref=max)`` POR AMOSTRA e z-score por
  amostra — mapa ``(time_steps, feature_dim)`` = (100, 80) — consumida por
  Conformer/Res2Net/AST/CCT;
- **tabular-63**: 11 estatísticas temporais + 26 MFCC + 26 RASTA-PLP —
  consumida por SVM/Random Forest (nomes canônicos em
  ``app/domain/xai/tabular.py``).

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

#: Janela-fonte canônica do benchmark: 3 s @ 16 kHz.
#:
#: É exatamente a janela do dataset (docs/data/dataset-protocol.md). O casamento
#: importa: `fit_length_tile` REPETE o clipe para alcançar a janela pedida, então
#: uma janela-fonte maior que a do dataset faz todo lote passar por repetição —
#: o oposto do que o protocolo garante ao descartar os pares curtos. Com 80.000
#: (a antiga janela de 5 s), cada amostra de 3 s era repetida 1,67× antes do
#: log-Mel e do vetor tabular.
DEFAULT_SOURCE_SAMPLES = 48000
DEFAULT_SAMPLE_RATE = 16000
DEFAULT_FEATURE_DIM = 80
DEFAULT_TIME_STEPS = 100
#: Alvo do front-end raw, igual à janela-fonte para que o recorte central do
#: dataset chegue intacto ao modelo. O valor anterior (64.600, a convenção de
#: ~4,04 s do RawNet2/AASIST) obrigava a repetir 1,35× cada amostra de 3 s.
DEFAULT_RAW_TARGET = 48000

N_TABULAR_FEATURES = 63

#: Sobreposição mínima entre janelas consecutivas do log-mel.
#:
#: O salto é imposto pelo contrato da arquitetura (`ceil(T / time_steps)`), mas
#: a janela era a constante 512 — e as duas eram INDEPENDENTES no código. Com
#: 100 quadros em 3 s o salto fica em 480 amostras: janelas consecutivas se
#: sobrepunham em 32 amostras, e o taper de Hann é ~0 nas duas pontas.
#:
#: Medido: o envelope de soma-e-sobreposição ia de 1,0 a EXATAMENTE 0 — **27%
#: do sinal caía em regiões de peso desprezível**, invisíveis à análise. Pior
#: para esta tarefa: um clique de 1 ms era 9× mais visível ou menos conforme a
#: posição em que caísse (razão mín/máx de 0,11), e artefato de síntese é
#: justamente transiente — descontinuidade de fase, ponto de emenda.
#:
#: Com 50% de sobreposição a razão sobe para 0,90 e o ponto cego desaparece,
#: SEM perda de detecção média (a janela maior mede 0,041–0,043 contra o melhor
#: caso 0,043 da janela curta). A 6% de sobreposição o fator dominante não era
#: resolução temporal, era o ponto cego.
MIN_STFT_OVERLAP = 0.5


def resolve_n_fft(hop_length: int, declared: Optional[int] = None) -> int:
    """Janela de análise compatível com o salto, ou a declarada pelo contrato.

    Um `n_fft` explícito da arquitetura sempre vence: o AST especifica 25 ms
    (400 amostras a 16 kHz) por definição do artigo, e com salto de 160 já
    obtém 60% de sobreposição.

    Sem declaração, a janela é derivada do salto para garantir
    `MIN_STFT_OVERLAP`, arredondando para a próxima potência de 2 (FFT mais
    rápida). Assim, mexer em `time_steps` no futuro não reintroduz o ponto
    cego: a relação passa a ser mantida pelo código, não pela memória de quem
    edita.
    """
    if declared:
        return int(declared)
    hop_length = max(1, int(hop_length))
    minimo = hop_length / max(1e-6, 1.0 - MIN_STFT_OVERLAP)
    n_fft = 1 << max(9, int(np.ceil(np.log2(minimo))))  # piso de 512
    return int(n_fft)

#: Mapa `input_type` do benchmark → identificador de front-end.
#:
#: FONTE ÚNICA (2026-07-28). Antes, quem precisava dessa correspondência a
#: reimplementava: o `registry` declarava `feature_frontend` em apenas 2 das 12
#: arquiteturas, e `scripts/reporting/rebuild_inference_contracts.py` mantinha um
#: mapa manual por nome de modelo que cobria 9 e esquecia Sonic Sleuth,
#: EfficientNet-LSTM, Ensemble, WavLM e HuBERT.
#:
#: Sem `feature_frontend` no contrato, o `FeaturePreparer` NÃO roteia para este
#: módulo e a inferência cai no front-end próprio do app (log-magnitude-mel,
#: hop 128, sem z-score), que não reproduz o do treino — as métricas do artigo
#: deixam de valer para o modelo em produção.
#:
#: A correspondência é mecânica porque `benchmarks/data.py::
#: prepare_input_for_architecture` decide o preparo pelo mesmo `input_type`.
_FRONTEND_BY_INPUT_TYPE = {
    "raw_audio": FRONTEND_RAW,
    "spectrogram": FRONTEND_LOGMEL,
    "tabular": FRONTEND_TABULAR,
    "tabular_audio_features": FRONTEND_TABULAR,
    "tabular_flattened": FRONTEND_TABULAR,
}


def frontend_for_input_type(input_type: Optional[str]) -> Optional[str]:
    """Front-end do benchmark correspondente a um ``input_type``.

    Devolve ``None`` para tipos que o benchmark não prepara (nesse caso o
    contrato NÃO deve alegar paridade com o front-end do benchmark).
    """
    if not input_type:
        return None
    return _FRONTEND_BY_INPUT_TYPE.get(str(input_type).strip().lower())


def fit_length_tile(
    flat: np.ndarray,
    target_len: int,
    *,
    crop_strategy: str = "center",
    seed: Optional[int] = None,
) -> np.ndarray:
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
        available = flat.shape[1] - target_len
        strategy = str(crop_strategy).lower()
        if strategy == "center":
            start = max(0, available // 2)
            return flat[:, start : start + target_len]
        if strategy == "random":
            rng = np.random.default_rng(seed)
            starts = rng.integers(0, available + 1, size=len(flat))
            return np.stack(
                [row[start : start + target_len] for row, start in zip(flat, starts)]
            )
        raise ValueError("crop_strategy deve ser 'center' ou 'random'")
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


def raw_audio_batch(
    X: np.ndarray,
    target_len: int = DEFAULT_RAW_TARGET,
    *,
    crop_strategy: str = "center",
    seed: Optional[int] = None,
) -> np.ndarray:
    """Janela raw do benchmark: ``(N, ·)`` → ``(N, target_len, 1)`` z-scored."""
    flat = np.asarray(X, dtype="float32").reshape(len(X), -1)
    raw = fit_length_tile(
        flat,
        max(1, int(target_len)),
        crop_strategy=crop_strategy,
        seed=seed,
    )
    raw = normalize_per_sample(raw)
    return raw[..., np.newaxis]


def raw_audio_single(y: np.ndarray, target_len: int = DEFAULT_RAW_TARGET) -> np.ndarray:
    """Versão single-sample de :func:`raw_audio_batch` → ``(target_len, 1)``."""
    return raw_audio_batch(np.asarray(y, dtype="float32")[np.newaxis, :], target_len)[0]


def raw_audio_multicrop_batch(
    X: np.ndarray,
    target_len: int = DEFAULT_RAW_TARGET,
    *,
    num_crops: int = 3,
) -> np.ndarray:
    """Retorna (N, C, target_len, 1) com crops início/centro/fim."""
    flat = np.asarray(X, dtype="float32").reshape(len(X), -1)
    target_len = max(1, int(target_len))
    num_crops = max(1, int(num_crops))
    batches = []
    for row in flat:
        if len(row) <= target_len:
            crop = fit_length_tile(row[np.newaxis, :], target_len)[0]
            crops = np.repeat(crop[np.newaxis, :], num_crops, axis=0)
        else:
            maximum = len(row) - target_len
            starts = np.rint(np.linspace(0, maximum, num=num_crops)).astype(int)
            crops = np.stack([row[start : start + target_len] for start in starts])
        batches.append(normalize_per_sample(crops))
    return np.asarray(batches, dtype="float32")[..., np.newaxis]


def raw_audio_single_multicrop(
    y: np.ndarray,
    target_len: int = DEFAULT_RAW_TARGET,
    *,
    num_crops: int = 3,
) -> np.ndarray:
    """Versão single-sample multicrop: (C, target_len, 1)."""
    return raw_audio_multicrop_batch(
        np.asarray(y, dtype="float32")[np.newaxis, :],
        target_len,
        num_crops=num_crops,
    )[0]


def log_mel_batch(
    X: np.ndarray,
    sample_rate: int = DEFAULT_SAMPLE_RATE,
    feature_dim: int = DEFAULT_FEATURE_DIM,
    time_steps: int = DEFAULT_TIME_STEPS,
    n_fft: Optional[int] = None,
) -> np.ndarray:
    """Log-mel do benchmark: ``(N, T)`` raw → ``(N, time_steps, feature_dim)``.

    hop dinâmico ``ceil(T/time_steps)`` (≥64), potência, dB com ``ref=max`` POR
    AMOSTRA, ajuste do eixo temporal e z-score por amostra. A janela de análise
    (``n_fft``) é derivada do salto quando a arquitetura não a declara — ver
    `resolve_n_fft`.
    """
    import librosa

    flat = np.asarray(X, dtype="float32").reshape(len(X), -1)
    sample_rate = int(sample_rate or DEFAULT_SAMPLE_RATE)
    feature_dim = int(feature_dim or DEFAULT_FEATURE_DIM)
    time_steps = int(time_steps or DEFAULT_TIME_STEPS)
    hop_length = max(64, int(np.ceil(flat.shape[1] / max(time_steps, 1))))
    n_fft = resolve_n_fft(hop_length, n_fft)

    # AJUSTE 2026-07-31: era `specs = []` + `specs.append(...)` por amostra e
    # `np.asarray(specs, ...)` no final — para lotes grandes (o treino do
    # benchmark converte o split inteiro de uma vez, nao em chunks) a lista de
    # arrays soltos e a copia final materializada ficam vivas ao mesmo tempo,
    # dobrando o pico bem no fim da funcao. Confirmado no SpectrogramTransformer
    # (grade 300x128 = 38.400 floats/amostra, a maior entre as arquiteturas
    # spectrogram): o pico cruzava os 24 GB do container exatamente nesse
    # `np.asarray` final e o processo era morto pelo OOM-killer sem traceback
    # nenhum. Escrever direto num array pre-alocado elimina essa duplicacao.
    out = np.empty((len(flat), time_steps, feature_dim), dtype="float32")
    for i, y in enumerate(flat):
        mel = librosa.feature.melspectrogram(
            y=y.astype("float32"),
            sr=sample_rate,
            n_fft=n_fft,
            hop_length=hop_length,
            n_mels=feature_dim,
            power=2.0,
        )
        mel_db = librosa.power_to_db(mel + 1e-10, ref=np.max).T
        mel_db = _resize_time_axis(mel_db[np.newaxis, ...], max(1, time_steps))[0]
        out[i] = mel_db[:, :feature_dim]
    return normalize_per_sample(out)


def log_mel_single(
    y: np.ndarray,
    sample_rate: int = DEFAULT_SAMPLE_RATE,
    feature_dim: int = DEFAULT_FEATURE_DIM,
    time_steps: int = DEFAULT_TIME_STEPS,
    source_samples: int = DEFAULT_SOURCE_SAMPLES,
    n_fft: Optional[int] = None,
) -> np.ndarray:
    """Versão single-sample de :func:`log_mel_batch` → ``(time_steps, F)``.

    Primeiro ajusta o clipe à janela-fonte do benchmark (``source_samples``,
    3 s por padrão) para que o hop dinâmico seja o MESMO do treino.
    """
    flat = fit_length_tile(
        np.asarray(y, dtype="float32")[np.newaxis, :], int(source_samples)
    )
    return log_mel_batch(
        flat,
        sample_rate=sample_rate,
        feature_dim=feature_dim,
        time_steps=time_steps,
        n_fft=n_fft,
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
                y.astype("float32"),
                sr=16000,
                frame_length=512,
                hop_length=256,
                n_plp=n_plp,
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

    Ordem canônica (nomes em ``app/domain/xai/tabular.py``): 11 estatísticas
    temporais, 26 MFCC (média+desvio de 13), 26 RASTA-PLP (média+desvio de 13).
    """
    flat = np.asarray(X, dtype="float32").reshape(len(X), -1)
    feats = [
        flat.mean(axis=1),
        flat.std(axis=1),
        np.mean(np.abs(flat), axis=1),
        np.sqrt(np.mean(flat**2, axis=1)),
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
            mfcc_stats.append(np.concatenate([mfcc.mean(axis=1), mfcc.std(axis=1)]))
        feats.append(np.asarray(mfcc_stats, dtype="float32").T)
    except Exception:  # noqa: BLE001 - librosa opcional (idem benchmark)
        pass

    feats.append(_rasta_plp_stats(flat))
    return np.vstack(feats).T.astype("float32")


def tabular_features_single(
    y: np.ndarray, source_samples: int = DEFAULT_SOURCE_SAMPLES
) -> np.ndarray:
    """Versão single-sample de :func:`tabular_features_batch` → ``(63,)``.

    Ajusta o clipe à janela-fonte de 3 s antes da extração (as estatísticas
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
    raw_num_crops: int = 1,
    n_fft: Optional[int] = None,
) -> np.ndarray:
    """Roteia uma amostra pelo front-end do benchmark declarado no contrato.

    Args:
        y: áudio mono (T,) float32 já na taxa do contrato.
        feature_frontend: um de :data:`BENCHMARK_FRONTENDS`.
        add_channel_dim: para log-mel, acrescenta eixo de canal quando o
            modelo espera ``(T, F, 1)``; ``None`` mantém ``(T, F)``.
    """
    if feature_frontend == FRONTEND_RAW:
        if int(raw_num_crops) > 1:
            return raw_audio_single_multicrop(
                y,
                target_len=target_sequence_length,
                num_crops=raw_num_crops,
            )
        return raw_audio_single(y, target_len=target_sequence_length)
    if feature_frontend == FRONTEND_LOGMEL:
        spec = log_mel_single(
            y,
            sample_rate=sample_rate,
            feature_dim=feature_dim,
            time_steps=time_steps,
            source_samples=source_samples,
            # PARIDADE: sem repassar, o AST — treinado com a janela de 25 ms
            # que o artigo especifica (400 amostras) — inferia com outra.
            n_fft=n_fft,
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
