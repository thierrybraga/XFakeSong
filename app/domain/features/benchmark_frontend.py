"""Front-end de características do BENCHMARK — fonte única treino↔inferência.

Este módulo contém, verbatim, as transformações de entrada usadas pelo
benchmark (``benchmarks/data.py``) para treinar os modelos promovidos em
``data/models/benchmark_final``:

- **raw**: janela center-crop (ou repetição p/ clipes curtos) + z-score por
  amostra + canal — AASIST/RawGAT-ST usam a janela canonica e multicrop;
- **log-mel**: librosa ``melspectrogram`` (hop dinâmico p/ fixar ``time_steps``
  quadros; ``n_fft`` DERIVADO do hop por `resolve_n_fft` para garantir
  ``MIN_STFT_OVERLAP`` — com hop 480 dá 1024, não 512), ``power_to_db(ref=max)``
  POR AMOSTRA e z-score por amostra — mapa ``(time_steps, feature_dim)``
  = (100, 80) — consumida por Conformer/Res2Net/CCT. O AST declara o próprio
  ``n_fft`` (400 = 25 ms) e consome (300, 128);
- **tabular-63** (v1) e **tabular-183** (v2 = v1 + LFCC com Δ/ΔΔ) —
  consumidas por SVM/Random Forest (nomes canônicos em
  ``app/domain/xai/tabular.py``). O v2 é o que treina desde 2026-08-09; o v1
  continua resolvível para os artefatos anteriores.

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
from typing import Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# Identificadores gravados no input_contract dos modelos do benchmark.
FRONTEND_RAW = "benchmark_raw_v1"
FRONTEND_LOGMEL = "benchmark_logmel_v1"
FRONTEND_TABULAR = "benchmark_tabular_v1"
#: Vetor tabular de 183 descritores: o v1 inteiro + bloco LFCC com Δ/ΔΔ.
#:
#: MOTIVAÇÃO (2026-08-09). A 5 dB — o SNR NÃO VISTO — SVM e Random Forest
#: mantinham a AUC (0,849 e 0,838) e perdiam o ponto de operação: acurácia
#: 0,5000 e 0,6274, com recall 0,0000 e 0,2851 sob o limiar fixo de 0,5. A
#: separação continua lá; o vetor inteiro é que TRANSLADA sob ruído, porque 8
#: dos 11 descritores temporais do v1 (desvio, média |x|, RMS, mín, máx,
#: energia da diferença, ZCR e os percentis) crescem monotonicamente com a
#: potência do ruído. `mín`/`máx` são estatísticas de ORDEM sobre 48.000
#: amostras: a 5 dB medem o ruído, não a voz.
#:
#: O bloco novo é LFCC — o front-end do baseline CM do ASVspoof2019/2021
#: (Todisco et al.). A escala linear em frequência não comprime os agudos, que
#: é onde vocoder deixa artefato; a mel comprime. Δ e ΔΔ são diferenças ENTRE
#: quadros, logo invariantes a qualquer offset constante de canal ou nível.
#:
#: RETRATAÇÃO: a análise que motivou este bloco também sugeriu CMVN antes da
#: agregação. Está errado — o pooling do vetor é média⊕desvio POR
#: coeficiente, e CMVN zera exatamente essas duas estatísticas (média 0,
#: desvio 1 por construção). Aplicar CMVN aqui transformaria 40 colunas em
#: constantes. Δ/ΔΔ entrega a invariância pretendida sem esse efeito.
FRONTEND_TABULAR_V2 = "benchmark_tabular_v2"
BENCHMARK_FRONTENDS = (
    FRONTEND_RAW,
    FRONTEND_LOGMEL,
    FRONTEND_TABULAR,
    FRONTEND_TABULAR_V2,
)

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


#: Corte da CORREÇÃO DE BANDA, em Hz.
#:
#: MOTIVAÇÃO (2026-08-19). O corpus pareado junta duas fontes com taxas de
#: origem distintas: CETUC em 16 kHz nativo (bonafide) e fake_voices em 24 kHz
#: reamostrado para 16 kHz (spoof). A taxa de origem prediz a classe com 100%
#: de acurácia — 49.264 arquivos de cada lado, sem exceção — e o filtro
#: anti-aliasing do `soxr_hq` deixa assinatura na banda superior.
#:
#: MEDIDO: a energia relativa em 7,9–8,0 kHz é 4,3e-06 na classe bonafide e
#: 4,3e-10 na spoof, quatro ordens de grandeza. Uma regressão logística sobre
#: 40 energias de banda alcança AUC 1,0000 no corpus inteiro; com a banda
#: acima de 7,5 kHz removida, cai para 0,83. O atalho vive em 100 Hz de
#: espectro, e nenhum modelo precisava aprender nada sobre vocoders para
#: explorá-lo.
#:
#: A varredura do ponto de corte mostra o penhasco inteiramente entre 8,0 e
#: 7,5 kHz (1,0000 -> 0,8317) e estabilidade abaixo disso (7,0 kHz -> 0,8293),
#: o que situa 7.500 Hz como o menor corte que remove o artefato sem descartar
#: sinal útil.
#:
#: DESCARTADA: igualar por round-trip 16->24->16 na classe bonafide. Testado, e
#: PIORA — a bonafide sobe por interpolação e desce, a spoof nasceu em 24 kHz e
#: desceu uma vez; as assinaturas ficam diferentes e a AUC da banda isolada vai
#: de 0,9810 para 1,0000.
BAND_CORRECTION_HZ = 7500

#: Ordem do FIR da correção de banda. 255 dá transição de ~150 Hz a 16 kHz,
#: estreita o bastante para não invadir a banda de fala e larga o bastante
#: para evitar o toque (ringing) de um corte abrupto.
BAND_CORRECTION_TAPS = 255


#: Alvo de RMS da renormalização, em dBFS. Igual ao usado no build do corpus.
BAND_CORRECTION_RMS_DBFS = -26.0

#: Teto de pico da política de nível do corpus, em dBFS.
CORPUS_PEAK_CEILING_DBFS = -1.0


def normalize_corpus_level(
    y: np.ndarray,
    target_rms_dbfs: float = BAND_CORRECTION_RMS_DBFS,
    peak_ceiling_dbfs: float = CORPUS_PEAK_CEILING_DBFS,
) -> Tuple[np.ndarray, dict]:
    """Política de nível ÚNICA do projeto: RMS alvo com teto de pico.

    FONTE ÚNICA de propósito. O nível era definido em dois lugares que
    divergiam: o corpus normaliza a −26 dBFS
    (``build_paired_pt_corpus.TARGET_RMS_DB``) e a inferência normalizava a
    −23 LUFS (``app.utils.silero_vad.TARGET_LUFS``) — 3 dB, fator 1,41. Para
    raw e log-Mel isso é inconsequente (o z-score por amostra e o dB-ref-max
    são invariantes a reescala linear), mas o vetor tabular do SVM/RandomForest
    é calculado direto sobre a amplitude, sem normalização: toda coluna linear
    em amplitude (RMS, média |x|, mín/máx, percentis) saía 1,41× fora do que o
    modelo viu no treino.

    Devolve ``(audio, info)``; ``info`` traz o ganho aplicado para o manifesto.
    """
    audio = np.asarray(y, dtype="float32")
    rms = float(np.sqrt(np.mean(np.square(audio, dtype="float64"))))
    if rms <= 0:
        return audio, {"applied_gain_db": 0.0}
    source_rms_db = 20.0 * float(np.log10(rms))
    source_peak_db = 20.0 * float(np.log10(float(np.abs(audio).max()) + 1e-12))
    gain = 10.0 ** ((target_rms_dbfs - source_rms_db) / 20.0)
    ceiling = 10.0 ** (peak_ceiling_dbfs / 20.0)
    peak = float(np.abs(audio).max()) * gain
    limited = peak > ceiling
    if limited:
        gain *= ceiling / peak
    return (audio * gain).astype("float32"), {
        "source_rms_db": round(source_rms_db, 3),
        "source_peak_db": round(source_peak_db, 3),
        "applied_gain_db": round(20.0 * float(np.log10(gain)), 3),
        "peak_limited": limited,
    }


def apply_band_correction(
    X: np.ndarray,
    cutoff_hz: float = BAND_CORRECTION_HZ,
    sample_rate: int = DEFAULT_SAMPLE_RATE,
    chunk: int = 512,
    remove_dc: bool = True,
    renormalize_rms_dbfs: float | None = BAND_CORRECTION_RMS_DBFS,
) -> np.ndarray:
    """Passa-baixas de fase zero aplicado a TODAS as amostras, sem exceção.

    O ponto é a uniformidade: aplicar às duas classes remove a assinatura de
    reamostragem que distinguia uma delas. Aplicar a só uma reintroduziria o
    problema com outro sinal.

    Processa em blocos porque a partição de treino sozinha ocupa ~2,3 GB em
    float32, e `filtfilt` aloca cópias intermediárias.
    """
    from scipy.signal import filtfilt, firwin

    forma = X.shape
    plano = np.asarray(X, dtype="float32").reshape(len(X), -1)
    taps = firwin(
        BAND_CORRECTION_TAPS, cutoff_hz, fs=sample_rate, window="hamming"
    ).astype("float64")
    # `filtfilt` exige o sinal MAIOR que `padlen`, que por default e 3x a ordem
    # do filtro (765 aqui). As janelas do protocolo tem 48.000 amostras, mas a
    # funcao e chamada tambem por `extract_window` no build e por testes com
    # sinais curtos -- sem este ajuste ela levanta
    # "The length of the input vector x must be greater than padlen".
    # Adaptar o padlen preserva o comportamento nas janelas reais e mantem a
    # funcao correta em qualquer comprimento, em vez de pular em silencio.
    n_amostras = plano.shape[1]
    padlen = min(3 * len(taps), max(n_amostras - 1, 0))
    saida = np.empty_like(plano)
    for ini in range(0, len(plano), chunk):
        bloco = plano[ini : ini + chunk].astype("float64")
        saida[ini : ini + chunk] = filtfilt(
            taps, [1.0], bloco, axis=1, padlen=padlen
        ).astype("float32")
    # ── DC e nível ────────────────────────────────────────────────────────
    #
    # SEGUNDO ATALHO, medido em 2026-08-19: o offset DC separa as classes com
    # AUC 0,7387. Não é artefato de síntese -- conversores AD introduzem DC na
    # gravação, vocoders neurais saem com média ~zero. Aqui a bonafide tem
    # 2,04e-03 contra 1,38e-04 da spoof, razão de 14,7x.
    #
    # A ORDEM IMPORTA, e a ingênua quebra. Remover o DC sozinho eleva o atalho
    # de RMS de 0,5126 para 0,7503: a normalização do build foi feita sobre o
    # sinal COM DC, e como a bonafide tinha 14,7x mais, tirá-lo desiguala o RMS
    # que estava equalizado. Por isso a renormalização vem logo atrás, sobre o
    # sinal já sem DC -- o que devolve o RMS ao acaso (0,5106).
    #
    # O que sobrevive de propósito: fator de crista (0,697) e curtose (0,650).
    # Esses SÃO candidatos a artefato de síntese -- a compressão de faixa
    # dinâmica que a Seção de artefatos do TCC prevê. O ganho aplicado no build
    # confirma a assimetria de origem: a bonafide precisou de +3,3 dB com
    # desvio de 6,6 dB, a spoof de -8,7 dB com desvio de 0,8 dB.
    if remove_dc:
        saida -= saida.mean(axis=1, keepdims=True)
    if renormalize_rms_dbfs is not None:
        alvo = float(10.0 ** (renormalize_rms_dbfs / 20.0))
        rms = np.sqrt(np.mean(saida.astype("float64") ** 2, axis=1, keepdims=True))
        saida = (saida * (alvo / (rms + 1e-12))).astype("float32")

    logger.info(
        "[protocolo] correção aplicada a %d amostras: passa-baixas %.0f Hz "
        "(FIR %d taps, fase zero)%s%s",
        len(plano),
        cutoff_hz,
        BAND_CORRECTION_TAPS,
        ", DC removido" if remove_dc else "",
        f", RMS renormalizado a {renormalize_rms_dbfs:.0f} dBFS"
        if renormalize_rms_dbfs is not None else "",
    )
    return saida.reshape(forma)

#: Coeficientes LFCC por quadro (convenção do baseline CM do ASVspoof2019).
N_LFCC = 20
#: 20 estáticos + 20 Δ + 20 ΔΔ, cada bloco com média e desvio sobre os quadros.
N_LFCC_FEATURES = 6 * N_LFCC
#: Largura do vetor v2: o v1 inteiro (subconjunto, mesma ordem) + LFCC.
N_TABULAR_FEATURES_V2 = N_TABULAR_FEATURES + N_LFCC_FEATURES

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
#:
#: O ``input_type`` tabular resolve para o **v2** desde 2026-08-09: é o que os
#: clássicos passam a treinar. O v1 continua um front-end de primeira classe —
#: quem o resolve é o `feature_frontend` gravado no contrato do artefato, não
#: este mapa, então modelo antigo segue lendo o vetor com que foi treinado.
_FRONTEND_BY_INPUT_TYPE = {
    "raw_audio": FRONTEND_RAW,
    "spectrogram": FRONTEND_LOGMEL,
    "tabular": FRONTEND_TABULAR_V2,
    "tabular_audio_features": FRONTEND_TABULAR_V2,
    "tabular_flattened": FRONTEND_TABULAR_V2,
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
    """Média/desvio por coeficiente RASTA-PLP → ``(2·n_plp, N)``.

    Degrada para zeros na amostra que falha, mas NUNCA em silêncio — ver as
    duas guardas abaixo. O extrator vive no próprio repositório
    (``app/domain/features/extractors/cepstral/components/plp.py``), então
    falha de import é ambiente quebrado, não dependência opcional ausente: até
    2026-08-09 esse caminho devolvia 26 colunas de zeros sem um aviso.
    """
    from app.domain.features.extractors.cepstral.components.plp import (
        extract_rasta_plp_features,
    )

    rows = []
    degraded = 0
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
            degraded += 1
            rows.append(np.zeros(2 * n_plp, dtype="float32"))
    # DEGRADAÇÃO SILENCIOSA (corrigido em 2026-08-09): a queda para zeros era
    # por amostra e não deixava rastro nenhum — 26 dos 63 descritores podiam
    # ser constantes num lote inteiro sem uma linha de log. Falha total é erro
    # de ambiente (extrator quebrado), não caso de borda de uma amostra.
    if degraded == len(flat) and len(flat):
        raise RuntimeError(
            "RASTA-PLP falhou em TODAS as %d amostras do lote: 26 dos 63 "
            "descritores sairiam zerados. Verifique "
            "app.domain.features.extractors.cepstral." % len(flat)
        )
    if degraded:
        logger.warning(
            "[tabular] RASTA-PLP degradado a zeros em %d de %d amostras "
            "(%.2f%%) — 26 descritores constantes nessas linhas",
            degraded,
            len(flat),
            100.0 * degraded / max(1, len(flat)),
        )
    arr = np.nan_to_num(
        np.asarray(rows, dtype="float32"), nan=0.0, posinf=0.0, neginf=0.0
    )
    return arr.T


def _linear_filterbank(
    n_filters: int, n_fft: int, sample_rate: int
) -> np.ndarray:
    """Banco de filtros triangulares igualmente espaçados em Hz → ``(n_filters, 1+n_fft//2)``.

    O análogo linear de ``librosa.filters.mel``: mesmos triângulos sobrepostos
    a meia altura, só que os vértices caem numa grade LINEAR de frequência. É
    o que separa LFCC de MFCC.
    """
    n_bins = 1 + n_fft // 2
    fft_freqs = np.linspace(0.0, sample_rate / 2.0, n_bins, dtype="float64")
    # n_filters+2 vértices: cada filtro usa (anterior, centro, próximo).
    edges = np.linspace(0.0, sample_rate / 2.0, n_filters + 2, dtype="float64")
    fb = np.zeros((n_filters, n_bins), dtype="float64")
    for i in range(n_filters):
        left, center, right = edges[i], edges[i + 1], edges[i + 2]
        rising = (fft_freqs - left) / max(center - left, 1e-12)
        falling = (right - fft_freqs) / max(right - center, 1e-12)
        fb[i] = np.maximum(0.0, np.minimum(rising, falling))
    # Normalização de área (Slaney), idêntica à do banco mel do librosa: sem
    # ela os filtros largos dominariam só por integrarem mais bins.
    widths = edges[2 : n_filters + 2] - edges[:n_filters]
    fb *= (2.0 / np.maximum(widths, 1e-12))[:, np.newaxis]
    return fb.astype("float32")


def _lfcc_stats(
    flat: np.ndarray,
    n_lfcc: int = N_LFCC,
    n_fft: int = 512,
    hop_length: int = 256,
    sample_rate: int = DEFAULT_SAMPLE_RATE,
) -> np.ndarray:
    """Estatísticas LFCC do vetor v2: ``(N, T)`` → ``(6·n_lfcc, N)``.

    Ordem por amostra: média e desvio dos coeficientes ESTÁTICOS, depois de Δ,
    depois de ΔΔ. Transposto (features nas linhas) para casar com o empilhamento
    de :func:`tabular_features_batch`.
    """
    import librosa
    from scipy.fftpack import dct

    fb = _linear_filterbank(n_lfcc, n_fft, sample_rate)
    rows = []
    for y in flat:
        spec = (
            np.abs(
                librosa.stft(
                    y.astype("float32"),
                    n_fft=n_fft,
                    hop_length=hop_length,
                    window="hann",
                    center=True,
                )
            )
            ** 2
        )
        # log da energia por banda; o piso evita -inf em banda muda.
        log_fb = np.log(fb @ spec + 1e-10)
        static = dct(log_fb, type=2, axis=0, norm="ortho")[:n_lfcc]
        # width=9 é o default do librosa e exige ≥9 quadros; com 3 s e hop 256
        # são 188, mas um clipe curto num consumidor externo cairia aqui.
        width = min(9, static.shape[1] if static.shape[1] % 2 else static.shape[1] - 1)
        if width >= 3:
            delta = librosa.feature.delta(static, width=width, order=1)
            delta2 = librosa.feature.delta(static, width=width, order=2)
        else:
            delta = np.zeros_like(static)
            delta2 = np.zeros_like(static)
        rows.append(
            np.concatenate(
                [
                    static.mean(axis=1),
                    static.std(axis=1),
                    delta.mean(axis=1),
                    delta.std(axis=1),
                    delta2.mean(axis=1),
                    delta2.std(axis=1),
                ]
            )
        )
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
    # DEGRADAÇÃO SILENCIOSA (corrigido em 2026-08-09): este bloco ficava sob
    # `except Exception: pass`. Sem librosa o vetor caía de 63 para 37 colunas
    # sem um único aviso, e `N_TABULAR_FEATURES` — declarado desde sempre —
    # não era referenciado em lugar nenhum do projeto. Um modelo treinado
    # assim declararia 37 no contrato e passaria por válido.
    import librosa

    mfcc_stats = []
    for y in flat:
        mfcc = librosa.feature.mfcc(y=y, sr=16000, n_mfcc=13)
        mfcc_stats.append(np.concatenate([mfcc.mean(axis=1), mfcc.std(axis=1)]))
    feats.append(np.asarray(mfcc_stats, dtype="float32").T)

    feats.append(_rasta_plp_stats(flat))
    out = np.vstack(feats).T.astype("float32")
    if out.shape[1] != N_TABULAR_FEATURES:
        raise RuntimeError(
            f"vetor tabular v1 com {out.shape[1]} colunas, esperado "
            f"{N_TABULAR_FEATURES} — front-end e contrato divergiriam"
        )
    return out


def tabular_features_v2_batch(X: np.ndarray) -> np.ndarray:
    """Vetor tabular v2 de 183 descritores: ``(N, ·)`` → ``(N, 183)``.

    Superset estrito do v1, na mesma ordem, seguido do bloco LFCC (20
    coeficientes estáticos, Δ e ΔΔ, cada um com média e desvio sobre os
    quadros). Ser superset é deliberado: qualquer diferença de desempenho
    contra o v1 é atribuível ao bloco novo ou à dimensionalidade, nunca à
    remoção de um descritor.
    """
    flat = np.asarray(X, dtype="float32").reshape(len(X), -1)
    base = tabular_features_batch(flat)
    lfcc = _lfcc_stats(flat).T
    out = np.concatenate([base, lfcc], axis=1).astype("float32")
    if out.shape[1] != N_TABULAR_FEATURES_V2:
        raise RuntimeError(
            f"vetor tabular v2 com {out.shape[1]} colunas, esperado "
            f"{N_TABULAR_FEATURES_V2} — front-end e contrato divergiriam"
        )
    return out


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


def tabular_features_v2_single(
    y: np.ndarray, source_samples: int = DEFAULT_SOURCE_SAMPLES
) -> np.ndarray:
    """Versão single-sample de :func:`tabular_features_v2_batch` → ``(183,)``."""
    flat = fit_length_tile(
        np.asarray(y, dtype="float32")[np.newaxis, :], int(source_samples)
    )
    return tabular_features_v2_batch(flat)[0]


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
    if feature_frontend == FRONTEND_TABULAR_V2:
        return tabular_features_v2_single(y, source_samples=source_samples)
    raise ValueError(
        f"feature_frontend desconhecido: {feature_frontend!r} "
        f"(esperado um de {BENCHMARK_FRONTENDS})"
    )
