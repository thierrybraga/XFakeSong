"""Configuração do benchmark."""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

# Manifesto oficial do recorte experimental. Ele fixa nomes, variantes e
# runners usados no TCC; listas derivadas abaixo devem ser consumidas pelos
# scripts para evitar divergência entre treino, consolidação e LaTeX.
# Os rótulos `variant` são a PROVENIÊNCIA que vai para os resultados (o runner
# os copia para `architectures[<nome>].provenance`). Precisam descrever a
# configuração REALMENTE treinada — em 2026-07-27 vários estavam defasados após
# mudanças de arquitetura: "ast_vit_base_scratch" quando o AST passara a partir
# de pesos AudioSet, "rawgat_st_multiply_stride4" com um stride que só vale nas
# variantes legadas, e um "rawnet2_paper_like" que não dizia QUAL RawNet2
# (verificação de locutor vs. baseline anti-spoofing — arquiteturas diferentes
# com o mesmo nome). Ao alterar uma arquitetura, atualize o rótulo junto.
OFFICIAL_TCC_MODEL_MANIFEST: List[Dict[str, Any]] = [
    {
        "benchmark_name": "RandomForest",
        "result_key": "RandomForest",
        "display_name": "Random Forest",
        "variant": "sklearn_random_forest_gridsearch",
        "runner": "benchmarks.runner:classical",
        "input_type": "tabular_features",
        "family": "classical-tabular",
        "scope": "official",
    },
    {
        "benchmark_name": "SVM",
        "result_key": "SVM",
        "display_name": "SVM",
        "variant": "sklearn_svc_gridsearch_linear_rbf",
        "runner": "benchmarks.runner:classical",
        "input_type": "tabular_features",
        "family": "classical-tabular",
        "scope": "official",
    },
    {
        "benchmark_name": "Hybrid CNN-Transformer",
        "result_key": "CCT",
        "display_name": "CCT",
        "variant": "cct_hassani2021_conv64_128_4layers_4heads_256d_no_se",
        "runner": "benchmarks.runner:keras",
        "input_type": "spectrogram",
        "family": "spectral-attention",
        "scope": "official",
    },
    {
        "benchmark_name": "SpectrogramTransformer",
        "result_key": "AST",
        "display_name": "AST",
        "variant": "ast_gong2021_vit_base_audioset_pretrained_in300x128",
        "runner": "benchmarks.runner:keras",
        "input_type": "spectrogram",
        "family": "spectral-attention",
        "scope": "official",
    },
    {
        "benchmark_name": "MultiscaleCNN",
        "result_key": "Res2Net",
        "display_name": "Res2Net",
        "variant": "res2net50_gao2021_scale4_basewidth26_no_se",
        "runner": "benchmarks.runner:keras",
        "input_type": "spectrogram",
        "family": "spectral-convolutional",
        "scope": "official",
    },
    {
        "benchmark_name": "Conformer",
        "result_key": "Conformer",
        "display_name": "Conformer",
        "variant": "conformer_m_gulati2020_16blocks_256d_4heads_kernel31",
        "runner": "benchmarks.runner:keras",
        "input_type": "spectrogram",
        "family": "spectral-attention",
        "scope": "official",
    },
    {
        "benchmark_name": "RawNet2",
        "result_key": "RawNet2",
        "display_name": "RawNet2",
        "variant": "rawnet2_antispoofing_tak2021_sinc20_3xgru1024",
        "runner": "benchmarks.runner:keras",
        "input_type": "raw_audio",
        "family": "waveform-end-to-end",
        "scope": "official",
    },
    {
        "benchmark_name": "AASIST",
        "result_key": "AASIST",
        "display_name": "AASIST",
        "variant": "aasist_jung2022_sinc_fixo_gat_temp2_hsgal_temp100",
        "runner": "benchmarks.runner:keras",
        "input_type": "raw_audio",
        "family": "waveform-end-to-end",
        "scope": "official",
    },
    {
        "benchmark_name": "RawGAT-ST",
        "result_key": "RawGAT-ST",
        "display_name": "RawGAT-ST",
        "variant": "rawgat_st_tak2021_sinc_fixo_temp2_2_100_pool05_07_05_topk12",
        "runner": "benchmarks.runner:keras",
        "input_type": "raw_audio",
        "family": "waveform-end-to-end",
        "scope": "official",
    },
    {
        "benchmark_name": "WavLM Original",
        "result_key": "WavLM Original",
        "display_name": "WavLM Original",
        # base-PLUS, nao base: o runner passou a `microsoft/wavlm-base-plus` em
        # 2026-07-15 (mesma arquitetura, pre-treino de 94k h em vez de 960 h) e
        # este rotulo ficou para tras, declarando nos resultados um checkpoint
        # diferente do que foi realmente treinado — exatamente o que o
        # comentario no topo deste arquivo proibe.
        "variant": "microsoft/wavlm-base-plus:pytorch_runner_frozen_backbone",
        "runner": "scripts.benchmark.run_wavlm_original_benchmark:ssl_original",
        "input_type": "raw_audio_16khz_16000",
        "family": "ssl-pretrained",
        "scope": "official",
    },
    {
        "benchmark_name": "HuBERT Original",
        "result_key": "HuBERT Original",
        "display_name": "HuBERT Original",
        "variant": "facebook/hubert-base-ls960:pytorch_runner_frozen_backbone",
        "runner": "scripts.benchmark.run_wavlm_original_benchmark:ssl_original",
        "input_type": "raw_audio_16khz_16000",
        "family": "ssl-pretrained",
        "scope": "official",
    },
    # ── Por que NÃO há entradas com fine-tuning (revisto em 2026-08-11) ────
    #
    # Existiram "WavLM AASIST" e "HuBERT AASIST" por dois dias, sob a premissa
    # de que destravar o front-end era "a receita de campeonato" e que o
    # probing congelado não respondia à pergunta certa. A premissa estava
    # DESATUALIZADA:
    #
    #   - ASVspoof 5 (2024): os baselines oficiais da Track 1 são RawNet2 e
    #     AASIST, sem front-end SSL; e os sistemas de TOPO usam WavLM,
    #     wav2vec 2.0, HuBERT e afins como upstreams CONGELADOS;
    #   - há resultado publicado de front-end congelado batendo o treinável
    #     com folga na mesma comparação (8,76% contra 21,67% de EER);
    #   - a evidência pró-fine-tuning (Wang & Yamagishi, Odyssey 2022) é de
    #     2022 e o campo se moveu na direção oposta.
    #
    # Além disso, o resultado de referência daquela receita (Tak et al.,
    # Odyssey 2022 — 0,82% de EER no ASVspoof21 LA) usa wav2vec 2.0 XLS-R
    # (~300M, 24 camadas), não WavLM/HuBERT base (94,5M, 12 camadas). Combinar
    # esses backbones com o grafo AASIST seria uma abordagem NOVA, não a
    # reprodução de uma configuração documentada — e o objetivo aqui é
    # benchmark.
    #
    # As entradas `Original` acima JÁ SÃO a configuração documentada: backbone
    # congelado, soma ponderada de camadas, pooling e cabeça treinada.
    #
    # O código do grafo (`app/domain/models/architectures/torch_ssl_aasist.py`)
    # e as flags `--backend aasist`/`--no-freeze-backbone` do runner SSL
    # permanecem no projeto, testados, como ABLAÇÃO disponível fora do escopo
    # oficial. Se algum dia o eixo a explorar for o back-end sobre o backbone
    # congelado — que é o que a literatura recente estuda —, a peça está lá.
]

EXTENDED_MODEL_MANIFEST: List[Dict[str, Any]] = [
    {
        "benchmark_name": "Sonic Sleuth",
        "display_name": "Sonic Sleuth",
        "runner": "benchmarks.runner:keras",
        "input_type": "spectrogram",
        "family": "extended",
        "scope": "extended",
    },
    {
        "benchmark_name": "EfficientNet-LSTM",
        "display_name": "EfficientNet-LSTM",
        "runner": "benchmarks.runner:keras",
        "input_type": "spectrogram",
        "family": "extended",
        "scope": "extended",
    },
    {
        "benchmark_name": "Ensemble",
        "display_name": "Ensemble",
        "runner": "benchmarks.runner:keras",
        "input_type": "multi_representation",
        "family": "extended",
        "scope": "extended",
    },
    # WavLM/HuBERT no caminho **Keras** — distintos dos "* Original" do escopo
    # oficial, que rodam pelo runner PyTorch. Aqui o state_dict do checkpoint
    # HuggingFace é portado para Keras (`architectures/ssl_backbone.py`), o
    # backbone fica INTEIRAMENTE congelado e treinam apenas a soma ponderada de
    # hidden-states (receita SUPERB) e a cabeça.
    #
    # Até 2026-07-27 eles não constavam de manifesto algum e o benchmark gravava
    # `provenance: null` justamente nos dois modelos em que a proveniência (qual
    # checkpoint) É a definição do experimento.
    #
    # `fallback_variant` existe porque este caminho DEGRADA para um CNN-1D
    # treinado do zero quando o checkpoint não está acessível. O runner publica
    # esse rótulo quando a degradação ocorre, para que nenhum artefato alegue
    # backbone pré-treinado onde não houve (ver `runner._architecture_provenance`
    # e `ssl_utils.record_ssl_backbone_status`).
    {
        "benchmark_name": "WavLM",
        "display_name": "WavLM (port Keras)",
        "variant": (
            "microsoft/wavlm-base:keras_port_frozen_backbone_superb_weighted_sum"
        ),
        "fallback_variant": "wavlm_fallback_cnn1d_scratch_nao_e_o_ssl_real",
        "runner": "benchmarks.runner:keras",
        "input_type": "raw_audio_16khz_16000",
        # `family` acompanha a derivação de MODEL_FAMILIES (que só monta as cinco
        # famílias a partir do manifesto OFICIAL): estes pertencem à lista
        # "extended". A natureza SSL está declarada no `variant`.
        "family": "extended",
        "scope": "extended",
    },
    {
        "benchmark_name": "HuBERT",
        "display_name": "HuBERT (port Keras)",
        "variant": (
            "facebook/hubert-base-ls960:"
            "keras_port_frozen_backbone_superb_weighted_sum"
        ),
        "fallback_variant": "hubert_fallback_cnn1d_scratch_nao_e_o_ssl_real",
        "runner": "benchmarks.runner:keras",
        "input_type": "raw_audio_16khz_16000",
        "family": "extended",
        "scope": "extended",
    },
]

MODEL_FAMILIES: Dict[str, List[str]] = {
    family: [
        item["benchmark_name"]
        for item in OFFICIAL_TCC_MODEL_MANIFEST
        if item["family"] == family
    ]
    for family in (
        "classical-tabular",
        "spectral-convolutional",
        "spectral-attention",
        "waveform-end-to-end",
        "ssl-pretrained",
    )
}
MODEL_FAMILIES["extended"] = [
    item["benchmark_name"] for item in EXTENDED_MODEL_MANIFEST
]

OFFICIAL_TCC_RESULT_ORDER = [item["result_key"] for item in OFFICIAL_TCC_MODEL_MANIFEST]

OFFICIAL_TCC_DISPLAY_NAMES = {
    item["result_key"]: item["display_name"] for item in OFFICIAL_TCC_MODEL_MANIFEST
}

#: Prefixo do runner SSL dedicado. A checagem é pelo MÓDULO, não pelo sufixo:
#: com `endswith(":ssl_original")` as variantes `:ssl_finetuned` caíam em
#: `ALL_TCC_ARCHITECTURES` — a lista dos modelos que `benchmarks.runner` sabe
#: treinar — e o orquestrador tentaria rodá-las pelo caminho Keras.
_SSL_RUNNER_MODULE = "scripts.benchmark.run_wavlm_original_benchmark"


def _is_ssl_runner(item: Dict[str, Any]) -> bool:
    return str(item.get("runner", "")).startswith(_SSL_RUNNER_MODULE)


# Modelos suportados diretamente por benchmarks.runner/run_benchmark.py.
ALL_TCC_ARCHITECTURES = [
    item["benchmark_name"]
    for item in OFFICIAL_TCC_MODEL_MANIFEST
    if not _is_ssl_runner(item)
]

# WavLM/HuBERT reais são treinados no mesmo fluxo WSL/Docker, mas por um runner
# SSL PyTorch/Hugging Face dedicado. Duas receitas convivem ali: `ssl_original`
# congela o backbone e treina só a cabeça sobre embeddings em cache;
# `ssl_finetuned` destrava o backbone e liga a sequência a um grafo AASIST.
SSL_DOCKER_ARCHITECTURES = [
    item["benchmark_name"]
    for item in OFFICIAL_TCC_MODEL_MANIFEST
    if _is_ssl_runner(item)
]

#: Só as variantes com fine-tuning — usadas pelo orquestrador para escolher as
#: flags (`--backend aasist --no-freeze-backbone`) e o timeout, muito maior.
#:
#: **VAZIA desde 2026-08-11**, e de propósito: o escopo oficial só tem SSL
#: congelado (ver a justificativa no fim de `OFFICIAL_TCC_MODEL_MANIFEST`). A
#: derivação continua aqui, e não como lista literal, para que reintroduzir uma
#: entrada com `runner: ...:ssl_finetuned` volte a acionar as flags certas sem
#: nenhuma outra edição.
SSL_FINETUNED_ARCHITECTURES = [
    item["benchmark_name"]
    for item in OFFICIAL_TCC_MODEL_MANIFEST
    if str(item.get("runner", "")).endswith(":ssl_finetuned")
]

DOCKER_TRAINING_ARCHITECTURES = [
    item["benchmark_name"] for item in OFFICIAL_TCC_MODEL_MANIFEST
]

CLASSICAL_TCC_ARCHITECTURES = ["RandomForest", "SVM"]

NEURAL_TCC_ARCHITECTURES = [
    arch for arch in ALL_TCC_ARCHITECTURES if arch not in CLASSICAL_TCC_ARCHITECTURES
]

NEURAL_DOCKER_ARCHITECTURES = [
    arch
    for arch in DOCKER_TRAINING_ARCHITECTURES
    if arch not in CLASSICAL_TCC_ARCHITECTURES
]


@dataclass
class BenchmarkConfig:
    """Parâmetros de uma execução de benchmark.

    Attributes:
        architectures: nomes (display) das arquiteturas do registry a avaliar.
        dataset_path: caminho .npz com X_train/y_train (e opcional X_test/...).
            Se None, usa um dataset sintético separável (modo de verificação).
        epochs: épocas de treino por arquitetura.
        batch_size: tamanho de batch.
        seed: semente para splits e geração sintética (reprodutibilidade).
        snr_levels_db: níveis de SNR (dB) do teste de robustez AWGN.
        latency_runs: nº de inferências para medir a latência (mediana).
        output_dir: pasta de saída dos artefatos (JSON/CSV/LaTeX/figuras).
        models_dir: pasta padrão para modelos treinados pelo benchmark. Por
            padrão usa o mesmo diretório carregado pela Gradio/API.
        run_api_probe: se True, roda o teste de sistema da API (TestClient).
        synthetic_n / synthetic_shape: tamanho/forma do dataset sintético.
        converge_auc_threshold: AUC mínimo (no limpo) para marcar "convergiu".
        converge_accuracy_threshold: acurácia mínima no threshold de decisão.
    """

    architectures: List[str] = field(default_factory=lambda: ["MultiscaleCNN", "SVM"])
    dataset_path: Optional[str] = None
    epochs: int = 100
    batch_size: int = 32
    seed: int = 42
    # 30/20/10 dB coincidem com `train_aug_snr_db` e medem robustez em CONDIÇÃO
    # CASADA. 5 dB é deliberadamente NÃO VISTO no augmentation: sem ao menos um
    # nível fora do treino, a tabela de robustez não distingue "aprendeu a lidar
    # com ruído" de "decorou os níveis que viu". Custa só avaliação — nenhum
    # treino extra. Mantenha 5 dB FORA de `train_aug_snr_db` ao ajustar.
    snr_levels_db: List[int] = field(default_factory=lambda: [30, 20, 10, 5])
    latency_runs: int = 30
    output_dir: str = "data/results/benchmark"
    models_dir: str = "data/models"
    run_api_probe: bool = False
    synthetic_n: int = 360
    synthetic_shape: tuple = (32, 16)
    converge_auc_threshold: float = 0.60
    converge_accuracy_threshold: float = 0.55
    preset_name: str = "custom"
    device_profile: str = "auto"
    optimize_hyperparameters: bool = True
    training_overrides: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    # P0 — split por fonte/gerador (anti-vazamento). Quando True e o dataset
    # carrega `groups` (fonte por amostra), o split mantém os grupos DISJUNTOS
    # entre train/val/test (StratifiedGroupKFold), evitando que o mesmo
    # falante/gerador apareça em treino e teste (números inflados).
    group_split: bool = False
    # P0.4 — protocolo cross-generator: nome do gerador (ex.: "fkvoice") a
    # segurar fora do treino; o teste passa a conter SÓ esse gerador (+ reais
    # disjuntos). Mede generalização a gerador inédito. None desativa.
    holdout_generator: Optional[str] = None

    # Tier `large` — split DISJUNTO POR FALANTE (usuários não vistos). Quando True
    # e o dataset carrega `speaker_ids`, mantém cada falante inteiramente em um
    # único conjunto (StratifiedGroupKFold por falante). Espelha `group_split`.
    speaker_split: bool = False
    # Protocolo holdout-speaker: segura este falante fora do treino e o usa (só
    # ele + reais reservados) como teste — generalização a usuário inédito.
    # Espelha `holdout_generator`. None desativa.
    holdout_speaker: Optional[str] = None
    # Tier informativo do dataset (test/small/medium/large/custom), só para o plano.
    tier: Optional[str] = None

    # Protocolo AWGN canônico: o ruído é aplicado à forma de onda antes de
    # qualquer frontend. Uma cópia ruidosa por amostra mantém o custo de memória
    # em ~2x e distribui os níveis de SNR de forma balanceada e reprodutível.
    waveform_noise_augmentation: bool = True
    # False usa a mesma copia AWGN estatica para todas as arquiteturas. True
    # habilita otimizacoes por arquitetura e deve ser reportado como ablacao,
    # nunca misturado a tabela comparativa principal.
    architecture_specific_augmentation: bool = False

    train_aug_snr_db: List[int] = field(default_factory=lambda: [30, 20, 10])
    train_noise_copies: int = 1
    waveform_noise_batch_size: int = 64
    # Em datasets reais, recusa NPZs que já contenham apenas features quando há
    # avaliação AWGN. Datasets sintéticos internos continuam aceitos para smoke.
    strict_waveform_awgn: bool = True
    # Compatibilidade com execuções antigas. Mantido desligado: ativá-lo volta
    # a adicionar ruído no vetor tabular e invalida o protocolo comparável.
    classical_noise_augmentation: bool = False

    # Controles metodológicos comuns. Os hiperparâmetros arquiteturais continuam
    # específicos, mas todos os modelos neurais usam o mesmo orçamento e a mesma
    # regra de seleção: 100 épocas completas e melhor checkpoint em val limpa.
    fixed_epoch_budget: bool = True
    select_best_checkpoint: bool = True
    #: Métrica de seleção do checkpoint: ``val_loss`` (padrão, o que produziu os
    #: artefatos publicados) ou ``val_eer``.
    #:
    #: Custo medido da escolha no run `clean_benchmark_15k`, comparando a época
    #: de menor `val_loss` com a de maior `val_accuracy`: 0,00 p.p. em cinco das
    #: nove neurais, ≤0,62 p.p. em três, e **5,29 p.p. no RawGAT-ST** (época 17
    #: contra 88). É por isso que o campo existe e o padrão não muda: só uma
    #: arquitetura paga o descompasso, e trocar o padrão invalidaria as outras
    #: dez sem ganho.
    checkpoint_monitor: str = "val_loss"
    #: Corte do passa-baixas de CORREÇÃO DE BANDA, em Hz; ``None`` desliga.
    #:
    #: O corpus pareado tem taxa de ORIGEM perfeitamente correlacionada com a
    #: classe (CETUC 16 kHz nativo x fake_voices 24 kHz reamostrado), e o
    #: filtro anti-aliasing deixa assinatura acima de 7,5 kHz que separa as
    #: classes com AUC 0,98 sozinha. Aplicar o mesmo passa-baixas às DUAS
    #: classes remove o atalho. Ver
    #: `benchmark_frontend.apply_band_correction`.
    #:
    #: Default ``None`` preserva a reprodutibilidade dos artefatos anteriores;
    #: runs corrigidos passam 7500.
    band_correction_hz: float | None = None
    decision_threshold: float = 0.5
    metric_threshold_policy: str = "fixed_0.5_comparison"
    experiment_scope: str = "official"
    # Rigor acadêmico (2026-07-14): IC 95% de bootstrap (EER/AUC/accuracy)
    # nas métricas limpas e de robustez. 1000 reamostragens ≈ segundos por
    # condição; 0 desliga (testes/smokes).
    bootstrap_ci_samples: int = 1000

    # Rigor acadêmico (2026-07-27): REPETIÇÕES com sementes de treino distintas.
    #
    # O bootstrap acima mede a variância de AMOSTRAGEM DO TESTE; ele não diz
    # nada sobre a variância de TREINO (inicialização, dropout, ordem de batch,
    # realização do ruído de augmentation). Com uma execução por modelo, uma
    # diferença de 1–2 pp de EER entre duas arquiteturas pode ser apenas ruído
    # de execução — e o artigo não teria como distinguir.
    #
    # `n_seeds > 1` roda cada arquitetura N vezes e reporta média ± desvio.
    # IMPORTANTE: apenas a semente de TREINO varia. O split (teste selado) e a
    # realização do ruído de AVALIAÇÃO permanecem presos a `seed`, para que
    # todas as repetições sejam medidas exatamente no mesmo conjunto e nas
    # mesmas condições de ruído.
    n_seeds: int = 1

    @property
    def training_seeds(self) -> List[int]:
        """Sementes de TREINO das repetições (a de dados continua sendo `seed`)."""
        return [int(self.seed) + i for i in range(max(1, int(self.n_seeds)))]

    # Robustez a CODEC com perdas (round-trip via ffmpeg, na forma de onda,
    # antes dos frontends — mesmo ponto do AWGN). Ex.: ["mp3", "opus"].
    # Desligado por padrão (custo: ~2 chamadas ffmpeg por amostra de teste).
    # Atalho de dominio: a classe majoritaria por fonte nao pode superar este
    # limite em execucoes academicas.
    source_oracle_threshold: float = 0.55
    fail_on_source_shortcut: bool = False
    codec_eval: List[str] = field(default_factory=list)
    # Selo do teste validado (benchmarks/test_lock.py). Preenchido pelo CLI
    # quando `--test-lock` é passado; vai para os resultados como prova de que o
    # teste conferido era o mesmo congelado antes do treino.
    test_lock: Optional[Dict[str, Any]] = None
    preserve_predefined_splits: bool = True
    fail_on_split_overlap: bool = True

    @classmethod
    def quick(cls, **overrides) -> "BenchmarkConfig":
        """Preset rápido (sintético, 1 época) — para verificação do harness."""
        base = dict(
            architectures=["SVM"],
            dataset_path=None,
            epochs=1,
            snr_levels_db=[20],
            latency_runs=5,
            synthetic_n=200,
            synthetic_shape=(8, 8),
            preset_name="quick",
            optimize_hyperparameters=False,
        )
        base.update(overrides)
        return cls(**base)

    @classmethod
    def full_tcc(cls, **overrides) -> "BenchmarkConfig":
        """Preset TCC para o runner direto: nove modelos não SSL."""
        base = dict(
            architectures=list(ALL_TCC_ARCHITECTURES),
            epochs=100,
            snr_levels_db=[30, 20, 10, 5],
            run_api_probe=True,
            preset_name="full_tcc",
            optimize_hyperparameters=True,
        )
        base.update(overrides)
        return cls(**base)

    @classmethod
    def full_all_architectures(cls, **overrides) -> "BenchmarkConfig":
        """Alias explícito para o preset completo do artigo."""
        return cls.full_tcc(**overrides)

    @classmethod
    def cross_generator_tcc(
        cls, holdout_generator: str = "fkvoice", **overrides
    ) -> "BenchmarkConfig":
        """Preset P0.4 — reteste cross-generator (anti-vazamento de fonte).

        Treina SEM o gerador `holdout_generator` (default XTTS=fkvoice) e o usa
        como teste. Mede generalização a gerador inédito — o reteste
        metodológico mais importante antes da defesa.
        """
        base = dict(
            architectures=list(ALL_TCC_ARCHITECTURES),
            epochs=100,
            snr_levels_db=[30, 20, 10, 5],
            run_api_probe=False,
            preset_name=f"cross_generator:{holdout_generator}",
            optimize_hyperparameters=True,
            holdout_generator=holdout_generator,
        )
        base.update(overrides)
        return cls(**base)

    @classmethod
    def unseen_speaker_tcc(
        cls, holdout_speaker: Optional[str] = None, **overrides
    ) -> "BenchmarkConfig":
        """Preset tier `large` — protocolo de USUÁRIO NÃO VISTO (unseen speaker).

        Com `holdout_speaker`, segura um falante fora do treino e testa nele
        (mais reais reservados, p/ manter ambas as classes no teste). Sem ele,
        usa split disjunto por falante (`speaker_split=True`). Requer um `.npz`
        com `speaker_ids` (exportado a partir de um dataset tier `large`).
        """
        base = dict(
            architectures=list(ALL_TCC_ARCHITECTURES),
            epochs=100,
            snr_levels_db=[30, 20, 10, 5],
            run_api_probe=False,
            preset_name=(
                f"unseen_speaker:{holdout_speaker}"
                if holdout_speaker
                else "unseen_speaker"
            ),
            optimize_hyperparameters=True,
            tier="large",
            speaker_split=holdout_speaker is None,
            holdout_speaker=holdout_speaker,
        )
        base.update(overrides)
        return cls(**base)

    @classmethod
    def group_tcc(cls, **overrides) -> "BenchmarkConfig":
        """Preset P0 — split disjunto por fonte/gerador (StratifiedGroupKFold).

        Atenção: se as fontes forem fortemente correlacionadas à classe (poucos
        grupos), o teste pode ficar de classe única — nesse caso prefira
        `cross_generator_tcc`.
        """
        base = dict(
            architectures=list(ALL_TCC_ARCHITECTURES),
            epochs=100,
            snr_levels_db=[30, 20, 10, 5],
            run_api_probe=False,
            preset_name="group_tcc",
            optimize_hyperparameters=True,
            group_split=True,
        )
        base.update(overrides)
        return cls(**base)

    @classmethod
    def neural_tcc(cls, **overrides) -> "BenchmarkConfig":
        """Preset neural do TCC: sete arquiteturas neurais do artigo, sem SVM/RF."""
        base = dict(
            architectures=list(NEURAL_TCC_ARCHITECTURES),
            epochs=100,
            snr_levels_db=[30, 20, 10, 5],
            run_api_probe=False,
            preset_name="neural_tcc",
            optimize_hyperparameters=True,
        )
        base.update(overrides)
        return cls(**base)

    @classmethod
    def rawnet2_100e(cls, **overrides) -> "BenchmarkConfig":
        """Preset individual do RawNet2 para benchmark real em 100 epocas."""
        base = dict(
            architectures=["RawNet2"],
            epochs=100,
            batch_size=16,
            snr_levels_db=[30, 20, 10, 5],
            run_api_probe=False,
            preset_name="single:RawNet2",
            device_profile="gpu",
            optimize_hyperparameters=True,
        )
        base.update(overrides)
        return cls(**base)

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)
