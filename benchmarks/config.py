"""Configuração do benchmark."""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


# Manifesto oficial do recorte experimental. Ele fixa nomes, variantes e
# runners usados no TCC; listas derivadas abaixo devem ser consumidas pelos
# scripts para evitar divergência entre treino, consolidação e LaTeX.
OFFICIAL_TCC_MODEL_MANIFEST: List[Dict[str, Any]] = [
    {
        "benchmark_name": "RandomForest",
        "result_key": "RandomForest",
        "display_name": "Random Forest",
        "variant": "sklearn_random_forest_gridsearch",
        "runner": "benchmarks.runner:classical",
        "input_type": "tabular_features",
    },
    {
        "benchmark_name": "SVM",
        "result_key": "SVM",
        "display_name": "SVM",
        "variant": "sklearn_svc_rbf_gridsearch",
        "runner": "benchmarks.runner:classical",
        "input_type": "tabular_features",
    },
    {
        "benchmark_name": "Hybrid CNN-Transformer",
        "result_key": "CCT",
        "display_name": "CCT",
        "variant": "cct",
        "runner": "benchmarks.runner:keras",
        "input_type": "spectrogram",
    },
    {
        "benchmark_name": "SpectrogramTransformer",
        "result_key": "AST",
        "display_name": "AST",
        "variant": "ast_vit_base_scratch",
        "runner": "benchmarks.runner:keras",
        "input_type": "spectrogram",
    },
    {
        "benchmark_name": "MultiscaleCNN",
        "result_key": "Res2Net",
        "display_name": "Res2Net",
        "variant": "res2net50_scale4_no_se",
        "runner": "benchmarks.runner:keras",
        "input_type": "spectrogram",
    },
    {
        "benchmark_name": "Conformer",
        "result_key": "Conformer",
        "display_name": "Conformer",
        "variant": "conformer",
        "runner": "benchmarks.runner:keras",
        "input_type": "spectrogram",
    },
    {
        "benchmark_name": "RawNet2",
        "result_key": "RawNet2",
        "display_name": "RawNet2",
        "variant": "rawnet2_paper_like_gru1024_dense1024",
        "runner": "benchmarks.runner:keras",
        "input_type": "raw_audio",
    },
    {
        "benchmark_name": "AASIST",
        "result_key": "AASIST",
        "display_name": "AASIST",
        "variant": "aasist",
        "runner": "benchmarks.runner:keras",
        "input_type": "raw_audio",
    },
    {
        "benchmark_name": "RawGAT-ST",
        "result_key": "RawGAT-ST",
        "display_name": "RawGAT-ST",
        "variant": "rawgat_st_multiply_stride4",
        "runner": "benchmarks.runner:keras",
        "input_type": "raw_audio",
    },
    {
        "benchmark_name": "WavLM Original",
        "result_key": "WavLM Original",
        "display_name": "WavLM Original",
        "variant": "microsoft/wavlm-base:frozen_backbone",
        "runner": "scripts.run_wavlm_original_benchmark:ssl_original",
        "input_type": "raw_audio_16khz_16000",
    },
    {
        "benchmark_name": "HuBERT Original",
        "result_key": "HuBERT Original",
        "display_name": "HuBERT Original",
        "variant": "facebook/hubert-base-ls960:frozen_backbone",
        "runner": "scripts.run_wavlm_original_benchmark:ssl_original",
        "input_type": "raw_audio_16khz_16000",
    },
]

OFFICIAL_TCC_RESULT_ORDER = [
    item["result_key"] for item in OFFICIAL_TCC_MODEL_MANIFEST
]

OFFICIAL_TCC_DISPLAY_NAMES = {
    item["result_key"]: item["display_name"] for item in OFFICIAL_TCC_MODEL_MANIFEST
}

# Modelos suportados diretamente por benchmarks.runner/run_benchmark.py.
ALL_TCC_ARCHITECTURES = [
    item["benchmark_name"]
    for item in OFFICIAL_TCC_MODEL_MANIFEST
    if not str(item["runner"]).endswith(":ssl_original")
]

# WavLM/HuBERT reais são treinados no mesmo fluxo WSL/Docker, mas por um runner
# SSL PyTorch/Hugging Face dedicado. O runner baixa o checkpoint base, congela o
# backbone e treina somente a cabeça classificadora.
SSL_DOCKER_ARCHITECTURES = [
    item["benchmark_name"]
    for item in OFFICIAL_TCC_MODEL_MANIFEST
    if str(item["runner"]).endswith(":ssl_original")
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

    architectures: List[str] = field(
        default_factory=lambda: ["MultiscaleCNN", "SVM"]
    )
    dataset_path: Optional[str] = None
    epochs: int = 20
    batch_size: int = 32
    seed: int = 42
    snr_levels_db: List[int] = field(default_factory=lambda: [30, 20, 10])
    latency_runs: int = 30
    output_dir: str = "results/benchmark"
    models_dir: str = "app/models"
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

    # P2 — augmentation ruidoso para os modelos clássicos (SVM/RF). Anexa
    # cópias do conjunto de treino com AWGN nos MESMOS níveis de SNR avaliados
    # (mesmo espaço de feature em que a robustez é medida), atacando o colapso
    # a ~50% sob ruído.
    classical_noise_augmentation: bool = True

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
            snr_levels_db=[30, 20, 10],
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
    def cross_generator_tcc(cls, holdout_generator: str = "fkvoice",
                            **overrides) -> "BenchmarkConfig":
        """Preset P0.4 — reteste cross-generator (anti-vazamento de fonte).

        Treina SEM o gerador `holdout_generator` (default XTTS=fkvoice) e o usa
        como teste. Mede generalização a gerador inédito — o reteste
        metodológico mais importante antes da defesa.
        """
        base = dict(
            architectures=list(ALL_TCC_ARCHITECTURES),
            epochs=100,
            snr_levels_db=[30, 20, 10],
            run_api_probe=False,
            preset_name=f"cross_generator:{holdout_generator}",
            optimize_hyperparameters=True,
            holdout_generator=holdout_generator,
        )
        base.update(overrides)
        return cls(**base)

    @classmethod
    def unseen_speaker_tcc(cls, holdout_speaker: Optional[str] = None,
                           **overrides) -> "BenchmarkConfig":
        """Preset tier `large` — protocolo de USUÁRIO NÃO VISTO (unseen speaker).

        Com `holdout_speaker`, segura um falante fora do treino e testa nele
        (mais reais reservados, p/ manter ambas as classes no teste). Sem ele,
        usa split disjunto por falante (`speaker_split=True`). Requer um `.npz`
        com `speaker_ids` (exportado a partir de um dataset tier `large`).
        """
        base = dict(
            architectures=list(ALL_TCC_ARCHITECTURES),
            epochs=100,
            snr_levels_db=[30, 20, 10],
            run_api_probe=False,
            preset_name=(
                f"unseen_speaker:{holdout_speaker}" if holdout_speaker
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
            snr_levels_db=[30, 20, 10],
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
            snr_levels_db=[30, 20, 10],
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
            snr_levels_db=[30, 20, 10],
            run_api_probe=False,
            preset_name="single:RawNet2",
            device_profile="gpu",
            optimize_hyperparameters=True,
        )
        base.update(overrides)
        return cls(**base)

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)
