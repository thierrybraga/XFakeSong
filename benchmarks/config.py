"""Configuração do benchmark."""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


# Ordem CANÔNICA de execução do benchmark: do MENOS para o MAIS custoso de
# treinar. Roda primeiro o barato (clássico, CNN de espectrograma leve) — que
# valida o pipeline cedo e falha rápido — e deixa por último o caro (raw-audio
# GAT/SSL e o Ensemble, que depende dos demais). Heurística por: família
# (clássico < neural), domínio de entrada (espectrograma < áudio bruto de 16k×5s),
# complexidade (CNN < LSTM < Transformer < GAT < SSL) e cap de batch por VRAM
# (ver benchmarks/planning.py::_fit_to_device). Refinável com a latência/tempo
# medidos de uma execução completa. `run_models_sequential.py` e o caderno
# `04_all_architectures_full_benchmark` consomem esta ordem.
ALL_TCC_ARCHITECTURES = [
    # --- Clássico (single fit, CPU) ---
    "RandomForest",
    "SVM",
    # --- Espectrograma: CNN leve -> LSTM -> Transformer ---
    "MultiscaleCNN",
    "Sonic Sleuth",
    "EfficientNet-LSTM",
    "Conformer",
    "Hybrid CNN-Transformer",
    "SpectrogramTransformer",
    # --- Áudio bruto: CNN/GRU -> GAT -> SSL ---
    "RawNet2",
    "AASIST",
    "RawGAT-ST",
    "WavLM",
    "HuBERT",
    # --- Fusão (depende dos demais; mais custoso) ---
    "Ensemble",
]

CLASSICAL_TCC_ARCHITECTURES = ["RandomForest", "SVM"]

NEURAL_TCC_ARCHITECTURES = [
    arch for arch in ALL_TCC_ARCHITECTURES if arch not in CLASSICAL_TCC_ARCHITECTURES
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
        default_factory=lambda: ["MultiscaleCNN", "Ensemble"]
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
        """Preset completo do TCC: todas as arquiteturas + baselines clássicos."""
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
        """Alias explícito para o preset completo com as 14 arquiteturas."""
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
        """Preset neural do TCC: todas as arquiteturas neurais, sem SVM/RF."""
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
