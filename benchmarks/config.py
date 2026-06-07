"""Configuração do benchmark."""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from typing import List, Optional


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
    run_api_probe: bool = False
    synthetic_n: int = 360
    synthetic_shape: tuple = (32, 16)
    converge_auc_threshold: float = 0.60
    converge_accuracy_threshold: float = 0.55

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
        )
        base.update(overrides)
        return cls(**base)

    @classmethod
    def full_tcc(cls, **overrides) -> "BenchmarkConfig":
        """Preset do TCC: as arquiteturas-chave + baselines clássicos.

        Inclui SVM e Random Forest (clássicos, leves) — baselines que o cenário
        CPU/dados-escassos favorece e que o artigo original havia omitido.
        """
        base = dict(
            architectures=[
                "MultiscaleCNN",
                "Ensemble",
                "EfficientNet-LSTM",
                "AASIST",
                "RawNet2",
                "SVM",
                "RandomForest",
            ],
            epochs=20,
            snr_levels_db=[30, 20, 10],
            run_api_probe=True,
        )
        base.update(overrides)
        return cls(**base)

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)
