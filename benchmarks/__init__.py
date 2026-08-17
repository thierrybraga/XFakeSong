"""Sistema de benchmark e teste do XFakeSong (para o TCC).

Gera, de forma reprodutível e usando o PIPELINE REAL (TrainingService →
ModelLoader → Predictor → MetricsCalculator) e a API (FastAPI TestClient),
os dados empíricos do trabalho:

- desempenho por arquitetura (acurácia, precisão, recall, F1, EER, AUC-ROC,
  min-tDCF) em um conjunto de teste held-out;
- eficiência computacional (parâmetros, tamanho em disco, latência);
- robustez sob ruído AWGN em múltiplos níveis de SNR;
- teste de sistema da API REST (status + latência por endpoint).

Saídas: JSON + CSV (legíveis por máquina) e tabelas LaTeX + figuras prontas
para a monografia.

Uso programático:
    from benchmarks import BenchmarkConfig, run_benchmark
    cfg = BenchmarkConfig.quick()
    results = run_benchmark(cfg)

Uso por CLI:
    python scripts/benchmark/run_benchmark.py --quick
    python scripts/benchmark/run_benchmark.py --dataset data.npz --archs MultiscaleCNN SVM RandomForest
"""

from __future__ import annotations

from benchmarks.config import (
    ALL_TCC_ARCHITECTURES,
    CLASSICAL_TCC_ARCHITECTURES,
    DOCKER_TRAINING_ARCHITECTURES,
    EXTENDED_MODEL_MANIFEST,
    MODEL_FAMILIES,
    NEURAL_TCC_ARCHITECTURES,
    NEURAL_DOCKER_ARCHITECTURES,
    OFFICIAL_TCC_DISPLAY_NAMES,
    OFFICIAL_TCC_MODEL_MANIFEST,
    OFFICIAL_TCC_RESULT_ORDER,
    SSL_DOCKER_ARCHITECTURES,
    BenchmarkConfig,
)

# `plan_benchmark`/`run_benchmark` entram SOB DEMANDA (PEP 562).
#
# `benchmarks.runner` importa TensorFlow no topo, e importa-lo aqui fazia
# QUALQUER acesso ao pacote puxar o stack de treino inteiro — inclusive
# `from benchmarks.config import ...`, que e so dataclasses. Na pratica isso
# impedia consolidar resultados, reconstruir resumos e auditar artefatos num
# checkout sem o ambiente de treino instalado: passos que apenas leem e
# reescrevem JSON exigiam ~600 MB de dependencia de GPU.
#
# O contrato publico nao muda: `from benchmarks import run_benchmark` continua
# funcionando, so que resolvido na primeira vez que o nome e usado.
_LAZY = {"plan_benchmark", "run_benchmark"}


def __getattr__(name: str):
    if name in _LAZY:
        from benchmarks import runner

        return getattr(runner, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(__all__)


__all__ = [
    "ALL_TCC_ARCHITECTURES",
    "CLASSICAL_TCC_ARCHITECTURES",
    "DOCKER_TRAINING_ARCHITECTURES",
    "EXTENDED_MODEL_MANIFEST",
    "MODEL_FAMILIES",
    "NEURAL_TCC_ARCHITECTURES",
    "NEURAL_DOCKER_ARCHITECTURES",
    "OFFICIAL_TCC_DISPLAY_NAMES",
    "OFFICIAL_TCC_MODEL_MANIFEST",
    "OFFICIAL_TCC_RESULT_ORDER",
    "SSL_DOCKER_ARCHITECTURES",
    "BenchmarkConfig",
    "plan_benchmark",
    "run_benchmark",
]
