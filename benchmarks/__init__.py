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
    python scripts/run_benchmark.py --quick
    python scripts/run_benchmark.py --dataset data.npz --archs MultiscaleCNN Ensemble SVM
"""

from __future__ import annotations

from benchmarks.config import (
    ALL_TCC_ARCHITECTURES,
    CLASSICAL_TCC_ARCHITECTURES,
    NEURAL_TCC_ARCHITECTURES,
    BenchmarkConfig,
)
from benchmarks.runner import plan_benchmark, run_benchmark

__all__ = [
    "ALL_TCC_ARCHITECTURES",
    "CLASSICAL_TCC_ARCHITECTURES",
    "NEURAL_TCC_ARCHITECTURES",
    "BenchmarkConfig",
    "plan_benchmark",
    "run_benchmark",
]
