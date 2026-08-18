"""Repetições com sementes distintas e publicação da incerteza.

Duas lacunas metodológicas fechadas em 2026-07-27:

1. Os IC 95% de bootstrap eram calculados (1000 reamostragens por condição) e
   **descartados** na geração das tabelas — o artigo publicava pontos secos.
2. Cada arquitetura rodava **uma única vez**: o bootstrap mede a variância de
   amostragem do TESTE, não a de TREINO. Sem repetição, "A é melhor que B" não
   se distingue de "esta execução de A foi melhor".
"""

from __future__ import annotations

import numpy as np
import pytest

from benchmarks.config import BenchmarkConfig
from benchmarks.report import _pct_with_uncertainty, _uncertainty_note
from benchmarks.runner import _aggregate_seed_runs


def test_training_seeds_derive_from_base_seed():
    cfg = BenchmarkConfig(seed=42, n_seeds=3)
    assert cfg.training_seeds == [42, 43, 44]
    assert BenchmarkConfig(seed=7, n_seeds=1).training_seeds == [7]


def _run(seed: int, eer: float, acc: float) -> dict:
    return {
        "status": "ok",
        "training_seed": seed,
        "clean": {"eer": eer, "accuracy": acc, "auc_roc": 0.9, "n": 100},
        "robustness": {"20": {"eer": eer + 0.01, "accuracy": acc - 0.01}},
        "efficiency": {"latency_ms": 1.0},
        "wall_time_s": 1.0,
    }


def test_aggregation_reports_mean_and_sample_std():
    runs = [_run(42, 0.10, 0.90), _run(43, 0.20, 0.80), _run(44, 0.30, 0.70)]
    agg = _aggregate_seed_runs(runs)

    assert agg["n_seeds"] == 3
    assert agg["training_seeds"] == [42, 43, 44]
    assert agg["clean"]["eer"] == pytest.approx(0.20)
    # desvio AMOSTRAL (ddof=1), não populacional
    assert agg["clean"]["eer_seed_std"] == pytest.approx(np.std([0.1, 0.2, 0.3], ddof=1))
    assert agg["clean"]["eer_seed_values"] == [0.10, 0.20, 0.30]
    # a robustez por SNR também agrega
    assert agg["robustness"]["20"]["eer"] == pytest.approx(0.21)
    # cada execução fica preservada para auditoria, COM o tempo de parede
    # (a chave é `wall_time_s`; usar `duration_sec` deixava tudo nulo)
    assert len(agg["seed_runs"]) == 3
    assert all(run["wall_time_s"] == 1.0 for run in agg["seed_runs"])


def test_promoted_artifact_is_first_seed_not_best():
    """Escolher a melhor execução pelo teste seria seleção no conjunto de teste."""
    runs = [_run(42, 0.30, 0.70), _run(43, 0.05, 0.95), _run(44, 0.20, 0.80)]
    agg = _aggregate_seed_runs(runs)
    assert agg["promoted_artifact_seed"] == 42, "não pode promover a melhor seed"


def test_single_run_keeps_previous_schema():
    """Com n_seeds=1 o resultado é o de antes — sem campos de agregação."""
    run = _run(42, 0.10, 0.90)
    agg = _aggregate_seed_runs([run])
    assert agg is run
    assert "eer_seed_std" not in agg["clean"]


def test_tables_publish_uncertainty():
    """A incerteza precisa CHEGAR à tabela: ± entre seeds, ou IC do bootstrap."""
    multi = {"eer": 0.0446, "eer_seed_std": 0.0223}
    assert "$\\pm$" in _pct_with_uncertainty(multi, "eer")

    single = {"eer": 0.0446, "eer_ci95_low": 0.02, "eer_ci95_high": 0.07}
    rendered = _pct_with_uncertainty(single, "eer")
    assert "[" in rendered and ";" in rendered

    bare = {"eer": 0.0446}
    assert _pct_with_uncertainty(bare, "eer").endswith(r"\%")
    assert _pct_with_uncertainty({"eer": float("nan")}, "eer") == "---"


def test_uncertainty_note_declares_which_uncertainty():
    multi = {
        "architectures": {"A": {"status": "ok", "n_seeds": 3}},
        "config": {"bootstrap_ci_samples": 1000},
    }
    assert "3 execuções" in _uncertainty_note(multi)

    single = {
        "architectures": {"A": {"status": "ok", "n_seeds": 1}},
        "config": {"bootstrap_ci_samples": 1000},
    }
    note = _uncertainty_note(single)
    assert "bootstrap" in note
    # precisa declarar a limitação, não só a metodologia
    assert "variabilidade de treino não está representada" in note
