"""Testes do sistema de benchmark (benchmarks/).

Cobre as partes puras (split, AWGN, métricas, relatório) e um smoke de
integração ponta-a-ponta usando SVM — clássico e rápido, exercitando o runner
completo (treino→avaliação→robustez→eficiência→relatório) sem treino Keras lento.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")


def test_stratified_split_keeps_both_classes_in_test():
    from benchmarks.data import BenchmarkData

    d = BenchmarkData.synthetic(n=200, shape=(8, 8), seed=1)
    Xtr, ytr, Xv, yv, Xte, yte = d.stratified_split(seed=1)
    assert len(Xtr) + len(Xv) + len(Xte) == 200
    assert set(np.unique(yte)) == {0, 1}  # teste tem ambas as classes
    assert Xte.shape[1:] == (8, 8)


def test_awgn_preserves_shape_and_is_finite():
    from benchmarks.data import BenchmarkData

    X = np.random.default_rng(0).standard_normal((10, 8, 8)).astype("float32")
    Xn = BenchmarkData.add_awgn(X, snr_db=20, seed=0)
    assert Xn.shape == X.shape
    assert np.isfinite(Xn).all()
    # ruído mais forte (SNR menor) afasta mais do sinal original
    near = np.mean((BenchmarkData.add_awgn(X, 40, 0) - X) ** 2)
    far = np.mean((BenchmarkData.add_awgn(X, 5, 0) - X) ** 2)
    assert far > near


def test_evaluate_scores_perfect_separation():
    from benchmarks.evaluate import evaluate_scores

    y = np.array([0, 0, 1, 1])
    p = np.array([0.1, 0.2, 0.8, 0.9])  # separa perfeitamente
    m = evaluate_scores(y, p)
    assert m["auc_roc"] == 1.0
    assert m["eer"] == 0.0
    assert m["accuracy"] == 1.0
    assert "min_tdcf" in m


def test_report_write_all_creates_artifacts():
    from benchmarks.report import write_all

    fake = {
        "config": {"snr_levels_db": [20]},
        "environment": {"platform": "x", "python": "3.13", "tensorflow": "2.21",
                        "gpu": False, "device": "CPU"},
        "dataset": {"name": "synthetic", "n_total": 100, "n_test": 15,
                    "input_shape": [8, 8], "balance_test": {"real": 7, "fake": 8},
                    "y_test": [0, 1] * 7 + [0]},
        "architectures": {
            "MultiscaleCNN": {
                "status": "ok", "type": "neural", "converged": True,
                "clean": {"accuracy": 0.93, "precision": 0.9, "recall": 0.95,
                          "f1": 0.92, "auc_roc": 0.98, "eer": 0.06,
                          "min_tdcf": 0.12},
                "scores_clean": [0.2, 0.8] * 7 + [0.1],
                "robustness": {"20": {"accuracy": 0.8, "eer": 0.2,
                                      "auc_roc": 0.85}},
                "efficiency": {"params": 1000, "size_mb": 4.4,
                               "latency_ms": 12.3},
                "history": {"val_accuracy": [0.6, 0.8, 0.93]}, "epochs": 3,
            },
            "SVM": {"status": "error", "error": "falha simulada"},
        },
    }
    with tempfile.TemporaryDirectory() as td:
        write_all(fake, td)
        out = Path(td)
        assert (out / "results.json").exists()
        assert (out / "results.csv").exists()
        assert (out / "summary.md").exists()
        for t in ("tab_resultados", "tab_eficiencia", "tab_robustez"):
            tex = (out / "tables" / f"{t}.tex").read_text(encoding="utf-8")
            assert "\\begin{table}" in tex and "MultiscaleCNN" in tex
        # figuras desenhadas a partir de scores/história
        assert (out / "figures" / "roc.png").exists()
        assert (out / "figures" / "convergencia.png").exists()


def test_run_benchmark_quick_svm_integration():
    """Smoke ponta-a-ponta com SVM (clássico, rápido): runner + relatório."""
    from benchmarks import BenchmarkConfig, run_benchmark

    with tempfile.TemporaryDirectory() as td:
        cfg = BenchmarkConfig.quick(
            architectures=["SVM"], snr_levels_db=[20], output_dir=td,
            synthetic_n=160, synthetic_shape=(8, 8),
        )
        results = run_benchmark(cfg)
        svm = results["architectures"]["SVM"]
        assert svm["status"] == "ok", svm
        assert svm["type"] == "classical"
        assert "clean" in svm and "auc_roc" in svm["clean"]
        assert "20" in svm["robustness"]
        assert svm["efficiency"]["latency_ms"] is not None
        # artefatos
        saved = json.loads((Path(td) / "results.json").read_text("utf-8"))
        assert saved["dataset"]["n_test"] > 0
