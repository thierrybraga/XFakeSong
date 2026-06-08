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


def test_benchmark_data_validation_rejects_bad_labels():
    from benchmarks.data import BenchmarkData

    d = BenchmarkData(
        X=np.zeros((6, 8, 8), dtype="float32"),
        y=np.array([0, 1, 2, 0, 1, 2]),
    )
    try:
        d.validate()
    except ValueError as exc:
        assert "labels esperados" in str(exc)
    else:
        raise AssertionError("validate deveria rejeitar labels fora de {0,1}")


def test_prepare_for_architecture_adapts_input_contracts():
    from benchmarks.data import BenchmarkData

    d = BenchmarkData.synthetic(n=20, shape=(8, 8), seed=2)
    raw = d.prepare_for_architecture("RawNet2")
    spec = d.prepare_for_architecture("MultiscaleCNN")
    svm = d.prepare_for_architecture("SVM")

    assert raw.X.shape == (20, 16000, 1)
    assert spec.X.shape == (20, 100, 80)
    assert svm.X.shape == d.X.shape
    assert raw.metadata["input_type"] == "raw_audio"
    assert spec.metadata["input_type"] == "spectrogram"


def test_prepare_raw_audio_for_spectrogram_uses_logmel():
    from benchmarks.data import BenchmarkData

    rng = np.random.default_rng(3)
    d = BenchmarkData(
        X=rng.standard_normal((8, 16000, 1)).astype("float32"),
        y=np.array([0, 1] * 4),
    )
    spec = d.prepare_for_architecture("MultiscaleCNN")

    assert spec.X.shape == (8, 100, 80)
    assert np.isfinite(spec.X).all()
    assert spec.metadata["input_type"] == "spectrogram"


def test_prepare_raw_audio_for_classical_uses_compact_features():
    from benchmarks.data import BenchmarkData

    rng = np.random.default_rng(4)
    d = BenchmarkData(
        X=rng.standard_normal((8, 16000, 1)).astype("float32"),
        y=np.array([0, 1] * 4),
    )
    svm = d.prepare_for_architecture("SVM")

    assert svm.X.ndim == 2
    assert svm.X.shape[0] == 8
    assert svm.X.shape[1] < 16000
    assert np.isfinite(svm.X).all()
    assert svm.metadata["input_type"] == "tabular_audio_features"


def test_npz_metadata_json_is_loaded():
    from benchmarks.data import BenchmarkData

    with tempfile.TemporaryDirectory() as td:
        path = Path(td) / "dataset.npz"
        meta = {
            "source": "unit_test",
            "sample_rate": 16000,
            "duration_sec": 1.0,
            "splits": {"train": {"samples": 4, "real": 2, "fake": 2}},
        }
        np.savez_compressed(
            path,
            X=np.zeros((8, 16, 1), dtype="float32"),
            y=np.array([0, 1] * 4, dtype="int64"),
            metadata_json=np.asarray(json.dumps(meta)),
        )

        data = BenchmarkData.from_npz(str(path))

    assert data.metadata["source"] == "unit_test"
    assert data.metadata["sample_rate"] == 16000
    assert data.metadata["npz_path"].endswith("dataset.npz")


def test_evaluate_scores_perfect_separation():
    from benchmarks.evaluate import evaluate_scores

    y = np.array([0, 0, 1, 1])
    p = np.array([0.1, 0.2, 0.8, 0.9])  # separa perfeitamente
    m = evaluate_scores(y, p)
    assert m["auc_roc"] == 1.0
    assert m["eer"] == 0.0
    assert m["accuracy"] == 1.0
    assert "min_tdcf" in m


def test_evaluate_scores_sanitizes_nonfinite_scores():
    from benchmarks.evaluate import evaluate_scores

    metrics = evaluate_scores(
        np.array([0, 1, 0, 1]),
        np.array([np.nan, np.inf, -np.inf, 0.75]),
    )

    assert metrics["nonfinite_scores"] == 3
    assert np.isfinite(metrics["accuracy"])
    assert np.isfinite(metrics["eer"])


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
                "history": {"val_binary_accuracy": [0.6, 0.8, 0.93]},
                "epochs": 3,
            },
            "SVM": {"status": "error", "error": "falha simulada"},
        },
    }
    with tempfile.TemporaryDirectory() as td:
        write_all(fake, td)
        out = Path(td)
        assert (out / "results.json").exists()
        assert (out / "results.csv").exists()
        assert (out / "predictions_clean.csv").exists()
        assert (out / "summary.md").exists()
        assert (out / "tcc_report.md").exists()
        for t in ("tab_resultados", "tab_eficiencia", "tab_robustez"):
            tex = (out / "tables" / f"{t}.tex").read_text(encoding="utf-8")
            assert "\\begin{table}" in tex and "MultiscaleCNN" in tex
        resultados = (out / "tables" / "tab_resultados.tex").read_text(
            encoding="utf-8"
        )
        assert "\\begin{tabular}{lccccccc}" in resultados
        assert "\\multicolumn{7}{c}" in resultados
        # figuras desenhadas a partir de scores/história
        assert (out / "figures" / "roc.png").exists()
        assert (out / "figures" / "convergencia.png").exists()
        assert (out / "figures" / "confusion_matrices.png").exists()
        assert (out / "figures" / "score_distributions.png").exists()
        pred_csv = (out / "predictions_clean.csv").read_text("utf-8")
        assert "architecture,sample_index,y_true,p_fake,y_pred,correct" in pred_csv
        arch_dir = out / "architectures" / "multiscalecnn"
        assert (arch_dir / "metrics.json").exists()
        assert (arch_dir / "summary.md").exists()
        assert (arch_dir / "predictions_clean.csv").exists()
        assert (arch_dir / "robustness.csv").exists()
        assert (arch_dir / "confusion_matrix.png").exists()
        assert (arch_dir / "roc.png").exists()
        assert (arch_dir / "score_distribution.png").exists()
        assert (arch_dir / "convergence.png").exists()
        report = (out / "tcc_report.md").read_text("utf-8")
        assert "# Relatório de Benchmark para TCC" in report
        assert "![Curvas ROC](figures/roc.png)" in report
        assert "(architectures/multiscalecnn/confusion_matrix.png)" in report


def test_robustez_table_uses_dynamic_colspan_without_converged_models():
    from benchmarks.report import write_all

    fake = {
        "config": {"snr_levels_db": [20]},
        "environment": {"platform": "x", "python": "3.13"},
        "dataset": {
            "name": "synthetic",
            "n_total": 10,
            "n_test": 2,
            "input_shape": [8, 8],
            "balance_test": {"real": 1, "fake": 1},
            "y_test": [0, 1],
        },
        "architectures": {
            "SVM": {
                "status": "ok",
                "type": "classical",
                "converged": False,
                "clean": {"accuracy": 0.5, "eer": 0.5, "auc_roc": 0.5},
                "scores_clean": [0.4, 0.6],
                "robustness": {"20": {"accuracy": 0.5, "eer": 0.5}},
                "efficiency": {"params": None, "size_mb": 0.1, "latency_ms": 1.0},
                "history": None,
                "epochs": 1,
            }
        },
    }
    with tempfile.TemporaryDirectory() as td:
        write_all(fake, td)
        tex = (Path(td) / "tables" / "tab_robustez.tex").read_text("utf-8")
        assert "\\multicolumn{5}{c}{(nenhum modelo convergente)}" in tex


def test_quick_preset_is_classical_and_fast():
    from benchmarks import BenchmarkConfig

    cfg = BenchmarkConfig.quick()
    assert cfg.architectures == ["SVM"]
    assert cfg.synthetic_shape == (8, 8)


def test_convergence_requires_accuracy_threshold():
    from benchmarks import BenchmarkConfig, run_benchmark

    with tempfile.TemporaryDirectory() as td:
        cfg = BenchmarkConfig.quick(
            architectures=["SVM"],
            snr_levels_db=[],
            output_dir=td,
            synthetic_n=120,
            converge_accuracy_threshold=1.01,
        )
        results = run_benchmark(cfg)
        assert results["architectures"]["SVM"]["converged"] is False


def test_api_probe_uses_configured_openapi_path(monkeypatch):
    from benchmarks.api_probe import run_api_probe

    monkeypatch.setenv("XFAKE_API_ONLY", "1")
    monkeypatch.setenv("XFAKE_CREATE_DEFAULT_MODELS", "0")
    result = run_api_probe(max_endpoints=3)

    assert result["status"] == "ok"
    assert result["n_2xx"] >= 1
    assert result["endpoints"][0]["path"] == "/api/openapi.json"



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
        assert (Path(td) / "tcc_report.md").exists()
