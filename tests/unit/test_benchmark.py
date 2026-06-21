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
import pytest

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


def test_prepare_raw_audio_center_crops_long_clips_for_rawnet2():
    from benchmarks.data import BenchmarkData

    rng = np.random.default_rng(33)
    d = BenchmarkData(
        X=rng.standard_normal((8, 80000, 1)).astype("float32"),
        y=np.array([0, 1] * 4),
    )
    raw = d.prepare_for_architecture("RawNet2")

    assert raw.X.shape == (8, 16000, 1)
    assert raw.metadata["prepared_shape"] == [16000, 1]


def test_prepare_raw_audio_center_crops_long_clips_for_aasist():
    from benchmarks.data import BenchmarkData

    rng = np.random.default_rng(34)
    d = BenchmarkData(
        X=rng.standard_normal((8, 80000, 1)).astype("float32"),
        y=np.array([0, 1] * 4),
    )
    raw = d.prepare_for_architecture("AASIST")

    assert raw.X.shape == (8, 16000, 1)
    assert raw.metadata["input_type"] == "raw_audio"
    assert raw.metadata["prepared_shape"] == [16000, 1]


def test_prepare_raw_audio_center_crops_long_clips_for_ensemble():
    from benchmarks.data import BenchmarkData

    rng = np.random.default_rng(35)
    d = BenchmarkData(
        X=rng.standard_normal((8, 80000, 1)).astype("float32"),
        y=np.array([0, 1] * 4),
    )
    raw = d.prepare_for_architecture("Ensemble")

    assert raw.X.shape == (8, 16000, 1)
    assert raw.metadata["input_type"] == "raw_audio"
    assert raw.metadata["prepared_shape"] == [16000, 1]


def test_prepare_raw_audio_center_crops_long_clips_for_wavlm():
    from benchmarks.data import BenchmarkData

    rng = np.random.default_rng(36)
    d = BenchmarkData(
        X=rng.standard_normal((8, 80000, 1)).astype("float32"),
        y=np.array([0, 1] * 4),
    )
    raw = d.prepare_for_architecture("WavLM")

    assert raw.X.shape == (8, 16000, 1)
    assert raw.metadata["input_type"] == "raw_audio"
    assert raw.metadata["prepared_shape"] == [16000, 1]


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


def test_report_creates_convergence_placeholder_for_classical_models():
    from benchmarks.report import write_all

    fake = {
        "config": {"snr_levels_db": [20]},
        "environment": {"platform": "x", "python": "3.13"},
        "dataset": {
            "name": "synthetic",
            "n_total": 10,
            "n_test": 4,
            "input_shape": [8, 8],
            "balance_test": {"real": 2, "fake": 2},
            "y_test": [0, 1, 0, 1],
        },
        "architectures": {
            "SVM": {
                "status": "ok",
                "type": "classical",
                "converged": True,
                "clean": {"accuracy": 0.75, "eer": 0.25, "auc_roc": 0.75},
                "scores_clean": [0.1, 0.8, 0.4, 0.7],
                "robustness": {"20": {"accuracy": 0.75, "eer": 0.25}},
                "efficiency": {"params": None, "size_mb": 0.1, "latency_ms": 1.0},
                "history": None,
                "training_config": {"model_family": "classical", "fit_samples": 6},
                "final_training_metrics": {
                    "fit_samples": 6,
                    "n_features": 8,
                    "classes": [0, 1],
                },
                "epochs": 1,
            }
        },
    }
    with tempfile.TemporaryDirectory() as td:
        write_all(fake, td)
        out = Path(td)
        assert (out / "figures" / "convergencia.png").exists()
        assert (out / "architectures" / "svm" / "convergence.png").exists()
        assert "(architectures/svm/convergence.png)" in (
            out / "tcc_report.md"
        ).read_text("utf-8")


def test_report_creates_graph_placeholders_when_scores_are_missing():
    from benchmarks.report import write_all

    fake = {
        "config": {"snr_levels_db": [20]},
        "environment": {"platform": "x", "python": "3.13"},
        "dataset": {
            "name": "synthetic",
            "n_total": 10,
            "n_test": 4,
            "input_shape": [8, 8],
            "balance_test": {"real": 2, "fake": 2},
            "y_test": [0, 1, 0, 1],
        },
        "architectures": {
            "RawNet2": {
                "status": "ok",
                "type": "neural",
                "converged": False,
                "clean": {"accuracy": None, "eer": None, "auc_roc": None},
                "scores_clean": [],
                "robustness": {},
                "efficiency": {"params": None, "size_mb": None, "latency_ms": None},
                "history": {},
                "epochs": 1,
            }
        },
    }
    with tempfile.TemporaryDirectory() as td:
        write_all(fake, td)
        out = Path(td)
        for figure in [
            "roc.png",
            "robustez.png",
            "convergencia.png",
            "eficiencia.png",
            "confusion_matrices.png",
            "score_distributions.png",
        ]:
            assert (out / "figures" / figure).exists()
        for figure in [
            "confusion_matrix.png",
            "roc.png",
            "score_distribution.png",
            "convergence.png",
        ]:
            assert (out / "architectures" / "rawnet2" / figure).exists()


def test_tcc_pipeline_verifies_per_architecture_artifacts(tmp_path):
    from scripts.run_tcc_pipeline import _verify_outputs

    results = {
        "architectures": {
            "SVM": {
                "status": "ok",
            }
        }
    }
    root_files = [
        "results.csv",
        "predictions_clean.csv",
        "summary.md",
        "tcc_report.md",
        "dataset_manifest.json",
        "dataset.md",
        "tables/tab_resultados.tex",
        "tables/tab_eficiencia.tex",
        "tables/tab_robustez.tex",
        "figures/roc.png",
        "figures/robustez.png",
        "figures/convergencia.png",
        "figures/eficiencia.png",
        "figures/confusion_matrices.png",
        "figures/score_distributions.png",
    ]
    arch_files = [
        "metrics.json",
        "summary.md",
        "predictions_clean.csv",
        "robustness.csv",
        "confusion_matrix.png",
        "roc.png",
        "score_distribution.png",
    ]

    (tmp_path / "results.json").write_text(json.dumps(results), encoding="utf-8")
    for relative in root_files:
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("ok", encoding="utf-8")
    for relative in arch_files:
        path = tmp_path / "architectures" / "svm" / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("ok", encoding="utf-8")

    with pytest.raises(RuntimeError, match="architectures/svm/convergence.png"):
        _verify_outputs(tmp_path)


def test_quick_preset_is_classical_and_fast():
    from benchmarks import BenchmarkConfig

    cfg = BenchmarkConfig.quick()
    assert cfg.architectures == ["SVM"]
    assert cfg.synthetic_shape == (8, 8)


def test_benchmark_relative_paths_are_anchored_to_project_root(monkeypatch, tmp_path):
    from benchmarks import BenchmarkConfig, plan_benchmark
    from benchmarks.runner import PROJECT_ROOT

    monkeypatch.chdir(tmp_path)
    cfg = BenchmarkConfig.quick(
        output_dir="results/path_anchor_probe",
        models_dir="app/models/path_anchor_probe",
    )

    plan_benchmark(cfg, write=False)

    assert Path(cfg.output_dir) == PROJECT_ROOT / "results" / "path_anchor_probe"
    assert Path(cfg.models_dir) == PROJECT_ROOT / "app" / "models" / "path_anchor_probe"


def test_full_tcc_preset_includes_all_architectures():
    from benchmarks.config import (
        ALL_TCC_ARCHITECTURES,
        CLASSICAL_TCC_ARCHITECTURES,
        NEURAL_TCC_ARCHITECTURES,
        BenchmarkConfig,
    )

    cfg = BenchmarkConfig.full_tcc()

    assert cfg.architectures == ALL_TCC_ARCHITECTURES
    assert len(cfg.architectures) == 14
    assert "SpectrogramTransformer" in cfg.architectures
    assert "Spectrogram Transformer" not in cfg.architectures
    assert BenchmarkConfig.full_all_architectures().architectures == cfg.architectures
    assert CLASSICAL_TCC_ARCHITECTURES == ["SVM", "RandomForest"]
    assert len(NEURAL_TCC_ARCHITECTURES) == 12
    assert "SVM" not in NEURAL_TCC_ARCHITECTURES
    assert BenchmarkConfig.neural_tcc().architectures == NEURAL_TCC_ARCHITECTURES
    rawnet2 = BenchmarkConfig.rawnet2_100e()
    assert rawnet2.architectures == ["RawNet2"]
    assert rawnet2.epochs == 100
    assert rawnet2.batch_size == 16
    assert rawnet2.device_profile == "gpu"
    assert rawnet2.preset_name == "single:RawNet2"


def test_neural_benchmark_plan_uses_curated_hyperparameters():
    from benchmarks import BenchmarkConfig, plan_benchmark

    with tempfile.TemporaryDirectory() as td:
        cfg = BenchmarkConfig.neural_tcc(
            architectures=[
                "WavLM",
                "AASIST",
                "SpectrogramTransformer",
                "EfficientNet-LSTM",
            ],
            output_dir=td,
            synthetic_n=24,
            synthetic_shape=(8, 8),
            epochs=7,
            device_profile="cpu",
        )
        plan = plan_benchmark(cfg, write=True)

        wavlm = plan["architectures"]["WavLM"]["training_config"]
        aasist = plan["architectures"]["AASIST"]["training_config"]
        ast = plan["architectures"]["SpectrogramTransformer"]["training_config"]
        efficientnet = plan["architectures"]["EfficientNet-LSTM"]["training_config"]

        assert plan["preset"] == "neural_tcc"
        assert wavlm["learning_rate"] == 1e-4
        assert wavlm["batch_size"] <= 4
        assert wavlm["use_augmentation"] is False
        assert wavlm["use_mixed_precision"] is False
        assert aasist["learning_rate"] == 1e-4
        assert aasist["input_domain"] == "raw_audio"
        assert aasist["batch_size"] <= 4
        assert ast["learning_rate"] == 2e-5
        assert ast["batch_size"] <= 8
        assert ast["l2_reg_strength"] == 5e-5
        assert ast["weight_decay"] == 5e-5
        assert ast["use_augmentation"] is False
        assert ast["warmup_steps"] == 3000
        assert ast["checkpoint_best"] is True
        assert ast["early_stopping"] is True
        assert ast["early_stopping_patience"] == 20
        assert ast["epochs"] == 7
        assert ast["recommended_epochs"] == 100
        assert efficientnet["learning_rate"] == 1e-4
        assert efficientnet["optimizer"] == "Adam"
        assert efficientnet["lstm_units"] == 128
        assert efficientnet["pretrained"] is True
        assert efficientnet["batch_size"] <= 8


def test_rawnet2_100e_preset_uses_benchmark_hparams():
    from benchmarks import BenchmarkConfig, plan_benchmark

    with tempfile.TemporaryDirectory() as td:
        cfg = BenchmarkConfig.rawnet2_100e(output_dir=td)
        plan = plan_benchmark(cfg, write=True)

        rawnet2 = plan["architectures"]["RawNet2"]["training_config"]
        assert plan["preset"] == "single:RawNet2"
        assert rawnet2["epochs"] == 100
        assert rawnet2["batch_size"] <= 16
        assert rawnet2["learning_rate"] == 1e-4
        assert rawnet2["optimizer"] == "Adam"
        assert rawnet2["use_augmentation"] is False
        assert rawnet2["use_mixed_precision"] is False
        assert rawnet2["early_stopping"] is False


def test_benchmark_plan_is_written_before_training():
    from benchmarks import BenchmarkConfig, plan_benchmark

    with tempfile.TemporaryDirectory() as td:
        cfg = BenchmarkConfig.full_all_architectures(
            output_dir=td,
            synthetic_n=24,
            synthetic_shape=(8, 8),
            epochs=3,
            device_profile="cpu",
            run_api_probe=False,
        )
        plan = plan_benchmark(cfg, write=True)

        assert plan["preset"] == "full_tcc"
        assert len(plan["architectures"]) == 14
        assert plan["architectures"]["AASIST"]["training_config"]["batch_size"] <= 8
        assert plan["architectures"]["SVM"]["training_config"]["model_family"] == "classical"
        assert (Path(td) / "benchmark_plan.json").exists()
        assert (Path(td) / "benchmark_plan.md").exists()


def test_all_architectures_benchmark_smoke_contract(monkeypatch):
    """CI smoke barato: valida nomes, preparo, métricas e artefatos sem treino pesado."""
    from benchmarks import BenchmarkConfig, run_benchmark
    import benchmarks.runner as runner

    def fake_run_neural(_arch, _cfg, splits, _tmp, _models_dir):
        _Xtr, _ytr, _Xv, _yv, Xte, _yte = splits
        p = np.linspace(0.1, 0.9, len(Xte), dtype="float32")

        return {
            "predict_p_fake": lambda X: np.resize(p, len(X)),
            "predict_fn": lambda xb: np.zeros((len(xb), 1), dtype="float32"),
            "params": 1,
            "size_mb": 0.0,
            "history": {"val_accuracy": [0.5]},
            "training_config": {"epochs": _cfg.epochs, "batch_size": _cfg.batch_size},
            "final_metrics": {},
            "model_artifact": str(_models_dir / f"bench_{_arch}.keras"),
        }

    def fake_run_classical(arch, cfg, splits, tmp, models_dir):
        return fake_run_neural(arch, cfg, splits, tmp, models_dir)

    monkeypatch.setattr(runner, "_run_neural", fake_run_neural)
    monkeypatch.setattr(runner, "_run_classical", fake_run_classical)

    with tempfile.TemporaryDirectory() as td:
        cfg = BenchmarkConfig.full_all_architectures(
            output_dir=td,
            synthetic_n=24,
            synthetic_shape=(8, 8),
            epochs=1,
            latency_runs=1,
            snr_levels_db=[20],
            run_api_probe=False,
        )
        results = run_benchmark(cfg)

        assert len(results["architectures"]) == 14
        assert all(r["status"] == "ok" for r in results["architectures"].values())
        assert (Path(td) / "figures" / "confusion_matrices.png").exists()
        assert (Path(td) / "results.json").exists()


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



def test_run_benchmark_quick_svm_integration(monkeypatch):
    """Smoke ponta-a-ponta com SVM (clássico, rápido): runner + relatório."""
    from benchmarks import BenchmarkConfig, run_benchmark

    for name in (
        "MODELS_DIR",
        "DEEPFAKE_MODELS_DIR",
        "XFAKE_MODELS_DIR",
        "XFAKE_STORAGE_DIR",
        "DEEPFAKE_STORAGE_DIR",
    ):
        monkeypatch.delenv(name, raising=False)

    with tempfile.TemporaryDirectory() as td:
        models_dir = Path(td) / "models"
        cfg = BenchmarkConfig.quick(
            architectures=["SVM"],
            snr_levels_db=[20],
            output_dir=td,
            models_dir=str(models_dir),
            synthetic_n=160, synthetic_shape=(8, 8),
        )
        results = run_benchmark(cfg)
        svm = results["architectures"]["SVM"]
        assert svm["status"] == "ok", svm
        assert svm["type"] == "classical"
        assert svm["epochs"] is None
        assert svm["fit_strategy"]["kind"] == "single_fit"
        assert "clean" in svm and "auc_roc" in svm["clean"]
        assert "20" in svm["robustness"]
        assert svm["efficiency"]["latency_ms"] is not None
        model_artifact = Path(svm["model_artifact"])
        assert model_artifact.exists()
        assert model_artifact.parent == models_dir
        # artefatos
        saved = json.loads((Path(td) / "results.json").read_text("utf-8"))
        assert saved["dataset"]["n_test"] > 0
        assert (Path(td) / "tcc_report.md").exists()


def test_svm_optimized_benchmark_reports_real_fit_strategy(monkeypatch):
    """SVM otimizado deve reportar GridSearchCV+refit, não épocas artificiais."""
    from benchmarks import BenchmarkConfig, run_benchmark

    for name in (
        "MODELS_DIR",
        "DEEPFAKE_MODELS_DIR",
        "XFAKE_MODELS_DIR",
        "XFAKE_STORAGE_DIR",
        "DEEPFAKE_STORAGE_DIR",
    ):
        monkeypatch.delenv(name, raising=False)

    with tempfile.TemporaryDirectory() as td:
        models_dir = Path(td) / "models"
        cfg = BenchmarkConfig(
            architectures=["SVM"],
            output_dir=td,
            models_dir=str(models_dir),
            dataset_path=None,
            synthetic_n=160,
            synthetic_shape=(8, 8),
            epochs=100,
            snr_levels_db=[20],
            latency_runs=1,
            optimize_hyperparameters=True,
        )
        results = run_benchmark(cfg)
        svm = results["architectures"]["SVM"]
        assert svm["status"] == "ok", svm
        assert svm["type"] == "classical"
        assert svm["epochs"] is None
        assert svm["fit_strategy"]["kind"] == "grid_search_cv_then_refit"
        assert svm["fit_strategy"]["cv"] == 3
        assert svm["fit_strategy"]["n_candidates"] == 12
        assert svm["fit_strategy"]["n_fits"] == 36
        assert svm["fit_strategy"]["final_refit"] is True
        assert svm["fit_strategy"]["total_fit_calls_estimate"] == 37

        report = (Path(td) / "tcc_report.md").read_text("utf-8")
        assert "Treino executado: `CV 36+fit`" in report
        assert "Épocas executadas: `100`" not in report
