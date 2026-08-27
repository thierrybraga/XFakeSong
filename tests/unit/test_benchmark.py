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


def test_awgn_realized_snr_matches_target_per_sample():
    from benchmarks.data import BenchmarkData

    X = np.random.default_rng(7).standard_normal((6, 4000)).astype("float32")
    noisy = BenchmarkData.add_awgn(X, snr_db=20, seed=11)
    signal_power = np.mean(X ** 2, axis=1)
    noise_power = np.mean((noisy - X) ** 2, axis=1)
    measured = 10.0 * np.log10(signal_power / noise_power)
    assert np.allclose(measured, 20.0, atol=1e-3)


def test_mixed_awgn_is_balanced_and_reproducible():
    from benchmarks.data import BenchmarkData

    X = np.random.default_rng(3).standard_normal((10, 1000)).astype("float32")
    noisy_a, assigned_a = BenchmarkData.add_awgn_mixed(
        X, [30, 20, 10], seed=42
    )
    noisy_b, assigned_b = BenchmarkData.add_awgn_mixed(
        X, [30, 20, 10], seed=42
    )
    assert np.array_equal(assigned_a, assigned_b)
    assert np.array_equal(noisy_a, noisy_b)
    counts = np.unique(assigned_a, return_counts=True)[1]
    assert counts.max() - counts.min() <= 1


def test_protocol_adds_noise_before_frontend(monkeypatch):
    from benchmarks import BenchmarkConfig, runner

    rng = np.random.default_rng(5)
    raw_splits = (
        rng.standard_normal((8, 2000)).astype("float32"),
        np.array([0, 1] * 4),
        rng.standard_normal((4, 2000)).astype("float32"),
        np.array([0, 1] * 2),
        rng.standard_normal((4, 2000)).astype("float32"),
        np.array([0, 1] * 2),
    )
    seen = []

    def fake_frontend(X, _arch, **_kwargs):
        seen.append(np.asarray(X).copy())
        return np.asarray(X)[:, :4], "tabular_audio_features"

    monkeypatch.setattr(runner, "prepare_input_for_architecture", fake_frontend)
    cfg = BenchmarkConfig(
        architectures=["SVM"],
        dataset_path="raw.npz",
        train_aug_snr_db=[30, 20, 10],
        train_noise_copies=1,
        strict_waveform_awgn=True,
    )
    prepared = runner._prepare_protocol_splits("SVM", cfg, raw_splits)

    assert len(seen) == 4  # treino/val/teste limpos + treino ruidoso
    assert seen[0].shape == seen[3].shape == raw_splits[0].shape
    assert not np.array_equal(seen[0], seen[3])
    assert prepared[7]["evaluation_domain"] == "waveform"
    assert prepared[7]["frontend_after_noise"] is True
    assert len(prepared[1]) == 2 * len(raw_splits[1])

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

    assert raw.X.shape == (20, 48000, 1)
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

    assert raw.X.shape == (8, 48000, 1)
    assert raw.metadata["prepared_shape"] == [48000, 1]


def test_prepare_raw_audio_center_crops_long_clips_for_aasist():
    """AASIST usa janela 64.600 (~4,04s, protocolo ASVspoof2021 baseline
    compartilhado com RawGAT-ST) + crop_strategy multicrop na avaliação —
    diferente do RawNet2 (1s, sem TTA). Ver registry.py::input_requirements
    e tests/unit/test_rawgat_aasist_ssl_backends.py."""
    from benchmarks.data import BenchmarkData

    rng = np.random.default_rng(34)
    d = BenchmarkData(
        X=rng.standard_normal((8, 80000, 1)).astype("float32"),
        y=np.array([0, 1] * 4),
    )
    raw = d.prepare_for_architecture("AASIST")

    assert raw.X.shape == (8, 48000, 1)
    assert raw.metadata["input_type"] == "raw_audio"
    assert raw.metadata["prepared_shape"] == [48000, 1]


def test_prepare_raw_audio_center_crops_long_clips_for_ensemble():
    from benchmarks.data import BenchmarkData

    rng = np.random.default_rng(35)
    d = BenchmarkData(
        X=rng.standard_normal((8, 80000, 1)).astype("float32"),
        y=np.array([0, 1] * 4),
    )
    raw = d.prepare_for_architecture("Ensemble")

    assert raw.X.shape == (8, 48000, 1)
    assert raw.metadata["input_type"] == "raw_audio"
    assert raw.metadata["prepared_shape"] == [48000, 1]


def test_prepare_raw_audio_center_crops_long_clips_for_wavlm():
    from benchmarks.data import BenchmarkData

    rng = np.random.default_rng(36)
    d = BenchmarkData(
        X=rng.standard_normal((8, 80000, 1)).astype("float32"),
        y=np.array([0, 1] * 4),
    )
    raw = d.prepare_for_architecture("WavLM")

    assert raw.X.shape == (8, 48000, 1)
    assert raw.metadata["input_type"] == "raw_audio"
    assert raw.metadata["prepared_shape"] == [48000, 1]


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
    assert m["accuracy_at_eer"] == 1.0
    assert "min_tdcf" in m


def test_accuracy_at_eer_separates_threshold_collapse_from_failure():
    """P2: scores ordenados mas colapsados para ~0 (caso Ensemble sob ruído)
    despencam no limiar fixo 0.5, mas `accuracy_at_eer` revela o teto real."""
    import numpy as np

    from benchmarks.evaluate import evaluate_scores

    y = np.array([0] * 50 + [1] * 50)
    # Reais em [0, 0.001), fakes em [0.002, 0.01): AUC=1.0, mas tudo < 0.5.
    rng = np.random.default_rng(0)
    p = np.concatenate([rng.uniform(0, 0.001, 50), rng.uniform(0.002, 0.01, 50)])
    m = evaluate_scores(y, p)
    assert m["auc_roc"] > 0.99           # perfeitamente separável (ranking)
    assert m["accuracy"] < 0.6           # colapsa no limiar fixo 0.5
    assert m["accuracy_at_eer"] > 0.95   # recupera no limiar ótimo
    # E quando há sobreposição genuína, os dois ficam baixos:
    p_bad = np.concatenate([rng.uniform(0.3, 0.7, 50), rng.uniform(0.3, 0.7, 50)])
    mb = evaluate_scores(y, p_bad)
    assert mb["accuracy_at_eer"] < 0.7


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
        # 9 colunas desde 2026-07-27: Acur.@EER (oráculo) entrou ao lado da
        # acurácia no limiar fixo, para separar calibração de separabilidade.
        assert "\\begin{tabular}{lcccccccc}" in resultados
        assert "\\multicolumn{8}{c}" in resultados
        assert "Acur.@EER" in resultados
        assert "limiar fixo de decisão 0,5" in resultados
        # figuras desenhadas a partir de scores/história
        assert (out / "figures" / "roc.png").exists()
        assert (out / "figures" / "convergencia.png").exists()
        assert (out / "figures" / "confusion_matrices.png").exists()
        assert (out / "figures" / "score_distributions.png").exists()
        pred_csv = (out / "predictions_clean.csv").read_text("utf-8")
        # `ranking_score` acompanha o arquivo por arquitetura: nos clássicos é
        # dele que saem AUC/EER, e o `p_fake` calibrado não os reproduz.
        assert (
            "architecture,sample_index,y_true,p_fake,ranking_score,y_pred,correct"
            in pred_csv
        )
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


def test_robustez_table_usa_colspan_quando_nenhuma_arquitetura_conclui():
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
    # A marcação `\dagger` do modelo não convergido é verificada em
    # `test_benchmark_protocol_fixes.py::
    # test_robustness_table_keeps_non_converged_models_marked`, que também
    # cobre a legenda. Aqui fica só o que é exclusivo deste caso: o colspan
    # dinâmico quando NENHUMA arquitetura conclui (2026-08-17, deduplicação).
    empty = {**fake, "architectures": {"SVM": {"status": "error", "error": "x"}}}
    with tempfile.TemporaryDirectory() as td:
        write_all(empty, td)
        tex = (Path(td) / "tables" / "tab_robustez.tex").read_text("utf-8")
        assert "\\multicolumn{5}{c}{(nenhuma arquitetura concluiu)}" in tex


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
    from scripts.benchmark.run_tcc_pipeline import _verify_outputs

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
        output_dir="data/results/path_anchor_probe",
        models_dir="data/models/path_anchor_probe",
    )

    plan_benchmark(cfg, write=False)

    assert Path(cfg.output_dir) == PROJECT_ROOT / "data" / "results" / "path_anchor_probe"
    assert Path(cfg.models_dir) == PROJECT_ROOT / "data" / "models" / "path_anchor_probe"


def test_full_tcc_preset_includes_all_architectures():
    from benchmarks.config import (
        ALL_TCC_ARCHITECTURES,
        CLASSICAL_TCC_ARCHITECTURES,
        DOCKER_TRAINING_ARCHITECTURES,
        NEURAL_TCC_ARCHITECTURES,
        SSL_DOCKER_ARCHITECTURES,
        SSL_FINETUNED_ARCHITECTURES,
        BenchmarkConfig,
    )

    cfg = BenchmarkConfig.full_tcc()

    assert cfg.architectures == ALL_TCC_ARCHITECTURES
    assert len(cfg.architectures) == 9
    assert "SpectrogramTransformer" in cfg.architectures
    assert "Spectrogram Transformer" not in cfg.architectures
    assert "WavLM Original" not in cfg.architectures
    assert "HuBERT Original" not in cfg.architectures
    assert BenchmarkConfig.full_all_architectures().architectures == cfg.architectures
    assert CLASSICAL_TCC_ARCHITECTURES == ["RandomForest", "SVM"]
    # 2026-08-11: as duas variantes com fine-tuning saíram do escopo oficial.
    # Os sistemas de topo do ASVspoof 5 usam SSL CONGELADO, e o resultado de
    # referência da receita ajustada usa wav2vec2 XLS-R, não WavLM/HuBERT base
    # — combiná-los seria abordagem nova, não benchmark de configuração
    # documentada. As entradas `Original` já são a documentada.
    assert SSL_DOCKER_ARCHITECTURES == ["WavLM Original", "HuBERT Original"]
    # Vazia por consequência, não por literal: a derivação segue no lugar para
    # que reintroduzir uma entrada `:ssl_finetuned` volte a acionar as flags.
    assert SSL_FINETUNED_ARCHITECTURES == []
    # Nenhuma variante SSL pode vazar para a lista que `benchmarks.runner`
    # tenta treinar pelo caminho Keras — era o que `endswith(":ssl_original")`
    # deixava acontecer com as variantes `:ssl_finetuned`.
    assert set(ALL_TCC_ARCHITECTURES) & set(SSL_DOCKER_ARCHITECTURES) == set()
    assert DOCKER_TRAINING_ARCHITECTURES == [
        *ALL_TCC_ARCHITECTURES,
        *SSL_DOCKER_ARCHITECTURES,
    ]
    assert len(DOCKER_TRAINING_ARCHITECTURES) == 11
    assert len(NEURAL_TCC_ARCHITECTURES) == 7
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
        # WavLM/HuBERT (Original) rodam por um runner PyTorch/transformers
        # dedicado (scripts/benchmark/run_wavlm_original_benchmark.py), fora do
        # caminho benchmarks.runner:keras que plan_benchmark cobre; e
        # EfficientNet-LSTM/Ensemble/Sonic Sleuth ficam fora do recorte
        # oficial de 11 modelos do TCC (ver ARCH_ALIASES/NEURAL_BENCHMARK_
        # HPARAMS em benchmarks/planning.py — só RawNet2, AASIST, RawGAT-ST,
        # Conformer, CCT, AST, Res2Net, SVM, RandomForest).
        cfg = BenchmarkConfig.neural_tcc(
            architectures=[
                "RawGAT-ST",
                "AASIST",
                "SpectrogramTransformer",
                "Conformer",
            ],
            output_dir=td,
            synthetic_n=24,
            synthetic_shape=(8, 8),
            epochs=7,
            device_profile="cpu",
        )
        plan = plan_benchmark(cfg, write=True)

        rawgatst = plan["architectures"]["RawGAT-ST"]["training_config"]
        aasist = plan["architectures"]["AASIST"]["training_config"]
        ast = plan["architectures"]["SpectrogramTransformer"]["training_config"]
        conformer = plan["architectures"]["Conformer"]["training_config"]

        assert plan["preset"] == "neural_tcc"
        assert rawgatst["learning_rate"] == 5e-5
        assert rawgatst["input_domain"] == "raw_audio"
        assert rawgatst["batch_size"] <= 4
        assert rawgatst["use_augmentation"] is True
        assert rawgatst["use_mixed_precision"] is False
        assert aasist["learning_rate"] == 3e-4
        assert aasist["input_domain"] == "raw_audio"
        assert aasist["batch_size"] <= 4
        # AJUSTE 2026-07-14: LR/WD reduzidos (pre-LN + 87M params do zero).
        assert ast["learning_rate"] == 1e-5
        assert ast["batch_size"] <= 8
        assert ast["l2_reg_strength"] == 1e-5
        assert ast["weight_decay"] == 1e-5
        assert ast["use_augmentation"] is False
        assert ast["warmup_steps"] == 3000
        assert ast["clipnorm"] == 1.0
        assert ast["checkpoint_best"] is True
        # Protocolo 2026-07-12: orçamento fixo de 100 épocas —
        # fixed_epoch_budget=True desliga o early stopping no controle
        # experimental comum e a seleção fica por melhor checkpoint (val).
        assert ast["early_stopping"] is False
        assert ast["select_best_checkpoint"] is True
        assert ast["early_stopping_patience"] == 20
        assert ast["epochs"] == 7
        assert ast["recommended_epochs"] == 100
        # AJUSTE 2026-08-06: o Conformer colapsou para `loss = ln 2` a partir da
        # época ~14 em duas sessões independentes. LR de pico 1e-4 -> 5e-5,
        # warmup 1500 -> 3000 passos e `decay_steps` explícito em 76.100 — que é
        # o número REAL de passos do orçamento (ceil(24.324/32) x 100 épocas).
        # Com o default de 50.000 o cosseno zerava na época ~66. Ver
        # docs/evaluation/retraining-adjustments.md, seção 2026-08-06.
        assert conformer["learning_rate"] == 5e-5
        assert conformer["optimizer"] == "AdamW"
        assert conformer["weight_decay"] == 1e-4
        assert conformer["warmup_steps"] == 3000
        assert conformer["decay_steps"] == 76100
        assert conformer["clipnorm"] == 1.0
        assert conformer["batch_size"] <= 16


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
        assert len(plan["architectures"]) == 9
        assert plan["architectures"]["AASIST"]["training_config"]["batch_size"] <= 8
        assert plan["architectures"]["SVM"]["training_config"]["model_family"] == "classical"
        assert (Path(td) / "benchmark_plan.json").exists()
        assert (Path(td) / "benchmark_plan.md").exists()


def test_all_architectures_benchmark_smoke_contract(monkeypatch):
    """CI smoke barato: valida nomes, preparo, métricas e artefatos sem treino pesado."""
    import benchmarks.runner as runner
    from benchmarks import BenchmarkConfig, run_benchmark

    def fake_run_neural(_arch, _cfg, splits, _tmp, _models_dir, **_kwargs):
        # Protocolo 2026-07-12: _prepare_protocol_splits retorna 8 itens
        # (6 arrays + clean_train_count + protocol) — mesmo fatiamento do
        # runner real (_run_neural usa splits[:6]).
        _Xtr, _ytr, _Xv, _yv, Xte, _yte = splits[:6]
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

    def fake_run_classical(arch, cfg, splits, tmp, models_dir, **kwargs):
        # **kwargs acompanha `training_seed` (repetições com sementes distintas)
        return fake_run_neural(arch, cfg, splits, tmp, models_dir, **kwargs)

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

        assert len(results["architectures"]) == 9
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


def test_conformer_benchmark_smoke_generates_model_results_and_figures(monkeypatch):
    """Smoke P0: Conformer deve treinar 1 época e materializar artefatos."""
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
        root = Path(td)
        output_dir = root / "conformer_smoke"
        models_dir = root / "models"
        cfg = BenchmarkConfig(
            architectures=["Conformer"],
            dataset_path=None,
            epochs=1,
            batch_size=16,
            seed=123,
            snr_levels_db=[20],
            latency_runs=1,
            output_dir=str(output_dir),
            models_dir=str(models_dir),
            run_api_probe=False,
            synthetic_n=24,
            synthetic_shape=(16, 16),
            optimize_hyperparameters=False,
            training_overrides={
                "Conformer": {
                    "parameters": {
                        "dropout_rate": 0.3,
                        "learning_rate": 1e-4,
                        "weight_decay": 1e-4,
                    },
                    "use_mixed_precision": False,
                }
            },
        )

        results = run_benchmark(cfg)
        conformer = results["architectures"]["Conformer"]
        assert conformer["status"] == "ok", conformer
        assert conformer["type"] == "neural"
        assert conformer["epochs"] == 1
        assert conformer["model_parameters"]["dropout_rate"] == 0.3

        # AJUSTE 2026-08-09: `model_artifact` aponta para a cópia PRESERVADA no
        # run. `models_dir` é global e chaveado só pela arquitetura — qualquer
        # execução posterior sobrescreve o arquivo de lá (foi como o
        # bench_svm.pkl do clean_benchmark_15k virou um artefato de smoke).
        model_artifact = Path(conformer["model_artifact"])
        assert model_artifact.exists()
        assert model_artifact.parent == output_dir / "architectures" / "conformer" / "models"
        shared = Path(conformer["model_artifact_shared_copy"])
        assert shared.exists() and shared.parent == models_dir
        assert conformer["model_artifact_fingerprint"]["integrity"] == "recorded_at_run"
        assert (output_dir / "results.json").exists()

        saved = json.loads((output_dir / "results.json").read_text("utf-8"))
        assert saved["architectures"]["Conformer"]["status"] == "ok"

        for figure in (
            "roc.png",
            "robustez.png",
            "convergencia.png",
            "eficiencia.png",
            "confusion_matrices.png",
            "score_distributions.png",
        ):
            path = output_dir / "figures" / figure
            assert path.exists(), figure
            assert path.stat().st_size > 0, figure


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
        # O caminho clássico não deixava NENHUMA cópia no run — o `.pkl` só
        # existia no `models_dir` global. Agora acompanha os neurais.
        model_artifact = Path(svm["model_artifact"])
        assert model_artifact.exists()
        assert model_artifact.parent == Path(td) / "architectures" / "svm" / "models"
        shared = Path(svm["model_artifact_shared_copy"])
        assert shared.exists() and shared.parent == models_dir
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
        # 5 dobras (era 3) e 24 candidatos (era 12) desde 2026-08-09: o grid
        # passou a vir de `svm.py::SVM_PARAM_GRID` — antes o runner usava uma
        # cópia própria, com o eixo `gamma` duplicado (scale ≈ auto depois do
        # StandardScaler).
        assert svm["fit_strategy"]["cv"] == 5
        # 15 candidatos: 3 C x 4 gamma no RBF + 3 C no linear. Como dicionário
        # único seriam 24, com 9 lineares redundantes (gamma não afeta linear).
        assert svm["fit_strategy"]["n_candidates"] == 15
        assert svm["fit_strategy"]["n_fits"] == 75
        assert svm["fit_strategy"]["final_refit"] is True
        assert svm["fit_strategy"]["total_fit_calls_estimate"] == 76

        report = (Path(td) / "tcc_report.md").read_text("utf-8")
        assert "Treino executado: `CV 75+fit`" in report
        assert "Épocas executadas: `100`" not in report


def test_classical_grid_comes_from_the_architecture_modules():
    """O grid do runner É o da arquitetura — não uma quarta fonte própria.

    Até 2026-08-09 `_classical_search_space` carregava uma cópia divergente e
    os grids regularizados de `svm.py`/`random_forest.py` não tinham NENHUM
    chamador: o benchmark rodava `max_depth=None` e `min_samples_leaf=1` para o
    RF, exatamente o overfitting que aqueles grids existiam para corrigir.
    """
    from app.domain.models.architectures.random_forest import (
        RANDOM_FOREST_PARAM_GRID,
    )
    from app.domain.models.architectures.svm import SVM_PARAM_GRID
    from benchmarks.runner import _classical_search_space

    svm_grid, _, svm_step = _classical_search_space("SVM", 42)
    rf_grid, _, rf_step = _classical_search_space("RandomForest", 42)
    assert (svm_step, rf_step) == ("svm", "rf")
    assert svm_grid == SVM_PARAM_GRID
    assert rf_grid == RANDOM_FOREST_PARAM_GRID

    # Regressões concretas que o grid antigo do runner tinha:
    assert None not in rf_grid["rf__max_depth"], "profundidade ilimitada de volta"
    assert 1 not in rf_grid["rf__min_samples_leaf"], "folha de 1 amostra de volta"
    assert "rf__min_samples_split" in rf_grid, "min_samples_split não explorado"
    # O grid do SVM é uma LISTA de blocos: `gamma` só cruza com o kernel RBF.
    from sklearn.model_selection import ParameterGrid

    candidatos = list(ParameterGrid(svm_grid))
    assert len(candidatos) == 15
    gammas = {c["svm__gamma"] for c in candidatos if "svm__gamma" in c}
    # 'auto' == 'scale' depois do StandardScaler (ambos ≈1/n_features): manter
    # os dois desperdiçava metade do eixo.
    assert "auto" not in gammas
    assert any(isinstance(g, float) for g in gammas)
    assert "poly" not in {c["svm__kernel"] for c in candidatos}


def test_npz_predefined_splits_are_preserved_without_duplicate_aggregate(tmp_path):
    from benchmarks.data import BenchmarkData

    train_x = np.arange(8 * 1000, dtype="float32").reshape(8, 1000)
    val_x = np.arange(8 * 1000, 12 * 1000, dtype="float32").reshape(4, 1000)
    test_x = np.arange(12 * 1000, 16 * 1000, dtype="float32").reshape(4, 1000)
    train_y = np.array([0, 1] * 4)
    val_y = np.array([0, 1] * 2)
    test_y = np.array([0, 1] * 2)
    path = tmp_path / "predefined.npz"
    np.savez(
        path,
        X_train=train_x,
        y_train=train_y,
        X_val=val_x,
        y_val=val_y,
        X_test=test_x,
        y_test=test_y,
        X=np.zeros((2, 1000), dtype="float32"),
        y=np.array([0, 1]),
    )

    data = BenchmarkData.from_npz(str(path))
    splits = data.stratified_split(seed=999, preserve_predefined=True)

    assert len(data.y) == 16
    np.testing.assert_array_equal(splits[0], train_x)
    np.testing.assert_array_equal(splits[2], val_x)
    np.testing.assert_array_equal(splits[4], test_x)


def test_split_overlap_audit_rejects_exact_contamination():
    from benchmarks.runner import _audit_split_overlap

    shared = np.ones((1, 1000), dtype="float32")
    train = np.concatenate([shared, np.zeros((1, 1000), dtype="float32")])
    val = np.full((2, 1000), 2.0, dtype="float32")
    test = np.concatenate([shared, np.full((1, 1000), 3.0, dtype="float32")])
    y = np.array([0, 1])

    with pytest.raises(ValueError, match="Contaminação entre partições"):
        _audit_split_overlap((train, y, val, y, test, y), fail_on_overlap=True)


def test_plan_preserves_model_hparams_but_forces_common_training_controls():
    from benchmarks.config import BenchmarkConfig
    from benchmarks.planning import build_benchmark_plan

    cfg = BenchmarkConfig(
        architectures=["Conformer"],
        epochs=100,
        device_profile="cpu",
        training_overrides={
            "Conformer": {
                "epochs": 17,
                "learning_rate": 7e-5,
                "dropout_rate": 0.37,
            }
        },
    )
    plan = build_benchmark_plan(cfg)
    effective = plan["architectures"]["Conformer"]["training_config"]

    assert effective["epochs"] == 100
    assert effective["early_stopping"] is False
    assert effective["select_best_checkpoint"] is True
    assert effective["validation_condition"] == "clean"
    assert effective["decision_threshold"] == 0.5
    assert effective["learning_rate"] == 7e-5
    assert effective["dropout_rate"] == 0.37


def test_multiscalecnn_treina_em_float32_na_gpu():
    """MultiscaleCNN nao pode receber mixed_float16 no perfil GPU.

    Nao e preferencia de precisao: com `mixed_float16` o processo morre de
    SIGSEGV no BACKWARD do Res2Net. Reproduzido em 2026-08-02 num repro minimo
    (log-mel 100x80, batch 32, RTX 3060) — o primeiro passo de treino completa
    e o segundo mata o processo. Foi o `returncode=-11` aos 336 s no benchmark
    de 2026-08-01, que deixou a arquitetura sem nenhum artefato.

    Isolado: float32 roda limpo, forward puro em fp16 roda limpo,
    `TF_CUDNN_USE_AUTOTUNE=0` nao muda nada.
    """
    # `_fit_to_device` recebe o dispositivo como argumento, entao o caminho de
    # GPU e testavel numa maquina sem GPU — o que importa aqui e a decisao do
    # plano, nao o hardware de quem roda a suite.
    from benchmarks.planning import _base_recommended_hparams, _fit_to_device

    gpu = {"resolved_profile": "gpu"}
    for arch in ("MultiscaleCNN", "RawNet2"):
        tuned = _fit_to_device(_base_recommended_hparams(arch), arch, gpu)
        assert tuned["use_mixed_precision"] is False, (
            f"{arch} voltou a pedir mixed precision — o treino morre de "
            f"SIGSEGV no backward"
        )


def test_aasist_treina_em_float32_na_gpu():
    """AASIST nao pode receber mixed_float16 no perfil GPU.

    Falha diferente da do MultiscaleCNN: nao e SIGSEGV, e divergencia para
    NaN. Em 2026-08-04 o AASIST morreu no batch 502 da epoca 1, TRES execucoes
    seguidas com o mesmo seed (`TerminateOnNaN`), e o `ModelCheckpoint` chegou
    a promover pesos com 384 parametros nao-finitos.

    O carve-out anterior forcava fp16 aqui com a justificativa de que "Sinc e
    logits permanecem float32 e o encoder 2D/GAT usa loss scaling automatico".
    Isso cobre so metade: o loss scaling age no BACKWARD (detecta inf/NaN no
    gradiente e pula o passo) e nao protege contra overflow no FORWARD, que e
    o risco do softmax de atencao do GAT em float16.

    A/B com mesma LR (3e-4), mesmo lote (24), mesmo seed e mesmos dados: em
    float32 a epoca 1 fecha com loss=0.669 e val_accuracy=0.709. E sem custo
    de tempo — 11,1 min contra ~12 min em fp16, entao o argumento de
    velocidade nao se aplica a esta arquitetura.
    """
    from benchmarks.planning import _base_recommended_hparams, _fit_to_device

    gpu = {"resolved_profile": "gpu"}
    tuned = _fit_to_device(_base_recommended_hparams("AASIST"), "AASIST", gpu)
    assert tuned["use_mixed_precision"] is False, (
        "AASIST voltou a pedir mixed precision — o treino diverge para NaN "
        "no batch 502 da primeira epoca"
    )


def test_arquiteturas_sem_restricao_seguem_com_mixed_precision():
    """A blocklist e cirurgica: quem nao esta nela continua usando fp16.

    Sem esta guarda, alguem "consertaria" o SIGSEGV desligando mixed precision
    para todo mundo e o benchmark inteiro ficaria ~2x mais lento sem motivo.

    AASIST saiu desta lista em 2026-08-04 — entrou na blocklist com repro
    deterministico, ver `test_aasist_treina_em_float32_na_gpu`.
    """
    from benchmarks.planning import _base_recommended_hparams, _fit_to_device

    gpu = {"resolved_profile": "gpu"}
    for arch in ("Conformer", "Hybrid CNN-Transformer"):
        tuned = _fit_to_device(_base_recommended_hparams(arch), arch, gpu)
        assert tuned["use_mixed_precision"] is True, arch
