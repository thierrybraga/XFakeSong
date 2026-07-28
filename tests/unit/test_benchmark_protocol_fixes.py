"""Correções de rigor do pipeline de benchmark (revisão de 2026-07-27).

Cada teste amarra um achado da revisão técnico-acadêmica:

1. multicrop de avaliação valia só para AASIST/RawGAT-ST — test-time
   augmentation assimétrico dentro da tabela comparativa principal;
2. o limiar calibrado vinha de scores COM temperatura e era aplicado a scores
   SEM temperatura;
3. a tabela de robustez descartava arquiteturas por um critério medido no
   próprio conjunto de teste;
4. as tabelas publicavam acurácia@0,5 sem dizer, e `accuracy_at_eer_oracle`
   nunca chegava a artefato de publicação;
5. figuras/CSVs vêm de uma semente enquanto as tabelas mostram a média;
6. o relatório do TCC não carregava commit, checkpoints nem o selo do teste;
8. a disjunção de sementes de ruído treino↔avaliação era um comentário, não
   uma verificação;
9. um `early_stopping` explícito era sobrescrito em silêncio pelo plano.
"""

from __future__ import annotations

import numpy as np
import pytest

from benchmarks.config import BenchmarkConfig
from benchmarks.planning import _merge_effective_hparams
from benchmarks.report import (
    _bootstrap_unit,
    _provenance_lines,
    _table_resultados,
    _table_robustez,
    _test_lock_lines,
)

# ─────────────────────── 1. multicrop simétrico ───────────────────────

def test_multicrop_is_uniform_across_raw_audio_architectures():
    """A decisão não pode depender do NOME da arquitetura.

    Média sobre 3 crops reduz a variância do score: aplicá-la só a
    AASIST/RawGAT-ST dava a esses dois uma vantagem de avaliação sobre
    RawNet2/WavLM/HuBERT na mesma tabela.
    """
    import inspect

    from benchmarks import runner

    source = inspect.getsource(runner._benchmark_one)
    decision = [
        line for line in source.splitlines() if "use_multicrop =" in line
    ]
    assert decision, "linha de decisão do multicrop não encontrada"
    joined = " ".join(decision)
    for arch_name in ("aasist", "rawgatst"):
        assert arch_name not in joined, (
            "o multicrop voltou a depender do nome da arquitetura: "
            f"{joined.strip()}"
        )
    assert 'input_type") == "raw_audio"' in joined


def test_protocol_does_not_claim_multicrop_before_deciding():
    """O bloco de protocolo declarava multicrop até para quem usou crop central."""
    import inspect

    from benchmarks import runner

    source = inspect.getsource(runner._prepare_protocol_splits)
    assert '"eval_crop_strategy": "resolved_at_eval"' in source


# ─────────────────── 2. temperatura na escala do score ───────────────────

def test_calibrated_threshold_and_scores_share_the_same_scale():
    """T é aplicada na avaliação, como no limiar derivado em validação."""
    import inspect

    from benchmarks import runner

    source = inspect.getsource(runner._run_neural)
    assert "_apply_temperature" in source
    assert 'input_contract.get("temperature")' in source
    # e a temperatura efetiva precisa ficar registrada no artefato
    assert '"calibrated_temperature"' in source


def test_temperature_scaling_is_monotonic_so_eer_is_preserved():
    """A correção não pode alterar EER/AUC — só o ponto de operação."""
    from app.domain.services.detection.predictor import apply_temperature_scaling

    rng = np.random.default_rng(0)
    logits = rng.normal(size=(200, 2))
    probs = np.exp(logits) / np.exp(logits).sum(axis=1, keepdims=True)
    scaled = np.asarray(apply_temperature_scaling(probs, 2.0))

    order_before = np.argsort(probs[:, 1])
    order_after = np.argsort(scaled[:, 1])
    np.testing.assert_array_equal(order_before, order_after)
    # ... mas o ponto de corte 0,5 muda de lugar: é exatamente por isso que
    # aplicar o limiar na escala errada media outro ponto de operação.
    assert not np.allclose(probs[:, 1], scaled[:, 1])


# ─────────── 3 e 4. tabelas: seleção e política de limiar ───────────

def _results_fixture(converged_second: bool = False) -> dict:
    def block(acc, eer):
        return {
            "accuracy": acc,
            "accuracy_at_eer_oracle": 0.99,
            "eer": eer,
            "auc_roc": 0.99,
            "min_tdcf": 0.1,
            "bootstrap_unit": "cluster",
        }

    return {
        "config": {
            "snr_levels_db": [20],
            "bootstrap_ci_samples": 1000,
            "converge_auc_threshold": 0.60,
            "converge_accuracy_threshold": 0.55,
            "decision_threshold": 0.5,
        },
        "dataset": {"n_test": 100},
        "architectures": {
            "AASIST": {
                "status": "ok",
                "converged": True,
                "clean": block(0.95, 0.04),
                "robustness": {"20": block(0.90, 0.08)},
                "efficiency": {"latency_ms": 10.0, "params": 1000},
                "epochs": 100,
            },
            "Descalibrado": {
                "status": "ok",
                "converged": converged_second,
                "clean": block(0.50, 0.00),
                "robustness": {"20": block(0.50, 0.00)},
                "efficiency": {"latency_ms": 11.0, "params": 2000},
                "epochs": 100,
            },
        },
    }


def test_robustness_table_keeps_non_converged_models_marked():
    """Modelo separável mas descalibrado sumia da tabela sem nota alguma."""
    table = _table_robustez(_results_fixture(converged_second=False))
    assert "Descalibrado" in table, "arquitetura removida da tabela de robustez"
    assert "\\dagger" in table
    assert "próprio conjunto de teste" in table


def test_results_table_declares_threshold_policy_and_oracle_column():
    table = _table_resultados(_results_fixture())
    assert "Acur.@EER" in table
    assert "limiar fixo de decisão 0,5" in table
    assert "ORÁCULO" in table
    # o número do oráculo precisa realmente sair na linha
    assert "99,00" in table


def test_bootstrap_unit_reaches_the_caption():
    """"IC 95% bootstrap" sozinho não distingue cluster de amostra."""
    results = _results_fixture()
    assert _bootstrap_unit(results) == "cluster"
    assert "CLUSTERS" in _table_resultados(results)


# ──────────────── 5 e 6. proveniência no relatório ────────────────

def test_provenance_lines_carry_commit_and_checkpoints():
    lines = _provenance_lines(
        {
            "git": {"available": True, "commit_short": "abc123def456", "dirty": True},
            "pretrained_checkpoints": {"WavLM": "microsoft/wavlm-base"},
            "libraries": {"tensorflow": "2.21.0"},
        }
    )
    rendered = "\n".join(lines)
    assert "abc123def456" in rendered
    assert "SUJA" in rendered, "árvore suja precisa ser declarada"
    assert "microsoft/wavlm-base" in rendered
    assert "tensorflow 2.21.0" in rendered


def test_test_lock_absence_is_stated_not_omitted():
    assert "não verificado" in "\n".join(_test_lock_lines({}))
    validated = _test_lock_lines(
        {"test_lock": {"lock_path": "x.json", "dataset_sha256": "deadbeef"}}
    )
    assert "validado" in "\n".join(validated)
    assert "deadbeef" in "\n".join(validated)


def test_scores_seed_is_recorded_when_repeating():
    from benchmarks.runner import _aggregate_seed_runs

    runs = [
        {
            "status": "ok",
            "training_seed": seed,
            "clean": {"eer": eer, "accuracy": 0.9},
            "robustness": {},
            "wall_time_s": 1.0,
        }
        for seed, eer in ((42, 0.1), (43, 0.2))
    ]
    agg = _aggregate_seed_runs(runs)
    # as figuras/CSVs vêm desta semente, e o artefato precisa dizer qual é
    assert agg["scores_seed"] == 42


# ──────── SNR não visto: casada vs. generalização ────────

def test_default_protocol_includes_an_unseen_snr_level():
    """Sem um nível fora do treino, robustez e memorização se confundem."""
    cfg = BenchmarkConfig()
    trained = set(cfg.train_aug_snr_db)
    evaluated = set(cfg.snr_levels_db)
    assert evaluated - trained, (
        "todo SNR avaliado está no augmentation de treino: a tabela de "
        "robustez só mede condição casada"
    )
    assert trained <= evaluated, "os níveis casados devem continuar avaliados"
    # o preset acadêmico herda o mesmo protocolo
    assert set(BenchmarkConfig.full_tcc().snr_levels_db) - trained


def test_robustness_table_separates_matched_from_unseen():
    from benchmarks.report import _unseen_snr_levels

    results = _results_fixture()
    results["config"]["snr_levels_db"] = [20, 5]
    for r in results["architectures"].values():
        r["robustness"] = {
            "20": {"accuracy": 0.9, "eer": 0.08, "noise_condition": "matched"},
            "5": {"accuracy": 0.6, "eer": 0.30, "noise_condition": "unseen"},
        }

    assert _unseen_snr_levels(results) == {5}
    table = _table_robustez(results)
    assert "SNR 5\\,dB$^{*}$" in table
    assert "SNR 20\\,dB}" in table, "o nível casado não pode receber asterisco"
    assert "NÃO VISTO no treino" in table
    assert "generalização" in table


def test_unseen_levels_fall_back_to_config_for_old_results():
    from benchmarks.report import _unseen_snr_levels

    legacy = {
        "config": {"snr_levels_db": [30, 20, 10, 5], "train_aug_snr_db": [30, 20, 10]},
        "architectures": {"A": {"status": "ok", "robustness": {"5": {"eer": 0.3}}}},
    }
    assert _unseen_snr_levels(legacy) == {5}


# ──────────── 7. selo do teste nos DOIS entrypoints ────────────

def test_direct_entrypoint_can_verify_the_test_lock():
    """O selo era conferido só pelo orquestrador sequencial.

    `scripts/benchmark/run_benchmark.py` é o caminho documentado para `--full`
    e para modelo isolado: gravava o SHA da partição nos resultados e nunca o
    conferia contra o selo.
    """
    source = (
        __import__("pathlib")
        .Path(__file__)
        .resolve()
        .parents[2]
        .joinpath("scripts/benchmark/run_benchmark.py")
        .read_text(encoding="utf-8")
    )
    assert '"--test-lock"' in source
    assert "validate_dataset_against_lock" in source


def test_shared_test_lock_rejects_a_mutated_dataset(tmp_path):
    import json

    from benchmarks.test_lock import (
        TestLockError,
        inspect_npz,
        sha256_file,
        validate_dataset_against_lock,
    )

    dataset = tmp_path / "ds.npz"
    rng = np.random.default_rng(3)
    arrays = {
        "X_train": rng.normal(size=(4, 32)).astype("float32"),
        "y_train": np.array([0, 1, 0, 1]),
        "X_val": rng.normal(size=(2, 32)).astype("float32"),
        "y_val": np.array([0, 1]),
        "X_test": rng.normal(size=(2, 32)).astype("float32"),
        "y_test": np.array([0, 1]),
    }
    np.savez_compressed(dataset, **arrays)
    lock = tmp_path / "lock.json"
    lock.write_text(
        json.dumps(
            {
                "dataset_size_bytes": dataset.stat().st_size,
                "dataset_sha256": sha256_file(dataset),
                "test_archive_identity_sha256": inspect_npz(dataset)[
                    "test_archive_identity_sha256"
                ],
                "declared_untouched": True,
                "created_before_training": True,
            }
        ),
        encoding="utf-8",
    )
    assert validate_dataset_against_lock(dataset, lock)["validated"] is True

    # regravar o NPZ (mesmo com os mesmos dados) precisa invalidar o selo
    arrays["X_test"] = rng.normal(size=(2, 32)).astype("float32")
    np.savez_compressed(dataset, **arrays)
    with pytest.raises(TestLockError):
        validate_dataset_against_lock(dataset, lock)


# ──────────── 8. disjunção de sementes verificada ────────────

def test_train_and_eval_noise_seeds_cannot_collide_silently():
    """A garantia era um comentário com uma fórmula que não existia mais."""
    from benchmarks.runner import _prepare_protocol_splits

    rng = np.random.default_rng(0)
    n = 8
    waveform = rng.normal(0, 0.05, (n, 16000)).astype("float32")
    y = np.array([0, 1] * (n // 2))
    raw_splits = (waveform, y, waveform, y, waveform, y)

    cfg = BenchmarkConfig(
        architectures=["SVM"],
        dataset_path=None,
        snr_levels_db=[20],
        train_aug_snr_db=[20],
        waveform_noise_batch_size=4,
        seed=42,
    )
    # colisão forçada: com 8 amostras e batch 4 o ruído de treino usa
    # 42+10000+start para start em {0, 4}. Um nível de avaliação que caia
    # nesse mesmo espaço (42+20000-10000 == 42+10000+0) precisa ABORTAR.
    cfg.snr_levels_db = [-10000]
    with pytest.raises(ValueError, match="colisão de sementes"):
        _prepare_protocol_splits("SVM", cfg, raw_splits, training_seed=42)


def test_protocol_records_the_disjointness_check():
    from benchmarks.runner import _prepare_protocol_splits

    rng = np.random.default_rng(0)
    waveform = rng.normal(0, 0.05, (8, 16000)).astype("float32")
    y = np.array([0, 1] * 4)
    cfg = BenchmarkConfig(
        architectures=["SVM"], dataset_path=None, snr_levels_db=[20], seed=42
    )
    splits = _prepare_protocol_splits(
        "SVM", cfg, (waveform, y, waveform, y, waveform, y), training_seed=42
    )
    assert splits[7]["train_eval_noise_seeds_disjoint"] is True


# ──────────── 9. override explícito de early stopping ────────────

def test_explicit_early_stopping_override_survives_the_plan():
    cfg = BenchmarkConfig(architectures=["AASIST"], fixed_epoch_budget=False)
    cfg.training_overrides = {"AASIST": {"early_stopping": False}}
    params = _merge_effective_hparams(cfg, "AASIST", {"resolved_profile": "cpu"})
    assert params["early_stopping"] is False, (
        "--no-early-stopping era descartado quando fixed_epoch_budget=False"
    )

    # sem override, o orçamento fixo continua mandando
    cfg.training_overrides = {}
    params = _merge_effective_hparams(cfg, "AASIST", {"resolved_profile": "cpu"})
    assert params["early_stopping"] is True


# ──── paridade treino↔producao: o modelo do benchmark E o de producao ────

def test_every_benchmark_input_type_maps_to_a_frontend():
    """Sem `feature_frontend` no contrato, a inferencia usa OUTRO front-end."""
    from app.domain.features.benchmark_frontend import (
        BENCHMARK_FRONTENDS,
        frontend_for_input_type,
    )

    # tipos que `prepare_input_for_architecture` realmente produz
    for input_type in (
        "raw_audio", "spectrogram", "tabular",
        "tabular_audio_features", "tabular_flattened",
    ):
        assert frontend_for_input_type(input_type) in BENCHMARK_FRONTENDS

    # tipo desconhecido NAO pode alegar paridade
    assert frontend_for_input_type("unchanged") is None
    assert frontend_for_input_type(None) is None


def test_runner_stamps_the_frontend_on_the_sidecar(tmp_path):
    import json as _json

    from benchmarks.runner import _stamp_benchmark_frontend

    sidecar = tmp_path / "bench_x_config.json"
    sidecar.write_text(
        _json.dumps({"architecture": "X", "input_contract": {"temperature": 1.7}}),
        encoding="utf-8",
    )
    protocol = {
        "input_type": "spectrogram",
        "original_shape": [48000, 1],
        "prepared_shape": [100, 80],
        "train_crop_strategy": "random",
        "eval_crop_strategy": "multicrop",
    }
    contract = _stamp_benchmark_frontend(sidecar, {"temperature": 1.7}, protocol)

    assert contract["feature_frontend"] == "benchmark_logmel_v1"
    assert contract["time_steps"] == 100 and contract["feature_dim"] == 80
    assert contract["source_samples"] == 48000
    # a temperatura calibrada nao pode ser perdida no carimbo
    assert contract["temperature"] == 1.7
    # e precisa ficar GRAVADO, nao so devolvido
    gravado = _json.loads(sidecar.read_text(encoding="utf-8"))["input_contract"]
    assert gravado["feature_frontend"] == "benchmark_logmel_v1"


def test_unknown_input_type_does_not_claim_parity(tmp_path):
    from benchmarks.runner import _stamp_benchmark_frontend

    contract = _stamp_benchmark_frontend(
        tmp_path / "ausente.json", {"a": 1}, {"input_type": "unchanged"}
    )
    assert "feature_frontend" not in contract


def test_classical_models_get_a_contract_with_validation_threshold(tmp_path):
    """SVM/RF salvavam so o .pkl: iam para producao sem contrato nenhum."""
    import json as _json

    from benchmarks.runner import _classical_input_contract

    rng = np.random.default_rng(0)
    Xv = rng.normal(size=(40, 63)).astype("float32")
    yv = np.array([0, 1] * 20)

    def predict(X):
        # separavel: o limiar de EER tem de existir e ser finito
        return np.where(np.arange(len(X)) % 2 == 1, 0.8, 0.2)

    contract = _classical_input_contract(
        "SVM", tmp_path, "bench_svm", 63,
        {"original_shape": [48000, 1]}, predict, Xv, yv,
    )
    assert contract["feature_frontend"] == "benchmark_tabular_v1"
    assert contract["feature_dim"] == 63
    assert contract["normalization"] == "pipeline_interno"
    assert contract["threshold_source"] == "validation_eer"
    assert np.isfinite(contract["eer_threshold"])
    # sidecar gravado em disco, como o dos neurais
    gravado = _json.loads(
        (tmp_path / "bench_svm_config.json").read_text(encoding="utf-8")
    )
    assert gravado["input_contract"]["feature_frontend"] == "benchmark_tabular_v1"


def test_contract_rebuilder_covers_every_promoted_architecture():
    from scripts.reporting.rebuild_inference_contracts import ARCH_SPECS

    for esperado in (
        "sonic_sleuth", "efficientnet_lstm", "ensemble", "wavlm", "hubert",
    ):
        assert esperado in ARCH_SPECS, f"{esperado} iria para producao sem frontend"


def test_contract_rebuilder_preserves_calibrated_temperature():
    from scripts.reporting.rebuild_inference_contracts import build_contract

    spec = {"frontend": "benchmark_raw_v1", "architecture": "AASIST",
            "model_type": "tensorflow"}
    c = build_contract("aasist", spec, {"input_shape": [48000, 1]}, 0.04, 0.79, 1.83)
    assert c["temperature"] == 1.83, "regenerar o sidecar descartava a calibracao"
    # e a janela de fallback precisa ser a canonica
    c2 = build_contract("aasist", spec, {}, None, None)
    assert c2["input_shape"] == [48000, 1]
