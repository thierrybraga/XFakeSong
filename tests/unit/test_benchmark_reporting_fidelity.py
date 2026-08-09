"""Regressões das correções de 2026-08-09 (auditoria do `clean_benchmark_15k`).

Seis defeitos, todos de FIDELIDADE DO ARTEFATO — o que o JSON declara não batia
com o que o run fez. Nenhum altera métrica; todos alteram o que se pode afirmar
a partir delas.

P0 A janela do SSL era gravada como o literal `[16000, 1]` em quatro pontos
   enquanto o run usava `--target-samples` (64.000 por default), e
   `crop_strategy` dizia "center" mesmo quando `_fit_length` fazia tiling.
P0 O default de `--target-samples` (64.000) virou dívida com o dataset de 3 s:
   25% de cada entrada era repetição do próprio sinal.
P1 `converged` só olha o checkpoint selecionado — o Conformer colapsou por 84
   épocas e saiu `converged: True`.
P1 A latência mistura Keras/TF, PyTorch e sklearn sem declarar o runtime.
P1 Sem teste pareado, ICs sobrepostos eram lidos como "sem diferença".
P2 `fit_strategy` não dizia que os clássicos ajustam em treino+validação, e
   `codec_robustness: {}` não distinguia "não pedido" de "nada encontrado".

Ver docs/evaluation/retraining-adjustments.md, seção 2026-08-09.
"""

from __future__ import annotations

import ast
import inspect
import math
from pathlib import Path

import numpy as np
import pytest

from benchmarks import stability
from benchmarks.significance import (
    holm_adjust,
    mcnemar_test,
    paired_bootstrap_test,
)
from benchmarks.stability import analyze_training_stability

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_SSL_RUNNER = _PROJECT_ROOT / "scripts" / "benchmark" / "run_wavlm_original_benchmark.py"


# ─── P0. Janela do SSL ──────────────────────────────────────────────────────


def _ssl_module():
    """Importa o runner SSL sem executar o `main` (evita torch/transformers)."""
    import importlib.util

    spec = importlib.util.spec_from_file_location("_ssl_runner", _SSL_RUNNER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_input_preparation_declara_tiling_quando_a_janela_excede_o_clipe():
    """O clipe de 3 s com janela de 4 s é REPETIDO, não recortado no centro."""
    mod = _ssl_module()
    block = mod._input_preparation_block(48000, 64000)

    assert block["crop_strategy"] == "tile_repeat"
    assert block["prepared_shape"] == [64000, 1]
    assert block["tiled_padding_samples"] == 16000
    assert block["tiled_padding_ratio"] == pytest.approx(0.25)


def test_input_preparation_sem_tiling_quando_a_janela_casa_com_o_clipe():
    mod = _ssl_module()
    block = mod._input_preparation_block(48000, 48000)

    assert block["crop_strategy"] == "identity"
    assert block["prepared_shape"] == [48000, 1]
    assert "tiled_padding_samples" not in block


def test_input_preparation_recorta_o_centro_quando_a_janela_e_menor():
    mod = _ssl_module()
    assert mod._input_preparation_block(48000, 16000)["crop_strategy"] == "center_crop"


def test_fit_length_concorda_com_a_estrategia_declarada():
    """A descrição no artefato precisa bater com o que o código faz."""
    mod = _ssl_module()
    clip = np.arange(48000, dtype="float32").reshape(1, -1)

    tiled = mod._fit_length(clip, 64000)
    assert tiled.shape == (1, 64000)
    # A cauda é literalmente o começo do próprio clipe — é isso que
    # `tile_repeat` significa, e é o que "center" escondia.
    np.testing.assert_array_equal(tiled[0, 48000:], clip[0, :16000])
    assert mod._length_strategy(48000, 64000) == "tile_repeat"


def test_default_de_target_samples_casa_com_o_clipe_de_3s():
    """48.000 @16 kHz = 3 s: nem recorte, nem repetição."""
    source = _SSL_RUNNER.read_text(encoding="utf-8")
    tree = ast.parse(source)
    defaults = [
        kw.value.value
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and getattr(node.func, "attr", None) == "add_argument"
        and node.args
        and getattr(node.args[0], "value", None) == "--target-samples"
        for kw in node.keywords
        if kw.arg == "default"
    ]
    assert defaults == [48000], f"default inesperado: {defaults}"


def test_nenhum_literal_de_janela_sobrou_no_payload_do_ssl():
    """Os quatro `[16000, 1]` precisam sair — eram metadata falsa."""
    source = _SSL_RUNNER.read_text(encoding="utf-8")
    code_lines = [
        line
        for line in source.splitlines()
        if "[16000, 1]" in line and not line.lstrip().startswith("#")
    ]
    assert code_lines == [], f"literal de janela remanescente: {code_lines}"


# ─── P1. Estabilidade de treino ────────────────────────────────────────────


def _history(val_acc, val_loss=None):
    return {
        "val_accuracy": list(val_acc),
        "val_loss": list(val_loss if val_loss is not None else [1.0] * len(val_acc)),
    }


def test_colapso_do_conformer_e_detectado():
    """Padrão real: aprende, diverge na ~14 e fica no acaso até a 100."""
    val_acc = [0.55, 0.72, 0.88, 0.93] + [0.95] * 9 + [0.70, 0.52] + [0.5] * 85
    val_loss = [0.9, 0.5, 0.2, 0.15] + [0.13] * 9 + [0.4, 0.69] + [0.693] * 85
    report = analyze_training_stability(_history(val_acc, val_loss), epochs_budget=100)

    assert report["status"] == "collapsed"
    assert report["stable"] is False
    assert report["epochs_recorded"] == 100
    # O checkpoint promovido é bem anterior ao colapso — é justamente por isso
    # que `converged` não via problema nenhum.
    assert report["best_epoch"] <= 13
    assert report["collapse_epochs"] >= 15
    assert report["wasted_epoch_fraction"] > 0.5


def test_treino_saudavel_nao_e_marcado_como_colapso():
    rng = np.random.default_rng(0)
    val_acc = list(np.clip(0.90 + rng.normal(0, 0.01, 100), 0, 1))
    val_loss = list(np.clip(0.30 - np.linspace(0, 0.2, 100), 0.01, None))
    report = analyze_training_stability(_history(val_acc, val_loss), epochs_budget=100)

    assert report["status"] == "stable"
    assert report["stable"] is True


def test_inicio_lento_nunca_arma_a_guarda():
    """Warmup longo abaixo do acaso não é colapso — a guarda nem arma."""
    val_acc = [0.5] * 30 + list(np.linspace(0.55, 0.95, 70))
    report = analyze_training_stability(_history(val_acc), epochs_budget=100)

    assert report["status"] == "stable"


def test_colapso_com_recuperacao_e_distinguido_do_irreversivel():
    val_acc = [0.9] * 10 + [0.5] * 20 + [0.9] * 70
    report = analyze_training_stability(_history(val_acc), epochs_budget=100)

    assert report["status"] == "recovered_collapse"
    assert report["stable"] is True  # o orçamento ainda produziu resultado
    assert any("colapso tempor" in w for w in report["warnings"])


def test_val_loss_nao_finito_no_fim_e_divergencia():
    val_loss = [0.5] * 90 + [float("nan")] * 10
    report = analyze_training_stability(
        _history([0.9] * 100, val_loss), epochs_budget=100
    )

    assert report["status"] == "diverged_nonfinite"
    assert report["stable"] is False


def test_historico_truncado_por_retomada_vira_aviso():
    """RawNet2 gravou 17 de 100 épocas; o artefato não pode passar batido."""
    report = analyze_training_stability(_history([0.95] * 17), epochs_budget=100)

    assert report["epochs_recorded"] == 17
    assert any("17 épocas contra orçamento de 100" in w for w in report["warnings"])


def test_modelo_classico_sem_historico_nao_finge_veredito():
    report = analyze_training_stability(None)

    assert report["status"] == "unknown"
    assert report["stable"] is None


def test_criterio_pos_hoc_usa_os_mesmos_defaults_do_callback():
    """Se os dois divergirem, o artefato contradiz a guarda que rodou."""
    pytest.importorskip("tensorflow")
    from app.domain.models.training.trainer import CollapseAbort

    params = inspect.signature(CollapseAbort.__init__).parameters
    assert params["patience"].default == stability.DEFAULT_COLLAPSE_PATIENCE
    assert params["nan_patience"].default == stability.DEFAULT_NAN_PATIENCE
    assert params["arm_threshold"].default == stability.DEFAULT_ARM_THRESHOLD
    assert params["chance_accuracy"].default == stability.DEFAULT_CHANCE_ACCURACY
    assert params["tolerance"].default == stability.DEFAULT_TOLERANCE


# ─── P1. Runtime da latência ───────────────────────────────────────────────


def test_perfil_de_latencia_declara_o_runtime():
    from benchmarks.efficiency import measure_latency_profile

    profile = measure_latency_profile(
        lambda batch: batch + 1, np.zeros(8), runs=3, runtime="sklearn"
    )

    assert profile["status"] == "ok"
    assert profile["runtime"] == "sklearn"
    # A chave existe para que a legenda da figura possa ressalvar em vez de o
    # leitor supor que 17 ms de PyTorch e 53 ms de Keras estão na mesma escala.
    assert profile["cross_runtime_comparable"] is False


def test_runner_rotula_classico_e_neural_com_runtimes_diferentes():
    source = (_PROJECT_ROOT / "benchmarks" / "runner.py").read_text(encoding="utf-8")
    assert 'runtime="sklearn" if is_classical else "keras"' in source


# ─── P1. Teste pareado ─────────────────────────────────────────────────────


def test_mcnemar_ignora_acertos_e_erros_em_comum():
    """Só as discordâncias entram — é o ponto do teste pareado."""
    y = np.array([0, 1, 0, 1, 0, 1])
    a = np.array([0, 1, 0, 1, 1, 1])  # erra a amostra 4
    b = np.array([0, 1, 0, 1, 0, 0])  # erra a amostra 5

    result = mcnemar_test(y, a, b)

    assert result["discordant"] == 2
    assert result["only_a_correct"] == 1
    assert result["only_b_correct"] == 1
    assert result["p_value"] == pytest.approx(1.0)


def test_mcnemar_detecta_dominancia_sistematica():
    y = np.zeros(40, dtype=int)
    a = np.zeros(40, dtype=int)
    b = np.zeros(40, dtype=int)
    b[:12] = 1  # B erra 12 amostras que A acerta; A nunca erra sozinho

    result = mcnemar_test(y, a, b)

    assert result["only_a_correct"] == 12
    assert result["only_b_correct"] == 0
    assert result["p_value"] < 0.001


def test_mcnemar_por_cluster_reduz_o_n_efetivo():
    """8 amostras da mesma frase são 1 unidade, não 8 evidências."""
    y = np.zeros(24, dtype=int)
    a = np.zeros(24, dtype=int)
    b = np.zeros(24, dtype=int)
    b[:8] = 1
    clusters = np.repeat(["frase-1", "frase-2", "frase-3"], 8)

    por_amostra = mcnemar_test(y, a, b)
    por_cluster = mcnemar_test(y, a, b, cluster_ids=clusters)

    assert por_amostra["n_units"] == 24
    assert por_cluster["n_units"] == 3
    # Menos unidades independentes ⇒ p-valor maior. É exatamente o otimismo que
    # o bootstrap por amostra introduzia nos IC de WavLM/HuBERT.
    assert por_cluster["p_value"] > por_amostra["p_value"]


def test_bootstrap_pareado_nao_ve_diferenca_entre_modelos_iguais():
    rng = np.random.default_rng(7)
    y = np.repeat([0, 1], 60)
    scores = np.clip(y * 0.6 + rng.normal(0.2, 0.15, 120), 0, 1)

    result = paired_bootstrap_test(y, scores, scores, n_bootstrap=200)

    assert result["observed_difference"] == pytest.approx(0.0)
    assert result["significant_at_95"] is False


def test_bootstrap_pareado_separa_modelo_bom_de_ruim():
    rng = np.random.default_rng(11)
    y = np.repeat([0, 1], 120)
    bom = np.clip(y * 0.8 + rng.normal(0.1, 0.05, 240), 0, 1)
    ruim = np.clip(y * 0.15 + rng.normal(0.4, 0.30, 240), 0, 1)

    result = paired_bootstrap_test(y, bom, ruim, metric="eer", n_bootstrap=300)

    assert result["observed_difference"] < 0  # menor EER = melhor
    assert result["significant_at_95"] is True
    assert result["difference_ci95_high"] < 0


def test_bootstrap_pareado_nunca_devolve_p_igual_a_zero():
    """300 reamostragens não sustentam p=0; o +1 nas caudas evita a mentira."""
    rng = np.random.default_rng(3)
    y = np.repeat([0, 1], 100)
    bom = np.clip(y * 0.9 + rng.normal(0.05, 0.02, 200), 0, 1)
    ruim = np.clip(rng.uniform(0, 1, 200), 0, 1)

    result = paired_bootstrap_test(y, bom, ruim, n_bootstrap=300)

    assert result["p_value"] > 0.0


def test_holm_e_monotono_e_nao_ultrapassa_um():
    ajustado = holm_adjust([0.001, 0.02, 0.03, 0.5])

    assert ajustado == sorted(ajustado)
    assert all(0.0 <= p <= 1.0 for p in ajustado)
    assert ajustado[0] == pytest.approx(0.004)  # 4 x 0.001


def test_p_ajustado_nunca_fica_abaixo_do_bruto_por_arredondamento():
    """`round(2.2e-11, 6)` = 0.0 fazia o p de Holm sair MENOR que o bruto."""
    from benchmarks.significance import _round_p

    assert _round_p(7.275957614183426e-12) > 0.0
    assert _round_p(3 * 7.275957614183426e-12) > _round_p(7.275957614183426e-12)
    assert _round_p(0.5) == pytest.approx(0.5)
    assert _round_p(1.0) == 1.0


def test_holm_preserva_a_ordem_de_entrada():
    entrada = [0.5, 0.001, 0.03]
    ajustado = holm_adjust(entrada)

    assert ajustado[1] < ajustado[2] < ajustado[0]


def test_comparacao_par_a_par_cobre_todos_os_pares_e_ajusta():
    from benchmarks.significance import compare_models

    rng = np.random.default_rng(5)
    y = np.repeat([0, 1], 60)
    modelos = {
        f"m{i}": {"scores": np.clip(y * (0.9 - 0.25 * i) + rng.normal(0.1, 0.1, 120), 0, 1)}
        for i in range(3)
    }

    report = compare_models(y, modelos, n_bootstrap=100)

    assert report["protocol"]["n_comparisons"] == 3  # C(3,2)
    assert report["protocol"]["multiplicity_correction"] == "holm"
    for pair in report["pairs"]:
        assert pair["mcnemar"]["p_value_holm"] >= pair["mcnemar"]["p_value"]


# ─── P2. Declarações de protocolo ──────────────────────────────────────────


def test_classico_declara_que_ajusta_em_treino_mais_validacao():
    source = (_PROJECT_ROOT / "benchmarks" / "runner.py").read_text(encoding="utf-8")
    assert '"fit_splits": ["train", "val"] if len(yv) else ["train"]' in source
    assert '"fit_splits": ["train"],' in source  # contraparte neural


def test_codec_robustness_distingue_nao_pedido_de_vazio():
    source = (_PROJECT_ROOT / "benchmarks" / "runner.py").read_text(encoding="utf-8")
    assert '"status": "not_requested" if not codecs else "ok"' in source
    assert '"codec_eval_status": codec_eval_status,' in source


def test_cluster_ids_do_teste_sao_persistidos_para_o_teste_pareado():
    runner = (_PROJECT_ROOT / "benchmarks" / "runner.py").read_text(encoding="utf-8")
    ssl = _SSL_RUNNER.read_text(encoding="utf-8")
    assert '"test_cluster_ids"' in runner
    assert '"test_cluster_ids"' in ssl


def test_comparacao_pareada_recusa_conjuntos_de_teste_diferentes():
    """Misturar o run de 15k com o de 40k produziria saida com cara de valida."""
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "_consolidate",
        _PROJECT_ROOT / "scripts" / "reporting" / "consolidate_results.py",
    )
    consolidate = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(consolidate)

    rows = [
        {"model": "A", "slug": "a", "dataset": {"test_split_sha256": "aaa"}},
        {"model": "B", "slug": "b", "dataset": {"test_split_sha256": "bbb"}},
    ]
    extras = {
        "a": {"scores_clean": [0.1, 0.9], "y_test": [0, 1]},
        "b": {"scores_clean": [0.2, 0.8], "y_test": [0, 1]},
    }

    report = consolidate.build_significance_report(rows, extras, n_bootstrap=10)

    assert report["status"] == "skipped"
    assert "conjuntos de teste diferentes" in report["reason"]
    assert report["test_split_sha256"] == ["aaa", "bbb"]


def test_comparacao_pareada_roda_com_fingerprint_unico():
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "_consolidate2",
        _PROJECT_ROOT / "scripts" / "reporting" / "consolidate_results.py",
    )
    consolidate = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(consolidate)

    rng = np.random.default_rng(2)
    y = np.repeat([0, 1], 40).tolist()
    rows = [
        {
            "model": f"M{i}",
            "slug": f"m{i}",
            "decision_threshold": 0.5,
            "dataset": {"test_split_sha256": "mesmo"},
        }
        for i in range(2)
    ]
    extras = {
        f"m{i}": {
            "scores_clean": np.clip(
                np.asarray(y) * (0.8 - 0.3 * i) + rng.normal(0.1, 0.1, 80), 0, 1
            ).tolist(),
            "y_test": y,
        }
        for i in range(2)
    }

    report = consolidate.build_significance_report(rows, extras, n_bootstrap=50)

    assert report.get("status") != "skipped"
    assert report["protocol"]["n_comparisons"] == 1
    assert report["protocol"]["warning"]  # sem cluster_ids: declara o otimismo


def test_binomial_exato_bate_com_a_referencia():
    """Sanidade numérica do p-valor que sustenta o McNemar."""
    from benchmarks.significance import _binom_two_sided_p

    assert _binom_two_sided_p(0, 0) == 1.0
    assert _binom_two_sided_p(5, 10) == pytest.approx(1.0)
    assert _binom_two_sided_p(0, 10) == pytest.approx(2 / 1024)
    assert _binom_two_sided_p(1, 10) == pytest.approx(2 * 11 / 1024)
    assert not math.isnan(_binom_two_sided_p(3, 7))
