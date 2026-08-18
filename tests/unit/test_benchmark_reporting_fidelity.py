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
import json
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


# ─── Oscilação de treino (2026-08-09) ──────────────────────────────────────
#
# Séries `val_accuracy` REAIS do `clean_benchmark_15k`, copiadas do
# `history` dos artefatos. São a razão de o critério existir e os únicos dois
# casos que decidem os limiares: o RawGAT-ST tem de disparar e o AASIST — o
# segundo mais irregular do run — não. Sintetizar séries aqui deixaria os
# limiares livres para derivar sem ninguém notar.

_RAWGAT_ST_VAL_ACC = [
    0.8187, 0.8111, 0.7589, 0.7727, 0.8166, 0.7734, 0.8372, 0.8468, 0.8400,
    0.6168, 0.8523, 0.6889, 0.8043, 0.7328, 0.8640, 0.7205, 0.8242, 0.8729,
    0.7788, 0.8558, 0.8880, 0.8929, 0.8647, 0.7747, 0.8551, 0.8887, 0.8544,
    0.8743, 0.7720, 0.7850, 0.8427, 0.8551, 0.7630, 0.8633, 0.8152, 0.7424,
    0.8860, 0.8743, 0.8922, 0.8585, 0.8544, 0.8242, 0.8640, 0.8771, 0.8255,
    0.8640, 0.7864, 0.8805, 0.8434, 0.8111, 0.8963, 0.8166, 0.8681, 0.8681,
    0.8723, 0.8448, 0.8901, 0.8276, 0.8324, 0.8908, 0.8599, 0.8839, 0.8606,
    0.8441, 0.8915, 0.8716, 0.8530, 0.8771, 0.8819, 0.8661, 0.8379, 0.8201,
    0.8805, 0.8551, 0.8743, 0.8640, 0.8496, 0.8592, 0.8997, 0.8496, 0.8777,
    0.8571, 0.7658, 0.8448, 0.8880, 0.8578, 0.8551, 0.8764, 0.8826, 0.8489,
    0.8812,
]

_AASIST_VAL_ACC = [
    0.6978, 0.5316, 0.7953, 0.7232, 0.8620, 0.9114, 0.7493, 0.8736, 0.9052,
    0.9018, 0.8640, 0.9066, 0.9087, 0.9272, 0.9306, 0.9141, 0.9265, 0.9059,
    0.9533, 0.9464, 0.9354, 0.9444, 0.9217, 0.9327, 0.9320, 0.9251, 0.8963,
    0.9183, 0.8984, 0.8935, 0.8558, 0.8777, 0.8757, 0.8970, 0.9512, 0.9423,
    0.9354, 0.9299, 0.9492, 0.8269, 0.9087, 0.9334, 0.9402, 0.8262, 0.9547,
    0.9052, 0.9190, 0.9334, 0.9100, 0.9100, 0.9093, 0.9663, 0.9320, 0.9245,
    0.9492, 0.9258, 0.9588, 0.9293, 0.9389, 0.9245, 0.9368, 0.9588, 0.9375,
    0.9554, 0.9602, 0.9519, 0.9451, 0.9560, 0.9567, 0.9402, 0.9375, 0.9389,
    0.9183, 0.9299, 0.9348, 0.9416, 0.9389, 0.9499, 0.9492, 0.9389, 0.9423,
    0.9430, 0.9368, 0.9293, 0.9437, 0.9382, 0.9423, 0.9457, 0.9437, 0.9464,
    0.9512, 0.9464, 0.9485, 0.9457, 0.9471, 0.9457, 0.9478, 0.9464, 0.9451,
    0.9464,
]


def test_serie_real_do_rawgat_st_dispara_oscilacao():
    """Nunca cai ao acaso, então `collapsed` não pega — e saía como `stable`."""
    report = analyze_training_stability(
        _history(_RAWGAT_ST_VAL_ACC), epochs_budget=100
    )

    assert report["status"] == "unstable_oscillation"
    # Continua utilizável: o que muda é o artefato deixar de omitir o problema.
    assert report["stable"] is True
    assert report["max_epoch_drop"] == pytest.approx(0.2232, abs=1e-3)
    assert report["monitor_std_tail"] == pytest.approx(0.0274, abs=1e-3)
    assert any("oscila" in w for w in report["warnings"])


def test_serie_real_do_aasist_nao_dispara_oscilacao():
    """O segundo mais irregular do run fica de fora — o limiar não é frouxo."""
    report = analyze_training_stability(_history(_AASIST_VAL_ACC), epochs_budget=100)

    assert report["status"] == "stable"
    assert report["max_epoch_drop"] == pytest.approx(0.1662, abs=1e-3)
    assert report["monitor_std_tail"] == pytest.approx(0.0109, abs=1e-3)


def test_oscilacao_exige_os_dois_sinais():
    """Uma queda grande isolada não basta; o resto da série é plano."""
    val_acc = [0.95] * 40 + [0.70] + [0.95] * 59
    report = analyze_training_stability(_history(val_acc), epochs_budget=100)

    assert report["max_epoch_drop"] == pytest.approx(0.25, abs=1e-6)
    assert report["status"] == "stable"


def test_colapso_tem_precedencia_sobre_oscilacao():
    """O Conformer tem queda de 0,2761, mas o veredito certo é `collapsed`."""
    val_acc = [0.55, 0.72, 0.88, 0.93] + [0.95] * 9 + [0.70, 0.52] + [0.5] * 85
    report = analyze_training_stability(_history(val_acc), epochs_budget=100)

    assert report["status"] == "collapsed"
    # A cauda travada em 0,5 tem desvio zero — não é oscilação, é morte.
    assert report["monitor_std_tail"] == pytest.approx(0.0, abs=1e-9)


# ─── Custo do critério de seleção de checkpoint ────────────────────────────


def test_selection_gap_expoe_o_custo_no_padrao_do_rawgat_st():
    """Menor val_loss cai na época 8; o pico do monitor está na 79."""
    val_loss = [1.0] * len(_RAWGAT_ST_VAL_ACC)
    val_loss[7] = 0.5108  # mínimo real da série
    report = analyze_training_stability(
        _history(_RAWGAT_ST_VAL_ACC, val_loss), epochs_budget=100
    )

    assert report["best_epoch_by_val_loss"] == 8
    assert report["best_epoch_by_monitor"] == 79
    # 0,8468 na época escolhida contra 0,8997 no pico: -5,3 pp.
    assert report["selection_gap"] == pytest.approx(-0.0529, abs=1e-3)


def test_selection_gap_e_zero_quando_a_selecao_acerta_o_pico():
    val_acc = [0.80, 0.85, 0.95, 0.90]
    val_loss = [0.9, 0.6, 0.2, 0.5]
    report = analyze_training_stability(_history(val_acc, val_loss))

    assert report["best_epoch_by_val_loss"] == report["best_epoch_by_monitor"] == 3
    assert report["selection_gap"] == pytest.approx(0.0, abs=1e-9)


def test_selection_gap_nao_muda_a_selecao_efetiva():
    """A instrumentação é descritiva: quem escolhe continua sendo val_loss."""
    source = (
        _PROJECT_ROOT / "app" / "domain" / "models" / "training" / "trainer.py"
    ).read_text(encoding="utf-8")
    assert 'monitor="val_loss"' in source
    assert "best_epoch_by_monitor" not in source


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
    """8 discordâncias concentradas numa frase não são 8 evidências."""
    y = np.zeros(24, dtype=int)
    a = np.zeros(24, dtype=int)
    b = np.zeros(24, dtype=int)
    b[:8] = 1
    clusters = np.repeat(["frase-1", "frase-2", "frase-3"], 8)

    por_amostra = mcnemar_test(y, a, b)
    por_cluster = mcnemar_test(y, a, b, cluster_ids=clusters)

    assert por_amostra["n_units"] == 24
    assert por_cluster["n_units"] == 3
    assert por_cluster["n_samples"] == 24
    # As contagens de discordância continuam por amostra nos dois — o que muda
    # é a unidade de reamostragem do p-valor.
    assert por_cluster["only_a_correct"] == por_amostra["only_a_correct"] == 8
    # Menos unidades independentes ⇒ p-valor maior. É o otimismo que a binomial
    # exata sobre amostras correlacionadas introduzia.
    assert por_cluster["p_value"] > por_amostra["p_value"]


def test_mcnemar_por_cluster_nao_perde_discordancia_difusa():
    """Regressão: a agregação por MAIORIA zerava discordância espalhada.

    Cenário real: HuBERT sob duas janelas difere 3,9 pp de EER, mas os erros se
    espalham (2 de 8 amostras por frase). Votando maioria, nenhum cluster vira,
    o teste reportava `discordant=0, p=1` — indistinguível de "sem diferença" —
    enquanto o bootstrap pareado nos scores dava p < 0,001.
    """
    n_clusters, por_cluster_n = 40, 8
    y = np.zeros(n_clusters * por_cluster_n, dtype=int)
    a = np.zeros_like(y)
    b = np.zeros_like(y)
    clusters = np.repeat([f"frase-{i}" for i in range(n_clusters)], por_cluster_n)
    # 2 de cada 8 — minoria dentro de todo cluster, mas 25% do conjunto.
    for i in range(n_clusters):
        b[i * por_cluster_n : i * por_cluster_n + 2] = 1

    result = mcnemar_test(y, a, b, cluster_ids=clusters)

    assert result["discordant"] == 2 * n_clusters
    assert result["only_a_correct"] == 2 * n_clusters
    assert result["p_value"] < 0.01
    assert result["test"] == "mcnemar_cluster_bootstrap"


def test_mcnemar_com_um_unico_cluster_declara_a_limitacao():
    y = np.zeros(10, dtype=int)
    a = np.zeros(10, dtype=int)
    b = np.zeros(10, dtype=int)
    b[:4] = 1

    result = mcnemar_test(y, a, b, cluster_ids=np.repeat(["unica"], 10))

    assert result["unit"] == "sample"
    assert "cluster_warning" in result


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


def test_piso_do_p_valor_e_declarado():
    """2/(n+1) é o menor p que n reamostragens conseguem expressar."""
    rng = np.random.default_rng(13)
    y = np.repeat([0, 1], 100)
    bom = np.clip(y * 0.9 + rng.normal(0.05, 0.02, 200), 0, 1)
    ruim = np.clip(rng.uniform(0, 1, 200), 0, 1)

    r = paired_bootstrap_test(y, bom, ruim, n_bootstrap=500)

    assert r["p_value_floor"] == pytest.approx(2 / 501, rel=1e-3)
    assert r["p_value_at_floor"] is True
    assert r["p_value"] >= r["p_value_floor"]


def test_ic_da_diferenca_continua_informativo_quando_o_p_satura():
    """Regressão: o IC excluía zero e o veredito dizia "indistinguíveis".

    Com 55 comparações e 1.000 reamostragens, Holm multiplica o piso 0,002 por
    55 e trava todo p ajustado em 0,11 — nenhum par pode passar, quaisquer que
    sejam os dados. O IC não tem esse teto e é quem decide.
    """
    rng = np.random.default_rng(17)
    y = np.repeat([0, 1], 150)
    bom = np.clip(y * 0.85 + rng.normal(0.05, 0.03, 300), 0, 1)
    ruim = np.clip(y * 0.25 + rng.normal(0.35, 0.25, 300), 0, 1)

    r = paired_bootstrap_test(y, bom, ruim, n_bootstrap=500)

    assert r["p_value_at_floor"] is True
    assert r["significant_at_95"] is True  # o IC separa
    assert r["difference_ci95_high"] < 0


def test_compare_models_avisa_quando_holm_nao_consegue_resolver():
    from benchmarks.significance import compare_models

    rng = np.random.default_rng(19)
    y = np.repeat([0, 1], 50)
    modelos = {
        f"m{i}": {"scores": np.clip(y * (0.9 - 0.2 * i) + rng.normal(0.1, 0.1, 100), 0, 1)}
        for i in range(5)  # C(5,2) = 10 pares
    }

    escasso = compare_models(y, modelos, n_bootstrap=100)
    folgado = compare_models(y, modelos, n_bootstrap=5000)

    # 10 pares x piso 2/101 = 0,198 > 0,05 -> impossivel resolver
    assert escasso["protocol"]["min_resolvable_holm_p"] > 0.05
    assert "resolução insuficiente" in escasso["protocol"]["warning"]
    assert "n_bootstrap >=" in escasso["protocol"]["warning"]
    # 10 pares x piso 2/5001 = 0,004 -> resolve
    assert folgado["protocol"]["min_resolvable_holm_p"] < 0.05
    assert "warning" not in folgado["protocol"]


def test_os_dois_testes_recebem_a_mesma_resolucao():
    """Regressão: `n_bootstrap` só chegava ao bootstrap pareado.

    O McNemar por cluster ficava no default (2.000) e seu piso, 2/2001 × 55
    pares, travava todo p ajustado em 0,055 — logo acima de 0,05 — enquanto o
    bootstrap pareado, já com 5.000, resolvia. Discordância entre os dois tem
    de vir dos dados, não de resoluções diferentes.
    """
    from benchmarks.significance import compare_models

    rng = np.random.default_rng(23)
    y = np.repeat([0, 1], 60)
    clusters = np.repeat([f"f{i}" for i in range(30)], 4)
    modelos = {
        f"m{i}": {"scores": np.clip(y * (0.9 - 0.3 * i) + rng.normal(0.1, 0.1, 120), 0, 1)}
        for i in range(3)
    }

    report = compare_models(y, modelos, cluster_ids=clusters, n_bootstrap=3000)

    for pair in report["pairs"]:
        assert pair["mcnemar"]["bootstrap_samples"] == 3000
        assert pair["paired_bootstrap"]["bootstrap_samples"] <= 3000
        assert pair["mcnemar"]["p_value_floor"] == pytest.approx(
            pair["paired_bootstrap"]["p_value_floor"], rel=1e-6
        )


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


def test_classico_ajusta_so_no_treino_como_as_neurais():
    """A validação SAIU do ajuste dos clássicos (2026-08-09).

    Era `["train", "val"]`: além de dar às duas famílias um n efetivo
    diferente, o `eer_threshold` do contrato de inferência é derivado desse
    mesmo val — ou seja, era um limiar IN-SAMPLE.
    """
    source = (_PROJECT_ROOT / "benchmarks" / "runner.py").read_text(encoding="utf-8")
    assert '"fit_splits": ["train", "val"] if len(yv) else ["train"]' not in source
    assert source.count('"fit_splits": ["train"],') >= 2  # clássico + neural
    assert '"probability_calibration"' in source


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


# ─── Backfill de metadata em runs antigos ──────────────────────────────────


def _load_script(name: str):
    """Importa um script de `scripts/reporting/` pelo caminho (não é pacote)."""
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        f"_script_{name}", _PROJECT_ROOT / "scripts" / "reporting" / f"{name}.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _backfill():
    return _load_script("backfill_artifact_metadata")


_SPLITS_META = {
    "splits": {"train": {"samples": 12162}, "val": {"samples": 1456}}
}


def test_backfill_nao_carimba_keras_em_runner_pytorch():
    """WavLM/HuBERT sao `type: neural` mas rodam em PyTorch.

    Deduzir `estimator` do tipo carimbaria "keras" num modelo PyTorch — o
    backfill so pode declarar o que e derivavel, e `kind`/`estimator` sao de
    quem executou o ajuste.
    """
    mod = _backfill()
    ssl = {
        "type": "neural",
        "noise_protocol": {"train_noise_copies": 1},
        "fit_strategy": {"kind": "frozen_backbone_embedding_then_classifier_fit"},
    }

    block = mod._fit_strategy_block(ssl, _SPLITS_META)

    assert block["fit_splits"] == ["train"]
    assert "estimator" not in block
    assert "kind" not in block
    assert "fit_samples" not in block


def test_backfill_deriva_fit_samples_so_para_o_caminho_keras():
    mod = _backfill()
    keras = {"type": "neural", "noise_protocol": {"train_noise_copies": 1}}

    block = mod._fit_strategy_block(keras, _SPLITS_META)

    assert block["estimator"] == "keras"
    # 12.162 x (1 + 1 copia ruidosa). Bate com os 25.780 do classico menos os
    # 1.456 de validacao que so o classico incorpora.
    assert block["fit_samples"] == 24324
    assert "12162" in block["fit_samples_derivation"]


def test_backfill_usa_a_versao_do_run_e_nao_a_instalada():
    """A versao gravada tem de vir do `environment` do artefato."""
    mod = _backfill()
    env = {"libraries": {"tensorflow": "2.21.0", "sklearn": "1.8.0"}, "gpu": True}

    keras = mod._runtime_block({"type": "neural"}, env)
    classico = mod._runtime_block({"type": "classical"}, env)

    assert keras == {
        "runtime": "keras",
        "runtime_version": "2.21.0",
        "cross_runtime_comparable": False,
        "device": "gpu",
    }
    assert classico["runtime_version"] == "1.8.0"
    assert classico["device"] == "cpu"


def test_backfill_e_idempotente():
    """Rodar duas vezes nao pode reescrever nem duplicar nada."""
    mod = _backfill()
    arch = {
        "type": "neural",
        "training_stability": {"status": "stable"},
        "codec_eval_status": {"status": "not_requested"},
        "fit_strategy": {"fit_splits": ["train"]},
        "efficiency": {"latency_profile": {"status": "ok", "runtime": "keras"}},
    }

    _, fields = mod.plan_architecture(arch, {}, _SPLITS_META)

    assert fields == []


def test_backfill_recusa_cluster_ids_desalinhados(tmp_path):
    """Guarda central: y_test do artefato tem de bater com o do .npz."""
    mod = _backfill()
    npz = tmp_path / "ds.npz"
    np.savez(
        npz,
        y_train=np.zeros(6, dtype=int),
        y_val=np.zeros(2, dtype=int),
        y_test=np.array([0, 1, 0, 1]),
        cluster_ids=np.array([f"f{i // 3}" for i in range(12)]),
    )

    ok, basis = mod.test_cluster_ids_from_npz(npz, [0, 1, 0, 1])
    assert len(ok) == 4
    assert basis["slice"] == [8, 12]
    assert basis["y_test_match"] is True

    with pytest.raises(ValueError, match="nao bate"):
        mod.test_cluster_ids_from_npz(npz, [1, 1, 1, 1])
    with pytest.raises(ValueError, match="nao bate"):
        mod.test_cluster_ids_from_npz(npz, [0, 1])


def test_backfill_carimba_proveniencia():
    """Um artefato completado nao pode passar por um artefato reexecutado."""
    mod = _backfill()

    stamp = mod._stamp(["training_stability"], {"environment": True})

    assert stamp["script"].endswith("backfill_artifact_metadata.py")
    assert "DERIVADOS" in stamp["note"]
    assert stamp["fields"] == ["training_stability"]
    assert stamp["applied_at_utc"]


def test_backfill_nunca_sobrescreve_o_backup(tmp_path):
    mod = _backfill()
    alvo = tmp_path / "results.json"
    alvo.write_text('{"v": 1}', encoding="utf-8")

    primeiro = mod._backup(alvo)
    assert primeiro is not None and primeiro.exists()

    alvo.write_text('{"v": 2}', encoding="utf-8")
    segundo = mod._backup(alvo)

    assert segundo is None  # nao refez
    assert json.loads(primeiro.read_text(encoding="utf-8")) == {"v": 1}


def test_backfill_deriva_speaker_ids_do_npz(tmp_path):
    """Mesma fatia e a mesma verificacao de alinhamento dos cluster_ids."""
    mod = _backfill()
    npz = tmp_path / "ds.npz"
    np.savez(
        npz,
        y_train=np.zeros(6, dtype=int),
        y_val=np.zeros(2, dtype=int),
        y_test=np.array([0, 1, 0, 1]),
        cluster_ids=np.array([f"f{i // 3}" for i in range(12)]),
        speaker_ids=np.array([f"s{i // 6}" for i in range(12)]),
    )

    ids, basis = mod.provenance_ids_from_npz(npz, "speaker_ids", [0, 1, 0, 1])

    assert ids == ["s1"] * 4
    assert basis["key"] == "speaker_ids"
    assert basis["slice"] == [8, 12]
    assert basis["n_groups"] == 1
    with pytest.raises(ValueError, match="nao bate"):
        mod.provenance_ids_from_npz(npz, "speaker_ids", [1, 1, 1, 1])


def test_backfill_recomputa_estabilidade_de_criterio_antigo():
    """Bloco sem os limiares de oscilacao e de uma versao anterior."""
    mod = _backfill()
    antigo = {"status": "stable", "criteria": {"collapse_patience": 15}}
    novo = {"status": "stable", "criteria": {"collapse_patience": 15,
                                             "max_epoch_drop": 0.2}}

    assert mod._needs_stability_refresh({"training_stability": antigo}) is True
    assert mod._needs_stability_refresh({"training_stability": novo}) is False
    # Classico (`unknown`, sem `criteria`) nao tem o que recomputar — se
    # entrasse aqui, o backfill deixaria de ser idempotente para SVM/RF.
    assert mod._needs_stability_refresh(
        {"training_stability": {"status": "unknown", "stable": None}}
    ) is False


def test_backfill_detecta_artefato_substituido(tmp_path):
    """O caso real do bench_svm.pkl: run declarou 3,61 MB, disco tem 0,045."""
    mod = _backfill()
    artefato = tmp_path / "bench_svm.pkl"
    artefato.write_bytes(b"x" * 47_261)

    block = mod._artifact_fingerprint_block(
        {"model_artifact": str(artefato), "efficiency": {"size_mb": 3.61}}
    )

    assert block["integrity"] == "size_mismatch"
    assert "substituido" in block["reason"]
    # Não carimba sha256 de um arquivo que sabidamente não é o do run.
    assert "sha256" not in block


def test_backfill_verifica_artefato_integro_por_tamanho(tmp_path):
    mod = _backfill()
    artefato = tmp_path / "bench_x.keras"
    artefato.write_bytes(b"y" * (2 * 1024 * 1024))

    block = mod._artifact_fingerprint_block(
        {"model_artifact": str(artefato), "efficiency": {"size_mb": 2.0}}
    )

    assert block["integrity"] == "verified_by_size"
    assert len(block["sha256"]) == 64


def test_backfill_nao_acusa_o_runner_ssl_por_tamanho(tmp_path):
    """O `size_mb` do SSL soma cabeca + backbone: divergir e o esperado."""
    mod = _backfill()
    artefato = tmp_path / "bench_wavlm_original.pt"
    artefato.write_bytes(b"z" * 1_587_969)

    block = mod._artifact_fingerprint_block(
        {
            "model_artifact": str(artefato),
            "efficiency": {"size_mb": 361.63},
            "provenance": {"runner": "run_wavlm_original_benchmark:ssl_original"},
        }
    )

    assert block["integrity"] == "not_verifiable"
    assert "backbone" in block["reason"]


def test_backfill_marca_artefato_ausente(tmp_path):
    mod = _backfill()

    block = mod._artifact_fingerprint_block(
        {"model_artifact": str(tmp_path / "sumiu.keras"), "efficiency": {}}
    )

    assert block["integrity"] == "missing_artifact"


# ─── A. Promoção: métricas do results.json e guardas de recusa ─────────────


def _sync():
    return _load_script("sync_completed_benchmark_artifacts")


def _fake_run(tmp_path, *, model="M", clean=None, stability=None,
              fingerprint=None, sha="abc123", artifact_bytes=b"peso"):
    """Run mínimo com um modelo: run_summary.json + <slug>/results.json."""
    run = tmp_path / "run"
    arch_dir = run / "m" / "architectures" / "m" / "models"
    arch_dir.mkdir(parents=True)
    artifact = arch_dir / "bench_m.keras"
    artifact.write_bytes(artifact_bytes)

    arch = {
        "status": "ok",
        "clean": clean if clean is not None else {"accuracy": 0.5, "eer": 0.5},
        "efficiency": {"size_mb": 1.0},
        "model_artifact": str(artifact),
        "training_stability": stability or {"status": "stable", "stable": True},
    }
    if fingerprint is not None:
        arch["model_artifact_fingerprint"] = fingerprint
    (run / "m" / "results.json").write_text(
        json.dumps(
            {"architectures": {model: arch}, "dataset": {"test_split_sha256": sha}}
        ),
        encoding="utf-8",
    )
    summary = run / "run_summary.json"
    summary.write_text(
        json.dumps(
            {
                "models": [
                    {
                        "model": model,
                        "status": "ok",
                        "output_dir": str(run / "m"),
                        "model_artifact": str(artifact),
                        # DEFASADO de propósito: é o valor que a promoção NÃO
                        # pode usar (o caso WavLM/HuBERT 64.000 x 48.000).
                        "clean": {"accuracy": 0.9761, "eer": 0.0217},
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    return summary


def test_promocao_usa_a_metrica_do_results_json_e_nao_a_do_resumo(tmp_path):
    """Reproduz o caso real: resumo em 97,61% e artefato real em 93,92%."""
    mod = _sync()
    summary = _fake_run(tmp_path, clean={"accuracy": 0.9392, "eer": 0.0593})

    promotable = mod.collect_promotable(summary)

    assert len(promotable) == 1
    assert promotable[0]["metrics"]["accuracy"] == 0.9392
    assert promotable[0]["metrics"]["eer"] == 0.0593


def test_promocao_recusa_treino_colapsado(tmp_path):
    mod = _sync()
    summary = _fake_run(
        tmp_path,
        stability={"status": "collapsed", "stable": False, "reason": "84 épocas"},
    )

    with pytest.raises(mod.PromotionRefused, match="collapsed"):
        mod.collect_promotable(summary)


def test_promocao_recusa_artefato_com_tamanho_divergente(tmp_path):
    """Veredito do backfill sobre o bench_svm.pkl trocado."""
    mod = _sync()
    summary = _fake_run(
        tmp_path,
        fingerprint={
            "integrity": "size_mismatch",
            "reason": "o run registrou 3.61 MB e o arquivo tem 0.045 MB",
        },
    )

    with pytest.raises(mod.PromotionRefused, match="3.61 MB"):
        mod.collect_promotable(summary)


def test_promocao_recusa_sha256_divergente(tmp_path):
    mod = _sync()
    summary = _fake_run(
        tmp_path, fingerprint={"integrity": "recorded_at_run", "sha256": "0" * 64}
    )

    with pytest.raises(mod.PromotionRefused, match="sha256"):
        mod.collect_promotable(summary)


def test_promocao_aceita_sha256_conferido(tmp_path):
    mod = _sync()
    import hashlib

    digest = hashlib.sha256(b"peso").hexdigest()
    summary = _fake_run(
        tmp_path, fingerprint={"integrity": "recorded_at_run", "sha256": digest}
    )

    assert len(mod.collect_promotable(summary)) == 1


def test_promocao_recusa_artefato_ausente(tmp_path):
    mod = _sync()
    summary = _fake_run(tmp_path)
    payload = json.loads((tmp_path / "run" / "m" / "results.json").read_text())
    payload["architectures"]["M"]["model_artifact"] = str(tmp_path / "nao_existe.keras")
    (tmp_path / "run" / "m" / "results.json").write_text(json.dumps(payload))

    with pytest.raises(mod.PromotionRefused, match="ausente"):
        mod.collect_promotable(summary)


def test_promocao_recusa_conjuntos_de_teste_diferentes(tmp_path):
    """É o erro que misturar os runs de 15k e 40k cometeria."""
    mod = _sync()
    summary = _fake_run(tmp_path)
    run = summary.parent
    outro = run / "n"
    (outro / "architectures" / "n" / "models").mkdir(parents=True)
    artefato = outro / "architectures" / "n" / "models" / "bench_n.keras"
    artefato.write_bytes(b"peso")
    (outro / "results.json").write_text(
        json.dumps(
            {
                "architectures": {
                    "N": {
                        "status": "ok",
                        "clean": {"accuracy": 0.9},
                        "efficiency": {},
                        "model_artifact": str(artefato),
                        "training_stability": {"stable": True},
                    }
                },
                "dataset": {"test_split_sha256": "outro_fingerprint"},
            }
        ),
        encoding="utf-8",
    )
    payload = json.loads(summary.read_text())
    payload["models"].append(
        {"model": "N", "status": "ok", "output_dir": str(outro), "clean": {}}
    )
    summary.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(mod.PromotionRefused, match="conjuntos de teste diferentes"):
        mod.collect_promotable(summary)


def test_dry_run_nao_escreve_nada(tmp_path):
    mod = _sync()
    summary = _fake_run(tmp_path)
    final_dir = tmp_path / "benchmark_final"

    index = mod.sync_completed(summary, final_dir, dry_run=True)

    assert index["synced_count"] == 1
    assert index["dry_run"] is True
    assert not final_dir.exists()


# ─── A1. Artefato preservado dentro do run ─────────────────────────────────


def test_artefato_e_copiado_para_dentro_do_run(tmp_path):
    """`data/models/` e global: outro run sobrescreve e o artefato some."""
    from benchmarks.runner import _preserve_run_artifact

    shared = tmp_path / "models"
    shared.mkdir()
    origem = shared / "bench_svm.pkl"
    origem.write_bytes(b"modelo do benchmark")
    (shared / "bench_svm_config.json").write_text('{"architecture": "SVM"}')
    arch_dir = tmp_path / "run" / "architectures" / "svm"

    destino, fingerprint = _preserve_run_artifact(origem, arch_dir)

    assert destino == arch_dir / "models" / "bench_svm.pkl"
    assert destino.read_bytes() == b"modelo do benchmark"
    # O sidecar acompanha: `.pkl` sem contrato de entrada não é promovível.
    assert (arch_dir / "models" / "bench_svm_config.json").is_file()
    assert fingerprint["integrity"] == "recorded_at_run"
    assert len(fingerprint["sha256"]) == 64

    # Um smoke sobrescrevendo a cópia global não alcança a do run.
    origem.write_bytes(b"smoke de 8 amostras")
    assert destino.read_bytes() == b"modelo do benchmark"


def test_preserve_artifact_tolera_artefato_inexistente(tmp_path):
    from benchmarks.runner import _preserve_run_artifact

    assert _preserve_run_artifact(None, tmp_path) == (None, None)
    assert _preserve_run_artifact(tmp_path / "nada.keras", tmp_path) == (None, None)


def test_runner_ssl_tambem_grava_o_fingerprint_do_pt():
    """O `size_mb` do SSL soma o backbone, então não serve de proxy.

    Sem o sha256 gravado na hora, o `.pt` de WavLM/HuBERT fica em
    `not_verifiable` para sempre e a promoção não consegue detectar troca.
    """
    ssl = (
        _PROJECT_ROOT / "scripts" / "benchmark" / "run_wavlm_original_benchmark.py"
    ).read_text(encoding="utf-8")

    assert '"model_artifact_fingerprint": _file_fingerprint(artifact)' in ssl
    assert "_file_fingerprint," in ssl  # importado do runner canônico


# ─── B2. speaker_ids persistido nos dois runners ───────────────────────────


def test_os_dois_runners_persistem_test_speaker_ids():
    keras = (_PROJECT_ROOT / "benchmarks" / "runner.py").read_text(encoding="utf-8")
    ssl = (
        _PROJECT_ROOT / "scripts" / "benchmark" / "run_wavlm_original_benchmark.py"
    ).read_text(encoding="utf-8")

    assert '"test_speaker_ids"' in keras
    assert '"test_speaker_ids"' in ssl
    # E alimentam o agrupamento por locutor nos dois lados.
    assert 'grouped_clean["speaker"]' in keras
    assert 'grouped_clean["speaker"]' in ssl


def test_grouped_scores_por_locutor_expoe_o_pior_grupo():
    """O agregado esconde dispersão; `worst_group_accuracy` é o que informa."""
    from benchmarks.evaluate import evaluate_grouped_scores

    y = np.array([0, 1] * 10)
    # Locutor "bom": scores separados. Locutor "ruim": tudo invertido.
    scores = np.array(([0.1, 0.9] * 5) + ([0.9, 0.1] * 5))
    speakers = np.array(["bom"] * 10 + ["ruim"] * 10)

    grouped = evaluate_grouped_scores(y, scores, speakers, threshold=0.5)

    assert grouped["n_groups"] == 2
    assert grouped["per_group"]["bom"]["accuracy"] == pytest.approx(1.0)
    assert grouped["worst_group_accuracy"] == pytest.approx(0.0)


def test_binomial_exato_bate_com_a_referencia():
    """Sanidade numérica do p-valor que sustenta o McNemar."""
    from benchmarks.significance import _binom_two_sided_p

    assert _binom_two_sided_p(0, 0) == 1.0
    assert _binom_two_sided_p(5, 10) == pytest.approx(1.0)
    assert _binom_two_sided_p(0, 10) == pytest.approx(2 / 1024)
    assert _binom_two_sided_p(1, 10) == pytest.approx(2 * 11 / 1024)
    assert not math.isnan(_binom_two_sided_p(3, 7))


def test_oscilacao_registra_quedas_curtas_ao_acaso():
    """`unstable_oscillation` não pode afirmar "sem cair ao nível do acaso".

    O retreino do Conformer (2026-08-11) oscilou com 5 épocas em 0,5000, sendo
    4 consecutivas — abaixo da paciência de 15 que caracterizaria colapso. O
    veredito estava certo; a justificativa que o acompanhava dizia o contrário
    do histórico.
    """
    from benchmarks.stability import analyze_training_stability

    # sobe, mergulha 4 épocas, recupera e estabiliza
    val_acc = (
        [0.70, 0.90, 0.96, 0.98, 0.99]
        + [0.96, 0.91, 0.59]
        + [0.50] * 4
        + [0.93, 0.99] + [0.9924] * 33
    )
    hist = {"val_accuracy": val_acc, "val_loss": [0.3] * len(val_acc)}
    out = analyze_training_stability(hist, epochs_budget=len(val_acc))

    assert out["status"] == "unstable_oscillation"
    assert out["stable"] is True
    assert out["chance_level_epoch_count"] == 4
    assert out["longest_chance_run"] == 4
    assert "nível do acaso" in out["reason"]
    assert "sem cair ao nível do acaso" not in out["reason"]


def test_oscilacao_sem_queda_ao_acaso_mantem_a_redacao_original():
    from benchmarks.stability import analyze_training_stability

    val_acc = [0.70, 0.95, 0.99, 0.72, 0.98, 0.75, 0.97, 0.74, 0.96, 0.99] * 5
    hist = {"val_accuracy": val_acc, "val_loss": [0.3] * len(val_acc)}
    out = analyze_training_stability(hist, epochs_budget=len(val_acc))

    assert "chance_level_epochs" not in out
    if out["status"] == "unstable_oscillation":
        assert "sem cair ao nível do acaso" in out["reason"]


# ─── writers de predição sob ruído ─────────────────────────────────────────
#
# Movidos em 2026-08-17 de `test_resume_guards_and_artifacts.py`, que agrupava
# por DATA de correção e não por sujeito. O assunto é fidelidade do artefato de
# relatório, que é o deste arquivo.


def test_predicoes_sob_ruido_precisam_de_scores_robustness(tmp_path):
    """O writer canônico lê o DICIONÁRIO, não o disco.

    Foi assim que WavLM/HuBERT Original ficaram sem nenhuma predição por
    amostra sob ruído: o runner SSL gravava o arquivo certo e o `write_all`
    regravava por cima a partir de um dicionário sem `scores_robustness`.
    """
    from benchmarks.report import _write_arch_predictions_noisy_csv

    y_true = np.array([0, 1, 0, 1])
    scores = {"30": [0.1, 0.9, 0.2, 0.8], "10": [0.3, 0.7, 0.4, 0.6]}

    com_chave = tmp_path / "com.csv"
    _write_arch_predictions_noisy_csv(
        "X", {"scores_robustness": scores}, y_true, com_chave
    )
    linhas = com_chave.read_text(encoding="utf-8").strip().splitlines()
    assert len(linhas) == 1 + len(y_true) * len(scores)
    assert linhas[0].startswith("snr_db,sample_index,y_true,p_fake")

    # Sem a chave: exatamente o artefato defeituoso do clean_benchmark_15k.
    sem_chave = tmp_path / "sem.csv"
    _write_arch_predictions_noisy_csv("X", {}, y_true, sem_chave)
    assert len(sem_chave.read_text(encoding="utf-8").strip().splitlines()) == 1


def test_runner_ssl_nao_reintroduz_writers_paralelos():
    """As funções removidas escreviam um schema próprio e eram sobrescritas.

    Ressuscitá-las traria de volta a ilusão de que o runner SSL controla esses
    arquivos — que foi o que escondeu o CSV vazio por um run inteiro.
    """
    fonte = Path(__file__).resolve().parents[2] / (
        "scripts/benchmark/run_wavlm_original_benchmark.py"
    )
    arvore = ast.parse(fonte.read_text(encoding="utf-8"))
    definidas = {n.name for n in ast.walk(arvore) if isinstance(n, ast.FunctionDef)}
    proibidas = {"_write_predictions", "_write_predictions_noisy", "_write_robustness"}
    assert not (definidas & proibidas), (
        "o writer canônico é benchmarks.report.write_all; estas duplicam o "
        f"schema e são sobrescritas: {sorted(definidas & proibidas)}"
    )
