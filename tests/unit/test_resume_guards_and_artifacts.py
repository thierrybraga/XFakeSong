"""Regressões das correções de 2026-08-06 (diagnóstico do `clean_benchmark_15k`).

Quatro defeitos distintos, todos encontrados auditando aquele run:

1. O Conformer divergiu na época ~14 e ficou em `loss = ln 2` / `acc = 0.500`
   da época 22 à 100 — 85 épocas de GPU produzindo nada, em duas sessões
   independentes. Daí o `CollapseAbort`.
2. `model.fit()` devolve só as épocas da execução corrente, então uma retomada
   via `BackupAndRestore` truncava o histórico: RawNet2 gravou 17 de 100
   épocas e RawGAT-ST 91, apesar de ambos terem treinado as 100. Daí o
   `PersistentEpochHistory`.
3. `predictions_robustness.csv` de WavLM/HuBERT Original saiu só com o
   cabeçalho: o writer canônico lê `scores_robustness` do dicionário de
   resultados, e o runner SSL não colocava a chave lá.
4. `global_clipnorm` do RawGAT-ST era literal no compile enquanto o registry
   declarava outro valor — config morto.

Ver docs/evaluation/retraining-adjustments.md, seção 2026-08-06.
"""

from __future__ import annotations

import inspect
import json
import math

import numpy as np
import pytest


# ─── 1. Guarda de colapso ──────────────────────────────────────────────────


def _collapse_callback(**kwargs):
    tf = pytest.importorskip("tensorflow")
    from app.domain.models.training.trainer import CollapseAbort

    model = tf.keras.Sequential(
        [tf.keras.layers.Input(shape=(2,)), tf.keras.layers.Dense(1)]
    )
    cb = CollapseAbort(**kwargs)
    cb.set_model(model)
    model.stop_training = False
    return cb, model


def _drive(cb, epochs):
    """Alimenta o callback com (val_loss, val_accuracy) por época."""
    for i, (loss, acc) in enumerate(epochs):
        cb.on_epoch_end(i, {"val_loss": loss, "val_accuracy": acc})


def test_collapse_abort_dispara_no_padrao_do_conformer():
    """Sobe, colapsa para o acaso e fica lá: é o caso que motivou a guarda."""
    cb, model = _collapse_callback(patience=15)
    subida = [(0.30, 0.95)] * 10          # arma a guarda
    colapso = [(0.6931, 0.5)] * 20        # ln 2 / acaso, sustentado
    _drive(cb, subida + colapso)

    assert cb.triggered
    assert model.stop_training is True
    assert "0.9500" in cb.reason  # registra o melhor valor já atingido


def test_collapse_abort_ignora_inicio_lento():
    """Nunca armado: um modelo que começa no acaso e SOBE não pode ser morto.

    É a diferença entre esta guarda e early stopping — sem o gatilho de
    armação, um warmup longo (Conformer: 3.000 passos) seria confundido com
    colapso.
    """
    cb, model = _collapse_callback(patience=5)
    _drive(cb, [(0.6931, 0.5)] * 30 + [(0.2, 0.97)] * 5)

    assert not cb.triggered
    assert model.stop_training is False


def test_collapse_abort_tolera_queda_isolada():
    """Oscilação que se recupera antes da paciência não dispara.

    O AASIST teve pior época em val_accuracy=0.5316 no run real; um modelo que
    encoste no acaso por poucas épocas e volte não pode ser abortado.
    """
    cb, model = _collapse_callback(patience=10)
    _drive(cb, [(0.3, 0.95)] * 5 + [(0.69, 0.50)] * 9 + [(0.3, 0.96)] * 5)

    assert not cb.triggered
    assert model.stop_training is False


def test_collapse_abort_pega_loss_nao_finita():
    """NaN/Inf sustentado aborta mesmo sem a guarda de acurácia."""
    cb, model = _collapse_callback(nan_patience=3)
    _drive(cb, [(0.3, 0.95)] * 3 + [(float("nan"), 0.95)] * 3)

    assert cb.triggered
    assert "não-finito" in cb.reason
    assert model.stop_training is True


def test_collapse_abort_desligavel_pela_config():
    """`abort_on_collapse=False` tem de remover o callback da montagem."""
    pytest.importorskip("tensorflow")
    from app.core.config.settings import TrainingConfig

    assert TrainingConfig().abort_on_collapse is True
    assert TrainingConfig(abort_on_collapse=False).abort_on_collapse is False


# ─── 2. Histórico resistente a retomadas ───────────────────────────────────


def _history_callback(path):
    pytest.importorskip("tensorflow")
    from app.domain.models.training.trainer import PersistentEpochHistory

    return PersistentEpochHistory(path)


def test_historico_sobrevive_a_retomada(tmp_path):
    """Cenário REAL do RawNet2: 83 épocas, queda, retomada na 84 até a 100.

    Antes desta correção o `metrics.json` guardava 17 entradas para um treino
    de 100 épocas, e a figura de convergência mostrava um fragmento.
    """
    path = tmp_path / "epoch_history.jsonl"

    sessao1 = _history_callback(path)
    sessao1.on_train_begin()
    for e in range(83):
        sessao1.on_epoch_end(e, {"loss": 1.0 - e * 0.01, "val_loss": 0.5})

    # BackupAndRestore devolve o índice ABSOLUTO da época na retomada.
    sessao2 = _history_callback(path)
    sessao2.on_train_begin()
    assert len(sessao2._records) == 83, "não recuperou a sessão anterior"
    for e in range(83, 100):
        sessao2.on_epoch_end(e, {"loss": 0.2, "val_loss": 0.42})

    merged = sessao2.merged()
    assert len(merged["loss"]) == 100
    assert len(merged["val_loss"]) == 100
    assert merged["loss"][0] == pytest.approx(1.0), "época 1 da sessão 1 perdida"
    assert merged["loss"][83] == pytest.approx(0.2)


def test_historico_nao_desloca_series_com_metrica_faltante(tmp_path):
    """Uma época sem `val_loss` não pode empurrar os demais valores."""
    cb = _history_callback(tmp_path / "h.jsonl")
    cb.on_epoch_end(0, {"loss": 1.0, "val_loss": 0.9})
    cb.on_epoch_end(1, {"loss": 0.8})
    cb.on_epoch_end(2, {"loss": 0.6, "val_loss": 0.7})

    merged = cb.merged()
    assert len(merged["loss"]) == len(merged["val_loss"]) == 3
    assert math.isnan(merged["val_loss"][1])
    assert merged["val_loss"][2] == pytest.approx(0.7)


def test_historico_regrava_mesma_epoca_sem_duplicar(tmp_path):
    """Retomada que repete uma época já gravada é idempotente."""
    path = tmp_path / "h.jsonl"
    cb = _history_callback(path)
    cb.on_epoch_end(5, {"loss": 1.0})
    cb.on_epoch_end(5, {"loss": 0.5})

    linhas = [ln for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]
    assert len(linhas) == 1
    assert json.loads(linhas[0])["loss"] == pytest.approx(0.5)


def test_historico_fica_fora_do_backup_dir(tmp_path):
    """O `backup_dir` é APAGADO ao fim do treino (delete_checkpoint=True).

    Se o histórico morasse lá, seria removido junto justamente nos treinos que
    terminam bem — que são os que viram artefato.
    """
    pytest.importorskip("tensorflow")
    from app.core.config.settings import TrainingConfig
    from app.domain.models.training.trainer import (
        ModelTrainer,
        PersistentEpochHistory,
    )

    checkpoint = tmp_path / "models" / "best_checkpoint.weights.h5"
    checkpoint.parent.mkdir(parents=True)
    backup = tmp_path / "training_backup"

    # `use_mixed_precision=False` evita que o auto-detect altere a policy
    # global do Keras e contamine os demais testes da sessão.
    trainer = ModelTrainer(
        TrainingConfig(early_stopping=False), use_mixed_precision=False
    )
    callbacks = trainer._prepare_callbacks(
        checkpoint_path=str(checkpoint), backup_dir=str(backup)
    )
    historicos = [c for c in callbacks if isinstance(c, PersistentEpochHistory)]
    assert historicos, "callback de histórico não foi montado"
    destino = historicos[0].path.resolve()
    assert backup.resolve() not in destino.parents
    assert destino.parent == checkpoint.parent.resolve()


# ─── 3. predictions_robustness.csv ─────────────────────────────────────────


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
    import ast
    from pathlib import Path

    fonte = Path(__file__).resolve().parents[2] / (
        "scripts/benchmark/run_wavlm_original_benchmark.py"
    )
    arvore = ast.parse(fonte.read_text(encoding="utf-8"))
    definidas = {
        n.name for n in ast.walk(arvore) if isinstance(n, ast.FunctionDef)
    }
    proibidas = {"_write_predictions", "_write_predictions_noisy", "_write_robustness"}
    assert not (definidas & proibidas), (
        "o writer canônico é benchmarks.report.write_all; estas duplicam o "
        f"schema e são sobrescritas: {sorted(definidas & proibidas)}"
    )


# ─── 4. Sincronia das três fontes de hiperparâmetro ────────────────────────


@pytest.mark.parametrize(
    "arch_name,module_name,plan_key",
    [("RawGAT-ST", "rawgat_st", "rawgatst"), ("AASIST", "aasist", "aasist")],
)
def test_hparams_batem_nas_tres_fontes(arch_name, module_name, plan_key):
    """CLAUDE.md exige registry + create_model + planning em sincronia.

    A guarda existente (`test_default_params_are_accepted_by_builder`) checa se
    a CHAVE é aceita; esta checa se o VALOR é o mesmo. Um default divergente no
    `create_model` faz o caminho do app/Gradio treinar com uma receita
    diferente da que o benchmark documenta.
    """
    import importlib

    from app.domain.models.architectures.registry import ArchitectureRegistry
    from benchmarks.planning import NEURAL_BENCHMARK_HPARAMS

    info = ArchitectureRegistry().get_architecture(arch_name)
    plano = NEURAL_BENCHMARK_HPARAMS[plan_key]
    modulo = importlib.import_module(info.module_path)
    assinatura = inspect.signature(getattr(modulo, info.function_name)).parameters

    comparaveis = [
        k for k in info.default_params if k in plano and k in assinatura
    ]
    assert comparaveis, f"{arch_name}: nada em comum para comparar"

    for chave in comparaveis:
        registry_v = info.default_params[chave]
        plano_v = plano[chave]
        builder_v = assinatura[chave].default
        assert registry_v == pytest.approx(plano_v), (
            f"{arch_name}.{chave}: registry={registry_v} != planning={plano_v}"
        )
        assert registry_v == pytest.approx(builder_v), (
            f"{arch_name}.{chave}: registry={registry_v} != "
            f"create_model={builder_v}"
        )


def test_global_clipnorm_do_rawgat_chega_ao_construtor():
    """Era literal 0.7 no compile enquanto o registry declarava 0.5.

    Não basta existir nas três fontes: o runner precisa PROMOVER a chave para
    `parameters`, senão ela não chega ao `create_model` e volta a ser config
    morto.
    """
    import ast
    from pathlib import Path

    from app.domain.models.architectures import rawgat_st

    assinatura = inspect.signature(rawgat_st.create_model).parameters
    assert "global_clipnorm" in assinatura

    runner = Path(__file__).resolve().parents[2] / "benchmarks/runner.py"
    fonte = runner.read_text(encoding="utf-8")
    inicio = fonte.find('elif compact in {"aasist", "rawgatst"}')
    assert inicio > 0, "ramo de promoção do rawgatst não encontrado"
    assert '"global_clipnorm"' in fonte[inicio:inicio + 1200], (
        "global_clipnorm não está no whitelist de promoção do runner"
    )
    # Silenciosamente ler ast garante que o arquivo segue parseável.
    ast.parse(fonte)
