"""Guardas que protegem um treino longo: aborto de colapso e histórico.

SUJEITO: os callbacks de `app/domain/models/training/trainer.py` que decidem
**parar** um treino ou **preservar** o que ele produziu —
:class:`CollapseAbort` e :class:`PersistentEpochHistory`. A seleção do
checkpoint publicado é assunto de `test_checkpoint_selection.py`.

Consolidado em 2026-08-17 a partir de `test_resume_guards_and_artifacts.py`
(nomeado pela DATA da correção, 2026-08-06) e `test_collapse_never_learns.py`.
A divisão custou caro: os dois arquivos cobriam o MESMO callback com contratos
contraditórios, e o `arm_deadline` acrescentado em 16/08 quebrou uma regressão
de 06/08 sem que nada apontasse o conflito — as duas curvas de `val_accuracy`
eram idênticas. Um sujeito, um arquivo.

Origem de cada bloco:

1. **CollapseAbort** — o Conformer divergiu na época ~14 e ficou em
   `loss = ln 2` / `acc = 0.500` da época 22 à 100, 85 épocas de GPU produzindo
   nada, em duas sessões independentes. Depois, o braço (d) do retune do
   RawGAT-ST ficou 25 épocas em `val_accuracy` 0,5000 exato com o TREINO em
   95,4% — falha distinta, que o guarda original não pegava.
2. **PersistentEpochHistory** — `model.fit()` devolve só as épocas da execução
   corrente, então uma retomada via `BackupAndRestore` truncava o histórico:
   RawNet2 gravou 17 de 100 épocas e RawGAT-ST 91, apesar de ambos terem
   treinado as 100.
"""

from __future__ import annotations

import json
import math

import pytest

# `trainer.py` importa TensorFlow no topo, então sem ele a coleção deste
# arquivo QUEBRA em vez de pular. Guarda no módulo, uma vez, em lugar de
# repetir `importorskip` em cada helper — era assim nos dois arquivos de
# origem, e mascarava que a dependência é do módulo inteiro.
pytest.importorskip("tensorflow")


# ═══════════════════════════════════════════════════════════════════════════
# 1. CollapseAbort
# ═══════════════════════════════════════════════════════════════════════════


class _ModeloFalso:
    """Substituto de `tf.keras.Model` — o callback só toca `stop_training`."""

    def __init__(self) -> None:
        self.stop_training = False


def _guarda(**kwargs):
    from app.domain.models.training.trainer import CollapseAbort

    cb = CollapseAbort(label="teste", **kwargs)
    cb.set_model(_ModeloFalso())
    return cb


def _roda(curva, **kwargs):
    """Alimenta o callback com (accuracy de treino, val_accuracy) por época.

    Um ponto escalar vale para os dois eixos — atalho para as curvas em que a
    folga de generalização não é o que está sob teste.
    """
    cb = _guarda(**kwargs)
    for i, ponto in enumerate(curva):
        treino, val = ponto if isinstance(ponto, tuple) else (ponto, ponto)
        cb.on_epoch_end(
            i, {"accuracy": treino, "val_accuracy": val, "val_loss": 0.5}
        )
        if cb.triggered:
            break
    return cb


def _roda_perdas(cb, epocas):
    """Alimenta (val_loss, val_accuracy) — para os casos guiados pela perda."""
    for i, (loss, acc) in enumerate(epocas):
        cb.on_epoch_end(i, {"val_loss": loss, "val_accuracy": acc})


#: Curva medida do braço (d) do retune: treino sai de 0,50 e chega a 0,95
#: enquanto a validação fica em 0,5000 exato.
#: Ver data/results/rawgat_arm_d/.../epoch_history.jsonl.
_BRACO_D = [(0.50 + 0.03 * i, 0.5) for i in range(30)]


# ─── 1a. colapso: esteve bom e caiu ───────────────────────────────────────


def test_aborta_no_padrao_do_conformer():
    """Sobe, colapsa para o acaso e fica lá: o caso que motivou a guarda."""
    cb = _guarda(patience=15)
    _roda_perdas(cb, [(0.30, 0.95)] * 10 + [(0.6931, 0.5)] * 20)

    assert cb.triggered
    assert cb.model.stop_training is True
    assert "0.9500" in cb.reason, "não registra o melhor valor já atingido"
    assert "nível do acaso" in cb.reason


def test_tolera_queda_isolada():
    """Oscilação que se recupera antes da paciência não dispara.

    O AASIST teve pior época em `val_accuracy` 0,5316 no run real; encostar no
    acaso por poucas épocas e voltar não pode abortar.
    """
    cb = _guarda(patience=10)
    _roda_perdas(cb, [(0.3, 0.95)] * 5 + [(0.69, 0.50)] * 9 + [(0.3, 0.96)] * 5)

    assert not cb.triggered
    assert cb.model.stop_training is False


def test_pega_loss_nao_finita():
    """NaN/Inf sustentado aborta mesmo sem a guarda de acurácia."""
    cb = _guarda(nan_patience=3)
    _roda_perdas(cb, [(0.3, 0.95)] * 3 + [(float("nan"), 0.95)] * 3)

    assert cb.triggered
    assert "não-finito" in cb.reason
    assert cb.model.stop_training is True


def test_desligavel_pela_config():
    """`abort_on_collapse=False` tem de remover o callback da montagem."""
    from app.core.config.settings import TrainingConfig

    assert TrainingConfig().abort_on_collapse is True
    assert TrainingConfig(abort_on_collapse=False).abort_on_collapse is False


# ─── 1b. nunca generalizou: o treino sobe e a validação não ───────────────


def test_aborta_quando_nunca_cruza_o_limiar():
    """A curva real do braço (d): val travada no acaso, treino a 95,4%."""
    cb = _roda(_BRACO_D)
    assert cb.triggered
    assert "nunca alcançou" in cb.reason
    assert "memoriza o treino" in cb.reason
    assert cb.model.stop_training


def test_aborta_no_prazo_configurado_e_nao_antes():
    cb = _guarda(arm_deadline=15)
    logs = {"accuracy": 0.95, "val_accuracy": 0.5, "val_loss": 0.5}
    for i in range(14):
        cb.on_epoch_end(i, dict(logs))
    assert not cb.triggered, "não pode abortar antes do prazo"
    cb.on_epoch_end(14, dict(logs))
    assert cb.triggered, "deve abortar exatamente na época do prazo"


def test_prazo_nao_dispara_se_ja_armou():
    """Cruzou o limiar cedo: o prazo não pode abortar depois disso."""
    cb = _roda([(0.99, 0.8)] + [(0.99, 0.75)] * 40)
    assert not cb.triggered


@pytest.mark.parametrize(
    "nome,primeira_epoca_boa",
    [("RawGAT-ST", 3), ("CCT", 1), ("Conformer", 1), ("AASIST", 1)],
)
def test_nao_mata_run_legitimo(nome, primeira_epoca_boa):
    """Nenhuma das nove arquiteturas do run oficial cruzou 0,6 após a época 3."""
    curva = [(0.5, 0.5)] * (primeira_epoca_boa - 1) + [(0.90, 0.85)] * 30
    cb = _roda(curva)
    assert not cb.triggered, f"{nome} seria morto indevidamente"


# ─── 1c. a fronteira entre 1a e 1b ────────────────────────────────────────
#
# Estes três testes são a razão de o arquivo existir. Olhando só
# `val_accuracy`, "nunca aprendeu" e "warmup longo" são a MESMA curva; o que os
# separa é a folga treino-validação. Quando os dois blocos moravam em arquivos
# diferentes, o `arm_deadline` foi calibrado contra um deles e quebrou o outro.


def test_ignora_inicio_lento():
    """Nunca armado: começa no acaso e SOBE — não pode ser morto.

    É a diferença entre esta guarda e early stopping. Um warmup longo
    (Conformer: 3.000 passos) seria confundido com colapso.
    """
    cb = _guarda(patience=5)
    _roda_perdas(cb, [(0.6931, 0.5)] * 30 + [(0.2, 0.97)] * 5)

    assert not cb.triggered
    assert cb.model.stop_training is False


def test_nao_aborta_sem_folga_de_generalizacao():
    """Treino e validação no acaso JUNTOS: é warmup, não memorização.

    Sem folga a guarda se cala mesmo passado o prazo — quem limita esse caso é
    o orçamento fixo de épocas.
    """
    cb = _roda([(0.5, 0.5)] * 30)
    assert not cb.triggered
    assert not cb.model.stop_training


def test_nao_aborta_quando_a_metrica_de_treino_falta():
    """Sem `accuracy` nos logs não há como distinguir os dois casos.

    Na dúvida a guarda se cala: matar um treino bom custa mais do que deixar um
    ruim correr até o fim do orçamento.
    """
    cb = _guarda()
    for i in range(30):
        cb.on_epoch_end(i, {"val_accuracy": 0.5, "val_loss": 0.5})
    assert not cb.triggered


# ═══════════════════════════════════════════════════════════════════════════
# 2. PersistentEpochHistory
# ═══════════════════════════════════════════════════════════════════════════


def _historico(path):
    from app.domain.models.training.trainer import PersistentEpochHistory

    return PersistentEpochHistory(path)


def test_historico_sobrevive_a_retomada(tmp_path):
    """Cenário REAL do RawNet2: 83 épocas, queda, retomada na 84 até a 100.

    Antes da correção o `metrics.json` guardava 17 entradas para um treino de
    100 épocas, e a figura de convergência mostrava um fragmento.
    """
    path = tmp_path / "epoch_history.jsonl"

    sessao1 = _historico(path)
    sessao1.on_train_begin()
    for e in range(83):
        sessao1.on_epoch_end(e, {"loss": 1.0 - e * 0.01, "val_loss": 0.5})

    # BackupAndRestore devolve o índice ABSOLUTO da época na retomada.
    sessao2 = _historico(path)
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
    cb = _historico(tmp_path / "h.jsonl")
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
    cb = _historico(path)
    cb.on_epoch_end(5, {"loss": 1.0})
    cb.on_epoch_end(5, {"loss": 0.5})

    linhas = [
        ln for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()
    ]
    assert len(linhas) == 1
    assert json.loads(linhas[0])["loss"] == pytest.approx(0.5)


def test_historico_fica_fora_do_backup_dir(tmp_path):
    """O `backup_dir` é APAGADO ao fim do treino (delete_checkpoint=True).

    Se o histórico morasse lá, seria removido junto justamente nos treinos que
    terminam bem — que são os que viram artefato.
    """
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
