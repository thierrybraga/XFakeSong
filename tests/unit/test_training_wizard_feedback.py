"""Feedback do assistente de treino: custo à frente, parada e sobreajuste.

Um treino do assistente pode levar horas — o RawGAT-ST chega a ~54 h de GPU no
orçamento de 100 épocas. Antes desta revisão o usuário: (a) não sabia disso ao
clicar, (b) não tinha como interromper pela interface, e (c) precisava
perceber sozinho, olhando quatro números por época, quando a validação parava
de melhorar.
"""

from __future__ import annotations

import re

import pytest

pytest.importorskip("gradio")

from app.interfaces.gradio.tabs.training_wizard import (  # noqa: E402
    _PEDIDO_DE_PARADA,
    _estimativa_de_treino,
    _train_status_html,
)


def _historico(val_loss):
    n = len(val_loss)
    return {
        "loss": [0.5] * n,
        "acc": [0.8] * n,
        "val_loss": list(val_loss),
        "val_acc": [0.7] * n,
    }


# ─────────────── custo antes de começar ───────────────

def test_estimativa_reflete_o_custo_medido_da_arquitetura():
    """Reaproveita as horas de `benchmarks.planning`, não um chute."""
    caro = _estimativa_de_treino("RawGAT-ST", 100, 66452)
    barato = _estimativa_de_treino("Sonic Sleuth", 100, 66452)

    assert caro and barato
    assert "considere reduzir" in caro, "custo alto precisa ser sinalizado"
    # o mais caro do recorte tem de aparecer como muito maior que o mais barato
    horas = lambda t: int(re.search(r"<b>(\d+)", t).group(1))  # noqa: E731
    assert horas(caro) > 10 * horas(barato)


def test_estimativa_escala_com_epocas_e_amostras():
    cheio = _estimativa_de_treino("AASIST", 100, 66452)
    poucas_epocas = _estimativa_de_treino("AASIST", 10, 66452)
    poucas_amostras = _estimativa_de_treino("AASIST", 100, 6645)

    assert cheio != poucas_epocas != poucas_amostras
    # 10x menos épocas não pode continuar avisando "reduza as épocas"
    assert "considere reduzir" in cheio


def test_estimativa_omite_o_que_nao_sabe():
    """Arquitetura sem custo medido não inventa número."""
    assert _estimativa_de_treino("Arquitetura Inexistente", 100, 1000) == ""


def test_separador_de_milhar_nao_corrompe_a_frase():
    """O `.replace(',', '.')` ingênuo transformava a vírgula da frase em ponto."""
    texto = _estimativa_de_treino("AASIST", 100, 66452)
    assert "66.452 amostras" in texto
    assert "épocas, " in texto, "a vírgula da enumeração foi corrompida"


# ─────────────── interrupção ───────────────

def test_sinal_de_parada_e_reversivel():
    _PEDIDO_DE_PARADA.clear()
    assert not _PEDIDO_DE_PARADA.is_set()
    _PEDIDO_DE_PARADA.set()
    assert _PEDIDO_DE_PARADA.is_set()
    _PEDIDO_DE_PARADA.clear()
    assert not _PEDIDO_DE_PARADA.is_set()


def test_treino_respeita_o_pedido_de_parada():
    """O Keras não interrompe no meio da época — só em `stop_training`."""
    import inspect

    from app.interfaces.gradio.tabs import training_wizard

    fonte = inspect.getsource(training_wizard._run_training)
    assert "_PEDIDO_DE_PARADA.is_set()" in fonte
    assert "stop_training = True" in fonte
    # e o pedido pendente de um treino anterior não pode matar o próximo
    assert "_PEDIDO_DE_PARADA.clear()" in fonte


# ─────────────── sobreajuste ao vivo ───────────────

def test_aviso_de_sobreajuste_so_aparece_quando_ha_estagnacao():
    melhorando = _train_status_html(
        "AASIST", "GPU", 8, 100, _historico([.9, .8, .7, .6, .5, .4, .3, .2]), 120.0
    )
    assert "val_loss sem melhorar" not in melhorando

    estagnada = _train_status_html(
        "AASIST", "GPU", 10, 100,
        _historico([.9, .8, .7, .3, .4, .5, .6, .7, .8, .9]), 120.0,
    )
    assert "val_loss sem melhorar" in estagnada
    assert "sem melhorar há 6 épocas" in estagnada


def test_aviso_nao_alarma_no_inicio_do_treino():
    """Oscilação nas primeiras épocas é normal."""
    curto = _train_status_html(
        "AASIST", "GPU", 4, 100, _historico([.3, .5, .6, .7]), 60.0
    )
    assert "val_loss sem melhorar" not in curto


def test_aviso_informa_que_o_melhor_checkpoint_esta_salvo():
    """Sem isso, interromper parece perder o trabalho."""
    html = _train_status_html(
        "AASIST", "GPU", 12, 100,
        _historico([.9, .8, .2, .3, .4, .5, .6, .7, .8, .9, 1.0, 1.1]), 300.0,
    )
    assert "melhor checkpoint" in html


def test_aviso_ausente_quando_o_treino_terminou():
    html = _train_status_html(
        "AASIST", "GPU", 10, 10,
        _historico([.9, .8, .2, .3, .4, .5, .6, .7, .8, .9]), 300.0,
        phase="done",
    )
    assert "val_loss sem melhorar" not in html
