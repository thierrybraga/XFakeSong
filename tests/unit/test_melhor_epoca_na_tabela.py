"""A época reportada é a do critério que REALMENTE selecionou o checkpoint.

SUJEITO: ``scripts/reporting/consolidate_results.py::_best_epoch`` e a coluna
``Treino`` de ``update_tcc_latex.py::build_results_table``.

Por que este teste existe
-------------------------
O protocolo restaura o checkpoint da melhor época — ``select_best_checkpoint``
com ``--checkpoint-monitor val_eer`` no compose, propagado até o
``ModelCheckpoint`` do trainer. Os pesos avaliados são os DAQUELA época, não os
da última.

Duas coisas estavam erradas, e nenhuma quebrava nada visivelmente:

1. ``_best_epoch`` calculava sempre pelo ``val_loss``, qualquer que fosse o
   monitor do run. A época publicada podia ser outra que não a avaliada — os
   dois critérios coincidem por acaso com frequência, o que torna o defeito
   difícil de notar.
2. A coluna ``Treino`` mostrava só o orçamento (``100`` para todo neural, por
   ``fixed_epoch_budget``), sugerindo que o modelo publicado era o do fim do
   treino.

Há ainda uma assimetria real, agora declarada em vez de escondida: as nove
entradas do caminho Keras selecionam por ``val_eer``; WavLM Original e HuBERT
Original rodam por um runner PyTorch com laço próprio e selecionam por perda de
validação. Comparar as épocas sem dizer isso é comparar escolhas feitas sob
regras diferentes.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


@pytest.fixture(scope="module")
def cons():
    import scripts.reporting.consolidate_results as m
    return m


def test_epoca_segue_o_monitor_declarado_nao_o_val_loss(cons):
    """O caso que o defeito antigo errava: monitores discordando.

    `val_eer` tem mínimo na época 3; `val_loss`, na 5. Com o monitor `val_eer`
    declarado, a resposta certa é 3 — a função antiga devolveria 5.
    """
    history = {
        "val_eer": [0.20, 0.15, 0.05, 0.09, 0.12],
        "val_loss": [0.9, 0.7, 0.6, 0.55, 0.40],
    }
    assert cons._best_epoch(history, "val_eer") == 3
    assert cons._best_epoch(history, "val_loss") == 5


def test_direcao_por_metrica_min_para_erro_max_para_acerto(cons):
    """Errar a direção reporta a PIOR época como a melhor."""
    history = {"val_accuracy": [0.50, 0.95, 0.60]}
    assert cons._best_epoch(history, "val_accuracy") == 2
    history = {"val_eer": [0.50, 0.05, 0.60]}
    assert cons._best_epoch(history, "val_eer") == 2


def test_runner_ssl_cai_para_a_perda_de_validacao(cons):
    """O history do runner PyTorch não tem `val_eer`; a queda tem de funcionar.

    Sem isso, as duas entradas SSL sairiam sem época na tabela.
    """
    history = {"val_loss": [0.9, 0.4, 0.6], "val_accuracy": [0.7, 0.9, 0.8]}
    assert cons._best_epoch(history, "val_eer") == 2
    assert cons._serie_de_selecao(history, "val_eer")[0] == "val_loss"


def test_sem_history_nao_inventa_epoca(cons):
    """Clássicos (SVM/RandomForest) não têm épocas — `None`, não `1`."""
    assert cons._best_epoch(None, "val_eer") is None
    assert cons._best_epoch({}, "val_eer") is None


def test_a_coluna_treino_mostra_melhor_sobre_orcamento():
    """Formato `47/100`, não `100`.

    O leitor precisa ver quanto do orçamento foi útil: uma seleção na época 6
    de 100 (o caso do HuBERT no run corrigido) diz que o modelo parou de
    melhorar quase imediatamente — informação que a coluna antiga escondia.
    """
    from scripts.reporting.update_tcc_latex import build_results_table

    linhas = build_results_table([
        {"key": "CCT", "accuracy": 0.9928, "eer": 0.0058, "min_tdcf": 0.0135,
         "auc": 0.9997, "f1": 0.9928, "latency": 33.24, "epochs": 100,
         "best_epoch": 47, "converged": True},
        {"key": "SVM", "accuracy": 0.9175, "eer": 0.0709, "min_tdcf": 0.1454,
         "auc": 0.9773, "f1": 0.9108, "latency": 0.75, "converged": True},
    ]).split("\n")

    assert "47/100" in linhas[0], f"esperado '47/100' na linha do CCT: {linhas[0]}"
    assert "& 100 &" not in linhas[0], "coluna ainda anuncia só o orçamento"
    assert "CV+fit" in linhas[1], "clássicos devem seguir marcados como CV+fit"


def test_sem_melhor_epoca_a_coluna_degrada_para_o_orcamento():
    """Artefato antigo, sem `best_epoch`: mostra o orçamento, não quebra."""
    from scripts.reporting.update_tcc_latex import build_results_table

    linha = build_results_table([
        {"key": "AASIST", "accuracy": 0.9, "eer": 0.05, "min_tdcf": 0.1,
         "auc": 0.98, "f1": 0.9, "latency": 10.0, "epochs": 100,
         "converged": True},
    ])
    assert "& 100 &" in linha


def test_o_criterio_de_selecao_e_declarado_no_rodape():
    """A assimetria Keras/SSL precisa estar escrita, não inferida.

    Sem a nota, a coluna compara épocas escolhidas por `val_eer` com épocas
    escolhidas por perda de validação como se fossem a mesma medida.
    """
    fonte = (ROOT / "scripts" / "reporting" / "update_tcc_latex.py").read_text(
        encoding="utf-8"
    )
    assert "val\\_eer" in fonte, "o rodapé não nomeia o monitor do caminho Keras"
    assert "perda de validação" in fonte, "o rodapé não declara o critério do SSL"
    assert "melhor época" in fonte, "a legenda não explica a coluna Treino"
