"""A guarda de XLA casa com os NOMES REAIS do manifesto oficial.

SUJEITO: ``scripts/benchmark/run_models_sequential.py``
::``_XLA_UNFRIENDLY_TRAINING_ARCHITECTURES``.

Por que este teste existe
-------------------------
O conjunto é um literal (``{"multiscalecnn", "aasist", "rawgatst",
"rawnet2"}``) comparado contra uma chave DERIVADA do nome do manifesto. Até
2026-08-20 a derivação usava ``_slug``, que troca separador por underscore:
``_slug("RawGAT-ST")`` devolve ``"rawgat_st"``, que não pertence ao conjunto.

O defeito ficou escondido porque as outras três arquiteturas não têm hífen no
nome — casavam por acaso. E o sintoma não é um erro: o treino simplesmente roda
com o auto-JIT do XLA LIGADO, que é a condição que o comentário do próprio
conjunto documenta ter matado o run oficial em 2026-07-31 (OOM-killer do host
para MultiscaleCNN/AASIST/RawGAT-ST, ``CUDA_ERROR_OUT_OF_MEMORY`` para
RawNet2).

Medido no run ``clean_benchmark_15k`` antes da correção:
``rawgat_st/run.log`` registrava ``XLA=True`` e nenhuma linha ``[XLA] auto-JIT
desligado``, enquanto ``aasist/run.log`` e ``rawnet2/run.log`` registravam
``XLA=False``. A arquitetura de MAIOR custo estimado do escopo oficial (73,4 h
de GPU) rodava exatamente na condição que a guarda existe para evitar.

O teste percorre os nomes REAIS do manifesto em vez de repetir o literal — é a
única forma de a asserção continuar valendo quando alguém acrescentar uma
arquitetura cujo nome tenha hífen, ponto ou espaço.
"""

from __future__ import annotations

import importlib

import pytest


@pytest.fixture(scope="module")
def runner():
    return importlib.import_module("scripts.benchmark.run_models_sequential")


#: Nomes de manifesto que PRECISAM desligar o auto-JIT, com o motivo medido.
#: Escritos como aparecem em `benchmarks/config.py`, com hífen e tudo.
NOMES_QUE_PRECISAM_DESLIGAR_XLA = {
    "MultiscaleCNN": "OOM-killer do host (STFT in-graph + laços do Bottle2neck)",
    "AASIST": "OOM-killer do host (graph attention dinâmico)",
    "RawGAT-ST": "OOM-killer do host (graph attention dinâmico) — o do hífen",
    "RawNet2": "CUDA_ERROR_OUT_OF_MEMORY na GPU (GRU(1024) sob auto-JIT)",
}


def test_guarda_xla_casa_com_os_nomes_do_manifesto(runner):
    """Cada nome que precisa da mitigação tem de casar com o conjunto."""
    falhas = []
    for nome, motivo in NOMES_QUE_PRECISAM_DESLIGAR_XLA.items():
        chave = runner._compact(nome)
        if chave not in runner._XLA_UNFRIENDLY_TRAINING_ARCHITECTURES:
            falhas.append(f"{nome!r} -> {chave!r} não está no conjunto ({motivo})")
    assert not falhas, (
        "a guarda de XLA não alcança arquiteturas que dependem dela:\n  "
        + "\n  ".join(falhas)
    )


def test_guarda_nao_desliga_xla_para_quem_nao_precisa(runner):
    """Desligar o auto-JIT onde não é preciso custa desempenho de graça."""
    for nome in ("Conformer", "Hybrid CNN-Transformer", "SpectrogramTransformer",
                 "SVM", "RandomForest", "WavLM Original", "HuBERT Original"):
        chave = runner._compact(nome)
        assert chave not in runner._XLA_UNFRIENDLY_TRAINING_ARCHITECTURES, (
            f"{nome!r} não precisa da mitigação de XLA, mas casou com o conjunto"
        )


def test_todo_nome_do_conjunto_existe_no_manifesto(runner):
    """Nenhuma chave órfã: um nome que ninguém produz é guarda morta.

    Se uma arquitetura sair do escopo, a entrada correspondente tem de sair
    junto — senão o conjunto acumula chaves que não protegem nada e escondem,
    pelo tamanho, as que deixaram de casar.
    """
    from benchmarks.config import OFFICIAL_TCC_MODEL_MANIFEST

    conhecidos = {
        runner._compact(str(item.get("benchmark_name", "")))
        for item in OFFICIAL_TCC_MODEL_MANIFEST
    }
    orfas = runner._XLA_UNFRIENDLY_TRAINING_ARCHITECTURES - conhecidos
    assert not orfas, (
        f"chaves sem arquitetura correspondente no manifesto oficial: {sorted(orfas)}"
    )


def test_a_derivacao_usada_e_a_compacta_nao_a_com_underscore(runner):
    """Trava a causa-raiz, não só o sintoma.

    `_slug` e `_compact` divergem exatamente nos nomes com separador. Este
    teste documenta a diferença e prova que é `_compact` que serve ao conjunto —
    se alguém trocar de volta para `_slug`, o teste acima reprova, e este
    explica por quê.
    """
    assert runner._slug("RawGAT-ST") == "rawgat_st"
    assert runner._compact("RawGAT-ST") == "rawgatst"
    assert runner._slug("RawGAT-ST") not in runner._XLA_UNFRIENDLY_TRAINING_ARCHITECTURES
    assert runner._compact("RawGAT-ST") in runner._XLA_UNFRIENDLY_TRAINING_ARCHITECTURES
