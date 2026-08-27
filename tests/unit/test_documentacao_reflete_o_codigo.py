"""A documentação descreve o estado ATUAL do código, não um anterior.

SUJEITO: as afirmações verificáveis de ``docs/models/architectures.md``,
``docs/evaluation/benchmark.md`` e ``CLAUDE.md`` sobre protocolo e arquiteturas.

Por que este teste existe
-------------------------
Documentação desatualizada não quebra nada — ela apenas faz alguém acreditar em
algo falso. A auditoria de 2026-08-21 encontrou, entre outros:

- ``architectures.md`` declarando que o RawNet2 **default** é a variante de
  verificação de locutor, quando o escopo oficial passara a treinar o baseline
  anti-spoofing;
- o mesmo arquivo descrevendo o banco sinc como "aprendível" depois de ele ter
  virado FIXO nas três arquiteturas;
- ``benchmark.md`` afirmando "seleção uniforme: melhor checkpoint pela menor
  val_loss" quando o protocolo passara a selecionar por ``val_eer``;
- ``CLAUDE.md`` apontando as camadas customizadas para ``layers.py`` depois da
  separação por arquitetura.

Nenhuma dessas afirmações produziria erro de execução. Todas produziriam um TCC
que descreve um sistema diferente do que gerou os números.

Os testes abaixo checam apenas o que é DERIVÁVEL do código — nunca a redação.
Um documento pode ser reescrito à vontade desde que não contradiga a fonte.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
ARQUITETURAS = ROOT / "docs" / "models" / "architectures.md"
BENCHMARK = ROOT / "docs" / "evaluation" / "benchmark.md"
GUIA = ROOT / "CLAUDE.md"


def _texto(caminho: Path) -> str:
    assert caminho.exists(), f"documento ausente: {caminho}"
    return caminho.read_text(encoding="utf-8")


def test_docs_nao_anunciam_o_rawnet2_de_locutor_como_o_do_escopo_oficial():
    """O manifesto e a doc precisam nomear a MESMA variante.

    A verificação é indireta de propósito: compara o rótulo de proveniência do
    manifesto (que viaja para o `results.json` e daí para o TCC) com o que a
    doc afirma.
    """
    from benchmarks.config import OFFICIAL_TCC_MODEL_MANIFEST

    entrada = next(
        i for i in OFFICIAL_TCC_MODEL_MANIFEST
        if i.get("benchmark_name") == "RawNet2"
    )
    variante = str(entrada.get("variant", "")).lower()
    texto = _texto(ARQUITETURAS)

    if "antispoofing" in variante:
        assert "é esta que o escopo oficial treina" in texto.lower(), (
            "o manifesto declara a variante anti-spoofing, mas a doc não diz "
            "que é ela que o escopo oficial treina"
        )
        # A frase antiga afirmava o oposto.
        assert "`rawnet2` (**default**) = **Improved RawNet**" not in texto, (
            "a doc ainda anuncia o Improved RawNet de locutor como o default "
            "do escopo oficial"
        )


def test_doc_nao_descreve_o_banco_sinc_como_aprendivel():
    """Se o código fixa o banco, a doc não pode chamá-lo de aprendível."""
    from app.domain.models.architectures import (  # noqa: F401
        aasist_layers, rawgat_layers, rawnet2_layers,
    )
    import inspect

    fixo_por_padrao = []
    for mod, classe in (
        (aasist_layers, "AasistSincConv"),
        (rawgat_layers, "RawGatSincConv"),
        (rawnet2_layers, "RawNet2SincConv"),
    ):
        assinatura = inspect.signature(getattr(mod, classe).__init__)
        param = assinatura.parameters.get("trainable_filters")
        fixo_por_padrao.append(param is not None and param.default is False)

    assert all(fixo_por_padrao), (
        "alguma camada sinc voltou a ser treinável por padrão; a doc e o "
        "manifesto declaram banco FIXO"
    )
    texto = _texto(ARQUITETURAS)
    assert "banco de filtros passa-banda aprendível" not in texto, (
        "a doc ainda descreve o banco sinc como aprendível"
    )


def test_doc_do_benchmark_declara_o_monitor_de_selecao_real():
    """`val_eer` é o critério do compose — a doc não pode anunciar `val_loss`."""
    compose = (ROOT / "docker" / "compose" / "benchmark.nvidia.yml").read_text(
        encoding="utf-8"
    )
    if "val_eer" not in compose:
        pytest.skip("compose não declara checkpoint-monitor; nada a checar")

    texto = _texto(BENCHMARK)
    assert "seleção uniforme: melhor checkpoint pela menor val_loss limpa" not in texto, (
        "benchmark.md ainda afirma seleção por val_loss enquanto o compose "
        "roda com --checkpoint-monitor val_eer"
    )
    assert "val_eer" in texto, "benchmark.md não menciona o monitor real"


def test_guia_aponta_as_camadas_para_os_modulos_que_existem():
    """Os módulos citados no CLAUDE.md precisam existir de fato."""
    texto = _texto(GUIA)
    citados = re.findall(r"architectures/(\w+_layers)\.py", texto)
    assert citados, "CLAUDE.md não cita nenhum módulo de camadas por arquitetura"
    for modulo in set(citados):
        caminho = ROOT / "app" / "domain" / "models" / "architectures" / f"{modulo}.py"
        assert caminho.exists(), f"CLAUDE.md cita {modulo}.py, que não existe"


def test_guia_registra_a_separacao_de_camadas():
    """A separação é a mudança estrutural mais recente; precisa estar no guia."""
    texto = _texto(GUIA)
    assert "Camadas PRÓPRIAS de cada arquitetura" in texto, (
        "o CLAUDE.md não distingue camadas próprias das compartilhadas"
    )


def test_limitacoes_declaradas_batem_com_o_compose():
    """Se o compose roda uma semente, o guia tem de declarar isso como limitação.

    A `--seeds` existe no runner e NÃO é passada pelo compose. Declarar a
    limitação é o mínimo; escondê-la faria o TCC ordenar modelos separados por
    ruído amostral.
    """
    compose = (ROOT / "docker" / "compose" / "benchmark.nvidia.yml").read_text(
        encoding="utf-8"
    )
    uma_semente = "--seeds" not in compose
    if not uma_semente:
        pytest.skip("compose passa --seeds; a limitação não se aplica")
    texto = _texto(GUIA)
    assert "Uma semente por arquitetura" in texto, (
        "o compose roda com uma semente e o guia não declara a limitação"
    )
