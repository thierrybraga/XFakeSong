"""Gate de vulnerabilidades da release (`tools/ops/check_pip_audit.py`).

SUJEITO: a decisão de aprovar ou reprovar, a partir do relatório do pip-audit e da
lista de exceções. O arquivo real de exceções também é conferido contra os pinos do
projeto: sem isso, um pino alterado só apareceria na hora de publicar a versão.
"""

from __future__ import annotations

import json
import re
import subprocess
from datetime import date
from pathlib import Path

import pytest

from tools.ops import check_pip_audit as gate

ROOT = Path(__file__).resolve().parents[2]
HOJE = date(2026, 11, 2)


def _excecao(**campos) -> dict:
    base = {
        "package": "torch",
        "version": "2.5.1",
        "review_by": "2027-02-01",
        "reason": "só lê artefatos locais",
        "blocked_by": "segfault em GPU",
        "ids": ["PYSEC-1"],
    }
    return {**base, **campos}


def _lista(tmp_path: Path, *entradas) -> Path:
    arquivo = tmp_path / "excecoes.json"
    arquivo.write_text(json.dumps({"exceptions": list(entradas)}), encoding="utf-8")
    return arquivo


def _pacote(nome="torch", versao="2.5.1", avisos=()) -> dict:
    return {
        "name": nome,
        "version": versao,
        "vulns": [
            {"id": i, "fix_versions": ["9.9"], "aliases": list(a), "description": ""}
            for i, *a in avisos
        ],
    }


def _relatorio(*pacotes) -> dict:
    return {"dependencies": list(pacotes), "fixes": []}


def _avaliar(tmp_path, relatorio, *entradas, hoje=HOJE):
    return gate.evaluate(
        relatorio, gate.load_exceptions(_lista(tmp_path, *entradas)), hoje
    )


def test_ambiente_limpo_e_sem_excecoes_passa(tmp_path):
    falhas, notas = _avaliar(tmp_path, _relatorio(_pacote("numpy", "2.2.6")))
    assert (falhas, notas) == ([], [])


def test_aviso_sem_excecao_reprova_e_nomeia_pacote_e_id(tmp_path):
    falhas, _ = _avaliar(
        tmp_path, _relatorio(_pacote("gradio", "6.17.3", [("GHSA-X",)]))
    )
    assert len(falhas) == 1
    assert "gradio" in falhas[0] and "GHSA-X" in falhas[0]


def test_aviso_coberto_passa(tmp_path):
    falhas, notas = _avaliar(
        tmp_path, _relatorio(_pacote(avisos=[("PYSEC-1",)])), _excecao()
    )
    assert (falhas, notas) == ([], [])


def test_aviso_novo_no_pacote_com_excecao_reprova(tmp_path):
    # O caso que justifica o gate: a exceção vale para os IDs avaliados, não para
    # tudo o que aparecer depois naquele pacote.
    falhas, _ = _avaliar(
        tmp_path, _relatorio(_pacote(avisos=[("PYSEC-1",), ("PYSEC-2",)])), _excecao()
    )
    assert len(falhas) == 1 and "PYSEC-2" in falhas[0]


def test_aviso_coberto_por_alias_passa(tmp_path):
    # A base pode trocar o ID principal; o alias continua identificando o aviso.
    falhas, _ = _avaliar(
        tmp_path,
        _relatorio(_pacote(avisos=[("GHSA-NOVO", "PYSEC-1")])),
        _excecao(),
    )
    assert falhas == []


def test_aviso_repetido_no_relatorio_gera_uma_falha(tmp_path):
    # O pip-audit repete o mesmo ID quando ele vem de mais de uma fonte.
    falhas, _ = _avaliar(
        tmp_path, _relatorio(_pacote("gradio", "6.17.3", [("GHSA-X",), ("GHSA-X",)]))
    )
    assert len(falhas) == 1


def test_excecao_vence_no_dia_seguinte_ao_prazo(tmp_path):
    relatorio = _relatorio(_pacote(avisos=[("PYSEC-1",)]))
    assert _avaliar(tmp_path, relatorio, _excecao(), hoje=date(2027, 2, 1))[0] == []
    falhas, _ = _avaliar(tmp_path, relatorio, _excecao(), hoje=date(2027, 2, 2))
    assert len(falhas) == 1 and "venceu" in falhas[0]


def test_versao_diferente_da_registrada_reprova_e_nao_cobre_os_avisos(tmp_path):
    falhas, _ = _avaliar(
        tmp_path, _relatorio(_pacote(versao="2.6.0", avisos=[("PYSEC-1",)])), _excecao()
    )
    assert any("avaliou 2.5.1" in f for f in falhas)
    assert any("PYSEC-1" in f for f in falhas)


def test_excecao_de_pacote_ausente_reprova(tmp_path):
    falhas, _ = _avaliar(tmp_path, _relatorio(_pacote("numpy", "2.2.6")), _excecao())
    assert len(falhas) == 1 and "torch" in falhas[0]


def test_pacote_nao_auditado_reprova(tmp_path):
    relatorio = _relatorio({"name": "local", "skip_reason": "não está no PyPI"})
    falhas, _ = _avaliar(tmp_path, relatorio)
    assert len(falhas) == 1 and "não auditado" in falhas[0]


def test_id_que_saiu_do_relatorio_so_gera_nota(tmp_path):
    falhas, notas = _avaliar(tmp_path, _relatorio(_pacote()), _excecao())
    assert falhas == []
    assert len(notas) == 1 and "PYSEC-1" in notas[0]


def test_nome_do_pacote_e_normalizado(tmp_path):
    falhas, _ = _avaliar(
        tmp_path,
        _relatorio(_pacote("Pillow", "12.3.0", [("PYSEC-1",)])),
        _excecao(package="pillow", version="12.3.0"),
    )
    assert falhas == []


@pytest.mark.parametrize(
    "campos",
    [
        {"reason": "  "},
        {"blocked_by": None},
        {"ids": []},
        {"ids": "PYSEC-1"},
        {"review_by": "em breve"},
        {"review_by": None},
        {"version": ""},
    ],
)
def test_excecao_incompleta_invalida_a_lista(tmp_path, campos):
    with pytest.raises(ValueError):
        gate.load_exceptions(_lista(tmp_path, _excecao(**campos)))


def test_excecao_duplicada_invalida_a_lista(tmp_path):
    with pytest.raises(ValueError, match="duplicada"):
        gate.load_exceptions(_lista(tmp_path, _excecao(), _excecao(package="Torch")))


def test_main_devolve_1_ao_reprovar_e_0_ao_aprovar(tmp_path, capsys):
    excecoes = _lista(tmp_path, _excecao())
    relatorio = tmp_path / "relatorio.json"
    args = [
        "--exceptions",
        str(excecoes),
        "--report",
        str(relatorio),
        "--today",
        "2026-11-02",
    ]

    relatorio.write_text(
        json.dumps(_relatorio(_pacote(avisos=[("PYSEC-1",)]))), "utf-8"
    )
    assert gate.main(args) == 0

    relatorio.write_text(
        json.dumps(_relatorio(_pacote(avisos=[("PYSEC-9",)]))), "utf-8"
    )
    assert gate.main(args) == 1
    assert "PYSEC-9" in capsys.readouterr().out


@pytest.mark.parametrize("conteudo", ["", "{}", '{"dependencies": []}', "[]"])
def test_main_devolve_2_para_relatorio_sem_pacotes(tmp_path, conteudo):
    relatorio = tmp_path / "relatorio.json"
    relatorio.write_text(conteudo, "utf-8")
    args = ["--exceptions", str(_lista(tmp_path)), "--report", str(relatorio)]
    assert gate.main(args) == 2


@pytest.mark.parametrize(
    ("codigo", "saida"),
    [(2, ""), (1, ""), (1, "ERROR: sem rede")],
    ids=["codigo-de-erro", "saida-vazia", "saida-nao-json"],
)
def test_pip_audit_que_nao_conclui_nao_aprova(tmp_path, monkeypatch, codigo, saida):
    # O pip-audit também sai com 1 em erro geral: o 1 sozinho não prova auditoria.
    def falso_run(*_args, **_kwargs):
        return subprocess.CompletedProcess([], codigo, stdout=saida, stderr="falhou")

    monkeypatch.setattr(gate.subprocess, "run", falso_run)
    assert gate.main(["--exceptions", str(_lista(tmp_path))]) == 2


def test_auditoria_propria_com_avisos_chega_a_avaliacao(tmp_path, monkeypatch):
    saida = json.dumps(_relatorio(_pacote(avisos=[("PYSEC-1",)])))

    def falso_run(*_args, **_kwargs):
        return subprocess.CompletedProcess([], 1, stdout=saida, stderr="")

    monkeypatch.setattr(gate.subprocess, "run", falso_run)
    args = ["--exceptions", str(_lista(tmp_path, _excecao())), "--today", "2026-11-02"]
    assert gate.main(args) == 0


def _pino(arquivo: str, pacote: str) -> str:
    texto = (ROOT / arquivo).read_text(encoding="utf-8")
    return re.search(rf"(?mi)^{re.escape(pacote)}==([^\s#]+)", texto).group(1)


def test_lista_real_e_valida_e_acompanha_os_pinos():
    excecoes = gate.load_exceptions(ROOT / gate.DEFAULT_EXCEPTIONS)
    assert sorted(excecoes) == ["pillow", "torch", "transformers"]
    assert excecoes["torch"].version == _pino("requirements.txt", "torch")
    assert excecoes["transformers"].version == _pino("requirements.txt", "transformers")
    assert excecoes["pillow"].version == _pino("constraints.txt", "pillow")


def test_o_workflow_de_release_usa_o_gate_e_os_pinos():
    workflow = (ROOT / ".github/workflows/release-gate.yml").read_text(encoding="utf-8")
    assert "python tools/ops/check_pip_audit.py" in workflow
    # `pip-audit` puro reprovaria toda release; com `|| true`, nenhuma.
    assert not re.search(r"(?m)^\s*(run: )?pip-audit\b", workflow)
    # Sem os pinos, a CI auditaria versões diferentes das que a imagem publica.
    assert workflow.count("-c constraints.txt") == 2
