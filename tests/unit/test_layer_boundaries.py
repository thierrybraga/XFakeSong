"""Fronteiras de camada em `app/`: quem pode importar quem.

SUJEITO: o grafo de imports do pacote, lido do FONTE (AST) — sem executar
nada, então vale também sem TensorFlow/Gradio instalados. Conta todo import:
de topo, tardio dentro de função, sob `TYPE_CHECKING`, relativo
(`from ...core import x`) e `importlib.import_module("...")` com literal. Um
import tardio continua sendo dependência; foi assim que `core/db/setup.py`
manteve o ciclo `core → domain` depois de as entidades ORM saírem do domínio.

Regras (docs/architecture/overview.md):

- `app.core` é infraestrutura: não importa `app.domain`, `app.interfaces` nem
  a composição `app.dependencies` (que importa o domínio);
- `app.domain` não importa `app.interfaces` nem `app.dependencies`, nem
  framework de interface ou de plotagem (gradio, fastapi, matplotlib…) —
  gráfico é apresentação (P1-7);
- nada em `app/` importa `tools` — são CLIs operacionais sobre o `app/`,
  nunca o contrário.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
APP = ROOT / "app"


def _pacote(arquivo: Path) -> str:
    """Pacote contra o qual os imports relativos do arquivo resolvem."""
    partes = list(arquivo.relative_to(ROOT).with_suffix("").parts)
    return ".".join(partes[:-1])


def _alvos(codigo: str, pacote: str) -> list[tuple[int, str]]:
    """``(linha, módulo)`` de cada import do código, já absolutos."""
    alvos: list[tuple[int, str]] = []
    for no in ast.walk(ast.parse(codigo)):
        if isinstance(no, ast.Import):
            alvos.extend((no.lineno, alias.name) for alias in no.names)
        elif isinstance(no, ast.ImportFrom):
            if no.level:
                base = pacote.split(".")[: len(pacote.split(".")) - (no.level - 1)]
                modulo = ".".join(base + ([no.module] if no.module else []))
            else:
                modulo = no.module or ""
            alvos.append((no.lineno, modulo))
            # `from app import dependencies` importa o submódulo pelo nome.
            alvos.extend((no.lineno, f"{modulo}.{alias.name}") for alias in no.names)
        elif isinstance(no, ast.Call):
            nome = getattr(no.func, "attr", None) or getattr(no.func, "id", None)
            if (
                nome in ("import_module", "__import__")
                and no.args
                and isinstance(no.args[0], ast.Constant)
                and isinstance(no.args[0].value, str)
            ):
                alvos.append((no.lineno, no.args[0].value))
    return alvos


def _violacoes(camada: str, proibidos: tuple[str, ...]) -> list[str]:
    encontradas = []
    for arquivo in sorted((ROOT / camada.replace(".", "/")).rglob("*.py")):
        codigo = arquivo.read_text(encoding="utf-8")
        for linha, alvo in _alvos(codigo, _pacote(arquivo)):
            if any(alvo == p or alvo.startswith(p + ".") for p in proibidos):
                encontradas.append(
                    f"{arquivo.relative_to(ROOT).as_posix()}:{linha} → {alvo}"
                )
    return sorted(set(encontradas))


@pytest.mark.parametrize(
    ("camada", "proibidos"),
    [
        ("app.core", ("app.domain", "app.interfaces", "app.dependencies")),
        ("app.domain", ("app.interfaces", "app.dependencies")),
        (
            "app.domain",
            ("gradio", "fastapi", "starlette", "matplotlib", "plotly"),
        ),
        ("app", ("tools",)),
    ],
    ids=[
        "core-e-infraestrutura",
        "dominio-sem-interfaces",
        "dominio-sem-ui-nem-plotagem",
        "app-sem-tools",
    ],
)
def test_camada_nao_importa_o_que_esta_acima_dela(camada, proibidos):
    violacoes = _violacoes(camada, proibidos)
    assert (
        not violacoes
    ), f"`{camada}` não pode importar {', '.join(proibidos)}:\n" + "\n".join(violacoes)


def test_varredura_resolve_imports_relativos_tardios_e_dinamicos():
    """O detector precisa enxergar o que as regras proíbem — senão passa vazio."""
    codigo = (
        "from ...interfaces.web import x\n"
        "def f():\n"
        "    from ... import dependencies\n"
        "    import importlib\n"
        "    importlib.import_module('tools.dataset.download_datasets')\n"
    )
    alvos = {alvo for _, alvo in _alvos(codigo, "app.domain.services")}
    assert "app.interfaces.web" in alvos
    assert "app.dependencies" in alvos
    assert "tools.dataset.download_datasets" in alvos

    # E num arquivo real: o import relativo do domínio para o ORM em `core`.
    arquivo = APP / "domain" / "services" / "detection_service.py"
    alvos_reais = _alvos(arquivo.read_text(encoding="utf-8"), _pacote(arquivo))
    assert any(alvo.startswith("app.core.db.models") for _, alvo in alvos_reais)
