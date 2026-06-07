"""Garante que os notebooks ativos permanecem funcionais.

Trava o bug que motivou a regeneração: células de código com indentação
espúria (IndentationError) e markdown/JSON malformados. Cada célula de código
de cada notebook ATIVO (fora de `legacy/`) deve compilar; o JSON deve ser
nbformat 4 válido; e a estrutura esperada (index + features + 14 models +
3 pipeline) deve existir.

Não executa os notebooks (treino é lento) — apenas valida sintaxe e estrutura.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[2]
_NB = _ROOT / "notebooks"


def _active_notebooks():
    if not _NB.exists():
        return []
    return [p for p in sorted(_NB.rglob("*.ipynb")) if "legacy" not in p.parts]


def test_expected_notebook_structure():
    names = {p.relative_to(_NB).as_posix() for p in _active_notebooks()}
    assert "00_index.ipynb" in names
    assert "features/01_feature_extraction_study.ipynb" in names
    assert sum(1 for n in names if n.startswith("models/")) == 14
    assert sum(1 for n in names if n.startswith("pipeline/")) == 3


@pytest.mark.parametrize(
    "nb_path", _active_notebooks(),
    ids=[p.relative_to(_NB).as_posix() for p in _active_notebooks()],
)
def test_notebook_code_cells_compile(nb_path):
    nbj = json.loads(nb_path.read_text(encoding="utf-8"))
    assert nbj.get("nbformat") == 4, f"{nb_path.name}: nbformat inesperado"
    code_cells = [c for c in nbj["cells"] if c["cell_type"] == "code"]
    assert code_cells, f"{nb_path.name}: sem células de código"
    for i, cell in enumerate(code_cells):
        src = "".join(cell["source"])
        # Compila: pega IndentationError/SyntaxError (o bug original).
        compile(src, f"{nb_path.name}#code{i}", "exec")
