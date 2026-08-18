"""Regressões para manter a documentação da suíte de testes sincronizada."""

from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
TESTS = ROOT / "tests"
# O documento foi movido em b18a32f (reestruturação de docs/); o caminho antigo
# (docs/06_QUALIDADE_TESTES.md) deixava este teste falhando com FileNotFoundError.
DOC = ROOT / "docs" / "development" / "quality-and-testing.md"
CATEGORIES = ("unit", "api", "functional", "integration", "smoke")


def test_test_documentation_counts_match_tree():
    text = DOC.read_text(encoding="utf-8")
    counts = {
        category: len(list((TESTS / category).glob("test_*.py")))
        for category in CATEGORIES
    }
    total = sum(counts.values())

    assert f"Total atual: **{total} arquivos de teste**." in text
    for category, count in counts.items():
        label = "API" if category == "api" else category.capitalize()
        assert f"| {label} | `{category}` | {count} |" in text


def test_test_documentation_mentions_standard_entrypoints():
    text = DOC.read_text(encoding="utf-8")
    for term in (
        "./scripts/ops/run_tests.sh fast",
        "./scripts/ops/run_tests.sh cov",
        "make test",
        "pytest -m smoke tests/smoke/",
        ".github/workflows/ci.yml",
        ".github/workflows/static.yml",
        ".github/workflows/notebooks-execute.yml",
    ):
        assert term in text


# ─── convenção do caderno de testes (2026-08-17) ───────────────────────────
#
# A contagem acima detecta arquivo NOVO não documentado. Não detectava o que
# de fato corroeu a suíte: arquivos nomeados pelo episódio que os motivou
# ("P1", "Tier-1", "as correções de tal data"), sem docstring dizendo o
# sujeito, e dois arquivos cobrindo o mesmo módulo sem que nada apontasse. Foi
# assim que `CollapseAbort` acabou com dois donos e contratos contraditórios.


def _arquivos_de_teste():
    return sorted(TESTS.rglob("test_*.py"))


def test_todo_arquivo_de_teste_tem_docstring_de_modulo():
    """Primeira linha = o sujeito. Sem ela o arquivo só se explica pelo nome.

    Usa `ast`, não prefixo de texto: `tests/smoke/test_app_startup.py` abre com
    shebang (é executável standalone) e uma checagem por `startswith` o
    reprovaria por engano — foi exatamente esse falso negativo que fez a
    inserção automática das docstrings duplicar a dele.
    """
    import ast

    sem_doc = []
    for arquivo in _arquivos_de_teste():
        texto = arquivo.read_text(encoding="utf-8", errors="replace")
        if not ast.get_docstring(ast.parse(texto)):
            sem_doc.append(arquivo.relative_to(ROOT).as_posix())
    assert not sem_doc, (
        "arquivos de teste sem docstring de módulo — a convenção está em "
        f"docs/development/quality-and-testing.md: {sem_doc}"
    )


def test_nenhum_arquivo_nomeado_pelo_episodio():
    """O nome é o SUJEITO. Rótulo de fase/data envelhece e esconde duplicata."""
    import re

    proibidos = re.compile(
        r"test_(p\d_|tier\d|fase\d|sprint|wip_|tmp_|old_|new_|fix_\d)", re.I
    )
    ruins = [
        f.relative_to(ROOT).as_posix()
        for f in _arquivos_de_teste()
        if proibidos.search(f.name)
    ]
    assert not ruins, (
        "renomeie pelo módulo/comportamento sob contrato, não pela fase do "
        f"backlog que o motivou: {ruins}"
    )


def test_nome_de_teste_e_unico_no_repositorio():
    """Homônimo em arquivos diferentes quebra `-k` e esconde duplicação real.

    Quando duas camadas testam a mesma rota, o nome tem de dizer qual é qual
    (`..._via_api` x `..._no_servico`).
    """
    import collections
    import re

    vistos = collections.defaultdict(list)
    for arquivo in _arquivos_de_teste():
        texto = arquivo.read_text(encoding="utf-8", errors="replace")
        for m in re.finditer(r"^\s*(?:async )?def (test_\w+)", texto, re.M):
            vistos[m.group(1)].append(arquivo.relative_to(ROOT).as_posix())

    colisoes = {n: sorted(set(v)) for n, v in vistos.items() if len(set(v)) > 1}
    assert not colisoes, f"nomes de teste repetidos entre arquivos: {colisoes}"
