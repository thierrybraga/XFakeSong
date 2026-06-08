"""Regressões para manter a documentação da suíte de testes sincronizada."""

from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
TESTS = ROOT / "tests"
DOC = ROOT / "docs" / "06_TESTES.md"
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
        "./scripts/run_tests.sh fast",
        "./scripts/run_tests.sh cov",
        "make test",
        "pytest -m smoke tests/smoke/",
        ".github/workflows/ci.yml",
        ".github/workflows/static.yml",
        ".github/workflows/notebooks-execute.yml",
    ):
        assert term in text
