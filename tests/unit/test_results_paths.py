"""Contratos da raiz canônica de resultados."""

from pathlib import Path

import pytest

from app.core.config.paths import resolve_results_dir, resolve_results_output

RESULTS_ENV_VARS = (
    "RESULTS_DIR",
    "DEEPFAKE_RESULTS_DIR",
    "XFAKE_RESULTS_DIR",
    "XFAKE_STORAGE_DIR",
    "DEEPFAKE_STORAGE_DIR",
)


@pytest.fixture(autouse=True)
def clean_results_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in RESULTS_ENV_VARS:
        monkeypatch.delenv(name, raising=False)


def test_results_dir_defaults_to_project_data_results() -> None:
    assert resolve_results_dir(Path("/project")) == Path("/project/data/results")


@pytest.mark.parametrize(
    ("name", "value"),
    (
        ("RESULTS_DIR", "first"),
        ("DEEPFAKE_RESULTS_DIR", "second"),
        ("XFAKE_RESULTS_DIR", "third"),
    ),
)
def test_results_dir_accepts_all_aliases(
    monkeypatch: pytest.MonkeyPatch,
    name: str,
    value: str,
) -> None:
    monkeypatch.setenv(name, value)
    assert resolve_results_dir(Path("/project")) == Path("/project") / value


def test_results_alias_precedence(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("XFAKE_RESULTS_DIR", "third")
    monkeypatch.setenv("DEEPFAKE_RESULTS_DIR", "second")
    monkeypatch.setenv("RESULTS_DIR", "first")
    assert resolve_results_dir() == Path("first")


def test_storage_fallback_uses_results_subdirectory(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("XFAKE_STORAGE_DIR", "storage")
    assert resolve_results_dir() == Path("storage/results")


def test_configured_results_prefix_is_remapped(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("DEEPFAKE_RESULTS_DIR", "custom-results")
    resolved = resolve_results_output(
        "data/results/classical_benchmark",
        default_subdir="benchmark",
        base_dir=Path("/project"),
    )
    assert resolved == Path("/project/custom-results/classical_benchmark")


def test_absolute_output_has_priority(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RESULTS_DIR", "ignored")
    absolute = Path("/explicit/run")
    assert (
        resolve_results_output(
            absolute,
            default_subdir="benchmark",
            base_dir=Path("/project"),
        )
        == absolute
    )
