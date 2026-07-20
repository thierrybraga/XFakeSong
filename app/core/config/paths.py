"""Resolução centralizada dos diretórios operacionais do projeto."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable

RESULTS_DIR_ENV_VARS = (
    "RESULTS_DIR",
    "DEEPFAKE_RESULTS_DIR",
    "XFAKE_RESULTS_DIR",
)


def _first_env_path(names: Iterable[str]) -> Path | None:
    """Retorna o primeiro caminho não vazio definido no ambiente."""
    for name in names:
        value = os.getenv(name)
        if value and value.strip():
            return Path(value.strip()).expanduser()
    return None


def resolve_results_dir(base_dir: str | Path | None = None) -> Path:
    """Resolve a raiz única de resultados."""
    path = _first_env_path(RESULTS_DIR_ENV_VARS)
    if path is None:
        storage = _first_env_path(("XFAKE_STORAGE_DIR", "DEEPFAKE_STORAGE_DIR"))
        path = storage / "results" if storage is not None else Path("data/results")

    if base_dir is not None and not path.is_absolute():
        path = Path(base_dir) / path
    return path


def resolve_results_output(
    configured: str | Path | None,
    *,
    default_subdir: str | Path,
    base_dir: str | Path | None = None,
) -> Path:
    """Resolve uma saída dentro da raiz canônica."""
    if configured is not None:
        candidate = Path(configured).expanduser()
        if candidate.is_absolute():
            return candidate
        parts = candidate.parts
        if len(parts) >= 2 and parts[:2] == ("data", "results"):
            relative = Path(*parts[2:])
        elif parts and parts[0] == "results":
            relative = Path(*parts[1:])
        else:
            relative = candidate
    else:
        relative = Path(default_subdir)

    return resolve_results_dir(base_dir) / relative
