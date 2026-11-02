"""Leitura centralizada do ambiente e do arquivo ``.env``.

Fonte única para os helpers que antes estavam copiados em ``dependencies``,
``main_fastapi``, ``auth_handler``, ``middleware`` e ``performance``.

O ``.env`` era carregado só como efeito colateral do import de
``app.core.security``: a CLI, os scripts e a resolução de diretórios feita
pelo ``main.py`` antes de subir a API ignoravam o arquivo. Os pontos de
entrada chamam :func:`load_environment` explicitamente.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

PROJECT_ROOT = Path(__file__).resolve().parents[3]
_TRUE_VALUES = frozenset({"1", "true", "yes", "on"})
_loaded = False


def load_environment(dotenv_path: str | Path | None = None) -> bool:
    """Carrega o ``.env`` da raiz uma única vez por processo.

    Variáveis já definidas no ambiente têm precedência (``override=False``).
    Retorna ``True`` quando um arquivo foi lido nesta chamada.
    """
    global _loaded
    if _loaded:
        return False
    _loaded = True
    try:
        from dotenv import load_dotenv
    except ImportError:  # python-dotenv é opcional fora do runtime web
        return False
    path = Path(dotenv_path) if dotenv_path else PROJECT_ROOT / ".env"
    return bool(load_dotenv(path, override=False))


def env_flag(name: str, default: bool = False) -> bool:
    """Interpreta ``1/true/yes/on`` (sem diferenciar maiúsculas) como verdadeiro."""
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in _TRUE_VALUES


def env_int(name: str, default: int) -> int:
    """Inteiro do ambiente; valor ausente ou inválido devolve ``default``."""
    try:
        return int(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        return default


def env_positive_int(name: str) -> Optional[int]:
    """Inteiro estritamente positivo, ou ``None`` quando ausente/inválido."""
    raw = os.getenv(name)
    if raw in (None, ""):
        return None
    try:
        value = int(raw)
    except ValueError:
        return None
    return value if value > 0 else None


def first_env(*names: str) -> Optional[str]:
    """Primeiro valor não vazio entre ``names``, sem espaços nas bordas."""
    for name in names:
        value = os.getenv(name)
        if value and value.strip():
            return value.strip()
    return None
