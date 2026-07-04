"""Utilitários compartilhados pelos CLIs de ``scripts/``.

Todo script executável deste pacote é chamado a partir da raiz do
repositório (``python scripts/<categoria>/<nome>.py``). Como o Python coloca
apenas o diretório do próprio arquivo em ``sys.path``, cada script inicia com
o cabeçalho padrão de três linhas::

    ROOT = Path(__file__).resolve().parents[2]
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))

Após esse cabeçalho, os helpers abaixo ficam importáveis via
``from scripts._bootstrap import ...`` e evitam duplicação de código de
infraestrutura (resolução da raiz e configuração de logging).
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

#: Raiz do repositório (pai de ``scripts/``).
REPO_ROOT: Path = Path(__file__).resolve().parents[1]


def ensure_repo_root_on_sys_path() -> Path:
    """Garante que a raiz do repositório esteja em ``sys.path``.

    Idempotente; retorna :data:`REPO_ROOT` para conveniência.
    """
    root = str(REPO_ROOT)
    if root not in sys.path:
        sys.path.insert(0, root)
    return REPO_ROOT


def setup_logging(level: int = logging.INFO, name: str | None = None) -> logging.Logger:
    """Configura logging de console no formato padrão dos CLIs.

    Args:
        level: nível mínimo (padrão ``INFO``).
        name: nome do logger; ``None`` usa o logger do módulo chamador.

    Returns:
        Logger pronto para uso.
    """
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    return logging.getLogger(name or "scripts")
