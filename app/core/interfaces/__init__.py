"""Backward-compatible aliases for core contracts.

The canonical package is :mod:`app.core.contracts`. This module exists so older
notebooks, scripts, and examples that still import ``app.core.interfaces`` keep
working during the architecture migration.
"""

from app.core.contracts.audio import *  # noqa: F403
from app.core.contracts.base import *  # noqa: F403
from app.core.contracts.services import *  # noqa: F403
