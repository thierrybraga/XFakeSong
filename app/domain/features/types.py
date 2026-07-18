"""Tipos e classes base para extração de características.

Re-exports canônicos — a fonte de verdade está em:
  app/core/contracts/audio.py  (FeatureType)
  app/core/contracts/base.py   (ProcessingResult, ProcessingStatus)
"""

from app.core.contracts.audio import FeatureType  # noqa: F401
from app.core.contracts.base import ProcessingResult, ProcessingStatus  # noqa: F401
