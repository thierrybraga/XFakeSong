"""Tipos e classes base para extração de características.

Re-exports canônicos — a fonte de verdade está em:
  app/core/interfaces/audio.py  (FeatureType)
  app/core/interfaces/base.py   (ProcessingResult, ProcessingStatus)
"""

from app.core.interfaces.audio import FeatureType  # noqa: F401
from app.core.interfaces.base import ProcessingResult, ProcessingStatus  # noqa: F401
