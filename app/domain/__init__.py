"""Camada de domínio: regras de negócio, ML, features e XAI.

Sem re-exports: importe do submódulo (``from
app.domain.services.detection_service import DetectionService``). Um
``__init__`` com imports eager fazia qualquer ``import app.domain.<x>``
carregar librosa e os doze extratores de features.
"""
