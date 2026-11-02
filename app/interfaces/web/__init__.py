"""Interface Web (FastAPI) do SpeechChecker.

Terceira interface de entrada do projeto, ao lado de `app.interfaces.gradio`
e `app.interfaces.cli`. Expõe a API REST (`routers/`, `schemas/`) e monta a
UI Gradio em `/gradio` dentro de `main_fastapi.py`; a raiz `/` redireciona
para a UI (ou para `/api/docs` em modo API-only).

Uso típico:
    from app.interfaces.web.main_fastapi import app
"""

from __future__ import annotations
