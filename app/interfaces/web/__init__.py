"""Interface Web (FastAPI + Jinja2) do XFakeSong.

Terceira interface de entrada do projeto, ao lado de `app.interfaces.gradio`
e `app.interfaces.cli`. Expõe a API REST (`routers/`, `schemas/`) e a página
HTML de entrada (`templates/`, `static/`) montadas em `main_fastapi.py`.

Uso típico:
    from app.interfaces.web.main_fastapi import app
"""

from __future__ import annotations
