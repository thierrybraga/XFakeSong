# Legacy Flask startup — superseded by app/main_fastapi.py (FastAPI + Gradio).
# Retained as a thin compatibility shim so existing imports do not break.

from app.main_fastapi import app


def create_app(config=None):
    """Compatibility shim — returns the FastAPI application instance."""
    return app
