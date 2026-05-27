#!/usr/bin/env python3
"""Smoke test de importação e inicialização do gradio_app.

Verifica que o módulo principal pode ser importado sem erros e que
o objeto 'demo' (Gradio Blocks) está presente.

Uso standalone:
    python tests/smoke/test_app_startup.py

Uso via pytest (inclui mark smoke):
    pytest -m smoke tests/smoke/test_app_startup.py
"""
from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

# tests/smoke/ → tests/ → raiz
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def main() -> int:
    try:
        logger.info("Importando gradio_app...")
        import gradio_app  # noqa: F401, E402

        logger.info("OK: gradio_app importado sem erros")

        if hasattr(gradio_app, "demo"):
            logger.info("OK: objeto 'demo' encontrado")
            print("READY: Gradio app pronto para iniciar")
        else:
            logger.warning("WARN: objeto 'demo' não encontrado — verifique gradio_app.py")
            return 1

    except Exception as e:
        logger.error(f"FAILED: {e}", exc_info=True)
        return 1

    return 0


# ── pytest integration ────────────────────────────────────────────────────────
import pytest  # noqa: E402


@pytest.mark.smoke
def test_gradio_app_startup() -> None:
    """gradio_app deve ser importável e expor 'demo'."""
    rc = main()
    assert rc == 0, "gradio_app startup falhou — veja saída acima"


# ── standalone ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    sys.exit(main())
