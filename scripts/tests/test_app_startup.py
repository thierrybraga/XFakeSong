#!/usr/bin/env python3
"""Smoke test de importação e inicialização do gradio_app.

Verifica que o módulo principal pode ser importado sem erros e que
o objeto 'demo' (Gradio Blocks) está presente.

Uso:
    python scripts/tests/test_app_startup.py
"""

import os
import sys
from pathlib import Path
import logging

# Adiciona a raiz do projeto ao PYTHONPATH
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

try:
    logger.info("Importando gradio_app...")
    import gradio_app  # noqa: E402
    logger.info("OK: gradio_app importado sem erros")

    if hasattr(gradio_app, "demo"):
        logger.info("OK: objeto 'demo' encontrado")
        print("READY: Gradio app pronto para iniciar")
    else:
        logger.warning("WARN: objeto 'demo' não encontrado — verifique gradio_app.py")
        sys.exit(1)

except Exception as e:
    logger.error(f"FAILED: {e}", exc_info=True)
    sys.exit(1)
