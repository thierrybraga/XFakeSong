"""Compatibilidade com huggingface_hub para bibliotecas que ainda esperam
``HfFolder`` (removido em huggingface_hub >= 0.16, mas ainda importado
transitivamente por Gradio v4 e por versões antigas de ``transformers``).
"""

import logging

logger = logging.getLogger(__name__)


def ensure_hf_folder_shim() -> None:
    """Cria um shim de ``HfFolder`` em ``huggingface_hub`` se necessário.

    Deve ser chamado ANTES de qualquer import que possa transitivamente
    precisar de ``HfFolder`` (Gradio, transformers antigos).
    """
    try:
        from huggingface_hub import HfFolder  # noqa: F401
    except ImportError:
        import huggingface_hub

        class _HfFolder:
            """Shim para HfFolder removido em huggingface_hub >= 0.16."""

        huggingface_hub.HfFolder = _HfFolder
        logger.debug("Shim HfFolder aplicado (huggingface_hub >= 0.16).")
