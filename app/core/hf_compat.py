"""Compatibilidade com bibliotecas que esperam ``huggingface_hub.HfFolder``.

Criado para o Gradio 4, que importava essa classe. Com os pinos atuais o shim não
faz nada: ``huggingface_hub<1.0`` ainda a oferece, e o Gradio 6 não a usa.
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
            """Shim para HfFolder, removido em huggingface_hub 1.0."""

        huggingface_hub.HfFolder = _HfFolder
        logger.debug("Shim HfFolder aplicado (huggingface_hub sem a classe).")
