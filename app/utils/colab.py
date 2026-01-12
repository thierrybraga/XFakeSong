import sys
import os
import logging

logger = logging.getLogger(__name__)


def is_colab() -> bool:
    """Verifica se o código está rodando no Google Colab."""
    return 'google.colab' in sys.modules


def mount_drive(mount_path: str = '/content/drive'):
    """Monta o Google Drive se estiver no Colab."""
    if is_colab():
        from google.colab import drive
        if not os.path.exists(mount_path):
            drive.mount(mount_path)
            logger.info(f"Google Drive montado em {mount_path}")
        else:
            logger.info("Google Drive já está montado.")
    else:
        logger.warning(
            "Não está rodando no Colab. Ignorando montagem do Drive."
        )


def setup_colab_env(repo_path: str = '/content/TCC'):
    """Configura o ambiente do Colab (paths, dependências)."""
    if is_colab():
        if repo_path not in sys.path:
            sys.path.append(repo_path)
            logger.info(f"Adicionado {repo_path} ao sys.path")

        # Opcional: Instalar dependências se não estiverem presentes
        # (Geralmente feito via célula do notebook, mas pode ser
        # automatizado aqui)
