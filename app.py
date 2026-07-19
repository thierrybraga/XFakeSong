import sys
from pathlib import Path

# Adicionar diretório raiz ao PYTHONPATH para garantir importações corretas
sys.path.insert(0, str(Path(__file__).parent))

from app.core.hf_compat import ensure_hf_folder_shim  # noqa: E402

ensure_hf_folder_shim()  # antes do import do Gradio (que precisa de HfFolder)

from app.interfaces.gradio.app import demo  # noqa: E402

if __name__ == "__main__":
    # Hugging Face Spaces disponibiliza a porta na variável de ambiente
    # Mas o demo.launch() lida com isso automaticamente se não
    # especificarmos server_name/port ou podemos ser explícitos.
    demo.launch()
