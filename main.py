#!/usr/bin/env python3
"""
Ponto de entrada principal do sistema XfakeSong.
Fornece acesso à interface de linha de comando (CLI) e interface gráfica (Gradio).
"""
import sys
import argparse
import logging
from pathlib import Path

# Adicionar diretório raiz ao PYTHONPATH
sys.path.insert(0, str(Path(__file__).parent))

from app.interfaces.cli.context import AppContext
from app.interfaces.cli.menus.main_menu import MainMenu

def setup_logging():
    """Configura o sistema de logging."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('system.log'),
            logging.StreamHandler()
        ]
    )

def main():
    """Função principal."""
    parser = argparse.ArgumentParser(description="XfakeSong System")
    parser.add_argument("--gui", action="store_true", help="Iniciar interface gráfica (Gradio)")
    parser.add_argument("--port", type=int, default=7860, help="Porta para interface gráfica")
    args = parser.parse_args()

    setup_logging()
    logger = logging.getLogger("Main")

    if args.gui:
        logger.info("Iniciando interface gráfica...")
        try:
            # Importar e lançar a aplicação Gradio
            from gradio_app import demo
            demo.launch(server_name="0.0.0.0", server_port=args.port, share=False)
        except ImportError as e:
            logger.error(f"Erro ao importar gradio_app: {e}")
            logger.info("Certifique-se de que todas as dependências estão instaladas.")
            sys.exit(1)
        except Exception as e:
            logger.error(f"Erro ao iniciar GUI: {e}")
            sys.exit(1)
    else:
        logger.info("Iniciando interface CLI...")
        try:
            # Inicializar contexto e menu principal
            context = AppContext()
            menu = MainMenu(context)
            menu.show()
        except KeyboardInterrupt:
            print("\nEncerrando...")
        except Exception as e:
            logger.error(f"Erro fatal: {e}")
            print(f"\n❌ Erro fatal: {e}")
            sys.exit(1)

if __name__ == "__main__":
    main()
