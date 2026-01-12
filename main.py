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
    parser.add_argument("--gradio", action="store_true", dest="gui", help="Alias para --gui")
    parser.add_argument("--port", type=int, default=7860, help="Porta para interface gráfica")
    parser.add_argument("--gradio-port", type=int, dest="port", help="Alias para --port")
    parser.add_argument("--bootstrap-dirs", action="store_true", help="Criar estrutura de diretórios e sair")
    parser.add_argument("--deploy", action="store_true", help="Iniciar assistente de deploy para Hugging Face")
    args = parser.parse_args()

    setup_logging()
    logger = logging.getLogger("Main")

    if args.deploy:
        try:
            from app.deploy_hf import deploy_interface
            deploy_interface()
            sys.exit(0)
        except ImportError as e:
            logger.error(f"Erro ao importar modulo de deploy: {e}")
            print("Certifique-se de que 'huggingface_hub' esta instalado (pip install huggingface_hub)")
            sys.exit(1)

    if args.bootstrap_dirs:
        logger.info("Criando estrutura de diretórios...")
        try:
            app_dir = Path(__file__).parent / "app"
            dirs = [
                app_dir / "datasets",
                app_dir / "models",
                app_dir / "results",
                app_dir / "datasets" / "samples",
                app_dir / "datasets" / "features"
            ]
            for d in dirs:
                d.mkdir(parents=True, exist_ok=True)
                logger.info(f"Diretório verificado/criado: {d}")
            print("Estrutura de diretórios criada com sucesso.")
            sys.exit(0)
        except Exception as e:
            logger.error(f"Erro ao criar diretórios: {e}")
            sys.exit(1)

    if args.gui:
        logger.info("Iniciando interface gráfica unificada...")
        try:
            # Importar e lançar a aplicação Unificada (Flask + Gradio)
            import uvicorn
            from gradio_app import create_unified_app
            
            # Criar app unificado
            app = create_unified_app(args.port)
            
            # Iniciar servidor Uvicorn
            logger.info(f"Servidor iniciado em http://0.0.0.0:{args.port}")
            uvicorn.run(app, host="0.0.0.0", port=args.port)
            
        except ImportError as e:
            logger.error(f"Erro ao importar dependencias da GUI: {e}")
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
