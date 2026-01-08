import gradio as gr
import os
import sys
from pathlib import Path
import logging

# Adicionar diretório app ao path
sys.path.insert(0, str(Path(__file__).parent))

# Configurar logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("gradio_app")

# Imports de Autenticação e DB
from app.domain.services.auth_service import AuthService
from app.core.db_setup import get_flask_app

# --- Funções de Autenticação e Navegação (DESATIVADAS) ---
# O sistema agora opera em modo aberto sem login obrigatório.
# As funções abaixo foram mantidas comentadas para referência futura ou reativação.

# def login(username, password): ...
# def register(...): ...
# def recover(email): ...
# def logout(): ...
# def show_login(): ...
# def show_register(): ...
# def show_recovery(): ...

# SEO e Metadados
_HEAD_HTML = """
<meta name="description" content="XfakeSong Platform - Plataforma avançada para detecção, análise e treinamento de modelos anti-spoofing.">
<meta name="keywords" content="deepfake, audio, detection, ai, machine learning, security, forensics">
<meta name="author" content="XfakeSong Team">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<meta property="og:title" content="XfakeSong Platform">
<meta property="og:description" content="Ferramenta profissional para análise de integridade de áudio e detecção de deepfakes.">
<meta property="og:type" content="website">
"""

# CSS Personalizado
_CUSTOM_CSS = """
.gradio-container { 
    font-family: 'Segoe UI', Roboto, Helvetica, Arial, sans-serif; 
    min-height: 100vh !important;
}
h1, h2, h3 { font-weight: 600 !important; color: #1e293b; }
footer { visibility: hidden !important; height: 0 !important; overflow: hidden !important; }
/* Melhoria em botões */
button.primary { font-weight: bold !important; transition: all 0.2s; }
button.primary:hover { transform: translateY(-1px); shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1); }
"""

# Importar abas modulares
def _create_error_tab(name, error_msg):
    def _tab():
        with gr.Tab(f"Erro: {name}"):
            gr.Markdown(f"### ⚠️ Falha ao carregar aba {name}\n\nErro: `{error_msg}`")
    return _tab

try:
    from app.interfaces.gradio.tabs.features import create_features_tab
except ImportError as e:
    logger.error(f"Erro ao importar features tab: {e}")
    create_features_tab = _create_error_tab("Features", str(e))

try:
    from app.interfaces.gradio.tabs.training import create_training_tab
except ImportError as e:
    logger.error(f"Erro ao importar training tab: {e}")
    create_training_tab = _create_error_tab("Training", str(e))

try:
    from app.interfaces.gradio.tabs.detection import create_detection_tab
except ImportError as e:
    logger.error(f"Erro ao importar detection tab: {e}")
    create_detection_tab = _create_error_tab("Detection", str(e))

try:
    from app.interfaces.gradio.tabs.optimization import create_optimization_tab
except ImportError as e:
    logger.error(f"Erro ao importar optimization tab: {e}")
    create_optimization_tab = _create_error_tab("Optimization", str(e))

try:
    from app.interfaces.gradio.tabs.history import create_history_tab
except ImportError as e:
    logger.error(f"Erro ao importar history tab: {e}")
    create_history_tab = _create_error_tab("History", str(e))

# Configuração do Tema
theme = gr.themes.Soft(
    primary_hue="blue",
    secondary_hue="slate",
    font=["Inter", "ui-sans-serif", "system-ui", "sans-serif"], # Remover GoogleFont para evitar dependência externa
).set(
    button_primary_background_fill="*primary_600",
    button_primary_background_fill_hover="*primary_700",
    block_title_text_weight="600",
)

# Interface Principal
with gr.Blocks(title="XfakeSong - XAI Enhanced") as demo:
    gr.HTML(f"<style>{_CUSTOM_CSS}</style>")
    
    # Estado de Login (Fixo como True para bypass)
    is_logged_in = gr.State(True)

    # --- 1. Container de Autenticação (DESATIVADO) ---
    # O código abaixo foi comentado para remover a Login Wall.
    # with gr.Column(visible=True) as auth_container: ...

    # --- 2. Container da Aplicação (Protegido) ---
    with gr.Column(visible=True) as app_container:
        with gr.Row(elem_id="app_header"):
            with gr.Column(scale=4):
                gr.Markdown("## 🛡️ XfakeSong")
        
        with gr.Tabs() as main_tabs:
            # Aba de Detecção
            create_detection_tab()
            
            # Aba de Histórico
            create_history_tab()
            
            # Aba de Configurações
            with gr.Tab("Configurações", id="tab_settings"):
                with gr.Tabs():
                    create_features_tab()
                    create_training_tab()
                    create_optimization_tab()

    # --- Eventos de Autenticação e Navegação (DESATIVADOS) ---
    
    # 1. Login
    # l_btn.click(...)
    # 2. Navegação para Registro
    # goto_register_btn.click(...)
    # 3. Navegação para Recuperação
    # goto_recovery_btn.click(...)
    # 4. Voltar para Login (de Registro)
    # back_login_r.click(...)
    # 5. Voltar para Login (de Recuperação)
    # back_login_rec.click(...)
    # 6. Registrar Usuário
    # r_btn.click(...)
    # 7. Recuperar Senha
    # rec_btn.click(...)
    # 8. Logout
    # logout_btn.click(...)



# Habilitar fila explicitamente para suporte a streaming e concorrência
# Importante para evitar erros de "queue/data" abortado
demo.queue(default_concurrency_limit=20, max_size=None)

def create_unified_app(port: int):
    from fastapi import FastAPI
    from fastapi.middleware.cors import CORSMiddleware
    from a2wsgi import WSGIMiddleware
    from app.core.db_setup import get_flask_app
    from app.domain.models.architectures.registry import architecture_registry
    from app.extensions import db

    flask_app = get_flask_app()
    if not flask_app.secret_key:
        flask_app.secret_key = 'dev-key-unified'

    with flask_app.app_context():
        db.create_all()
        architecture_registry.sync_defaults_to_db()

    app = FastAPI()
    
    # Adicionar CORS Middleware para evitar bloqueios de recursos
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
        allow_credentials=True,
    )

    flask_wsgi = WSGIMiddleware(flask_app)
    
    # Montar Gradio em /gradio (deve vir antes do mount root do Flask se possível, 
    # mas gr.mount_gradio_app modifica o app existente)
    # Nota: mount_gradio_app adiciona rotas ao app FastAPI.
    app = gr.mount_gradio_app(app, demo, path="/gradio")

    # Montar Flask na raiz para servir /, /static, /auth, /api/v1 etc.
    # O WSGIMiddleware em "/" captura tudo que não foi capturado antes.
    app.mount("/", flask_wsgi)

    return app

if __name__ == "__main__":
    import uvicorn
    chosen_port = int(os.getenv("GRADIO_SERVER_PORT", "7860"))
    app = create_unified_app(chosen_port)
    logger.info("================================================================")
    logger.info(f"SERVIDOR UNIFICADO INICIADO! ACESSE EM: http://localhost:{chosen_port}")
    logger.info(f" - Interface Gradio: http://localhost:{chosen_port}/")
    logger.info(f" - API Backend:      http://localhost:{chosen_port}/api/v1/system/bootstrap")
    logger.info("================================================================")
    uvicorn.run(app, host="0.0.0.0", port=chosen_port)
