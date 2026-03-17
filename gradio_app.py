import gradio as gr
import os
import sys
from pathlib import Path
import logging

# Adicionar diretório app ao path
sys.path.insert(0, str(Path(__file__).parent))

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("gradio_app")

# Imports de Autenticação e DB
# from app.domain.services.auth_service import AuthService  # Unused
# from app.core.db_setup import get_flask_app  # Unused at top level

# --- Funções de Autenticação e Navegação (DESATIVADAS) ---
# O sistema agora opera em modo aberto sem login obrigatório.
# As funções abaixo foram mantidas comentadas para referência futura
# ou reativação.

# def login(username, password): ...
# def register(...): ...
# def recover(email): ...
# def logout(): ...
# def show_login(): ...
# def show_register(): ...
# def show_recovery(): ...

# SEO e Metadados
_HEAD_HTML = """
<meta name="description"
 content="XfakeSong Platform - Plataforma avançada para detecção, análise e
 treinamento de modelos anti-spoofing.">
<meta name="keywords"
 content="deepfake, audio, detection, ai, machine learning, security,
 forensics">
<meta name="author" content="XfakeSong Team">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<meta property="og:title" content="XfakeSong Platform">
<meta property="og:description"
 content="Ferramenta profissional para análise de integridade de áudio e
 detecção de deepfakes.">
<meta property="og:type" content="website">
"""
# noqa: E501

# CSS Personalizado — Paleta Profissional + Animações + Tipografia
_CUSTOM_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

:root {
    --xf-bg:            #0f172a;
    --xf-surface:       #1e293b;
    --xf-surface-hover: #334155;
    --xf-border:        #334155;
    --xf-border-light:  #475569;
    --xf-text:          #f1f5f9;
    --xf-text-muted:    #94a3b8;
    --xf-primary:       #3b82f6;
    --xf-primary-hover: #2563eb;
    --xf-primary-glow:  rgba(59, 130, 246, 0.25);
    --xf-accent:        #06b6d4;
    --xf-success:       #10b981;
    --xf-warning:       #f59e0b;
    --xf-danger:        #ef4444;
    --xf-gradient:      linear-gradient(135deg, #3b82f6 0%, #06b6d4 100%);
    --xf-radius:        10px;
    --xf-radius-lg:     16px;
    --xf-shadow:        0 4px 24px rgba(0, 0, 0, 0.25);
    --xf-shadow-lg:     0 8px 40px rgba(0, 0, 0, 0.35);
    --xf-transition:    all 0.25s cubic-bezier(0.4, 0, 0.2, 1);
    --xf-font:          'Inter', ui-sans-serif, system-ui, -apple-system, sans-serif;
    --xf-mono:          'JetBrains Mono', ui-monospace, monospace;
}

.gradio-container {
    font-family: var(--xf-font) !important;
    min-height: 100vh !important;
    background: var(--xf-bg) !important;
    color: var(--xf-text) !important;
}

h1, h2, h3, h4, h5 {
    font-weight: 700 !important;
    color: var(--xf-text) !important;
    letter-spacing: -0.02em !important;
}
h1 { font-size: 1.75rem !important; }
h2 { font-size: 1.5rem !important; }
h3 { font-size: 1.25rem !important; }
h4 { font-size: 1.1rem !important; }

footer {
    visibility: hidden !important;
    height: 0 !important;
    overflow: hidden !important;
}

#app_header {
    background: var(--xf-gradient) !important;
    padding: 16px 24px !important;
    border-radius: var(--xf-radius-lg) !important;
    margin-bottom: 12px !important;
    box-shadow: var(--xf-shadow) !important;
}
#app_header h2 {
    color: #fff !important;
    font-size: 1.6rem !important;
    text-shadow: 0 2px 4px rgba(0,0,0,0.2);
}

.gr-group, .gr-box, .gr-panel {
    background: var(--xf-surface) !important;
    border: 1px solid var(--xf-border) !important;
    border-radius: var(--xf-radius) !important;
    box-shadow: var(--xf-shadow) !important;
    transition: var(--xf-transition) !important;
}
.gr-group:hover, .gr-box:hover {
    border-color: var(--xf-border-light) !important;
}

.tabs > .tab-nav {
    background: var(--xf-surface) !important;
    border-bottom: 2px solid var(--xf-border) !important;
    border-radius: var(--xf-radius) var(--xf-radius) 0 0 !important;
    gap: 2px !important;
    padding: 4px 8px 0 8px !important;
}
.tabs > .tab-nav > button {
    font-family: var(--xf-font) !important;
    font-weight: 500 !important;
    font-size: 0.9rem !important;
    color: var(--xf-text-muted) !important;
    border: none !important;
    border-radius: var(--xf-radius) var(--xf-radius) 0 0 !important;
    padding: 10px 20px !important;
    transition: var(--xf-transition) !important;
    background: transparent !important;
}
.tabs > .tab-nav > button:hover {
    color: var(--xf-text) !important;
    background: var(--xf-surface-hover) !important;
}
.tabs > .tab-nav > button.selected {
    color: var(--xf-primary) !important;
    font-weight: 600 !important;
    background: var(--xf-bg) !important;
    border-bottom: 3px solid var(--xf-primary) !important;
}

button.primary, button[variant="primary"] {
    background: var(--xf-gradient) !important;
    color: #fff !important;
    font-weight: 600 !important;
    font-family: var(--xf-font) !important;
    border: none !important;
    border-radius: var(--xf-radius) !important;
    padding: 10px 24px !important;
    transition: var(--xf-transition) !important;
    box-shadow: 0 2px 8px var(--xf-primary-glow) !important;
}
button.primary:hover, button[variant="primary"]:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 6px 20px var(--xf-primary-glow) !important;
    filter: brightness(1.1) !important;
}
button.primary:active, button[variant="primary"]:active {
    transform: translateY(0px) !important;
}
button.secondary, button[variant="secondary"] {
    background: var(--xf-surface) !important;
    color: var(--xf-text) !important;
    border: 1px solid var(--xf-border) !important;
    border-radius: var(--xf-radius) !important;
    transition: var(--xf-transition) !important;
}
button.secondary:hover, button[variant="secondary"]:hover {
    background: var(--xf-surface-hover) !important;
    border-color: var(--xf-primary) !important;
}
button.stop, button[variant="stop"] {
    background: rgba(239, 68, 68, 0.15) !important;
    color: var(--xf-danger) !important;
    border: 1px solid var(--xf-danger) !important;
    border-radius: var(--xf-radius) !important;
    transition: var(--xf-transition) !important;
}
button.stop:hover, button[variant="stop"]:hover {
    background: rgba(239, 68, 68, 0.3) !important;
    transform: translateY(-1px) !important;
}

input, textarea, select, .gr-input, .gr-textbox textarea {
    background: var(--xf-bg) !important;
    color: var(--xf-text) !important;
    border: 1px solid var(--xf-border) !important;
    border-radius: var(--xf-radius) !important;
    font-family: var(--xf-font) !important;
    transition: var(--xf-transition) !important;
    padding: 8px 12px !important;
}
input:focus, textarea:focus, select:focus {
    border-color: var(--xf-primary) !important;
    box-shadow: 0 0 0 3px var(--xf-primary-glow) !important;
    outline: none !important;
}
input::placeholder, textarea::placeholder {
    color: var(--xf-text-muted) !important;
    opacity: 0.6 !important;
}

input[type="range"] { accent-color: var(--xf-primary) !important; }

.gr-dataframe { border-radius: var(--xf-radius) !important; overflow: hidden !important; }
table thead th {
    background: var(--xf-surface) !important;
    color: var(--xf-text) !important;
    font-weight: 600 !important;
    font-size: 0.85rem !important;
    text-transform: uppercase !important;
    letter-spacing: 0.05em !important;
    padding: 12px 16px !important;
    border-bottom: 2px solid var(--xf-primary) !important;
}
table tbody td {
    padding: 10px 16px !important;
    border-bottom: 1px solid var(--xf-border) !important;
}
table tbody tr:hover { background: var(--xf-surface-hover) !important; }

.gr-accordion {
    border: 1px solid var(--xf-border) !important;
    border-radius: var(--xf-radius) !important;
    background: var(--xf-surface) !important;
}

.json-holder, pre, code {
    font-family: var(--xf-mono) !important;
    font-size: 0.85rem !important;
    background: var(--xf-bg) !important;
    border-radius: var(--xf-radius) !important;
}

.gr-file, .upload-container {
    border: 2px dashed var(--xf-border) !important;
    border-radius: var(--xf-radius) !important;
    transition: var(--xf-transition) !important;
    background: var(--xf-surface) !important;
}
.gr-file:hover, .upload-container:hover {
    border-color: var(--xf-primary) !important;
    background: rgba(59, 130, 246, 0.05) !important;
}

.gr-markdown { line-height: 1.7 !important; }
.gr-markdown hr { border-color: var(--xf-border) !important; margin: 16px 0 !important; }
.gr-markdown strong { color: var(--xf-text) !important; }

@keyframes fadeInUp {
    from { opacity: 0; transform: translateY(12px); }
    to   { opacity: 1; transform: translateY(0); }
}
@keyframes pulseGlow {
    0%, 100% { box-shadow: 0 0 8px var(--xf-primary-glow); }
    50%      { box-shadow: 0 0 20px var(--xf-primary-glow); }
}
@keyframes shimmer {
    0%   { background-position: -200% 0; }
    100% { background-position: 200% 0; }
}

.gr-group, .gr-box { animation: fadeInUp 0.4s ease-out !important; }
button.primary:focus { animation: pulseGlow 1.5s infinite !important; }

::-webkit-scrollbar { width: 8px; height: 8px; }
::-webkit-scrollbar-track { background: var(--xf-bg); }
::-webkit-scrollbar-thumb { background: var(--xf-border); border-radius: 4px; }
::-webkit-scrollbar-thumb:hover { background: var(--xf-border-light); }

.progress-bar { background: var(--xf-gradient) !important; border-radius: 4px !important; }
.gr-number input { font-variant-numeric: tabular-nums !important; }

@media (max-width: 768px) {
    .gr-row { flex-direction: column !important; }
    button { width: 100% !important; }
}
"""


# Importar abas modulares
def _create_error_tab(name, error_msg):
    def _tab():
        with gr.Tab(f"Erro: {name}"):
            gr.Markdown(
                f"### ⚠️ Falha ao carregar aba {name}\n\nErro: `{error_msg}`"
            )
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

try:
    from app.interfaces.gradio.tabs.forensic_analysis import create_forensic_analysis_tab
except ImportError as e:
    logger.error(f"Erro ao importar forensic analysis tab: {e}")
    create_forensic_analysis_tab = _create_error_tab("Forensic Analysis", str(e))

try:
    from app.interfaces.gradio.tabs.voice_profiles import create_voice_profiles_tab
except ImportError as e:
    logger.error(f"Erro ao importar voice profiles tab: {e}")
    create_voice_profiles_tab = _create_error_tab("Voice Profiles", str(e))


# Configuração do Tema — Dark Mode Profissional
theme = gr.themes.Base(
    primary_hue="blue",
    secondary_hue="slate",
    neutral_hue="slate",
    font=["Inter", "ui-sans-serif", "system-ui", "-apple-system", "sans-serif"],
    font_mono=["JetBrains Mono", "ui-monospace", "monospace"],
).set(
    # Backgrounds
    body_background_fill="#0f172a",
    body_background_fill_dark="#0f172a",
    background_fill_primary="#1e293b",
    background_fill_primary_dark="#1e293b",
    background_fill_secondary="#0f172a",
    background_fill_secondary_dark="#0f172a",
    # Borders
    border_color_primary="#334155",
    border_color_primary_dark="#334155",
    block_border_color="#334155",
    block_border_color_dark="#334155",
    # Text
    body_text_color="#f1f5f9",
    body_text_color_dark="#f1f5f9",
    body_text_color_subdued="#94a3b8",
    body_text_color_subdued_dark="#94a3b8",
    # Buttons
    button_primary_background_fill="linear-gradient(135deg, #3b82f6, #06b6d4)",
    button_primary_background_fill_hover="linear-gradient(135deg, #2563eb, #0891b2)",
    button_primary_text_color="#ffffff",
    button_secondary_background_fill="#1e293b",
    button_secondary_text_color="#f1f5f9",
    # Blocks
    block_background_fill="#1e293b",
    block_background_fill_dark="#1e293b",
    block_title_text_weight="600",
    block_title_text_color="#f1f5f9",
    block_title_text_color_dark="#f1f5f9",
    block_label_text_color="#94a3b8",
    block_label_text_color_dark="#94a3b8",
    block_label_text_weight="500",
    block_shadow="0 4px 24px rgba(0, 0, 0, 0.25)",
    block_shadow_dark="0 4px 24px rgba(0, 0, 0, 0.25)",
    # Inputs
    input_background_fill="#0f172a",
    input_background_fill_dark="#0f172a",
    input_border_color="#334155",
    input_border_color_dark="#334155",
    input_border_color_focus="#3b82f6",
    input_border_color_focus_dark="#3b82f6",
    # Shadow
    shadow_spread="8px",
)


# Interface Principal
with gr.Blocks(
    title="XfakeSong — Audio Deepfake Detection",
    theme=theme,
    css=_CUSTOM_CSS,
    head=_HEAD_HTML,
) as demo:

    # Estado de Login (Fixo como True para bypass)
    is_logged_in = gr.State(True)

    # --- Container da Aplicação ---
    with gr.Column(visible=True) as app_container:
        with gr.Row(elem_id="app_header"):
            with gr.Column(scale=4):
                gr.Markdown("## 🛡️ XfakeSong — Audio Integrity Platform")

        gr.Markdown(
            "<p style='margin:-8px 0 12px 2px;opacity:0.65;font-size:0.88rem'>"
            "Plataforma profissional para detecção, análise forense e "
            "verificação de autenticidade de áudio.</p>"
        )

        with gr.Tabs() as main_tabs:
            # Aba de Detecção
            create_detection_tab()

            # Aba de Análise Forense
            create_forensic_analysis_tab()

            # Aba de Perfis de Voz
            create_voice_profiles_tab()

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
    from app.core.db_setup import init_db
    from app.core.exceptions import setup_exception_handlers
    from app.core.middleware import setup_middleware
    from app.routers import (
        system, detection, features, training, history, datasets,
        voice_profiles,
    )
    from app.core.security import setup_security, limiter
    from slowapi.errors import RateLimitExceeded
    from slowapi import _rate_limit_exceeded_handler

    # Inicializar Banco de Dados
    init_db()

    app = FastAPI(
        title="XfakeSong Unified",
        description="API e UI unificada para detecção de deepfakes.",
        version="1.1.0",
        docs_url="/api/docs",
        redoc_url="/api/redoc",
        openapi_url="/api/openapi.json",
    )

    # Configurar Segurança, Middleware e Exception Handlers
    setup_security(app)
    setup_middleware(app)
    setup_exception_handlers(app)
    app.state.limiter = limiter
    app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

    # Incluir Routers
    app.include_router(system.router)
    app.include_router(detection.router)
    app.include_router(features.router)
    app.include_router(training.router)
    app.include_router(history.router)
    app.include_router(datasets.router)
    app.include_router(voice_profiles.router)

    # Montar Gradio na raiz "/" para manter a experiência do usuário
    # Se quiser em "/gradio", mude o path abaixo.
    app = gr.mount_gradio_app(
        app, 
        demo, 
        path="/", 
        allowed_paths=[os.path.abspath(".")]
    )

    return app


if __name__ == "__main__":
    import uvicorn
    chosen_port = int(os.getenv("GRADIO_SERVER_PORT", "7860"))
    app = create_unified_app(chosen_port)
    logger.info(
        "================================================================"
    )
    logger.info(
        f"SERVIDOR UNIFICADO INICIADO! ACESSE EM: http://localhost:{chosen_port}"  # noqa: E501
    )
    logger.info(
        f" - Interface Gradio: http://localhost:{chosen_port}/gradio/"
    )
    logger.info(
        f" - API Backend:      http://localhost:{chosen_port}/api/v1/system/bootstrap"  # noqa: E501
    )
    logger.info(
        "================================================================"
    )
    uvicorn.run(app, host="0.0.0.0", port=chosen_port)
