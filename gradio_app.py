import logging
import os
import sys
from pathlib import Path

from app.core.performance import configure_runtime_environment

configure_runtime_environment()

# Adicionar diretório app ao path ANTES de imports do projeto
sys.path.insert(0, str(Path(__file__).parent))

# === Patch para compatibilidade Pydantic v2 + Gradio v4 ===
# Deve ser ANTES de qualquer import de gradio ou pydantic
from app.gradio_schema_patch import patch_gradio_schema_validator  # noqa: E402

patch_gradio_schema_validator()

# === Compatibilidade huggingface_hub ===
# HfFolder foi removido em versoes >= 0.16. Criar shim para evitar
# erros de import transitive de dependencias antigas.
try:
    from huggingface_hub import HfFolder  # noqa: F401
except ImportError:
    import huggingface_hub

    class _HfFolder:
        """Shim para HfFolder removido em huggingface_hub >= 0.16."""

        pass

    huggingface_hub.HfFolder = _HfFolder

# FE.2: força backend matplotlib não-interativo ANTES de qualquer
# import que importe pyplot transitivamente. Imprescindível em
# ambiente Gradio (threaded) e Docker (headless).
import matplotlib  # noqa: E402

if not os.environ.get("MPLBACKEND"):
    matplotlib.use("Agg", force=True)

import gradio as gr  # noqa: E402

# Configurar logging
from app.core.feedback import configure_logging  # noqa: E402

configure_logging(level=logging.INFO, log_file="system.log", force=False)
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
<meta name="author" content="XFakeSong Team">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<meta property="og:title" content="XfakeSong Platform">
<meta property="og:description"
 content="Ferramenta profissional para análise de integridade de áudio e
 detecção de deepfakes.">
<meta property="og:type" content="website">
"""
# noqa: E501

# CSS Personalizado carregado de arquivo estático.
def _load_custom_css() -> str:
    css_path = Path(__file__).parent / "app" / "static" / "css" / "gradio_theme.css"
    try:
        return css_path.read_text(encoding="utf-8")
    except OSError as exc:
        logger.warning("Falha ao carregar CSS Gradio de %s: %s", css_path, exc)
        return ""


_CUSTOM_CSS = _load_custom_css()


# FE.6: imports vêm do __init__ que já trata falhas individualmente.
# Cada create_*_tab é uma função; se a tab original falhou ao importar,
# o __init__ cria uma placeholder de erro automaticamente — não precisamos
# mais replicar try/except por tab aqui.
# Cleanup.2: `create_training_tab` (legacy) não é mais importado aqui — o
# wizard substituiu o workflow original. O arquivo training.py permanece em
# disco caso algum teste/script o use, mas a UI principal só expõe o wizard.
from app.interfaces.gradio.tabs import (  # noqa: E402
    create_dashboard_tab,
    create_dataset_management_tab,
    create_detection_tab,
    create_features_tab,
    create_forensic_analysis_tab,
    create_history_tab,
    create_optimization_tab,
    create_training_wizard_tab,
    create_voice_profiles_tab,
)

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


# =====================================================================
# UI Fase 1: Status bar global (sempre visível no topo)
# =====================================================================


def _render_status_bar() -> str:
    """HTML do status bar global com counts e versão.

    Lê dados ao vivo a cada render via dashboard collectors (fallback seguro).
    """
    try:
        from app.interfaces.gradio.tabs.dashboard import (
            _count_models,
            _count_profiles,
            _gpu_status,
        )

        models = _count_models()
        profiles = _count_profiles()
        gpu = _gpu_status()
    except Exception:
        models, profiles, gpu = 0, 0, "?"

    try:
        from app.core.feedback import get_feedback_summary

        feedback = get_feedback_summary()
        unread = int(feedback.get("unread", 0))
        counts = feedback.get("counts", {})
        warning_count = int(counts.get("warning", 0))
        error_count = int(counts.get("error", 0)) + int(counts.get("critical", 0))
    except Exception:
        unread, warning_count, error_count = 0, 0, 0

    gpu_ok = gpu.startswith("✓")
    gpu_short = "GPU ✓" if gpu_ok else ("GPU ✗" if gpu.startswith("✗") else "GPU ?")
    gpu_class = "status-ok" if gpu_ok else "status-muted"
    feedback_class = (
        "status-danger"
        if error_count
        else ("status-warning" if warning_count else "status-muted")
    )
    unread_html = f'<span class="notif-badge">{unread}</span>' if unread else ""
    return f"""
    <div id="status_bar_inner">
      <span class="brand">
        🛡️ XFakeSong <span class="brand-version">v1.1</span>
      </span>
      <span class="status-pill status-online">
        <span class="dot"></span>
        <span class="label">Online</span>
      </span>
      <span class="status-pill {gpu_class}">
        {gpu_short}
      </span>
      <span class="status-pill status-muted">
        <span class="status-value">{models}</span> modelos
      </span>
      <span class="status-pill status-muted">
        <span class="status-value">{profiles}</span> perfis
      </span>
      <span class="status-pill {feedback_class}">
        Notificações {unread_html}
        <span class="status-detail">warn:{warning_count} err:{error_count}</span>
      </span>
    </div>
    """


def _render_feedback_panel() -> str:
    from app.interfaces.gradio.utils.notifications import (
        render_notification_center_html,
    )

    return render_notification_center_html(limit=20)


def _refresh_global_feedback():
    return _render_status_bar(), _render_feedback_panel()


def _mark_feedback_read_ui():
    from app.interfaces.gradio.utils.notifications import mark_all_read

    mark_all_read()
    return _render_status_bar(), _render_feedback_panel()


def _clear_feedback_ui():
    from app.interfaces.gradio.utils.notifications import clear_history, notify_info

    removed = clear_history()
    notify_info(f"Historico de feedback limpo ({removed} eventos removidos)")
    return _render_status_bar(), _render_feedback_panel()


# =====================================================================
# Interface Principal (UI Fase 1 — 5 tabs role-based)
# =====================================================================

with gr.Blocks(
    title="XFakeSong — Audio Deepfake Detection Platform",
    theme=theme,
    css=_CUSTOM_CSS,
    head=_HEAD_HTML,
) as demo:
    # Estado de Login (Fixo como True para bypass)
    is_logged_in = gr.State(True)

    # --- Container da Aplicação ---
    with gr.Column(visible=True) as app_container:
        # Status bar global + toolbar (sempre visível)
        with gr.Row(elem_id="topbar_row"):
            with gr.Column(scale=8, min_width=0):
                status_bar_html = gr.HTML(_render_status_bar(), elem_id="status_bar")
            with gr.Column(scale=1, min_width=120):
                # UI Fase 3 — Toggles de tema e idioma
                with gr.Row():
                    theme_toggle_btn = gr.Button(
                        "🌙",
                        elem_classes="toolbar-toggle",
                        size="sm",
                        min_width=44,
                        scale=0,
                    )
                    lang_toggle_btn = gr.Button(
                        "🇧🇷 PT",
                        elem_classes="toolbar-toggle",
                        size="sm",
                        min_width=70,
                        scale=0,
                    )

        with gr.Accordion("🔔 Notificações pendentes", open=False):
            feedback_html = gr.HTML(
                _render_feedback_panel(),
                elem_id="global_feedback_center",
            )
            with gr.Row():
                feedback_refresh_btn = gr.Button("🔄 Atualizar", size="sm", scale=0)
                feedback_read_btn = gr.Button("Marcar lidas", size="sm", scale=0)
                feedback_clear_btn = gr.Button("Limpar histórico", size="sm", scale=0)

        # Navbar consolidada (5 seções role-based):
        # 🏠 Painel · 🎯 Detectar · 🔬 Investigar · 🎓 Treinar · 🗂️ Gerenciar
        with gr.Tabs() as main_tabs:
            # 🏠 Painel — landing page com KPIs, status e atividade recente
            # (a aba define seu próprio rótulo top-level "🏠 Painel")
            create_dashboard_tab()

            # 🎯 Detectar — análise de áudio (single + lote) e perfis de voz
            with gr.Tab("🎯 Detectar", id="tab_detect"):
                with gr.Tabs():
                    create_detection_tab()
                    create_voice_profiles_tab()

            # 🔬 Investigar — análise forense + explicabilidade (a aba define
            # seu próprio rótulo top-level "🔬 Investigar")
            create_forensic_analysis_tab()

            # 🎓 Treinar — assistente linear + otimização
            with gr.Tab("🎓 Treinar", id="tab_train"):
                training_enabled = (
                    os.getenv("ENABLE_TRAINING", "true").strip().lower()
                    not in {"0", "false", "no", "off"}
                )
                if training_enabled:
                    with gr.Tabs():
                        create_training_wizard_tab()
                        create_optimization_tab()
                else:
                    gr.Markdown(
                        "### Modo demonstração\n\n"
                        "O treinamento está desativado neste ambiente. "
                        "Use a aba **Detectar** para inferência com os modelos "
                        "já treinados."
                    )

            # 🗂️ Gerenciar — datasets, features e histórico
            with gr.Tab("🗂️ Gerenciar", id="tab_admin"):
                with gr.Tabs():
                    create_dataset_management_tab()
                    create_features_tab()
                    create_history_tab()

        # Auto-refresh do status bar a cada 60s
        try:
            _sb_timer = gr.Timer(15.0)
            _sb_timer.tick(
                fn=_refresh_global_feedback,
                inputs=[],
                outputs=[status_bar_html, feedback_html],
            )
        except Exception:
            # gr.Timer não disponível em versões antigas — degradação graciosa
            pass

        feedback_refresh_btn.click(
            fn=_refresh_global_feedback,
            inputs=[],
            outputs=[status_bar_html, feedback_html],
        )
        feedback_read_btn.click(
            fn=_mark_feedback_read_ui,
            inputs=[],
            outputs=[status_bar_html, feedback_html],
        )
        feedback_clear_btn.click(
            fn=_clear_feedback_ui,
            inputs=[],
            outputs=[status_bar_html, feedback_html],
        )

        # ───── UI Fase 3: Tema + Idioma ─────
        # Estados em memória do servidor (resetam ao recarregar página)
        theme_state = gr.State("dark")
        lang_state = gr.State("pt")

        # NOTE: o toggle de tema usa um lambda inline com `js=` no .click()
        # abaixo (aplica data-theme no DOM + persiste em localStorage). Não há
        # função Python dedicada — a antiga `_toggle_theme` foi removida por ser
        # código morto (nunca era chamada).

        def _toggle_lang(current_lang: str):
            """Alterna PT <-> EN. Atualiza estado global do i18n."""
            from app.interfaces.gradio.utils.i18n import set_language

            new_lang = "en" if current_lang == "pt" else "pt"
            set_language(new_lang)
            new_label = "🇺🇸 EN" if new_lang == "en" else "🇧🇷 PT"
            try:
                from app.interfaces.gradio.utils import notify_info

                notify_info(
                    "Language: English (some tabs will keep Portuguese — full i18n WIP)"
                    if new_lang == "en"
                    else "Idioma: Português"
                )
            except Exception:
                pass
            return new_lang, gr.update(value=new_label), _render_status_bar()

        # JS injection helper: usa o atributo `js=` do click handler (Gradio 4.x)
        try:
            theme_toggle_btn.click(
                fn=lambda t: (
                    "light" if t == "dark" else "dark",
                    gr.update(value="☀" if t == "dark" else "🌙"),
                ),
                inputs=[theme_state],
                outputs=[theme_state, theme_toggle_btn],
                js="""(theme) => {
                    const next = theme === 'dark' ? 'light' : 'dark';
                    document.body.setAttribute('data-theme', next);
                    try { localStorage.setItem('xf_theme', next); } catch(e) {}
                    return [theme];
                }""",
            )
        except TypeError:
            # Versões mais antigas de Gradio não aceitam js= em click
            theme_toggle_btn.click(
                fn=lambda t: (
                    "light" if t == "dark" else "dark",
                    gr.update(value="☀" if t == "dark" else "🌙"),
                ),
                inputs=[theme_state],
                outputs=[theme_state, theme_toggle_btn],
            )

        lang_toggle_btn.click(
            fn=_toggle_lang,
            inputs=[lang_state],
            outputs=[lang_state, lang_toggle_btn, status_bar_html],
        )

        # Restaura tema do localStorage no load (se disponível)
        try:
            demo.load(
                fn=None,
                inputs=[],
                outputs=[],
                js="""() => {
                    try {
                        const saved = localStorage.getItem('xf_theme') || 'dark';
                        document.body.setAttribute('data-theme', saved);
                    } catch(e) {}
                    return [];
                }""",
            )
        except Exception:
            pass


# PROD.5: Queue com limites para evitar DoS via flood de requests:
# - default_concurrency_limit=20: workers paralelos (matches detection.py expectation)
# - max_size=100: requests aguardando — após isso, novas viram 503
# Sem max_size, RAM pode esgotar com uploads grandes em fila.
demo.queue(default_concurrency_limit=20, max_size=100)


def create_unified_app(port: int):
    from fastapi import FastAPI
    from fastapi.requests import Request
    from fastapi.responses import HTMLResponse
    from fastapi.staticfiles import StaticFiles
    from fastapi.templating import Jinja2Templates
    from slowapi import _rate_limit_exceeded_handler
    from slowapi.errors import RateLimitExceeded

    from app.core.db_setup import init_db
    from app.core.exceptions import setup_exception_handlers
    from app.core.gpu import describe_gpu_setup, setup_gpu
    from app.core.middleware import setup_middleware
    from app.core.security import limiter, setup_security
    from app.core.version_check import check_versions
    from app.routers import (
        datasets,
        detection,
        features,
        history,
        system,
        training,
        voice_profiles,
    )

    # BUG.Render.2: detecta incompatibilidades de versão (ex.: gradio<4.31 +
    # starlette>=0.36 → TypeError em runtime). Loga warning claro.
    check_versions(strict=False)

    # GPU.2: configura TF cedo (memory_growth + mixed precision) ANTES de
    # qualquer model.fit. Idempotente — chamadas múltiplas são no-op.
    gpu_info = setup_gpu()
    logger.info(f"GPU setup: {describe_gpu_setup()}")
    if gpu_info.get("errors"):
        for err in gpu_info["errors"]:
            logger.warning(f"GPU config warning: {err}")

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

    app.mount(
        "/static",
        StaticFiles(directory="app/static"),
        name="static",
    )
    templates = Jinja2Templates(directory="app/templates")

    @app.get("/", response_class=HTMLResponse, include_in_schema=False)
    async def index(request: Request):
        return templates.TemplateResponse(
            "index.html",
            {"request": request},
        )

    @app.get("/loading", response_class=HTMLResponse, include_in_schema=False)
    async def loading(request: Request):
        return templates.TemplateResponse(
            "pages/loading.html",
            {"request": request},
        )

    # Incluir Routers
    app.include_router(system.router)
    app.include_router(detection.router)
    app.include_router(features.router)
    app.include_router(training.router)
    app.include_router(history.router)
    app.include_router(datasets.router)
    app.include_router(voice_profiles.router)

    # PROD.3: allowed_paths precisa incluir TODOS os diretórios que o Gradio
    # vai servir arquivos. Restrito a só "." dava 403 (ERR_ABORTED) em:
    #   - /tmp/gradio (uploads + cache, default temp dir)
    #   - /tmp/gradio_cached (alguns componentes do Gradio)
    #   - /tmp (fallback geral)
    # Sem isto, drag-and-drop de áudio falha silenciosamente no browser.
    allowed_paths = [
        os.path.abspath("."),
        os.environ.get("GRADIO_TEMP_DIR", "/tmp/gradio"),
        "/tmp",  # cobre numba, matplotlib, hf caches também
    ]
    # Diretórios podem não existir em dev local — Gradio ignora os ausentes
    # mas não falha por isso.

    app = gr.mount_gradio_app(
        app,
        demo,
        path="/gradio",
        allowed_paths=allowed_paths,
    )

    return app


if __name__ == "__main__":
    import uvicorn

    chosen_port = int(os.getenv("GRADIO_SERVER_PORT", "7860"))
    app = create_unified_app(chosen_port)
    logger.info("================================================================")
    logger.info(
        f"SERVIDOR UNIFICADO INICIADO! ACESSE EM: http://localhost:{chosen_port}"  # noqa: E501
    )
    logger.info(f" - Interface Gradio: http://localhost:{chosen_port}/gradio/")
    logger.info(
        f" - API Backend:      http://localhost:{chosen_port}/api/v1/system/bootstrap"  # noqa: E501
    )
    logger.info("================================================================")
    uvicorn.run(app, host="0.0.0.0", port=chosen_port)
