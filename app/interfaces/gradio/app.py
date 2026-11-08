import html
import logging
import os
from pathlib import Path

from app import __version__
from app.core.hf_compat import ensure_hf_folder_shim
from app.core.performance import configure_runtime_environment

configure_runtime_environment()

# === Patch de schemas, criado para Pydantic v2 + Gradio v4 ===
# Deve ser ANTES de qualquer import de gradio ou pydantic
from app.interfaces.gradio.schema_patch import (  # noqa: E402
    patch_gradio_schema_validator,
)

patch_gradio_schema_validator()

ensure_hf_folder_shim()

# FE.2: força backend matplotlib não-interativo ANTES de qualquer
# import que importe pyplot transitivamente. Imprescindível em
# ambiente Gradio (threaded) e Docker (headless).
import matplotlib  # noqa: E402

if not os.environ.get("MPLBACKEND"):
    matplotlib.use("Agg", force=True)

import gradio as gr  # noqa: E402

from app.core.bootstrap import OperationalPaths  # noqa: E402

# Configurar logging
from app.core.feedback import configure_logging  # noqa: E402

configure_logging(
    level=logging.INFO,
    log_file=str(OperationalPaths.resolve().logs / "system.log"),
    force=False,
)
logger = logging.getLogger("gradio_app")

# SEO e Metadados. O ícone embutido evita o `GET /favicon.ico -> 404` a cada
# abertura e mantém a aplicação sem dependência externa.
_HEAD_HTML = """
<meta name="description"
 content="SpeechChecker Platform - Plataforma avançada para detecção, análise e
 treinamento de modelos anti-spoofing.">
<meta name="keywords"
 content="deepfake, audio, detection, ai, machine learning, security,
 forensics">
<meta name="author" content="SpeechChecker Team">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<meta name="color-scheme" content="dark light">
<meta name="mobile-web-app-capable" content="yes">
<meta property="og:title" content="SpeechChecker Platform">
<meta property="og:description"
 content="Ferramenta profissional para análise de integridade de áudio e
 detecção de deepfakes.">
<meta property="og:type" content="website">
<meta property="og:locale" content="pt_BR">
<meta name="theme-color" content="#0f172a">
<link rel="icon" href="data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg'
 viewBox='0 0 100 100'%3E%3Ctext y='.9em' font-size='90'%3E🛡️%3C/text%3E%3C/svg%3E">
"""
# noqa: E501


# CSS Personalizado carregado de arquivo estático.
def _load_custom_css() -> str:
    """Carrega o tema Gradio local e registra diagnostico de referencia."""
    # app/interfaces/gradio/app.py -> gradio -> interfaces -> app -> raiz
    _repo_root = Path(__file__).resolve().parents[3]
    css_path = (
        _repo_root / "app" / "interfaces" / "gradio" / "static" / "css" / "theme.css"
    )
    try:
        css = css_path.read_text(encoding="utf-8")
        if not css.strip():
            logger.error("CSS Gradio vazio: %s", css_path)
            return ""
        logger.info("CSS Gradio carregado: %s (%d bytes)", css_path, len(css))
        return css
    except OSError as exc:
        logger.error("Falha ao carregar CSS Gradio de %s: %s", css_path, exc)
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
    create_benchmark_tab,
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

# Tema com os dois modos declarados. Antes os valores "claros" repetiam os escuros e
# o modo claro era remendado por CSS, o que deixava componentes nativos de fora.
theme = gr.themes.Base(
    primary_hue="blue",
    secondary_hue="slate",
    neutral_hue="slate",
    font=["Inter", "ui-sans-serif", "system-ui", "-apple-system", "sans-serif"],
    font_mono=["JetBrains Mono", "ui-monospace", "monospace"],
).set(
    # Backgrounds
    body_background_fill="#f8fafc",
    body_background_fill_dark="#0f172a",
    background_fill_primary="#ffffff",
    background_fill_primary_dark="#1e293b",
    background_fill_secondary="#f1f5f9",
    background_fill_secondary_dark="#0f172a",
    # Borders
    border_color_primary="#e2e8f0",
    border_color_primary_dark="#334155",
    block_border_color="#e2e8f0",
    block_border_color_dark="#334155",
    # Text
    body_text_color="#0f172a",
    body_text_color_dark="#f1f5f9",
    body_text_color_subdued="#475569",
    body_text_color_subdued_dark="#a3b1c6",
    # Buttons
    # Gradiente levemente mais escuro que a marca para manter contraste AA
    # do texto branco sobre a extremidade ciano (espelha --xf-gradient no CSS).
    button_primary_background_fill="linear-gradient(135deg, #2563eb, #0e7490)",
    button_primary_background_fill_dark="linear-gradient(135deg, #2563eb, #0891b2)",
    button_primary_background_fill_hover="linear-gradient(135deg, #1d4ed8, #0e7490)",
    button_primary_background_fill_hover_dark=(
        "linear-gradient(135deg, #1d4ed8, #0e7490)"
    ),
    button_primary_text_color="#ffffff",
    button_primary_text_color_dark="#ffffff",
    button_secondary_background_fill="#ffffff",
    button_secondary_background_fill_dark="#1e293b",
    button_secondary_text_color="#0f172a",
    button_secondary_text_color_dark="#f1f5f9",
    # Blocks
    block_background_fill="#ffffff",
    block_background_fill_dark="#1e293b",
    block_title_text_weight="600",
    block_title_text_color="#0f172a",
    block_title_text_color_dark="#f1f5f9",
    block_label_text_color="#475569",
    block_label_text_color_dark="#a3b1c6",
    block_label_text_weight="500",
    block_shadow="0 1px 3px rgba(15, 23, 42, 0.08)",
    block_shadow_dark="0 1px 3px rgba(0, 0, 0, 0.3)",
    # Inputs: a borda precisa de contraste 3:1 com o cartão para o campo ser visível.
    input_background_fill="#ffffff",
    input_background_fill_dark="#0f172a",
    input_border_color="#94a3b8",
    input_border_color_dark="#64748b",
    input_border_color_focus="#2563eb",
    input_border_color_focus_dark="#60a5fa",
    # Shadow
    shadow_spread="4px",
)


# =====================================================================
# Barra superior (sempre visível)
# =====================================================================


def _render_status_bar() -> str:
    """HTML da barra superior: marca, estado real e contagens.

    Lê dados ao vivo a cada render via dashboard collectors (fallback seguro).
    O estado vem dos modelos: "Operacional" fixo não dizia nada, inclusive sem nenhum
    modelo para analisar. O texto completo da GPU fica no `title` e no Painel.
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

    pronto = models > 0
    estado_classe = "status-ok" if pronto else "status-warning"
    estado_texto = "Pronto" if pronto else "Sem modelos"
    hardware = "GPU" if gpu.startswith("✓") else "CPU"
    return f"""
    <div id="status_bar_inner">
      <h1 class="brand">
        <span aria-hidden="true">🛡️</span> SpeechChecker
        <span class="brand-version">v{__version__}</span>
      </h1>
      <span class="status-pill {estado_classe}">
        <span class="dot" aria-hidden="true"></span>{estado_texto}
      </span>
      <span class="status-pill" title="{html.escape(gpu, quote=True)}">{hardware}</span>
      <span class="status-pill">
        <span class="status-value">{models}</span> modelos
      </span>
      <span class="status-pill">
        <span class="status-value">{profiles}</span> perfis
      </span>
    </div>
    """


def _refresh_model_choices():
    from app.interfaces.gradio.tabs.detection import refresh_model_choices

    return gr.update(choices=refresh_model_choices())


def _notif_label() -> str:
    """Sino com a contagem do que pede atenção; sem número quando não há nada."""
    from app.interfaces.gradio.utils.notifications import attention_count

    pendentes = attention_count()
    return f"🔔 {pendentes}" if pendentes else "🔔"


def _render_feedback_panel() -> str:
    from app.interfaces.gradio.utils.notifications import (
        render_notification_center_html,
    )

    return render_notification_center_html(limit=20)


def _refresh_global_feedback():
    return _render_status_bar(), _render_feedback_panel(), _notif_label()


def _mark_feedback_read_ui():
    from app.interfaces.gradio.utils.notifications import mark_all_read

    mark_all_read()
    return _render_status_bar(), _render_feedback_panel(), _notif_label()


def _clear_feedback_ui():
    from app.interfaces.gradio.utils.notifications import clear_history, notify_info

    removed = clear_history()
    notify_info(f"Histórico de notificações limpo ({removed} removidas)")
    return _render_status_bar(), _render_feedback_panel(), _notif_label()


# Desde o Gradio 6 a aparência vai para `launch`/`mount_gradio_app`, não para
# `gr.Blocks`. Passada ao `Blocks` ela é ignorada com um aviso: a interface sobe e
# funciona, mas sem o tema e o CSS. Nenhum teste de lógica percebe.
APPEARANCE = {"theme": theme, "css": _CUSTOM_CSS, "head": _HEAD_HTML}

# O tema e o painel mudam só no navegador: resposta imediata e nenhuma ida ao servidor.
# A preferência é a classe `.dark` do próprio Gradio. O padrão continua escuro porque
# as figuras são geradas com fundo escuro.
_THEME_LABELS = """
    const rotularTema = (escuro) => {
        const botao = document.querySelector('#theme_toggle');
        if (!botao) return;
        const alvo = escuro ? 'Mudar para o tema claro' : 'Mudar para o tema escuro';
        botao.textContent = escuro ? '🌙' : '☀️';
        botao.setAttribute('aria-label', alvo);
        botao.setAttribute('title', alvo);
    };
"""
_THEME_TOGGLE_JS = "() => {" + _THEME_LABELS + """
    const escuro = !document.body.classList.contains('dark');
    document.body.classList.toggle('dark', escuro);
    try { localStorage.setItem('xf_theme', escuro ? 'dark' : 'light'); } catch (e) {}
    rotularTema(escuro);
}"""
_ON_LOAD_JS = "() => {" + _THEME_LABELS + """
    let salvo = 'dark';
    try { salvo = localStorage.getItem('xf_theme') || 'dark'; } catch (e) {}
    const escuro = salvo !== 'light';
    document.body.classList.toggle('dark', escuro);
    rotularTema(escuro);
    const sino = document.querySelector('#notif_toggle');
    if (sino) {
        sino.setAttribute('aria-label', 'Abrir ou fechar as notificações');
        sino.setAttribute('title', 'Notificações');
        sino.setAttribute('aria-expanded', 'false');
    }
}"""
_NOTIF_TOGGLE_JS = """() => {
    const painel = document.querySelector('#notif_panel');
    if (!painel) return;
    const aberto = painel.classList.toggle('open');
    document.querySelector('#notif_toggle')?.setAttribute('aria-expanded', String(aberto));
}"""


# =====================================================================
# Interface principal
# =====================================================================


def create_interface() -> "gr.Blocks":
    """Constroi e devolve a interface Gradio.

    Antes, o `with gr.Blocks(...)` ficava em nivel de MODULO: importar
    `app.interfaces.gradio.app` construia a UI inteira como efeito colateral —
    carregava modelos, tocava o banco, montava 11 abas. Nao dava para importar
    o modulo sem pagar por isso, nem para instanciar duas vezes, nem para
    testar uma aba isolada.

    O objeto `demo` de modulo continua existindo logo abaixo, porque
    `app.py` (raiz) e `app/interfaces/web/main_fastapi.py` fazem
    `from app.interfaces.gradio.app import demo`. Quem quiser controlar a
    construcao usa esta funcao.
    """
    with gr.Blocks(
        title="SpeechChecker — Audio Deepfake Detection Platform",
    ) as demo:
        # Login desativado (modo aberto): o `gr.State(True)` que existia aqui
        # nao era lido por nada desde a desativacao da autenticacao.

        # --- Container da Aplicação ---
        with gr.Column(visible=True):
            with gr.Row(elem_id="topbar_row"):
                status_bar_html = gr.HTML(
                    _render_status_bar(), elem_id="status_bar", scale=1
                )
                notif_toggle_btn = gr.Button(
                    _notif_label(),
                    elem_id="notif_toggle",
                    elem_classes="toolbar-toggle",
                    scale=0,
                    min_width=44,
                )
                theme_toggle_btn = gr.Button(
                    "🌙",
                    elem_id="theme_toggle",
                    elem_classes="toolbar-toggle",
                    scale=0,
                    min_width=44,
                )

            # Fechado por padrão: uma faixa fixa "Notificações pendentes" ocupava
            # espaço em todas as telas, mesmo sem nada para mostrar.
            with gr.Group(elem_id="notif_panel"):
                gr.Markdown("#### Notificações")
                feedback_html = gr.HTML(
                    _render_feedback_panel(),
                    elem_id="global_feedback_center",
                )
                with gr.Row(elem_classes="action-row"):
                    feedback_refresh_btn = gr.Button("Atualizar", size="sm")
                    feedback_read_btn = gr.Button("Marcar como lidas", size="sm")
                    feedback_clear_btn = gr.Button("Limpar histórico", size="sm")

            # Navegação: no máximo dois níveis até a tela de trabalho, agrupando pelo
            # que a pessoa quer fazer. Em ordem de uso: ver o estado, analisar,
            # aprofundar, comparar com uma voz conhecida, treinar e cuidar dos dados.
            with gr.Tabs(elem_id="main_tabs") as main_tabs:
                create_dashboard_tab(main_tabs)

                with gr.Tab("🎯 Detectar", id="tab_detect") as detect_tab:
                    with gr.Tabs():
                        model_select = create_detection_tab()
                        create_history_tab()
                # Um modelo salvo pelo assistente aparece sem recarregar a página.
                if model_select is not None:
                    detect_tab.select(
                        fn=_refresh_model_choices,
                        outputs=[model_select],
                        show_progress="hidden",
                    )

                with gr.Tab("🔬 Investigar", id="tab_investigate"):
                    with gr.Tabs():
                        create_forensic_analysis_tab()
                        create_features_tab()

                create_voice_profiles_tab()

                arch_select = None
                with gr.Tab("🎓 Treinar", id="tab_train"):
                    training_enabled = os.getenv(
                        "ENABLE_TRAINING", "true"
                    ).strip().lower() not in {"0", "false", "no", "off"}
                    if training_enabled:
                        with gr.Tabs():
                            arch_select = create_training_wizard_tab()
                            create_optimization_tab()
                    else:
                        gr.Markdown(
                            "### Modo demonstração\n\n"
                            "O treinamento está desativado neste ambiente. "
                            "Use a aba **Detectar** para inferência com os modelos "
                            "já treinados."
                        )

                # Os atalhos "testar na detecção" e "treinar esta arquitetura"
                # escrevem nos seletores das duas abas acima.
                create_benchmark_tab(main_tabs, model_select, arch_select)

                create_dataset_management_tab()

            # Auto-refresh da barra de status.
            #
            # O intervalo era 15 s FIXO, e o `tick` dispara para CADA cliente
            # conectado: com o servidor ocioso e um cliente aberto, sao 240
            # execucoes por hora de uma funcao que consulta GPU, contagem de
            # modelos e perfis. O default subiu para 60 s — o botao de
            # atualizar ao lado cobre quem precisa do estado imediato — e
            # XFAKE_STATUS_REFRESH_S permite ajustar sem editar codigo.
            # 0 (ou negativo) desliga o timer e deixa so o botao.
            feedback_outputs = [status_bar_html, feedback_html, notif_toggle_btn]
            try:
                _sb_interval = float(os.getenv("XFAKE_STATUS_REFRESH_S", "60"))
            except (TypeError, ValueError):
                _sb_interval = 60.0
            if _sb_interval > 0:
                try:
                    _sb_timer = gr.Timer(_sb_interval)
                    _sb_timer.tick(
                        fn=_refresh_global_feedback,
                        inputs=[],
                        outputs=feedback_outputs,
                        show_progress="hidden",
                    )
                except Exception:
                    # gr.Timer ausente em versões antigas — degradação graciosa
                    pass

            feedback_refresh_btn.click(
                fn=_refresh_global_feedback,
                inputs=[],
                outputs=feedback_outputs,
            )
            feedback_read_btn.click(
                fn=_mark_feedback_read_ui,
                inputs=[],
                outputs=feedback_outputs,
            )
            feedback_clear_btn.click(
                fn=_clear_feedback_ui,
                inputs=[],
                outputs=feedback_outputs,
            )
            notif_toggle_btn.click(fn=None, js=_NOTIF_TOGGLE_JS)
            theme_toggle_btn.click(fn=None, js=_THEME_TOGGLE_JS)

            # SELETOR DE IDIOMA REMOVIDO (2026-07-28).
            #
            # O botao existia, chamava `i18n.set_language()` e trocava o proprio
            # rotulo — mas NENHUMA aba chama `i18n.t()`, entao a interface seguia
            # 100% em portugues. A propria notificacao admitia: "some tabs will keep
            # Portuguese — full i18n WIP". Um controle visivel que nao faz o que
            # promete e pior do que nao existir.
            #
            # Havia um agravante: `_current_lang` e estado GLOBAL do processo. Se as
            # abas passarem a ler `t()`, um usuario trocando o idioma mudaria a
            # interface de todos os outros — o Gradio serve varias sessoes no mesmo
            # processo. Ao reintroduzir, o idioma precisa viver em `gr.State` (por
            # sessao) e ser passado a `t(key, lang=...)`, nunca no modulo.
            #
            # `utils/i18n.py` permanece, com o dicionario de traducoes intacto.

            demo.load(fn=None, js=_ON_LOAD_JS)

    # PROD.5: Queue com limites para evitar DoS via flood de requests:
    # - default_concurrency_limit=20: workers paralelos (matches detection.py expectation)
    # - max_size=100: requests aguardando — após isso, novas viram 503
    # Sem max_size, RAM pode esgotar com uploads grandes em fila.
    return demo


demo = create_interface()
demo.queue(default_concurrency_limit=20, max_size=100)


# A app FastAPI unificada (UI Gradio montada em /gradio + API REST em
# /api/v1/*) é montada uma única vez em `app.interfaces.web.main_fastapi`,
# que já importa `demo` daqui para fazer `gr.mount_gradio_app(...)`. Este
# módulo não duplica mais essa montagem (consolidação — havia uma segunda
# cópia quase idêntica de setup_security/middleware/routers
# aqui, com um bug de allowed_paths incompleto no `main_fastapi.py` que só
# foi corrigido num dos dois lugares até então).
#
# Para rodar a app unificada:
#     python main.py --gradio
# ou, programaticamente:
#     from app.interfaces.web.main_fastapi import app
