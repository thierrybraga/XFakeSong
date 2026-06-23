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

/* ===== UI Fase 1 — Status Bar + Dashboard + cores semânticas ===== */

/* Status bar global (sempre visível no topo) */
#status_bar {
    display: flex !important;
    align-items: center !important;
    gap: 24px !important;
    padding: 8px 16px !important;
    background: var(--xf-surface) !important;
    border: 1px solid var(--xf-border) !important;
    border-radius: var(--xf-radius) !important;
    margin-bottom: 12px !important;
    font-family: var(--xf-mono) !important;
    font-size: 0.82rem !important;
}
#status_bar .status-pill {
    display: inline-flex !important;
    align-items: center !important;
    gap: 6px !important;
    color: var(--xf-text-muted) !important;
}
#status_bar .status-pill .dot {
    width: 8px;
    height: 8px;
    border-radius: 50%;
    background: var(--xf-success);
    box-shadow: 0 0 6px var(--xf-success);
}
#status_bar .status-pill .label {
    color: var(--xf-text);
    font-weight: 500;
}
#status_bar .brand {
    color: var(--xf-text) !important;
    font-weight: 700 !important;
    font-family: var(--xf-font) !important;
    font-size: 0.95rem !important;
    letter-spacing: -0.02em;
}

/* KPI cards no Dashboard */
.kpi-row {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
    gap: 16px;
    margin: 8px 0 24px 0;
}
.kpi-card {
    background: var(--xf-surface);
    border: 1px solid var(--xf-border);
    border-radius: var(--xf-radius-lg);
    padding: 20px 24px;
    transition: var(--xf-transition);
    position: relative;
    overflow: hidden;
}
.kpi-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 3px;
    background: var(--xf-gradient);
}
.kpi-card:hover {
    transform: translateY(-2px);
    box-shadow: var(--xf-shadow-lg);
    border-color: var(--xf-border-light);
}
.kpi-card .kpi-icon {
    font-size: 1.6rem;
    margin-bottom: 8px;
}
.kpi-card .kpi-value {
    font-size: 2rem;
    font-weight: 700;
    color: var(--xf-text);
    line-height: 1;
    font-variant-numeric: tabular-nums;
}
.kpi-card .kpi-label {
    color: var(--xf-text-muted);
    font-size: 0.85rem;
    margin-top: 6px;
    text-transform: uppercase;
    letter-spacing: 0.05em;
}
.kpi-success::before { background: linear-gradient(135deg, var(--xf-success), #34d399); }
.kpi-info::before    { background: linear-gradient(135deg, var(--xf-accent), #67e8f9); }
.kpi-neutral::before { background: linear-gradient(135deg, var(--xf-text-muted), var(--xf-border-light)); }

/* Painel de status do sistema */
.status-panel {
    background: var(--xf-surface);
    border: 1px solid var(--xf-border);
    border-radius: var(--xf-radius);
    padding: 12px 16px;
    display: flex;
    flex-direction: column;
    gap: 10px;
}
.status-item {
    display: flex;
    align-items: center;
    gap: 10px;
    padding: 6px 0;
    border-bottom: 1px solid var(--xf-border);
}
.status-item:last-child { border-bottom: none; }
.status-item .status-icon {
    font-size: 1.1rem;
    width: 24px;
    text-align: center;
}
.status-item .status-label {
    flex: 0 0 110px;
    color: var(--xf-text-muted);
    font-size: 0.85rem;
}
.status-item .status-value {
    color: var(--xf-text);
    font-size: 0.9rem;
    font-family: var(--xf-mono);
}
.status-item.ok .status-icon   { color: var(--xf-success); }
.status-item.warn .status-icon { color: var(--xf-warning); }
.status-item.info .status-icon { color: var(--xf-accent); }

/* Tabela de análises recentes */
.recent-table {
    width: 100%;
    border-collapse: collapse;
    background: var(--xf-surface);
    border-radius: var(--xf-radius);
    overflow: hidden;
}
.recent-table thead th {
    background: rgba(59, 130, 246, 0.08);
    color: var(--xf-text-muted);
    text-transform: uppercase;
    font-size: 0.75rem;
    letter-spacing: 0.05em;
    padding: 10px 14px;
    text-align: left;
    border-bottom: 2px solid var(--xf-primary);
}
.recent-table tbody td {
    padding: 10px 14px;
    border-bottom: 1px solid var(--xf-border);
    color: var(--xf-text);
    font-size: 0.9rem;
}
.recent-table tbody tr:hover { background: var(--xf-surface-hover); }
.recent-table td.mono { font-family: var(--xf-mono); font-size: 0.85rem; }
.recent-table td.dim  { color: var(--xf-text-muted); font-size: 0.82rem; }

/* Empty state (nenhum dado) */
.empty-state {
    text-align: center;
    padding: 40px 20px;
    background: var(--xf-surface);
    border: 1px dashed var(--xf-border);
    border-radius: var(--xf-radius);
    color: var(--xf-text-muted);
}
.empty-state .empty-icon { font-size: 2.5rem; margin-bottom: 12px; }
.empty-state .empty-title {
    color: var(--xf-text);
    font-weight: 600;
    font-size: 1.1rem;
    margin-bottom: 6px;
}
.empty-state .empty-desc {
    font-size: 0.9rem;
    line-height: 1.5;
    max-width: 400px;
    margin: 0 auto;
}

/* Cores semânticas globais (reusáveis em qualquer tab) */
.result-real     { color: var(--xf-success) !important; font-weight: 600 !important; }
.result-fake     { color: var(--xf-danger)  !important; font-weight: 600 !important; }
.result-ood      { color: var(--xf-warning) !important; font-weight: 600 !important; }
.result-uncertain { color: #8b5cf6           !important; font-weight: 600 !important; }

/* ===== UI Fase 2 — Training Wizard ===== */

/* Stepper (indicador de progresso 1-2-3-4) */
.wizard-stepper {
    display: flex;
    align-items: center;
    justify-content: center;
    margin: 16px 0 32px 0;
    padding: 16px;
    background: var(--xf-surface);
    border-radius: var(--xf-radius-lg);
    border: 1px solid var(--xf-border);
}
.wizard-stepper .step {
    display: flex;
    flex-direction: column;
    align-items: center;
    gap: 8px;
    min-width: 120px;
}
.wizard-stepper .step-circle {
    width: 40px;
    height: 40px;
    border-radius: 50%;
    display: flex;
    align-items: center;
    justify-content: center;
    font-weight: 600;
    font-size: 1.1rem;
    transition: var(--xf-transition);
    border: 2px solid var(--xf-border);
}
.wizard-stepper .step-label {
    color: var(--xf-text-muted);
    font-size: 0.85rem;
    font-weight: 500;
}
.wizard-stepper .step-active .step-circle {
    background: var(--xf-gradient);
    color: #fff;
    border-color: var(--xf-primary);
    box-shadow: 0 0 12px var(--xf-primary-glow);
}
.wizard-stepper .step-active .step-label {
    color: var(--xf-text);
    font-weight: 600;
}
.wizard-stepper .step-done .step-circle {
    background: var(--xf-success);
    color: #fff;
    border-color: var(--xf-success);
}
.wizard-stepper .step-done .step-label {
    color: var(--xf-success);
}
.wizard-stepper .step-pending .step-circle {
    background: var(--xf-bg);
    color: var(--xf-text-muted);
}
.wizard-stepper .step-connector {
    flex: 1;
    height: 2px;
    background: var(--xf-border);
    margin: 0 8px;
    margin-bottom: 28px;  /* alinha com o círculo (não com o label) */
    max-width: 80px;
}

/* ===== UI Fase 3 — Modo claro ===== */
/* Tema claro ativado via body[data-theme='light']. CSS abaixo sobrescreve
   as variáveis :root quando o atributo está presente.
   Default permanece dark (sem o atributo) — compatibilidade total. */
body[data-theme="light"] {
    --xf-bg:            #f8fafc;
    --xf-surface:       #ffffff;
    --xf-surface-hover: #f1f5f9;
    --xf-border:        #e2e8f0;
    --xf-border-light:  #cbd5e1;
    --xf-text:          #0f172a;
    --xf-text-muted:    #64748b;
    --xf-primary:       #2563eb;
    --xf-primary-hover: #1d4ed8;
    --xf-primary-glow:  rgba(37, 99, 235, 0.18);
    --xf-accent:        #0891b2;
    --xf-shadow:        0 2px 12px rgba(0, 0, 0, 0.06);
    --xf-shadow-lg:     0 4px 24px rgba(0, 0, 0, 0.08);
}
body[data-theme="light"] .gradio-container {
    background: var(--xf-bg) !important;
    color: var(--xf-text) !important;
}
body[data-theme="light"] .gr-group,
body[data-theme="light"] .gr-box,
body[data-theme="light"] .gr-panel {
    background: var(--xf-surface) !important;
    border-color: var(--xf-border) !important;
}
body[data-theme="light"] .recent-table thead th {
    background: rgba(37, 99, 235, 0.06) !important;
}
body[data-theme="light"] .status-pill .dot {
    box-shadow: 0 0 6px rgba(16, 185, 129, 0.4);
}

/* Botão de tema/idioma no status bar */
.toolbar-toggle {
    display: inline-flex !important;
    align-items: center !important;
    gap: 4px !important;
    padding: 4px 10px !important;
    background: transparent !important;
    border: 1px solid var(--xf-border) !important;
    border-radius: var(--xf-radius) !important;
    color: var(--xf-text) !important;
    font-family: var(--xf-mono) !important;
    font-size: 0.82rem !important;
    cursor: pointer !important;
    transition: var(--xf-transition) !important;
}
.toolbar-toggle:hover {
    background: var(--xf-surface-hover) !important;
    border-color: var(--xf-primary) !important;
}

/* ===== Notification Center ===== */

.notif-list {
    display: flex;
    flex-direction: column;
    gap: 6px;
    max-height: 480px;
    overflow-y: auto;
    padding: 4px;
}
.notif-item {
    display: flex;
    gap: 12px;
    padding: 12px 14px;
    background: var(--xf-surface);
    border: 1px solid var(--xf-border);
    border-left: 3px solid var(--xf-border-light);
    border-radius: var(--xf-radius);
    transition: var(--xf-transition);
}
.notif-item:hover {
    background: var(--xf-surface-hover);
    transform: translateX(2px);
}

/* Borda lateral colorida por nível */
.notif-item.notif-success  { border-left-color: #10b981; }
.notif-item.notif-info     { border-left-color: #06b6d4; }
.notif-item.notif-warning  { border-left-color: #f59e0b; }
.notif-item.notif-error    { border-left-color: #ef4444; }
.notif-item.notif-critical {
    border-left-color: #dc2626;
    background: rgba(220, 38, 38, 0.04);
}

.notif-icon {
    flex: 0 0 24px;
    font-size: 1.2rem;
    line-height: 1.4;
    font-weight: 600;
}
.notif-body { flex: 1; min-width: 0; }
.notif-header {
    display: flex;
    align-items: center;
    gap: 8px;
    flex-wrap: wrap;
    margin-bottom: 2px;
}
.notif-title {
    color: var(--xf-text);
    font-weight: 600;
    font-size: 0.92rem;
}
.notif-code {
    background: var(--xf-bg);
    color: var(--xf-text-muted);
    font-family: var(--xf-mono);
    font-size: 0.7rem;
    padding: 1px 6px;
    border-radius: 4px;
    border: 1px solid var(--xf-border);
}
.notif-ago {
    color: var(--xf-text-muted);
    font-size: 0.75rem;
    margin-left: auto;
    font-family: var(--xf-mono);
}
.notif-msg {
    color: var(--xf-text-muted);
    font-size: 0.85rem;
    line-height: 1.4;
    margin-top: 2px;
}
.notif-hint {
    margin-top: 6px;
    padding: 6px 10px;
    background: rgba(59, 130, 246, 0.08);
    border-radius: 6px;
    color: var(--xf-text);
    font-size: 0.82rem;
    line-height: 1.45;
    border-left: 2px solid var(--xf-accent);
}
.notif-empty {
    text-align: center;
    padding: 40px 20px;
    color: var(--xf-text-muted);
    font-size: 0.9rem;
}

/* Badge de unread no botão do status bar */
.notif-badge {
    display: inline-block;
    min-width: 18px;
    height: 18px;
    padding: 0 6px;
    background: var(--xf-danger);
    color: #fff;
    border-radius: 9px;
    font-size: 0.7rem;
    font-weight: 700;
    line-height: 18px;
    text-align: center;
    margin-left: 4px;
    font-family: var(--xf-font);
}

/* Inline status indicators (✓ ✗ ⚠ ao lado de campos) */
.field-status {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    padding: 4px 10px;
    border-radius: var(--xf-radius);
    font-size: 0.85rem;
    margin-left: 8px;
}
.field-status.success { background: rgba(16, 185, 129, 0.12); color: #10b981; }
.field-status.warning { background: rgba(245, 158, 11, 0.12); color: #f59e0b; }
.field-status.error   { background: rgba(239, 68, 68, 0.12); color: #ef4444; }
.field-status.info    { background: rgba(6, 182, 212, 0.12); color: #06b6d4; }

/* GPU diagnóstico (Dashboard) */
.gpu-diag {
    background: var(--xf-bg-elev);
    border: 1px solid var(--xf-border);
    border-radius: var(--xf-radius);
    padding: 12px 16px;
}
.gpu-diag pre {
    margin: 0;
    font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
    font-size: 0.82rem;
    line-height: 1.55;
    color: var(--xf-text);
    white-space: pre-wrap;
    word-break: break-word;
}
.gpu-diag-msg {
    margin-top: 8px;
    padding: 8px 12px;
    border-radius: 6px;
    font-weight: 600;
    border-left: 3px solid;
}
.gpu-diag-info    { background: rgba(6, 182, 212, 0.10);  border-color: #06b6d4; color: #06b6d4; }
.gpu-diag-warning { background: rgba(245, 158, 11, 0.10); border-color: #f59e0b; color: #f59e0b; }
.gpu-diag-error   { background: rgba(239, 68, 68, 0.10);  border-color: #ef4444; color: #ef4444; }
.gpu-diag .badge {
    display: inline-block;
    background: rgba(59, 130, 246, 0.15);
    color: #3b82f6;
    border: 1px solid rgba(59, 130, 246, 0.3);
    border-radius: 4px;
    padding: 1px 6px;
    font-size: 0.7rem;
    margin-left: 4px;
}

/* Cards de modelos (Step 2) */
.model-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(240px, 1fr));
    gap: 12px;
    margin: 16px 0;
    max-height: 480px;
    overflow-y: auto;
    padding-right: 8px;
}
.model-card {
    background: var(--xf-surface);
    border: 2px solid var(--xf-border);
    border-radius: var(--xf-radius);
    padding: 14px 16px;
    cursor: pointer;
    transition: var(--xf-transition);
    position: relative;
}
.model-card:hover {
    border-color: var(--xf-border-light);
    transform: translateY(-1px);
    box-shadow: var(--xf-shadow);
}
.model-card-selected {
    border-color: var(--xf-primary) !important;
    background: rgba(59, 130, 246, 0.08) !important;
    box-shadow: 0 0 0 3px var(--xf-primary-glow) !important;
}
.model-card .model-icon { font-size: 1.6rem; margin-bottom: 4px; }
.model-card .model-name {
    font-weight: 600;
    color: var(--xf-text);
    font-size: 1rem;
}
.model-card .model-category {
    color: var(--xf-accent);
    font-size: 0.75rem;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    margin: 2px 0 8px 0;
}
.model-card .model-desc {
    color: var(--xf-text-muted);
    font-size: 0.82rem;
    line-height: 1.4;
}

/* ===== Painel de treino ao vivo (Wizard Step 4) ===== */
.train-live {
    background: var(--xf-surface);
    border: 1px solid var(--xf-border);
    border-radius: var(--xf-radius-lg);
    padding: 16px 20px;
    margin-bottom: 12px;
}
.train-live .tl-head {
    display: flex;
    align-items: baseline;
    justify-content: space-between;
    flex-wrap: wrap;
    gap: 8px;
    margin-bottom: 10px;
}
.train-live .tl-head-title {
    font-weight: 700;
    font-size: 1.05rem;
}
.train-live .tl-head-meta {
    color: var(--xf-text-muted);
    font-family: var(--xf-mono);
    font-size: 0.8rem;
}
.train-live .tl-bar-track {
    width: 100%;
    height: 10px;
    background: var(--xf-bg);
    border-radius: 6px;
    overflow: hidden;
    border: 1px solid var(--xf-border);
}
.train-live .tl-bar-fill {
    height: 100%;
    border-radius: 6px;
    transition: width 0.5s cubic-bezier(0.4, 0, 0.2, 1);
    background-size: 200% 100%;
    animation: shimmer 2s linear infinite;
}
.train-live .tl-cards {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(120px, 1fr));
    gap: 10px;
    margin-top: 14px;
}
.train-live .tl-card {
    background: var(--xf-bg);
    border: 1px solid var(--xf-border);
    border-radius: var(--xf-radius);
    padding: 10px 12px;
    text-align: center;
}
.train-live .tl-card-label {
    color: var(--xf-text-muted);
    font-size: 0.72rem;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    margin-bottom: 4px;
}
.train-live .tl-card-value {
    font-family: var(--xf-mono);
    font-size: 1.3rem;
    font-weight: 700;
    font-variant-numeric: tabular-nums;
    line-height: 1;
}
.train-live .tl-note {
    margin-top: 12px;
    padding: 8px 12px;
    background: rgba(6, 182, 212, 0.08);
    border-left: 3px solid var(--xf-accent);
    border-radius: 6px;
    color: var(--xf-text);
    font-size: 0.85rem;
    line-height: 1.45;
}

/* ===== Cabeçalho de página padronizado (page_header component) ===== */
.page-header {
    display: flex;
    align-items: center;
    gap: 14px;
    padding: 14px 18px;
    margin: 4px 0 18px 0;
    background: var(--xf-surface);
    border: 1px solid var(--xf-border);
    border-left: 4px solid var(--xf-primary);
    border-radius: var(--xf-radius);
    box-shadow: var(--xf-shadow);
}
.page-header .ph-icon {
    font-size: 1.9rem;
    line-height: 1;
    flex: 0 0 auto;
    filter: drop-shadow(0 2px 4px rgba(0,0,0,0.25));
}
.page-header .ph-text { min-width: 0; }
.page-header .ph-title {
    margin: 0 !important;
    font-size: 1.3rem !important;
    font-weight: 700 !important;
    color: var(--xf-text) !important;
    letter-spacing: -0.02em !important;
    line-height: 1.2 !important;
}
.page-header .ph-subtitle {
    margin: 4px 0 0 0;
    color: var(--xf-text-muted);
    font-size: 0.9rem;
    line-height: 1.45;
}

/* Divisor de seção sutil (section_divider component) */
.xf-divider {
    height: 1px;
    background: var(--xf-border);
    border: none;
    margin: 18px 0;
    opacity: 0.8;
}

/* Callout contextual (info_callout component) */
.xf-callout {
    padding: 10px 14px;
    border-radius: var(--xf-radius);
    border-left: 3px solid var(--xf-accent);
    background: rgba(6, 182, 212, 0.08);
    color: var(--xf-text);
    font-size: 0.88rem;
    line-height: 1.5;
    margin: 6px 0;
}
.xf-callout-info    { border-left-color: var(--xf-accent);  background: rgba(6, 182, 212, 0.08); }
.xf-callout-success { border-left-color: var(--xf-success); background: rgba(16, 185, 129, 0.08); }
.xf-callout-warning { border-left-color: var(--xf-warning); background: rgba(245, 158, 11, 0.08); }
.xf-callout-accent  { border-left-color: var(--xf-primary); background: rgba(59, 130, 246, 0.08); }
.xf-callout a { color: var(--xf-primary); font-weight: 600; text-decoration: none; }
.xf-callout a:hover { text-decoration: underline; }

/* ===== Responsividade global ===== */
.gradio-container {
    width: min(100%, 1440px) !important;
    margin: 0 auto !important;
    padding: 16px !important;
}
.gradio-container * {
    box-sizing: border-box !important;
}
.gradio-container img,
.gradio-container canvas,
.gradio-container svg,
.gradio-container video {
    max-width: 100% !important;
    height: auto !important;
}
.gradio-container p,
.gradio-container li,
.gradio-container label,
.gradio-container span,
.gradio-container code,
.gradio-container pre,
.gradio-container textarea,
.gradio-container button {
    overflow-wrap: anywhere !important;
}
.ph-icon,
.kpi-icon,
.status-icon,
.notif-icon,
.model-icon,
.empty-icon {
    flex-shrink: 0 !important;
    line-height: 1 !important;
}
.gr-row,
.gr-column,
.gr-form,
.gr-group,
.gr-box,
.gr-panel,
.svelte-vt1mxs,
.svelte-sa48pu {
    min-width: 0 !important;
}
.gr-plot,
.plot-container,
.js-plotly-plot,
.gr-image,
.gr-video,
.gr-audio,
.gr-dataframe,
.gr-json,
.gr-markdown {
    max-width: 100% !important;
    min-width: 0 !important;
}
.gr-dataframe,
.gr-dataframe > div,
.table-wrap,
.recent-table {
    overflow-x: auto !important;
    -webkit-overflow-scrolling: touch !important;
}
.gr-dataframe table,
.recent-table {
    min-width: 680px;
}
.tabs > .tab-nav {
    display: flex !important;
    flex-wrap: nowrap !important;
    overflow-x: auto !important;
    overflow-y: hidden !important;
    scrollbar-width: thin !important;
    -webkit-overflow-scrolling: touch !important;
}
.tabs > .tab-nav > button {
    flex: 0 0 auto !important;
    white-space: nowrap !important;
}
#topbar_row {
    align-items: stretch !important;
    gap: 10px !important;
    max-width: 100% !important;
    overflow: hidden !important;
}
#topbar_row > .gr-column:last-child {
    flex: 0 0 auto !important;
}
#topbar_row .svelte-vt1mxs,
#topbar_row .svelte-sa48pu,
#topbar_row .stretch {
    max-width: 100% !important;
    min-width: 0 !important;
}
#status_bar,
#status_bar > *,
#status_bar_inner {
    max-width: 100% !important;
    min-width: 0 !important;
}
#status_bar_inner {
    display: flex !important;
    align-items: center !important;
    gap: 10px 16px !important;
    flex-wrap: wrap !important;
    width: 100% !important;
}
#status_bar_inner .status-pill {
    min-width: 0 !important;
}
#global_feedback_center {
    max-height: min(70vh, 620px) !important;
    overflow-y: auto !important;
}

@media (max-width: 1100px) {
    .gradio-container {
        padding: 12px !important;
    }
    .gr-row {
        gap: 12px !important;
    }
    .kpi-row {
        grid-template-columns: repeat(2, minmax(0, 1fr));
    }
    .model-grid {
        grid-template-columns: repeat(auto-fit, minmax(190px, 1fr));
    }
    .wizard-stepper {
        justify-content: flex-start;
        overflow-x: auto;
        -webkit-overflow-scrolling: touch;
    }
    .wizard-stepper .step {
        min-width: 96px;
    }
    .wizard-stepper .step-connector {
        min-width: 36px;
        max-width: 48px;
    }
}

@media (max-width: 768px) {
    .gradio-container {
        padding: 8px !important;
    }
    h1 { font-size: 1.45rem !important; }
    h2 { font-size: 1.28rem !important; }
    h3 { font-size: 1.12rem !important; }

    #app_header {
        padding: 12px 14px !important;
        border-radius: var(--xf-radius) !important;
    }
    #app_header h2 {
        font-size: 1.25rem !important;
        line-height: 1.25 !important;
    }
    #topbar_row,
    .gr-row,
    .svelte-vt1mxs.gap,
    .svelte-sa48pu.stretch {
        flex-direction: column !important;
    }
    #topbar_row,
    #topbar_row > *,
    #topbar_row .svelte-vt1mxs,
    #topbar_row .svelte-sa48pu,
    #topbar_row .stretch {
        width: 100% !important;
        max-width: 100% !important;
        min-width: 0 !important;
    }
    #topbar_row > .gr-column,
    .gr-row > .gr-column {
        width: 100% !important;
        flex: 1 1 auto !important;
    }
    #topbar_row > .gr-column:last-child > .gr-row {
        flex-direction: row !important;
        justify-content: flex-start !important;
        flex-wrap: wrap !important;
    }
    #topbar_row > .gr-column:last-child .svelte-vt1mxs.gap {
        flex-direction: row !important;
        justify-content: flex-start !important;
        flex-wrap: wrap !important;
    }
    #topbar_row .svelte-sa48pu.stretch {
        flex-direction: row !important;
        flex-wrap: wrap !important;
    }
    #status_bar {
        flex-wrap: wrap !important;
        gap: 8px 12px !important;
        padding: 8px 10px !important;
        font-size: 0.76rem !important;
        width: 100% !important;
    }
    #status_bar_inner {
        display: grid !important;
        grid-template-columns: repeat(2, minmax(0, 1fr));
        gap: 8px 10px !important;
    }
    #status_bar .brand {
        grid-column: 1 / -1;
        width: 100%;
        font-size: 0.9rem !important;
    }
    #status_bar .status-pill {
        width: 100%;
        min-width: 0 !important;
        white-space: normal !important;
    }
    .toolbar-toggle {
        min-height: 36px !important;
        padding: 6px 10px !important;
    }
    .tabs > .tab-nav {
        border-radius: var(--xf-radius) !important;
        padding: 4px !important;
    }
    .tabs > .tab-nav > button {
        padding: 9px 12px !important;
        font-size: 0.82rem !important;
    }
    button {
        min-height: 40px !important;
        white-space: normal !important;
    }
    button.primary,
    button[variant="primary"],
    button.secondary,
    button[variant="secondary"],
    button.stop,
    button[variant="stop"] {
        padding: 9px 14px !important;
    }
    input,
    textarea,
    select,
    .gr-input,
    .gr-textbox textarea {
        min-height: 40px !important;
        font-size: 16px !important;
    }
    .page-header {
        align-items: flex-start;
        padding: 12px 14px;
        gap: 10px;
    }
    .page-header .ph-icon {
        font-size: 1.45rem;
    }
    .page-header .ph-title {
        font-size: 1.12rem !important;
    }
    .page-header .ph-subtitle,
    .xf-callout,
    .notif-msg {
        font-size: 0.84rem;
    }
    .kpi-row {
        grid-template-columns: 1fr;
        gap: 10px;
    }
    .kpi-card {
        padding: 14px 16px;
    }
    .kpi-card .kpi-value {
        font-size: 1.55rem;
    }
    .status-item {
        display: grid;
        grid-template-columns: 24px minmax(0, 1fr);
        align-items: flex-start;
        gap: 8px;
    }
    .status-item .status-icon {
        grid-row: 1 / span 2;
        width: 24px;
    }
    .status-item .status-label {
        grid-column: 2;
        flex-basis: auto;
        width: auto;
        font-size: 0.78rem;
    }
    .status-item .status-value {
        grid-column: 2;
        min-width: 0;
        width: 100%;
        white-space: normal;
        overflow-wrap: anywhere;
        word-break: normal;
        font-size: 0.78rem;
        line-height: 1.35;
    }
    .notif-item {
        padding: 10px 12px;
    }
    .notif-ago {
        margin-left: 0;
        width: 100%;
    }
    .model-grid {
        grid-template-columns: 1fr;
        max-height: none;
        padding-right: 0;
    }
    .train-live .tl-card-value {
        font-size: 1.08rem;
    }
}

@media (max-width: 520px) {
    .gradio-container {
        padding: 6px !important;
    }
    #status_bar .status-pill {
        display: flex !important;
        width: 100%;
        justify-content: flex-start;
    }
    #status_bar_inner {
        grid-template-columns: 1fr;
    }
    #topbar_row > .gr-column:last-child button {
        flex: 1 1 44px !important;
    }
    .tabs > .tab-nav > button {
        max-width: 74vw !important;
        overflow: hidden !important;
        text-overflow: ellipsis !important;
    }
    .wizard-stepper {
        padding: 10px;
        margin: 10px 0 18px 0;
    }
    .wizard-stepper .step {
        min-width: 72px;
    }
    .wizard-stepper .step-circle {
        width: 32px;
        height: 32px;
        font-size: 0.92rem;
    }
    .wizard-stepper .step-label {
        font-size: 0.72rem;
        text-align: center;
    }
    .wizard-stepper .step-connector {
        min-width: 20px;
        margin-bottom: 22px;
    }
    .gr-dataframe table,
    .recent-table {
        min-width: 560px;
    }
}

/* ===== Ajustes finos de abas, ícones, mídia e espaçamento ===== */
.tabs > .tab-nav {
    gap: 6px !important;
    padding-bottom: 3px !important;
}
.tabs > .tab-nav::-webkit-scrollbar {
    height: 5px !important;
}
.tabs > .tab-nav::-webkit-scrollbar-track {
    background: transparent !important;
}
.tabs > .tab-nav::-webkit-scrollbar-thumb {
    background: var(--xf-border-light) !important;
    border-radius: 999px !important;
}
.tabs > .tab-nav > button {
    min-height: 38px !important;
    line-height: 1.2 !important;
}
.label-wrap,
.gr-accordion .label-wrap,
.gr-accordion button {
    min-height: 36px !important;
    align-items: center !important;
}
.gr-accordion .label-wrap > span {
    line-height: 1.25 !important;
}
.toolbar-toggle {
    min-height: 34px !important;
}
.gradio-container button svg,
.gradio-container .icon,
.gradio-container [class*="icon"] {
    flex: 0 0 auto !important;
}

/* Aba Detectar */
.detect-main-grid {
    align-items: flex-start !important;
    gap: 16px !important;
}
.detect-column {
    min-width: min(420px, 100%) !important;
}
.detect-card {
    overflow: hidden !important;
}
.detect-card .gr-markdown h4,
.detect-card h4 {
    margin-top: 0 !important;
    margin-bottom: 10px !important;
    line-height: 1.25 !important;
}
.detect-audio-input {
    min-height: 232px !important;
}
.detect-audio-input audio {
    width: 100% !important;
}
.detect-audio-input svg,
.detect-audio-input .icon,
.detect-audio-input [class*="icon"] {
    max-width: 44px !important;
    max-height: 44px !important;
}
.detect-advanced-settings {
    margin-top: 12px !important;
}
.detect-primary-action {
    min-height: 48px !important;
    width: 100% !important;
    justify-content: center !important;
}
.detect-output-grid {
    align-items: stretch !important;
    gap: 12px !important;
}
.detect-label-output {
    min-height: 138px !important;
}
.detect-confidence-output {
    min-height: 138px !important;
    min-width: 150px !important;
}
.detect-label-output .output-class,
.detect-label-output .output-label {
    min-width: 0 !important;
}
.detect-label-output svg,
.detect-label-output .icon,
.detect-label-output [class*="icon"] {
    max-width: 48px !important;
    max-height: 48px !important;
    opacity: 0.65 !important;
}
.detect-result-card input,
.detect-result-card textarea {
    font-variant-numeric: tabular-nums !important;
}

@media (max-width: 768px) {
    .tabs > .tab-nav {
        scrollbar-width: none !important;
        padding-bottom: 4px !important;
    }
    .tabs > .tab-nav::-webkit-scrollbar {
        display: none !important;
    }
    .detect-main-grid,
    .detect-output-grid {
        gap: 12px !important;
    }
    .detect-column,
    .detect-confidence-output {
        min-width: 0 !important;
    }
    .detect-audio-input {
        min-height: 190px !important;
    }
    .detect-output-grid {
        flex-direction: column !important;
    }
    .detect-label-output,
    .detect-confidence-output {
        width: 100% !important;
        min-height: auto !important;
    }
}
"""


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
    feedback_color = (
        "#ef4444" if error_count else ("#f59e0b" if warning_count else "#94a3b8")
    )
    unread_html = f'<span class="notif-badge">{unread}</span>' if unread else ""
    return f"""
    <div id="status_bar_inner" style="display:flex;align-items:center;
        gap:24px;padding:8px 16px;background:#1e293b;
        border:1px solid #334155;border-radius:10px;margin-bottom:12px;
        font-family:'JetBrains Mono',monospace;font-size:0.82rem;">
      <span class="brand"
            style="color:#f1f5f9;font-weight:700;font-family:'Inter';
                   font-size:0.95rem;letter-spacing:-0.02em;">
        🛡️ XFakeSong <span style="color:#94a3b8;font-weight:400">v1.1</span>
      </span>
      <span class="status-pill" style="display:inline-flex;align-items:center;
            gap:6px;color:#94a3b8;">
        <span style="width:8px;height:8px;border-radius:50%;background:#10b981;
                     box-shadow:0 0 6px #10b981;"></span>
        <span style="color:#f1f5f9;font-weight:500;">Online</span>
      </span>
      <span class="status-pill" style="color:{"#10b981" if gpu_ok else "#94a3b8"}">
        {gpu_short}
      </span>
      <span class="status-pill" style="color:#94a3b8">
        <span style="color:#f1f5f9;font-weight:500;">{models}</span> modelos
      </span>
      <span class="status-pill" style="color:#94a3b8">
        <span style="color:#f1f5f9;font-weight:500;">{profiles}</span> perfis
      </span>
      <span class="status-pill" style="color:{feedback_color}">
        Notificações {unread_html}
        <span style="color:#94a3b8">warn:{warning_count} err:{error_count}</span>
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
