import logging
import os
import sys
from pathlib import Path

# Adicionar diretório app ao path ANTES de imports do projeto
sys.path.insert(0, str(Path(__file__).parent))

# === Patch para compatibilidade Pydantic v2 + Gradio v4 ===
# Deve ser ANTES de qualquer import de gradio ou pydantic
from app.gradio_schema_patch import patch_gradio_schema_validator
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
            _count_models, _count_profiles, _gpu_status,
        )
        models = _count_models()
        profiles = _count_profiles()
        gpu = _gpu_status()
    except Exception:
        models, profiles, gpu = 0, 0, "?"

    gpu_ok = gpu.startswith("✓")
    gpu_short = "GPU ✓" if gpu_ok else ("GPU ✗" if gpu.startswith("✗") else "GPU ?")
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
      <span class="status-pill" style="color:{'#10b981' if gpu_ok else '#94a3b8'}">
        {gpu_short}
      </span>
      <span class="status-pill" style="color:#94a3b8">
        <span style="color:#f1f5f9;font-weight:500;">{models}</span> modelos
      </span>
      <span class="status-pill" style="color:#94a3b8">
        <span style="color:#f1f5f9;font-weight:500;">{profiles}</span> perfis
      </span>
    </div>
    """


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

        # Tabs role-based (UI Fase 1: Dashboard / Detectar / Investigar / Treinar / Admin)
        with gr.Tabs() as main_tabs:
            # 🏠 Dashboard — landing page com KPIs + quick actions
            create_dashboard_tab()

            # 🎯 Detectar — consolida detecção e verificação de perfil
            with gr.Tab("🎯 Detectar", id="tab_detect"):
                with gr.Tabs():
                    create_detection_tab()
                    create_voice_profiles_tab()

            # 🔬 Investigar — análise forense detalhada
            create_forensic_analysis_tab()

            # 🎓 Treinar — Wizard linear (UI Fase 2)
            with gr.Tab("🎓 Treinar", id="tab_train"):
                with gr.Tabs():
                    create_training_wizard_tab()
                    create_optimization_tab()

            # ⚙ Admin — gestão de datasets, modelos, histórico, features
            with gr.Tab("⚙ Admin", id="tab_admin"):
                with gr.Tabs():
                    create_dataset_management_tab()
                    create_history_tab()
                    create_features_tab()

        # Auto-refresh do status bar a cada 60s
        try:
            _sb_timer = gr.Timer(60.0)
            _sb_timer.tick(
                fn=_render_status_bar,
                inputs=[],
                outputs=[status_bar_html],
            )
        except Exception:
            # gr.Timer não disponível em versões antigas — degradação graciosa
            pass

        # ───── UI Fase 3: Tema + Idioma ─────
        # Estados em memória do servidor (resetam ao recarregar página)
        theme_state = gr.State("dark")
        lang_state = gr.State("pt")

        def _toggle_theme(current_theme: str):
            """Alterna entre dark e light. Aplica via JS no body[data-theme]."""
            new_theme = "light" if current_theme == "dark" else "dark"
            new_label = "☀" if new_theme == "light" else "🌙"
            # JS para aplicar imediatamente no DOM (Gradio CSS reage ao atributo)
            js = f"""
            () => {{
                document.body.setAttribute('data-theme', '{new_theme}');
                try {{ localStorage.setItem('xf_theme', '{new_theme}'); }} catch(e) {{}}
                return null;
            }}
            """
            return new_theme, gr.update(value=new_label), js

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
                fn=lambda t: ("light" if t == "dark" else "dark",
                              gr.update(value="☀" if t == "dark" else "🌙")),
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
    from fastapi.middleware.cors import CORSMiddleware
    from app.core.db_setup import init_db
    from app.core.exceptions import setup_exception_handlers
    from app.core.gpu import setup_gpu, describe_gpu_setup
    from app.core.middleware import setup_middleware
    from app.core.version_check import check_versions
    from app.routers import (
        system, detection, features, training, history, datasets,
        voice_profiles,
    )
    from app.core.security import setup_security, limiter
    from slowapi.errors import RateLimitExceeded
    from slowapi import _rate_limit_exceeded_handler

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

    # Montar Gradio na raiz "/" para manter a experiência do usuário
    # Se quiser em "/gradio", mude o path abaixo.
    app = gr.mount_gradio_app(
        app,
        demo,
        path="/",
        allowed_paths=allowed_paths,
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
