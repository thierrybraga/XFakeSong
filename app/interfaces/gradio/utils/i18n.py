"""Internacionalização (i18n) para a UI Gradio — UI Fase 3.

Dicionários simples PT-BR / EN com função `t(key, lang)`. Uso pragmático:
strings novas (status bar, dashboard, wizard) são internacionalizadas;
strings legadas das tabs originais ficam em PT-BR por enquanto.

Uso:
    from app.interfaces.gradio.utils.i18n import t, set_language

    set_language("en")
    t("dashboard.kpi.analyses_24h")  # "Analyses in last 24h"

    set_language("pt")
    t("dashboard.kpi.analyses_24h")  # "Análises últimas 24h"

Padrão: chave em snake_case com namespaces separados por ponto.
Se chave não existir, retorna a própria chave como fallback (não quebra UI).
"""

from __future__ import annotations

import logging
import threading
from typing import Optional

logger = logging.getLogger(__name__)

# Lock para mudança thread-safe (Gradio queue tem múltiplos workers)
_lang_lock = threading.RLock()
_current_lang = "pt"
_supported = ("pt", "en")


# =====================================================================
# Dicionários
# =====================================================================

TRANSLATIONS = {
    "pt": {
        # Brand / status bar
        "brand.tagline": "Plataforma de Detecção de Áudio Deepfake",
        "status.online": "Online",
        "status.gpu_ok": "GPU ✓",
        "status.gpu_off": "GPU ✗",
        "status.gpu_unknown": "GPU ?",
        "status.models": "modelos",
        "status.profiles": "perfis",
        "status.theme.dark": "🌙",
        "status.theme.light": "☀",
        "status.lang.pt": "🇧🇷 PT",
        "status.lang.en": "🇺🇸 EN",
        # Tabs top-level
        "tab.dashboard": "🏠 Dashboard",
        "tab.detect": "🎯 Detectar",
        "tab.investigate": "🔬 Investigar",
        "tab.train": "🎓 Treinar",
        "tab.admin": "⚙ Admin",
        # Dashboard
        "dashboard.kpi.analyses_24h": "Análises últimas 24h",
        "dashboard.kpi.models": "Modelos carregados",
        "dashboard.kpi.profiles": "Perfis de voz",
        "dashboard.kpi.datasets": "Datasets totais",
        "dashboard.recent": "Últimas Análises",
        "dashboard.system_status": "Status do Sistema",
        "dashboard.empty.title": "Nenhuma análise ainda",
        "dashboard.empty.desc": (
            "Faça sua primeira análise na aba <strong>🎯 Detectar</strong> "
            "ou treine um modelo em <strong>🎓 Treinar</strong>."
        ),
        # Wizard
        "wizard.step1.title": "Step 1 — Escolha do Dataset",
        "wizard.step2.title": "Step 2 — Escolha do Modelo",
        "wizard.step3.title": "Step 3 — Hiperparâmetros",
        "wizard.step4.title": "Step 4 — Treinamento",
        "wizard.btn.back": "← Voltar",
        "wizard.btn.next": "Próximo →",
        "wizard.btn.start": "Iniciar Treinamento →",
        "wizard.btn.new": "← Novo Treino",
        "wizard.btn.validate": "🔍 Validar",
        "wizard.scan.waiting": "*Aguardando validação...*",
        # Confirmações
        "confirm.delete.title": "Confirmar Remoção",
        "confirm.delete.are_you_sure": "Tem certeza que deseja remover",
        "confirm.delete.cannot_undo": "Esta ação não pode ser desfeita.",
        "confirm.delete.click_again": "Clique novamente para confirmar.",
        "confirm.delete.removed": "removido com sucesso",
        # Comuns
        "common.loading": "Carregando...",
        "common.error": "Erro",
        "common.success": "Sucesso",
        "common.cancel": "Cancelar",
        "common.confirm": "Confirmar",
        "common.delete": "Remover",
        "common.save": "Salvar",
        "common.close": "Fechar",
    },
    "en": {
        "brand.tagline": "Audio Deepfake Detection Platform",
        "status.online": "Online",
        "status.gpu_ok": "GPU ✓",
        "status.gpu_off": "GPU ✗",
        "status.gpu_unknown": "GPU ?",
        "status.models": "models",
        "status.profiles": "profiles",
        "status.theme.dark": "🌙",
        "status.theme.light": "☀",
        "status.lang.pt": "🇧🇷 PT",
        "status.lang.en": "🇺🇸 EN",
        "tab.dashboard": "🏠 Dashboard",
        "tab.detect": "🎯 Detect",
        "tab.investigate": "🔬 Investigate",
        "tab.train": "🎓 Train",
        "tab.admin": "⚙ Admin",
        "dashboard.kpi.analyses_24h": "Analyses last 24h",
        "dashboard.kpi.models": "Models loaded",
        "dashboard.kpi.profiles": "Voice profiles",
        "dashboard.kpi.datasets": "Total datasets",
        "dashboard.recent": "Recent Analyses",
        "dashboard.system_status": "System Status",
        "dashboard.empty.title": "No analyses yet",
        "dashboard.empty.desc": (
            "Run your first analysis in the <strong>🎯 Detect</strong> tab "
            "or train a model in <strong>🎓 Train</strong>."
        ),
        "wizard.step1.title": "Step 1 — Choose Dataset",
        "wizard.step2.title": "Step 2 — Choose Model",
        "wizard.step3.title": "Step 3 — Hyperparameters",
        "wizard.step4.title": "Step 4 — Training",
        "wizard.btn.back": "← Back",
        "wizard.btn.next": "Next →",
        "wizard.btn.start": "Start Training →",
        "wizard.btn.new": "← New Training",
        "wizard.btn.validate": "🔍 Validate",
        "wizard.scan.waiting": "*Awaiting validation...*",
        "confirm.delete.title": "Confirm Removal",
        "confirm.delete.are_you_sure": "Are you sure you want to remove",
        "confirm.delete.cannot_undo": "This action cannot be undone.",
        "confirm.delete.click_again": "Click again to confirm.",
        "confirm.delete.removed": "successfully removed",
        "common.loading": "Loading...",
        "common.error": "Error",
        "common.success": "Success",
        "common.cancel": "Cancel",
        "common.confirm": "Confirm",
        "common.delete": "Remove",
        "common.save": "Save",
        "common.close": "Close",
    },
}


# =====================================================================
# API pública
# =====================================================================


def get_language() -> str:
    """Retorna o idioma corrente."""
    with _lang_lock:
        return _current_lang


def set_language(lang: str) -> str:
    """Define o idioma corrente. Aceita 'pt' ou 'en'. Retorna lang aplicado."""
    global _current_lang
    lang = (lang or "").strip().lower()[:2]
    if lang not in _supported:
        logger.warning(f"Idioma '{lang}' não suportado. Mantendo {_current_lang}")
        lang = _current_lang
    with _lang_lock:
        _current_lang = lang
    return lang


def t(key: str, lang: Optional[str] = None, **fmt) -> str:
    """Traduz uma chave para o idioma corrente (ou explicitamente passado).

    Args:
        key: chave em formato 'namespace.subkey' (ex: 'dashboard.kpi.models')
        lang: opcional, sobrescreve o idioma global
        **fmt: kwargs para .format() se a string tiver placeholders

    Returns:
        Tradução, ou a própria key se não encontrada (fallback gracioso).
    """
    if lang is None:
        lang = get_language()
    if lang not in TRANSLATIONS:
        lang = "pt"
    value = TRANSLATIONS.get(lang, {}).get(key)
    if value is None:
        # Fallback: tenta o outro idioma antes de devolver a key crua
        fallback_lang = "en" if lang == "pt" else "pt"
        value = TRANSLATIONS.get(fallback_lang, {}).get(key, key)
    if fmt:
        try:
            return value.format(**fmt)
        except Exception:
            return value
    return value


def supported_languages() -> tuple:
    """Lista de idiomas suportados."""
    return _supported


__all__ = [
    "get_language",
    "set_language",
    "supported_languages",
    "t",
    "TRANSLATIONS",
]
