"""Tabs Gradio do XFakeSong.

Cada tab é uma função `create_<nome>_tab()` que constrói o layout e
event handlers da aba. Os imports são "lazy" — se uma tab falhar ao
importar (dependência ausente, módulo quebrado), o erro é capturado e
uma tab de placeholder é criada via `_create_error_tab`. Isso evita
que UMA aba quebrada derrube a interface inteira.
"""

from __future__ import annotations

import logging
from typing import Callable

# === Compatibilidade huggingface_hub ===
# Gradio v4 ainda importa HfFolder, removido em huggingface_hub recente.
try:
    from huggingface_hub import HfFolder  # noqa: F401
except ImportError:
    import huggingface_hub

    class _HfFolder:
        """Shim para HfFolder removido em huggingface_hub >= 0.16."""

        pass

    huggingface_hub.HfFolder = _HfFolder

import gradio as gr

logger = logging.getLogger(__name__)


def _create_error_tab(name: str, error_msg: str) -> Callable[[], None]:
    """Fábrica de tab placeholder usada quando importação real falha.

    O placeholder renderiza um aviso amigável para o usuário sem
    quebrar a aplicação inteira.
    """
    def _tab() -> None:
        with gr.Tab(f"⚠️ {name} (erro)"):
            gr.Markdown(
                f"### Falha ao carregar a aba **{name}**\n\n"
                f"```\n{error_msg}\n```\n\n"
                f"Esta aba foi desabilitada para preservar o resto da interface. "
                f"Verifique os logs do servidor para mais detalhes."
            )
    return _tab


# ---- Imports protegidos individualmente ----
# Se UM tab quebrar, o resto continua funcionando.

try:
    from .dashboard import create_dashboard_tab
except Exception as e:  # noqa: BLE001
    logger.error(f"Falha ao carregar tab Dashboard: {e}", exc_info=True)
    create_dashboard_tab = _create_error_tab("Dashboard", str(e))

try:
    from .detection import create_detection_tab
except Exception as e:  # noqa: BLE001
    logger.error(f"Falha ao carregar tab Detection: {e}", exc_info=True)
    create_detection_tab = _create_error_tab("Detection", str(e))

try:
    from .features import create_features_tab
except Exception as e:
    logger.error(f"Falha ao carregar tab Features: {e}", exc_info=True)
    create_features_tab = _create_error_tab("Features", str(e))

try:
    from .training import create_training_tab
except Exception as e:
    logger.error(f"Falha ao carregar tab Training: {e}", exc_info=True)
    create_training_tab = _create_error_tab("Training", str(e))

# UI Fase 2: wizard de treino linear (substitui training.py na UI principal)
try:
    from .training_wizard import create_training_wizard_tab
except Exception as e:
    logger.error(f"Falha ao carregar tab Training Wizard: {e}", exc_info=True)
    create_training_wizard_tab = _create_error_tab("Training Wizard", str(e))

try:
    from .optimization import create_optimization_tab
except Exception as e:
    logger.error(f"Falha ao carregar tab Optimization: {e}", exc_info=True)
    create_optimization_tab = _create_error_tab("Optimization", str(e))

try:
    from .history import create_history_tab
except Exception as e:
    logger.error(f"Falha ao carregar tab History: {e}", exc_info=True)
    create_history_tab = _create_error_tab("History", str(e))

try:
    from .forensic_analysis import create_forensic_analysis_tab
except Exception as e:
    logger.error(f"Falha ao carregar tab Forensic Analysis: {e}", exc_info=True)
    create_forensic_analysis_tab = _create_error_tab("Forensic Analysis", str(e))

try:
    from .voice_profiles import create_voice_profiles_tab
except Exception as e:
    logger.error(f"Falha ao carregar tab Voice Profiles: {e}", exc_info=True)
    create_voice_profiles_tab = _create_error_tab("Voice Profiles", str(e))

try:
    from .dataset_management import create_dataset_management_tab
except Exception as e:
    logger.error(f"Falha ao carregar tab Dataset Management: {e}", exc_info=True)
    create_dataset_management_tab = _create_error_tab("Dataset Management", str(e))


__all__ = [
    "create_dashboard_tab",
    "create_dataset_management_tab",
    "create_detection_tab",
    "create_features_tab",
    "create_forensic_analysis_tab",
    "create_history_tab",
    "create_optimization_tab",
    "create_training_tab",
    "create_training_wizard_tab",
    "create_voice_profiles_tab",
]
