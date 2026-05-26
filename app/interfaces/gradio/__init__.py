"""Interface Gradio do XFakeSong.

Este pacote expõe as funções create_*_tab() de cada aba e os utilitários
compartilhados de UI (plotting helpers, locks, paleta de cores).

Uso típico:
    from app.interfaces.gradio import create_detection_tab, create_training_tab
    # ou
    from app.interfaces.gradio.tabs import create_detection_tab
    from app.interfaces.gradio.utils.plotting import make_figure, PLOT_ACCENT
"""

from __future__ import annotations

# Re-exporta os builders de aba (lazy imports através de tabs/__init__.py)
from app.interfaces.gradio.tabs import (
    create_dashboard_tab,
    create_dataset_management_tab,
    create_detection_tab,
    create_features_tab,
    create_forensic_analysis_tab,
    create_history_tab,
    create_optimization_tab,
    create_training_tab,
    create_training_wizard_tab,
    create_voice_profiles_tab,
)

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
