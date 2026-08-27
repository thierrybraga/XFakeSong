"""XAI (Explainable AI) — wrappers de explicabilidade do XFakeSong.

Camada de domínio (``app/domain``) — lógica de explicabilidade específica
dos modelos do projeto (não é infraestrutura genérica, por isso vive fora
de ``app/core``):

- :mod:`app.domain.xai.gradcam` — mapas de ativação Grad-CAM para modelos
  Keras (implementação pura TensorFlow, sem dependência externa);
- :mod:`app.domain.xai.shap_explainer` — wrapper da biblioteca ``shap``
  (import guardado) para os classificadores clássicos sobre o vetor tabular;
- :mod:`app.domain.xai.tabular` — contrato do vetor tabular (nomes na ordem
  canônica, ``benchmark_tabular_v1`` de 63 e ``v2`` de 183 descritores) e
  utilidades para artefatos scikit-learn.

CLI correspondente: ``scripts/reporting/run_shap_analysis.py``.
"""

from typing import TYPE_CHECKING

from app.domain.xai.shap_explainer import (
    explain_with_kernel_shap,
    explain_with_tree_shap,
    shap_available,
)
from app.domain.xai.tabular import (
    extract_sklearn_estimator,
    split_sklearn_pipeline,
    tabular_feature_names,
)

if TYPE_CHECKING:  # somente para type checkers; nao executa em runtime
    from app.domain.xai.gradcam import (  # noqa: F401
        compute_gradcam,
        compute_gradcam_auto,
        find_last_conv_layer,
        heatmap_to_input_grid,
        resize_heatmap,
    )

# Grad-CAM entra SOB DEMANDA (PEP 562): `gradcam` importa TensorFlow no topo, e
# importa-lo aqui fazia com que usar SÓ o contrato tabular — sklearn puro, sem
# rede neural nenhuma — exigisse o stack de treino inteiro. Era o que impedia
# `export_rf_feature_importance.py`, que apenas lê um .pkl e plota importância
# de Gini, de rodar num ambiente sem TensorFlow.
_LAZY_GRADCAM = {
    "compute_gradcam",
    "compute_gradcam_auto",
    "find_last_conv_layer",
    "heatmap_to_input_grid",
    "resize_heatmap",
}


def __getattr__(name: str):
    if name in _LAZY_GRADCAM:
        from app.domain.xai import gradcam

        return getattr(gradcam, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(__all__)


__all__ = [
    "compute_gradcam",
    "compute_gradcam_auto",
    "find_last_conv_layer",
    "heatmap_to_input_grid",
    "resize_heatmap",
    "explain_with_kernel_shap",
    "explain_with_tree_shap",
    "shap_available",
    "extract_sklearn_estimator",
    "split_sklearn_pipeline",
    "tabular_feature_names",
]
