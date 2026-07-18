"""XAI (Explainable AI) — wrappers de explicabilidade do XFakeSong.

Camada de domínio (``app/domain``) — lógica de explicabilidade específica
dos modelos do projeto (não é infraestrutura genérica, por isso vive fora
de ``app/core``):

- :mod:`app.domain.xai.gradcam` — mapas de ativação Grad-CAM para modelos
  Keras (implementação pura TensorFlow, sem dependência externa);
- :mod:`app.domain.xai.shap_explainer` — wrapper da biblioteca ``shap``
  (import guardado) para os classificadores clássicos sobre o vetor tabular;
- :mod:`app.domain.xai.tabular` — contrato do vetor tabular de 63 descritores
  (nomes na ordem canônica) e utilidades para artefatos scikit-learn.

CLI correspondente: ``scripts/reporting/run_shap_analysis.py``.
"""

from app.domain.xai.gradcam import (
    compute_gradcam,
    compute_gradcam_auto,
    find_last_conv_layer,
    heatmap_to_input_grid,
    resize_heatmap,
)
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
