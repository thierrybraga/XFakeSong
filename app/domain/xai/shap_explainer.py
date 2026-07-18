"""Wrapper da biblioteca ``shap`` para os classificadores clássicos.

Isola a dependência externa (import guardado, mensagem acionável quando
ausente) e normaliza as saídas para arrays simples:

- :func:`explain_with_tree_shap` — ``TreeExplainer`` para o Random Forest,
  aplicado no espaço de entrada que o estimador final enxerga (após o
  pré-processamento do pipeline);
- :func:`explain_with_kernel_shap` — ``KernelExplainer`` (agnóstico de
  modelo) para o SVM, com background comprimido por k-means para custo
  tratável.

As funções retornam a matriz SHAP da classe *spoof* com shape
``(n_amostras, n_features)``.
"""

from __future__ import annotations

import logging
from typing import Any, Callable

import numpy as np

logger = logging.getLogger(__name__)

_INSTALL_HINT = (
    "A biblioteca 'shap' não está instalada. Instale com "
    "`pip install -r requirements-dev.txt` (ou `pip install shap`) para "
    "executar a análise SHAP."
)


def shap_available() -> bool:
    """Indica se a biblioteca ``shap`` está disponível no ambiente."""
    try:
        import shap  # noqa: F401
    except ImportError:
        return False
    return True


def _require_shap():
    try:
        import shap
    except ImportError as exc:  # pragma: no cover - depende do ambiente
        raise RuntimeError(_INSTALL_HINT) from exc
    return shap


def _spoof_class_matrix(shap_values: Any, n_features: int) -> np.ndarray:
    """Normaliza o retorno do shap para a matriz da classe positiva (spoof).

    Versões/explicadores diferentes retornam ``list[classe] -> (n, f)``,
    ``(n, f)`` direto ou ``(n, f, n_classes)``.
    """
    if isinstance(shap_values, list):
        values = shap_values[1] if len(shap_values) > 1 else shap_values[0]
        return np.asarray(values)
    values = np.asarray(shap_values)
    if values.ndim == 3:
        return values[:, :, -1]
    if values.ndim == 2 and values.shape[1] == n_features:
        return values
    raise ValueError(f"Formato SHAP inesperado: shape={values.shape}")


def explain_with_tree_shap(
    estimator: Any,
    X_explain: np.ndarray,
) -> np.ndarray:
    """SHAP exato para modelos de árvore (Random Forest).

    Args:
        estimator: estimador de árvore já treinado (passo final do pipeline);
            as entradas devem estar no espaço pós-pré-processamento.
        X_explain: amostras a explicar ``(n, n_features)``.

    Returns:
        Matriz SHAP da classe *spoof* ``(n, n_features)``.
    """
    shap = _require_shap()
    X_explain = np.asarray(X_explain, dtype="float64")
    explainer = shap.TreeExplainer(estimator)
    values = explainer.shap_values(X_explain, check_additivity=False)
    matrix = _spoof_class_matrix(values, X_explain.shape[1])
    logger.info("TreeExplainer: %d amostras × %d features", *matrix.shape)
    return matrix


def explain_with_kernel_shap(
    predict_proba_spoof: Callable[[np.ndarray], np.ndarray],
    X_background: np.ndarray,
    X_explain: np.ndarray,
    background_clusters: int = 25,
    nsamples: int | str = "auto",
) -> np.ndarray:
    """SHAP agnóstico de modelo via ``KernelExplainer`` (usado para o SVM).

    Args:
        predict_proba_spoof: função ``X -> p_spoof`` (probabilidade da
            classe positiva), tipicamente ``pipeline.predict_proba[:, 1]``.
        X_background: amostras de referência ``(m, n_features)``; são
            comprimidas em ``background_clusters`` centróides via k-means.
        X_explain: amostras a explicar ``(n, n_features)``.
        background_clusters: nº de centróides do background.
        nsamples: orçamento de avaliações por amostra (``"auto"`` delega
            ao shap).

    Returns:
        Matriz SHAP ``(n, n_features)`` da saída ``p_spoof``.

    Nota de custo: o KernelExplainer avalia o modelo O(nsamples) vezes por
    amostra explicada — mantenha ``X_explain`` pequeno (dezenas).
    """
    shap = _require_shap()
    X_background = np.asarray(X_background, dtype="float64")
    X_explain = np.asarray(X_explain, dtype="float64")
    clusters = min(int(background_clusters), len(X_background))
    background = shap.kmeans(X_background, clusters)
    explainer = shap.KernelExplainer(predict_proba_spoof, background)
    values = explainer.shap_values(X_explain, nsamples=nsamples, silent=True)
    matrix = _spoof_class_matrix(values, X_explain.shape[1])
    logger.info(
        "KernelExplainer: %d amostras × %d features (background=%d)",
        matrix.shape[0], matrix.shape[1], clusters,
    )
    return matrix
