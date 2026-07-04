"""Contrato do vetor tabular (63 descritores) e utilidades scikit-learn.

Fonte única dos nomes das características consumidas por SVM e Random
Forest, na ordem exata de construção em
``benchmarks/data.py::_to_tabular_features`` (11 estatísticas temporais +
26 MFCC + 26 RASTA-PLP). Consumido por
``scripts/reporting/export_rf_feature_importance.py`` e pelo módulo SHAP.
"""

from __future__ import annotations

from typing import Any, Callable, Optional, Tuple

import numpy as np

N_TEMPORAL = 11
N_MFCC = 26
N_RASTA = 26
N_FEATURES = N_TEMPORAL + N_MFCC + N_RASTA


def tabular_feature_names() -> list[str]:
    """Nomes na ordem exata de ``benchmarks/data.py::_to_tabular_features``."""
    temporal = [
        "média",
        "desvio-padrão",
        "média |x|",
        "RMS",
        "mínimo",
        "máximo",
        "percentil 25",
        "percentil 50",
        "percentil 75",
        "energia da diferença",
        "ZCR",
    ]
    mfcc = [f"MFCC{i + 1} (média)" for i in range(13)] + [
        f"MFCC{i + 1} (desvio)" for i in range(13)
    ]
    rasta = [f"RASTA-PLP{i + 1} (média)" for i in range(13)] + [
        f"RASTA-PLP{i + 1} (desvio)" for i in range(13)
    ]
    names = temporal + mfcc + rasta
    assert len(names) == N_FEATURES
    return names


def feature_group(name: str) -> str:
    """Família de um descritor: ``Temporal``, ``MFCC`` ou ``RASTA-PLP``."""
    if name.startswith("MFCC"):
        return "MFCC"
    if name.startswith("RASTA"):
        return "RASTA-PLP"
    return "Temporal"


def extract_sklearn_estimator(obj: Any) -> Optional[Any]:
    """Resolve o estimador final dentro de um artefato scikit-learn.

    Aceita estimadores diretos, ``GridSearchCV`` (``best_estimator_``),
    ``Pipeline`` (último passo com ``predict``) e dicionários de artefatos.
    Retorna ``None`` quando nenhum estimador é encontrado.
    """
    if hasattr(obj, "best_estimator_"):
        return extract_sklearn_estimator(obj.best_estimator_)
    if hasattr(obj, "named_steps"):
        for step in reversed(list(obj.named_steps.values())):
            found = extract_sklearn_estimator(step)
            if found is not None:
                return found
        return None
    if isinstance(obj, dict):
        for value in obj.values():
            found = extract_sklearn_estimator(value)
            if found is not None:
                return found
        return None
    if hasattr(obj, "predict"):
        return obj
    return None


def split_sklearn_pipeline(
    obj: Any,
) -> Tuple[Callable[[np.ndarray], np.ndarray], Any]:
    """Separa um artefato em ``(transformar_entrada, estimador_final)``.

    Para ``Pipeline`` com passos de pré-processamento (por exemplo,
    ``StandardScaler``), retorna uma função que aplica todos os passos
    exceto o último — necessário para explicar o estimador final no espaço
    de entrada que ele realmente enxerga (caso do ``TreeExplainer``).
    Para estimadores diretos, a transformação é a identidade.
    """
    if hasattr(obj, "best_estimator_"):
        return split_sklearn_pipeline(obj.best_estimator_)
    if isinstance(obj, dict):
        for value in obj.values():
            try:
                return split_sklearn_pipeline(value)
            except ValueError:
                continue
        raise ValueError("Nenhum estimador encontrado no dicionário de artefato.")
    if hasattr(obj, "named_steps") and hasattr(obj, "steps"):
        steps = list(obj.steps)
        if not steps:
            raise ValueError("Pipeline vazio.")
        estimator = steps[-1][1]
        if len(steps) == 1:
            return (lambda X: np.asarray(X)), estimator

        def transform(X: np.ndarray) -> np.ndarray:
            out = np.asarray(X)
            for _, step in steps[:-1]:
                out = step.transform(out)
            return out

        return transform, estimator
    if hasattr(obj, "predict"):
        return (lambda X: np.asarray(X)), obj
    raise ValueError(f"Artefato não reconhecido: {type(obj)!r}")
