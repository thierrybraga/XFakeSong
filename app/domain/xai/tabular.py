"""Contrato do vetor tabular (63 no v1, 183 no v2) e utilidades scikit-learn.

Fonte única dos nomes das características consumidas por SVM e Random
Forest, na ordem exata de construção em
``benchmarks/data.py::_to_tabular_features``:

- **v1** (63): 11 estatísticas temporais + 26 MFCC + 26 RASTA-PLP;
- **v2** (183): o v1 inteiro, na mesma ordem, + 120 descritores LFCC
  (20 coeficientes estáticos, Δ e ΔΔ, com média e desvio de cada bloco).

Consumido por ``scripts/reporting/export_rf_feature_importance.py`` e pelo
módulo SHAP — que resolvem a largura pelo próprio artefato, então os dois
contratos convivem.
"""

from __future__ import annotations

from typing import Any, Callable, Optional, Tuple

import numpy as np

N_TEMPORAL = 11
N_MFCC = 26
N_RASTA = 26
N_FEATURES = N_TEMPORAL + N_MFCC + N_RASTA

N_LFCC = 20
N_LFCC_FEATURES = 6 * N_LFCC
N_FEATURES_V2 = N_FEATURES + N_LFCC_FEATURES


def tabular_feature_names() -> list[str]:
    """Nomes do vetor v1, na ordem de ``_to_tabular_features``."""
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


def tabular_feature_names_v2() -> list[str]:
    """Nomes do vetor v2: os do v1 seguidos do bloco LFCC (Δ e ΔΔ inclusos)."""
    lfcc: list[str] = []
    for label in ("LFCC", "ΔLFCC", "ΔΔLFCC"):
        lfcc += [f"{label}{i + 1} (média)" for i in range(N_LFCC)]
        lfcc += [f"{label}{i + 1} (desvio)" for i in range(N_LFCC)]
    names = tabular_feature_names() + lfcc
    assert len(names) == N_FEATURES_V2
    return names


def feature_names_for_width(n_features: int) -> list[str]:
    """Nomes correspondentes à largura de um artefato (63 ou 183).

    Existe para que os consumidores de XAI não tenham de adivinhar a versão do
    front-end: a largura do modelo carregado decide.
    """
    if int(n_features) == N_FEATURES_V2:
        return tabular_feature_names_v2()
    if int(n_features) == N_FEATURES:
        return tabular_feature_names()
    raise ValueError(
        f"largura tabular desconhecida: {n_features} "
        f"(esperado {N_FEATURES} no v1 ou {N_FEATURES_V2} no v2)"
    )


def feature_group(name: str) -> str:
    """Família: ``Temporal``, ``MFCC``, ``RASTA-PLP`` ou ``LFCC``."""
    if name.startswith("MFCC"):
        return "MFCC"
    if name.startswith("RASTA"):
        return "RASTA-PLP"
    # Cobre LFCC, ΔLFCC e ΔΔLFCC — as três compartilham o mesmo front-end.
    if "LFCC" in name:
        return "LFCC"
    return "Temporal"


def _unwrap_calibrated(estimator: Any) -> Any:
    """Estimador interno de um ``CalibratedClassifierCV``, ou ele mesmo."""
    calibrated = getattr(estimator, "calibrated_classifiers_", None)
    if not calibrated:
        return estimator
    inner = getattr(calibrated[0], "estimator", None)
    return inner if inner is not None else estimator


def extract_sklearn_estimator(obj: Any) -> Optional[Any]:
    """Resolve o estimador final dentro de um artefato scikit-learn.

    Aceita estimadores diretos, ``GridSearchCV`` (``best_estimator_``),
    ``Pipeline`` (último passo com ``predict``) e dicionários de artefatos.
    Retorna ``None`` quando nenhum estimador é encontrado.
    """
    if hasattr(obj, "best_estimator_"):
        return extract_sklearn_estimator(obj.best_estimator_)
    # CalibratedClassifierCV(ensemble=False) tem UM classificador calibrado,
    # cujo `.estimator` foi ajustado no conjunto inteiro. Sem desembrulhar, o
    # TreeExplainer receberia o invólucro de calibração e falharia — foi o que
    # ligar a calibração dos clássicos em 2026-08-09 introduziria.
    unwrapped = _unwrap_calibrated(obj)
    if unwrapped is not obj:
        return extract_sklearn_estimator(unwrapped)
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
        # Desembrulha a calibração: o TreeExplainer precisa da floresta, não do
        # CalibratedClassifierCV que a envolve. Com `ensemble=False` o
        # `.estimator` interno foi ajustado no conjunto inteiro, então explicar
        # ele é explicar o modelo — o que a calibração acrescenta é uma
        # transformação monotônica do score, que não muda a atribuição.
        estimator = _unwrap_calibrated(steps[-1][1])
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
