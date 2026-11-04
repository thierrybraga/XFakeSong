"""
Classical ML Helpers

This module provides base classes and helper functions for classical machine learning models
(Random Forest, SVM, etc.) to reduce code duplication.
"""

from __future__ import annotations

import logging
import os
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

import joblib
import numpy as np
from sklearn.pipeline import Pipeline

logger = logging.getLogger(__name__)


def wrap_calibration(
    classifier, calibrate: bool, cv: int = 3, method: str = "isotonic"
):
    """Envolve um classificador sklearn em `CalibratedClassifierCV` (opt-in).

    Calibra as probabilidades de saída (default isotônica) — útil sobretudo para
    Random Forest, cujas probabilidades são naturalmente mal-calibradas. Quando
    `calibrate=False` (padrão), retorna o classificador inalterado (sem mudança
    de comportamento). Tolerante à versão do sklearn (`estimator=` ≥1.2 vs
    `base_estimator=`).

    ``ensemble=False`` (2026-08-09): com o default ``True`` o resultado é uma
    MÉDIA de `cv` modelos, cada um ajustado em `1 - 1/cv` dos dados, e nenhum
    deles é "o modelo" — `feature_importances_` some, o SHAP não tem um
    estimador para explicar e o artefato promovido deixa de ter uma topologia
    única. Com ``False`` o calibrador é ajustado sobre predições
    OUT-OF-FOLD (`cross_val_predict`) e o estimador final é ajustado em TODOS
    os dados: uma árvore só, calibrada, explicável — e ainda mais barato.
    """
    if not calibrate:
        return classifier
    from sklearn.calibration import CalibratedClassifierCV

    try:
        return CalibratedClassifierCV(
            estimator=classifier, method=method, cv=cv, ensemble=False
        )
    except TypeError:  # sklearn < 1.2
        return CalibratedClassifierCV(
            base_estimator=classifier, method=method, cv=cv, ensemble=False
        )


def unwrap_calibrated(estimator):
    """Devolve o estimador interno de um ``CalibratedClassifierCV``, ou ele mesmo.

    Com ``ensemble=False`` há exatamente um ``calibrated_classifiers_``, cujo
    ``.estimator`` foi ajustado no conjunto inteiro — é o modelo de verdade.
    Usado para recuperar ``feature_importances_`` e para o SHAP enxergar a
    floresta em vez do invólucro de calibração.
    """
    calibrated = getattr(estimator, "calibrated_classifiers_", None)
    if not calibrated:
        return estimator
    inner = getattr(calibrated[0], "estimator", None)
    return inner if inner is not None else estimator


class BaseClassicalModel(ABC):
    """
    Base class for classical ML models in the project.
    """

    def __init__(self, name: str, **kwargs):
        self.name = name
        self.kwargs = kwargs
        self.pipeline: Optional[Pipeline] = None
        self.is_fitted = False
        self.classes_ = None
        self.feature_importances_ = None  # Only for models that support it
        # Default to True for consistency
        self.probability = kwargs.get("probability", True)

    @abstractmethod
    def _create_pipeline(self) -> Pipeline:
        """
        Abstract method to create the specific model pipeline.
        Must be implemented by subclasses.
        """
        pass

    def fit(self, X: np.ndarray, y: np.ndarray, **kwargs) -> "BaseClassicalModel":
        """
        Trains the model.
        """
        logger.info(
            f"Training {self.name} model with {X.shape[0]} samples and {X.shape[1]} features"
        )

        if len(X.shape) != 2:
            raise ValueError(f"Expected 2D input, got shape {X.shape}")

        if self.pipeline is None:
            self.pipeline = self._create_pipeline()

        self.pipeline.fit(X, y)
        self.is_fitted = True

        # Extract classes from the final estimator
        final_estimator = self.pipeline.steps[-1][1]
        if hasattr(final_estimator, "classes_"):
            self.classes_ = final_estimator.classes_

        # Extract feature importances if available.
        #
        # `unwrap_calibrated`: sob calibração o passo final é um
        # `CalibratedClassifierCV`, que não expõe `feature_importances_` —
        # sem isto, ligar a calibração do Random Forest quebraria em silêncio
        # `get_feature_importance()`, o `export_rf_feature_importance.py` e o
        # módulo SHAP, que são o material de XAI do trabalho.
        inner_estimator = unwrap_calibrated(final_estimator)
        if hasattr(inner_estimator, "feature_importances_"):
            self.feature_importances_ = inner_estimator.feature_importances_

        train_accuracy = self.pipeline.score(X, y)
        logger.info(f"Training completed. Training accuracy: {train_accuracy:.4f}")

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Makes predictions with the model.
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
        return self.pipeline.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Makes probability predictions with the model.
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")

        # Check if the underlying model supports probability
        final_estimator = self.pipeline.steps[-1][1]
        if hasattr(final_estimator, "predict_proba"):
            return self.pipeline.predict_proba(X)
        elif (
            hasattr(final_estimator, "probability") and not final_estimator.probability
        ):
            raise ValueError("Model was not trained with probability=True")
        else:
            raise ValueError(
                f"Model {self.name} does not support probability predictions"
            )

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Calculates model accuracy.
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before scoring")
        return self.pipeline.score(X, y)

    def get_params(self) -> Dict[str, Any]:
        """
        Returns model parameters.

        Retorna uma CÓPIA de ``self.kwargs``: subclasses (SVM/RF) fazem
        ``params = super().get_params(); params.update({...})`` — sem a cópia,
        o ``update`` envenenaria ``self.kwargs`` com chaves como ``kernel`` e a
        próxima chamada a ``_create_pipeline`` quebraria com "got multiple values
        for keyword argument 'kernel'".
        """
        return dict(self.kwargs)

    def save(self, filepath: str) -> None:
        """
        Saves the trained model.
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before saving")

        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        from app.utils.file_storage import break_hardlink

        break_hardlink(filepath)
        joblib.dump(self.pipeline, filepath)
        logger.info(f"{self.name} model saved to {filepath}")

    @classmethod
    def load(cls, filepath: str, model_class: type) -> "BaseClassicalModel":
        """
        Loads a saved model.
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Model file not found: {filepath}")

        pipeline = joblib.load(filepath)

        # Extract parameters from the pipeline's final step
        final_estimator = pipeline.steps[-1][1]
        params = final_estimator.get_params()

        # Filter params to match __init__ arguments if possible, or just pass as kwargs
        # This part depends on how the subclass __init__ is structured.
        # Ideally, we reconstruct the object using the params.

        instance = model_class(**params)
        instance.pipeline = pipeline
        instance.is_fitted = True

        if hasattr(final_estimator, "classes_"):
            instance.classes_ = final_estimator.classes_
        if hasattr(final_estimator, "feature_importances_"):
            instance.feature_importances_ = final_estimator.feature_importances_

        logger.info(f"{instance.name} model loaded from {filepath}")
        return instance
