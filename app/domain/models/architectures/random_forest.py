"""Random Forest Architecture Implementation

Implementação de Random Forest para detecção de deepfakes.
Esta implementação segue o padrão das outras arquiteturas do projeto,
mas utiliza sklearn para o modelo Random Forest clássico.
"""

from __future__ import annotations

import logging
from typing import List, Tuple, Optional, Any, Dict

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# Project imports
from app.domain.models.architectures.classical_ml_helpers import (
    BaseClassicalModel,
    optimize_hyperparameters,
    evaluate_model
)

# Configure logger for Random Forest
logger = logging.getLogger(__name__)
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter(
        "%(asctime)s %(levelname)s [RandomForest] %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
logger.setLevel(logging.INFO)


class RandomForestModel(BaseClassicalModel):
    """
    Wrapper para modelo Random Forest que segue a interface do projeto.
    Encapsula o modelo sklearn Random Forest para compatibilidade com o pipeline.
    """

    def __init__(self,
                 n_estimators: int = 100,
                 max_depth: Optional[int] = None,
                 min_samples_split: int = 2,
                 min_samples_leaf: int = 1,
                 max_features: str = 'sqrt',
                 bootstrap: bool = True,
                 random_state: int = 42,
                 n_jobs: int = -1,
                 **kwargs):
        """
        Inicializa o modelo Random Forest.
        """
        super().__init__(name="RandomForest", **kwargs)
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.bootstrap = bootstrap
        self.random_state = random_state
        self.n_jobs = n_jobs
        
        # Initialize pipeline immediately
        self.pipeline = self._create_pipeline()
        
        logger.info(
            f"Random Forest model initialized with {n_estimators} estimators, max_depth={max_depth}")

    def _create_pipeline(self) -> Pipeline:
        return Pipeline([
            ('scaler', StandardScaler()),
            ('rf', RandomForestClassifier(
                n_estimators=self.n_estimators,
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                min_samples_leaf=self.min_samples_leaf,
                max_features=self.max_features,
                bootstrap=self.bootstrap,
                random_state=self.random_state,
                n_jobs=self.n_jobs,
                **self.kwargs
            ))
        ])

    def get_feature_importance(
            self, feature_names: Optional[List[str]] = None) -> Dict[str, float]:
        """
        Retorna a importância das features.
        """
        if not self.is_fitted:
            raise ValueError(
                "Model must be fitted before getting feature importance")

        if getattr(self, 'feature_importances_', None) is None:
            # Try to get from pipeline if not set (e.g. if loaded from pickle without proper state restoration)
            if hasattr(self.pipeline.named_steps['rf'], 'feature_importances_'):
                self.feature_importances_ = self.pipeline.named_steps['rf'].feature_importances_
            else:
                raise ValueError("Underlying Random Forest model does not provide feature importances")

        if feature_names is None:
            feature_names = [f"feature_{i}" for i in range(
                len(self.feature_importances_))]

        return dict(zip(feature_names, self.feature_importances_))

    def get_params(self) -> Dict[str, Any]:
        """
        Retorna os parâmetros do modelo.
        """
        params = super().get_params()
        params.update({
            'n_estimators': self.n_estimators,
            'max_depth': self.max_depth,
            'min_samples_split': self.min_samples_split,
            'min_samples_leaf': self.min_samples_leaf,
            'max_features': self.max_features,
            'bootstrap': self.bootstrap,
            'random_state': self.random_state,
            'n_jobs': self.n_jobs
        })
        return params


def create_random_forest_model(
    input_shape: Tuple[int, ...],
    num_classes: int = 2,
    n_estimators: int = 100,
    max_depth: Optional[int] = None,
    min_samples_split: int = 2,
    min_samples_leaf: int = 1,
    max_features: str = 'sqrt',
    bootstrap: bool = True,
    random_state: int = 42,
    n_jobs: int = -1,
    architecture: str = 'random_forest',
    **kwargs
) -> RandomForestModel:
    """
    Cria um modelo Random Forest para detecção de deepfakes.
    """
    logger.info(f"Creating Random Forest model with {n_estimators} estimators")
    logger.info(f"Input shape: {input_shape}, num_classes: {num_classes}")

    model = RandomForestModel(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        max_features=max_features,
        bootstrap=bootstrap,
        random_state=random_state,
        n_jobs=n_jobs,
        **kwargs
    )

    return model


def optimize_random_forest_hyperparameters(
    X_train: np.ndarray,
    y_train: np.ndarray,
    param_grid: Optional[Dict[str, List]] = None,
    cv: int = 5,
    scoring: str = 'accuracy',
    n_jobs: int = -1
) -> Tuple[RandomForestModel, Dict[str, Any]]:
    """
    Otimiza hiperparâmetros do Random Forest usando Grid Search.
    """
    if param_grid is None:
        param_grid = {
            'rf__n_estimators': [50, 100, 200],
            'rf__max_depth': [None, 10, 20, 30],
            'rf__min_samples_split': [2, 5, 10],
            'rf__min_samples_leaf': [1, 2, 4],
            'rf__max_features': ['sqrt', 'log2', None]
        }

    return optimize_hyperparameters(
        model_class=RandomForestModel,
        X_train=X_train,
        y_train=y_train,
        param_grid=param_grid,
        base_estimator=RandomForestClassifier(random_state=42, n_jobs=n_jobs),
        step_name='rf',
        cv=cv,
        scoring=scoring,
        n_jobs=n_jobs
    )


def evaluate_random_forest_model(
    model: RandomForestModel,
    X_test: np.ndarray,
    y_test: np.ndarray,
    feature_names: Optional[List[str]] = None,
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Avalia o desempenho do modelo Random Forest.
    """
    return evaluate_model(model, X_test, y_test, feature_names, verbose)


def analyze_feature_importance(
    model: RandomForestModel,
    feature_names: Optional[List[str]] = None,
    top_n: int = 20
) -> Dict[str, Any]:
    """
    Analisa a importância das features do modelo Random Forest.
    """
    if not model.is_fitted:
        raise ValueError(
            "Model must be fitted before analyzing feature importance")

    feature_importance = model.get_feature_importance(feature_names)

    # Ordenar por importância
    sorted_features = sorted(
        feature_importance.items(),
        key=lambda x: x[1],
        reverse=True)

    # Top N features
    top_features = sorted_features[:top_n]

    # Estatísticas
    importances = list(feature_importance.values())
    stats = {
        'mean_importance': np.mean(importances),
        'std_importance': np.std(importances),
        'max_importance': np.max(importances),
        'min_importance': np.min(importances)
    }

    analysis = {
        'feature_importance': feature_importance,
        'top_features': dict(top_features),
        'statistics': stats,
        'total_features': len(feature_importance)
    }

    logger.info(
        f"Feature importance analysis completed for {len(feature_importance)} features"
    )
    logger.info(
        f"Top feature: {top_features[0][0]} "
        f"(importance: {top_features[0][1]:.4f})"
    )
            
    return analysis
