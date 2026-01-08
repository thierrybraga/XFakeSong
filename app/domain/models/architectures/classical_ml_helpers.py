"""
Classical ML Helpers

This module provides base classes and helper functions for classical machine learning models
(Random Forest, SVM, etc.) to reduce code duplication.
"""

from __future__ import annotations

import logging
import os
import joblib
import numpy as np
from abc import ABC, abstractmethod
from typing import List, Tuple, Optional, Any, Dict, Union
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

from app.domain.features.segmented_feature_loader import create_feature_loader

logger = logging.getLogger(__name__)


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
        self.probability = kwargs.get('probability', True)

    @abstractmethod
    def _create_pipeline(self) -> Pipeline:
        """
        Abstract method to create the specific model pipeline.
        Must be implemented by subclasses.
        """
        pass

    def fit(self, X: np.ndarray, y: np.ndarray, **kwargs) -> 'BaseClassicalModel':
        """
        Trains the model.
        """
        logger.info(f"Training {self.name} model with {X.shape[0]} samples and {X.shape[1]} features")

        if len(X.shape) != 2:
            raise ValueError(f"Expected 2D input, got shape {X.shape}")

        if self.pipeline is None:
            self.pipeline = self._create_pipeline()

        self.pipeline.fit(X, y)
        self.is_fitted = True
        
        # Extract classes from the final estimator
        final_estimator = self.pipeline.steps[-1][1]
        if hasattr(final_estimator, 'classes_'):
            self.classes_ = final_estimator.classes_
            
        # Extract feature importances if available
        if hasattr(final_estimator, 'feature_importances_'):
            self.feature_importances_ = final_estimator.feature_importances_

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
        if hasattr(final_estimator, 'predict_proba'):
             return self.pipeline.predict_proba(X)
        elif hasattr(final_estimator, 'probability') and not final_estimator.probability:
             raise ValueError("Model was not trained with probability=True")
        else:
             raise ValueError(f"Model {self.name} does not support probability predictions")

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
        """
        return self.kwargs

    def save(self, filepath: str) -> None:
        """
        Saves the trained model.
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before saving")

        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        joblib.dump(self.pipeline, filepath)
        logger.info(f"{self.name} model saved to {filepath}")

    @classmethod
    def load(cls, filepath: str, model_class: type) -> 'BaseClassicalModel':
        """
        Loads a saved model.
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Model file not found: {filepath}")

        pipeline = joblib.load(filepath)
        
        # Extract parameters from the pipeline's final step
        final_step_name = pipeline.steps[-1][0]
        final_estimator = pipeline.steps[-1][1]
        params = final_estimator.get_params()
        
        # Filter params to match __init__ arguments if possible, or just pass as kwargs
        # This part depends on how the subclass __init__ is structured.
        # Ideally, we reconstruct the object using the params.
        
        instance = model_class(**params)
        instance.pipeline = pipeline
        instance.is_fitted = True
        
        if hasattr(final_estimator, 'classes_'):
            instance.classes_ = final_estimator.classes_
        if hasattr(final_estimator, 'feature_importances_'):
            instance.feature_importances_ = final_estimator.feature_importances_

        logger.info(f"{instance.name} model loaded from {filepath}")
        return instance

    def train_with_segmented_features(self,
                                      segmented_path: str = "datasets/features/segmented",
                                      feature_types: Optional[List[str]] = None,
                                      aggregate_method: str = 'mean',
                                      test_size: float = 0.2,
                                      max_samples_per_class: Optional[int] = None,
                                      random_state: int = 42) -> Dict[str, Any]:
        """
        Generic training with segmented features.
        """
        logger.info(f"Training {self.name} with segmented features")
        logger.info(f"Segmented path: {segmented_path}")
        logger.info(f"Feature types: {feature_types}")

        feature_loader = create_feature_loader(
            segmented_path=segmented_path,
            feature_types=feature_types,
            normalize=False,  # Pipeline handles scaling
            aggregate_method=aggregate_method
        )

        if max_samples_per_class:
            X, y, feature_names = feature_loader.load_multiple_samples_per_class(
                classes=['real', 'fake'],
                max_samples_per_class=max_samples_per_class
            )
        else:
            X, y, feature_names = feature_loader.load_multiple_samples_per_class(
                classes=['real', 'fake']
            )

        X_train, X_test, y_train, y_test = feature_loader.prepare_train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )

        self.fit(X_train, y_train)
        
        train_score = self.score(X_train, y_train)
        test_score = self.score(X_test, y_test)
        
        y_pred = self.predict(X_test)
        try:
            y_proba = self.predict_proba(X_test)
        except ValueError:
            y_proba = None

        conf_matrix = confusion_matrix(y_test, y_pred)
        class_report = classification_report(y_test, y_pred, output_dict=True)

        feature_importance = None
        top_features = None
        if self.feature_importances_ is not None:
            feature_names_list = feature_names if feature_names else [f"feature_{i}" for i in range(len(self.feature_importances_))]
            feature_importance = dict(zip(feature_names_list, self.feature_importances_))
            top_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)[:20]

        results = {
            'train_accuracy': train_score,
            'test_accuracy': test_score,
            'confusion_matrix': conf_matrix.tolist(),
            'classification_report': class_report,
            'feature_count': X.shape[1],
            'train_samples': X_train.shape[0],
            'test_samples': X_test.shape[0],
            'feature_types_used': feature_types or feature_loader.FEATURE_TYPES,
            'aggregate_method': aggregate_method,
            'classes': feature_loader.label_encoder.classes_.tolist(),
            'y_test': y_test.tolist(),
            'y_proba': y_proba.tolist() if y_proba is not None else []
        }
        
        if feature_importance:
            results['feature_importance'] = feature_importance
            results['top_features'] = top_features

        logger.info(f"Training completed. Train acc: {train_score:.4f}, Test acc: {test_score:.4f}")
        return results


def optimize_hyperparameters(
    model_class: type,
    X_train: np.ndarray,
    y_train: np.ndarray,
    param_grid: Dict[str, List],
    base_estimator: Any,
    step_name: str,
    cv: int = 5,
    scoring: str = 'accuracy',
    n_jobs: int = -1
) -> Tuple[Any, Dict[str, Any]]:
    """
    Generic hyperparameter optimization.
    """
    logger.info(f"Starting hyperparameter optimization for {model_class.__name__}")

    base_pipeline = Pipeline([
        ('scaler', StandardScaler()),
        (step_name, base_estimator)
    ])

    grid_search = GridSearchCV(
        base_pipeline,
        param_grid,
        cv=cv,
        scoring=scoring,
        n_jobs=n_jobs,
        verbose=1
    )

    grid_search.fit(X_train, y_train)

    best_params = grid_search.best_params_
    
    # Extract params for the model class
    model_params = {
        k.replace(f'{step_name}__', ''): v 
        for k, v in best_params.items() 
        if k.startswith(f'{step_name}__')
    }

    optimized_model = model_class(**model_params)
    optimized_model.pipeline = grid_search.best_estimator_
    optimized_model.is_fitted = True
    
    final_estimator = grid_search.best_estimator_.named_steps[step_name]
    if hasattr(final_estimator, 'classes_'):
        optimized_model.classes_ = final_estimator.classes_
    if hasattr(final_estimator, 'feature_importances_'):
        optimized_model.feature_importances_ = final_estimator.feature_importances_

    logger.info(f"Optimization completed. Best score: {grid_search.best_score_:.4f}")
    return optimized_model, best_params


def evaluate_model(
    model: BaseClassicalModel,
    X_test: np.ndarray,
    y_test: np.ndarray,
    feature_names: Optional[List[str]] = None,
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Generic model evaluation.
    """
    y_pred = model.predict(X_test)
    try:
        y_proba = model.predict_proba(X_test)
    except ValueError:
        y_proba = None

    accuracy = accuracy_score(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)
    class_report = classification_report(y_test, y_pred, output_dict=True)

    feature_importance = None
    if model.feature_importances_ is not None:
        names = feature_names if feature_names else [f"feature_{i}" for i in range(len(model.feature_importances_))]
        feature_importance = dict(zip(names, model.feature_importances_))

    results = {
        'accuracy': accuracy,
        'confusion_matrix': conf_matrix.tolist(),
        'classification_report': class_report,
        'predictions': y_pred.tolist(),
        'probabilities': y_proba.tolist() if y_proba is not None else None
    }
    
    if feature_importance:
        results['feature_importance'] = feature_importance

    if verbose:
        logger.info(f"{model.name} Evaluation Results:")
        logger.info(f"Accuracy: {accuracy:.4f}")
        logger.info(f"Confusion Matrix:\n{conf_matrix}")
        
        if feature_importance:
            sorted_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
            logger.info("Top 10 Most Important Features:")
            for i, (feature, importance) in enumerate(sorted_features[:10]):
                logger.info(f"  {i + 1:2d}. {feature}: {importance:.4f}")

    return results
