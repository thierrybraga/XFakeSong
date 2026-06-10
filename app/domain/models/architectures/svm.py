"""SVM Architecture Implementation

Implementação de Support Vector Machine para detecção de deepfakes.
Esta implementação segue o padrão das outras arquiteturas do projeto,
mas utiliza sklearn para o modelo SVM clássico.
Agora integrado com features extraídas da pasta segmented.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

# Project imports
from app.domain.models.architectures.classical_ml_helpers import (
    BaseClassicalModel,
    evaluate_model,
    optimize_hyperparameters,
    wrap_calibration,
)

# Configure logger for SVM
logger = logging.getLogger(__name__)
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter(
        "%(asctime)s %(levelname)s [SVM] %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
logger.setLevel(logging.INFO)


class SVMModel(BaseClassicalModel):
    """
    Wrapper para modelo SVM que segue a interface do projeto.
    Encapsula o modelo sklearn SVM para compatibilidade com o pipeline.
    """

    def __init__(self,
                 kernel: str = 'rbf',
                 C: float = 1.0,
                 gamma: str = 'scale',
                 probability: bool = True,
                 random_state: int = 42,
                 calibrate: bool = False,
                 **kwargs):
        """
        Inicializa o modelo SVM.
        """
        # NÃO repassar `probability`/`calibrate` ao super: a base armazena tudo
        # em self.kwargs, e _create_pipeline já passa probability=self.probability
        # explicitamente ao SVC → repassar aqui causaria "got multiple values
        # for keyword argument 'probability'" no fit. (Mesmo padrão do RF.)
        super().__init__(name="SVM", **kwargs)
        self.kernel = kernel
        self.C = C
        self.gamma = gamma
        self.probability = probability
        self.random_state = random_state
        # Calibração isotônica opcional (CalibratedClassifierCV) — default off.
        self.calibrate = calibrate

        # Initialize pipeline immediately
        self.pipeline = self._create_pipeline()

        logger.info(
            f"SVM model initialized with kernel={kernel}, C={C}, gamma={gamma}")

    def _create_pipeline(self) -> Pipeline:
        # Com calibração externa (CalibratedClassifierCV), desliga o Platt
        # interno do SVC (probability=True): dupla calibração aninhada degrada
        # as probabilidades e duplica o custo de fit. O CalibratedClassifierCV
        # usa decision_function e fornece o predict_proba.
        clf = SVC(
            kernel=self.kernel,
            C=self.C,
            gamma=self.gamma,
            probability=self.probability and not self.calibrate,
            random_state=self.random_state,
            **self.kwargs
        )
        clf = wrap_calibration(clf, self.calibrate)
        return Pipeline([
            ('scaler', StandardScaler()),
            ('svm', clf),
        ])

    def get_params(self) -> Dict[str, Any]:
        """
        Retorna os parâmetros do modelo.
        """
        params = super().get_params()
        params.update({
            'kernel': self.kernel,
            'C': self.C,
            'gamma': self.gamma,
            'probability': self.probability,
            'random_state': self.random_state
        })
        return params


def create_svm_model(
    input_shape: Tuple[int, ...],
    num_classes: int = 2,
    kernel: str = 'rbf',
    C: float = 1.0,
    gamma: str = 'scale',
    probability: bool = True,
    random_state: int = 42,
    architecture: str = 'svm',
    **kwargs
) -> SVMModel:
    """
    Cria um modelo SVM para detecção de deepfakes.
    """
    logger.info(f"Creating SVM model with kernel={kernel}, C={C}")
    logger.info(f"Input shape: {input_shape}, num_classes: {num_classes}")

    model = SVMModel(
        kernel=kernel,
        C=C,
        gamma=gamma,
        probability=probability,
        random_state=random_state,
        **kwargs
    )

    return model


def optimize_svm_hyperparameters(
    X_train: np.ndarray,
    y_train: np.ndarray,
    param_grid: Optional[Dict[str, List]] = None,
    cv: int = 5,
    scoring: str = 'accuracy',
    n_jobs: int = -1
) -> Tuple[SVMModel, Dict[str, Any]]:
    """
    Otimiza hiperparâmetros do SVM usando Grid Search.
    """
    if param_grid is None:
        param_grid = {
            'svm__kernel': ['linear', 'rbf', 'poly'],
            'svm__C': [0.1, 1, 10, 100],
            'svm__gamma': ['scale', 'auto', 0.001, 0.01, 0.1, 1]
        }

    return optimize_hyperparameters(
        model_class=SVMModel,
        X_train=X_train,
        y_train=y_train,
        param_grid=param_grid,
        base_estimator=SVC(probability=True, random_state=42),
        step_name='svm',
        cv=cv,
        scoring=scoring,
        n_jobs=n_jobs
    )


def evaluate_svm_model(
    model: SVMModel,
    X_test: np.ndarray,
    y_test: np.ndarray,
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Avalia o desempenho do modelo SVM.
    """
    return evaluate_model(model, X_test, y_test, None, verbose)


# Função de compatibilidade com o sistema
def create_model(
    input_shape: Tuple[int, ...],
    num_classes: int = 2,
    **kwargs
) -> SVMModel:
    """
    Função de compatibilidade para criação do modelo SVM.
    Segue a interface padrão das outras arquiteturas.
    """
    return create_svm_model(
        input_shape=input_shape,
        num_classes=num_classes,
        **kwargs
    )


# Metadados da arquitetura
ARCHITECTURE_INFO = {
    'name': 'SVM',
    'description': 'Support Vector Machine para classificação de deepfakes',
    'type': 'classical_ml',
    'input_requirements': {
        'min_features': 1,
        'max_features': None,
        'feature_type': 'numerical'
    },
    'hyperparameters': {
        'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
        'C': 'float > 0',
        'gamma': ['scale', 'auto', 'float > 0'],
        'probability': 'bool'
    },
    'advantages': [
        'Efetivo em espaços de alta dimensionalidade',
        'Memória eficiente',
        'Versátil (diferentes kernels)',
        'Funciona bem com poucos dados'
    ],
    'disadvantages': [
        'Não fornece estimativas de probabilidade diretamente',
        'Sensível à escala das features',
        'Pode ser lento em datasets grandes'
    ]
}
