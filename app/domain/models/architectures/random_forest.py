"""Random Forest Architecture Implementation

Implementação de Random Forest para detecção de deepfakes.
Esta implementação segue o padrão das outras arquiteturas do projeto,
mas utiliza sklearn para o modelo Random Forest clássico.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# Project imports
from app.domain.models.architectures.classical_ml_helpers import (
    BaseClassicalModel,
    unwrap_calibrated,
    wrap_calibration,
)

# Convenção do projeto: logger de módulo SEM handlers/level próprios — a
# configuração de handlers, formatters e nível é responsabilidade da aplicação.
# Handlers locais duplicavam linhas de log quando o root já estava configurado.
logger = logging.getLogger(__name__)

#: Grid de busca do Random Forest — FONTE ÚNICA (treino do app e benchmark).
#:
#: AJUSTE (retune): o grid original selecionava max_depth=20/None com
#: min_samples_leaf=1 → `train_score=1.0` (overfit) e robustez ruim sob ruído.
#: Remove profundidade ilimitada e exige folhas/splits maiores.
#:
#: CORREÇÃO 2026-08-09: este grid NUNCA rodou. `benchmarks/runner.py::
#: _classical_search_space` mantinha uma cópia própria — uma 4ª fonte de
#: hiperparâmetros — com `max_depth: [None, 10, 20]`, `min_samples_leaf: [1,2]`
#: e SEM `min_samples_split` (preso no default 2 do sklearn). No
#: `clean_benchmark_15k` venceu `max_depth=None, min_samples_leaf=2` com
#: `mean_train_score = 1.0`: o overfitting que este grid existia para corrigir
#: seguiu intacto porque a correção morava no arquivo errado. Isso também
#: invalida a conclusão de 2026-07-06/07 de que "as métricas idênticas provam
#: que o ótimo já caía na faixa restrita" — comparava o grid do runner com ele
#: mesmo. Agora o runner importa daqui.
RANDOM_FOREST_PARAM_GRID = {
    "rf__n_estimators": [200, 300],
    "rf__max_depth": [10, 15, 20],
    "rf__min_samples_split": [5, 10, 20],
    "rf__min_samples_leaf": [2, 4, 8],
    "rf__max_features": ["sqrt", "log2"],
}


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
                 calibrate: bool = False,
                 calibration_cv=None,
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
        # Calibração isotônica opcional — RF tem probabilidades mal-calibradas
        # por natureza; útil para limiares de decisão confiáveis. Default off.
        self.calibrate = calibrate
        # Ver a nota em svm.py: dobras AGRUPADAS, não o cv=3 não agrupado.
        self.calibration_cv = calibration_cv

        # Initialize pipeline immediately
        self.pipeline = self._create_pipeline()

        logger.info(
            f"Random Forest model initialized with {n_estimators} estimators, max_depth={max_depth}")

    def _create_pipeline(self) -> Pipeline:
        clf = RandomForestClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            min_samples_split=self.min_samples_split,
            min_samples_leaf=self.min_samples_leaf,
            max_features=self.max_features,
            bootstrap=self.bootstrap,
            random_state=self.random_state,
            n_jobs=self.n_jobs,
            **self.kwargs
        )
        clf = wrap_calibration(
            clf, self.calibrate, cv=self.calibration_cv or 3
        )
        return Pipeline([
            ('scaler', StandardScaler()),
            ('rf', clf),
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
            # `unwrap_calibrated`: sob calibração o passo 'rf' é o
            # CalibratedClassifierCV, não a floresta.
            step = unwrap_calibrated(self.pipeline.named_steps['rf'])
            if hasattr(step, 'feature_importances_'):
                self.feature_importances_ = step.feature_importances_
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

    ``architecture`` é um SINK INTENCIONAL, não um parâmetro morto: a interface
    comum das arquiteturas passa o nome da variante, e sem absorvê-lo aqui a
    chave vazaria por ``**kwargs`` até o ``RandomForestClassifier`` do
    scikit-learn (TypeError). O RF não tem variantes topológicas.
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


