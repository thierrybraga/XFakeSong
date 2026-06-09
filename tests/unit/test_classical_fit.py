"""Regressão: modelos clássicos (SVM/RF) construídos pela factory devem TREINAR.

Pega o bug onde `SVMModel` repassava `probability` ao super (→ self.kwargs) e
`_create_pipeline` o passava de novo ao SVC → "got multiple values for keyword
argument 'probability'" no `.fit()`. Checks anteriores só CONSTRUÍam o modelo
(sem treinar), então o erro de fit nunca era exercido.
"""

from __future__ import annotations

import numpy as np
import pytest

TAB = (7616,)


def _data(n=40, dim=7616):
    rng = np.random.default_rng(0)
    return rng.standard_normal((n, dim)).astype("float32"), rng.integers(0, 2, n)


def test_svm_factory_fit_and_proba():
    from app.domain.models.architectures import svm as svm_mod

    X, y = _data()
    model = svm_mod.create_model(input_shape=TAB, num_classes=2)
    model.fit(X, y)  # não deve levantar (bug do probability duplicado)
    p = model.predict_proba(X[:3])
    assert p.shape == (3, 2)
    assert np.allclose(p.sum(axis=1), 1.0, atol=1e-3)


def test_random_forest_factory_fit_and_proba():
    from app.domain.models.architectures.random_forest import (
        create_random_forest_model,
    )

    X, y = _data()
    model = create_random_forest_model(input_shape=TAB, num_classes=2, n_estimators=20)
    model.fit(X, y)
    p = model.predict_proba(X[:3])
    assert p.shape == (3, 2)
    assert np.allclose(p.sum(axis=1), 1.0, atol=1e-3)


def _make(arch: str):
    if arch == "SVM":
        from app.domain.models.architectures.svm import create_svm_model as make
    else:
        from app.domain.models.architectures.random_forest import (
            create_random_forest_model as make,
        )
    return make(input_shape=(120,), num_classes=2)


@pytest.mark.parametrize("arch", ["SVM", "RandomForest"])
def test_get_params_does_not_poison_kwargs(arch):
    """Regressão: `get_params()` retornava `self.kwargs` por REFERÊNCIA e o
    `params.update({...})` da subclasse o envenenava com chaves como `kernel`,
    quebrando o próximo `_create_pipeline()` com "got multiple values for
    keyword argument 'kernel'". Chamamos get_params() ANTES de (re)montar o
    pipeline e treinar — a ordem que dispara o bug — e nada deve quebrar.
    Esta é exatamente a sequência da célula de inspeção dos notebooks 13/14.
    """
    model = _make(arch)
    snapshot = dict(model.kwargs)            # estado interno limpo
    _ = model.get_params()                   # antes: envenenava self.kwargs
    assert model.kwargs == snapshot, "get_params() mutou self.kwargs"

    pipe = model._create_pipeline()          # antes: levantava TypeError aqui
    assert pipe.steps[0][0] == "scaler"

    X = np.random.RandomState(0).standard_normal((30, 120)).astype("float32")
    y = np.r_[np.zeros(15), np.ones(15)].astype(int)
    model.fit(X, y)                          # treino completo após get_params()
    assert model.predict_proba(X[:2]).shape == (2, 2)
