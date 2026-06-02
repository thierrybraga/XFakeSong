"""Regressão: modelos clássicos (SVM/RF) construídos pela factory devem TREINAR.

Pega o bug onde `SVMModel` repassava `probability` ao super (→ self.kwargs) e
`_create_pipeline` o passava de novo ao SVC → "got multiple values for keyword
argument 'probability'" no `.fit()`. Checks anteriores só CONSTRUÍam o modelo
(sem treinar), então o erro de fit nunca era exercido.
"""

from __future__ import annotations

import numpy as np

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
