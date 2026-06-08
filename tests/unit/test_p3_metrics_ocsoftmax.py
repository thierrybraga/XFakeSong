"""Testes P3: min-tDCF, OC-Softmax, refinos de arquitetura e calibração."""

from __future__ import annotations

import numpy as np
import pytest


# ───────────────────────── min t-DCF ─────────────────────────

def _metrics():
    from app.domain.models.training import metrics as M
    cls = next(c for c in dir(M) if "Metric" in c or "Calc" in c)
    return getattr(M, cls)()


def test_min_tdcf_discriminative_beats_random():
    mc = _metrics()
    y = np.array([0, 0, 0, 0, 1, 1, 1, 1])  # 0=bonafide, 1=spoof
    good = np.array([0.05, 0.1, 0.2, 0.15, 0.9, 0.85, 0.95, 0.8])  # p_fake
    rnd = np.full(8, 0.5)
    t_good, _ = mc.calculate_min_tdcf(y, good)
    t_rand, _ = mc.calculate_min_tdcf(y, rnd)
    assert 0.0 <= t_good <= 1.5
    assert t_good < t_rand
    assert t_good < 0.2  # quase perfeito separa → ~0


def test_min_tdcf_single_class_safe():
    mc = _metrics()
    y = np.zeros(5, dtype=int)  # só bonafide
    t, thr = mc.calculate_min_tdcf(y, np.random.rand(5))
    assert t == 1.0  # sem spoof → custo máximo (degradação graciosa)


def test_all_metrics_includes_eer_and_tdcf():
    mc = _metrics()
    y = np.array([0, 0, 1, 1])
    proba = np.array([[0.9, 0.1], [0.8, 0.2], [0.2, 0.8], [0.1, 0.9]])
    m = mc.calculate_all_metrics(y, proba.argmax(1), proba)
    assert "eer" in m and "min_tdcf" in m


# ───────────────────────── OC-Softmax ─────────────────────────

def test_ocsoftmax_layer_shape_and_range():
    import tensorflow as tf

    from app.domain.models.architectures.layers import OCSoftmaxLayer

    emb = tf.random.normal((8, 128))
    s = OCSoftmaxLayer()(emb)
    assert tuple(s.shape) == (8, 1)
    assert bool(tf.reduce_all(s >= -1.0001)) and bool(tf.reduce_all(s <= 1.0001))


def test_ocsoftmax_loss_finite_and_grad():
    import tensorflow as tf

    from app.domain.models.architectures.layers import OCSoftmaxLayer, oc_softmax_loss

    layer = OCSoftmaxLayer()
    emb = tf.Variable(tf.random.normal((6, 64)))
    y = tf.constant([0, 0, 0, 1, 1, 1])
    loss_fn = oc_softmax_loss(m0=0.9, m1=0.2, alpha=20.0)
    with tf.GradientTape() as tape:
        s = layer(emb)
        loss = loss_fn(y, s)
    grad = tape.gradient(loss, emb)
    assert np.isfinite(float(loss))
    assert grad is not None and bool(tf.reduce_all(tf.math.is_finite(grad)))


# ───────────────────────── refinos de arquitetura ─────────────────────────

def test_aasist_has_six_residual_blocks():
    from app.domain.models.architectures import aasist

    m = aasist.create_model(input_shape=(48000, 1), num_classes=2, architecture="aasist")
    n = sum(1 for lyr in m.layers if "res_block" in lyr.name)
    assert n == 6  # paridade com o paper (RawNet2 encoder)


# ───────────────────────── calibração opt-in ─────────────────────────

@pytest.mark.parametrize("calibrate", [False, True])
def test_rf_calibrate_flag_fit_proba(calibrate):
    from app.domain.models.architectures.random_forest import (
        create_random_forest_model,
    )

    rng = np.random.default_rng(0)
    X = rng.standard_normal((60, 128)).astype("float32")
    y = rng.integers(0, 2, 60)
    model = create_random_forest_model(
        input_shape=(128,), num_classes=2, n_estimators=20, calibrate=calibrate
    )
    model.fit(X, y)
    p = model.predict_proba(X[:3])
    assert p.shape == (3, 2) and np.allclose(p.sum(1), 1.0, atol=1e-3)
