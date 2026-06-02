"""Testes P2: RawGAT-ST fiel ao paper + back-end SSL→AASIST."""

from __future__ import annotations

import numpy as np
import pytest


# ───────────────── RawGAT-ST (paper-faithful, raw audio) ─────────────────

def test_rawgatst_is_raw_audio_in_registry():
    from app.domain.models.architectures.registry import architecture_registry as reg

    info = reg.get_architecture("RawGAT-ST")
    assert info.input_requirements.get("input_type") == "raw_audio"


def test_rawgatst_builds_and_trains_no_nan():
    import tensorflow as tf

    from app.domain.models.architectures import rawgat_st

    m = rawgat_st.create_model(
        input_shape=(48000, 1), num_classes=2, architecture="rawgat_st"
    )
    assert m.output_shape == (None, 2)

    x = (0.3 * np.random.randn(4, 48000, 1)).astype("float32")
    y = m.predict(x, verbose=0)
    assert np.all(np.isfinite(y))
    assert np.allclose(y.sum(axis=1), 1.0, atol=1e-3)  # softmax 2-u

    h = m.fit(x, np.array([0, 1, 0, 1]), epochs=1, batch_size=2, verbose=0)
    assert np.isfinite(h.history["loss"][0])
    assert all(
        bool(tf.reduce_all(tf.math.is_finite(w)))
        for w in m.weights if "float" in str(w.dtype)
    )


# ───────────────── SSL → AASIST back-end ─────────────────

@pytest.mark.parametrize("arch", ["wavlm", "wavlm_aasist"])
def test_wavlm_backends_build_and_train(arch):
    import tensorflow as tf

    from app.domain.models.architectures import wavlm

    m = wavlm.create_model(input_shape=(48000, 1), num_classes=2, architecture=arch)
    x = (0.3 * np.random.randn(2, 48000, 1)).astype("float32")
    y = m.predict(x, verbose=0)
    assert np.all(np.isfinite(y)) and np.allclose(y.sum(1), 1.0, atol=1e-3)
    h = m.fit(x, np.array([0, 1]), epochs=1, batch_size=2, verbose=0)
    assert np.isfinite(h.history["loss"][0])
    assert all(
        bool(tf.reduce_all(tf.math.is_finite(w)))
        for w in m.weights if "float" in str(w.dtype)
    )


@pytest.mark.parametrize("arch", ["hubert", "hubert_aasist"])
def test_hubert_backends_build_and_train(arch):
    import tensorflow as tf

    from app.domain.models.architectures import hubert

    m = hubert.create_model(input_shape=(48000, 1), num_classes=2, architecture=arch)
    x = (0.3 * np.random.randn(2, 48000, 1)).astype("float32")
    y = m.predict(x, verbose=0)
    assert np.all(np.isfinite(y)) and np.allclose(y.sum(1), 1.0, atol=1e-3)
    h = m.fit(x, np.array([0, 1]), epochs=1, batch_size=2, verbose=0)
    assert np.isfinite(h.history["loss"][0])
    assert all(
        bool(tf.reduce_all(tf.math.is_finite(w)))
        for w in m.weights if "float" in str(w.dtype)
    )


def test_ssl_aasist_backend_helper_static_shape():
    """O back-end deve produzir um vetor (B, D) com D estático (não None)."""
    import tensorflow as tf

    from app.domain.models.architectures.ssl_utils import build_ssl_aasist_backend

    inp = tf.keras.Input(shape=(None, 768))  # T dinâmico, como saída SSL
    out = build_ssl_aasist_backend(inp, dropout_rate=0.2, name="t")
    # Keras 3: KerasTensor.shape é uma tupla
    assert len(out.shape) == 2
    assert out.shape[-1] is not None  # dimensão de features estática
