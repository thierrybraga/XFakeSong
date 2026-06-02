"""Testes das melhorias P0: front-end LFCC + RawBoost.

- Paridade train↔inference do LFCC (mesmo núcleo `lfcc_from_waveform`).
- Shape preservado (T, 80, 1) e troca de front-end (lfcc vs logmel).
- Fallback logmel para modelos legados (sem `feature_frontend`).
- RawBoost: shape preservado, saída finita, identidade em algo=0.
"""

from __future__ import annotations

import numpy as np
import pytest

SR = 16000
T = SR * 3  # 3s → 375 frames com n_fft=512/hop=128/pad_end


def _tone(freq: float = 150.0) -> np.ndarray:
    return (0.3 * np.sin(2 * np.pi * freq * np.arange(T) / SR)).astype("float32")


def test_lfcc_train_inference_parity():
    """O núcleo de treino e o wrapper de inferência devem casar bit-a-bit."""
    import tensorflow as tf

    from app.domain.services.detection.audio_preprocessing import (
        audio_to_lfcc,
        lfcc_from_waveform,
    )

    y = _tone()
    train = lfcc_from_waveform(
        tf.constant(y[None, :]), sample_rate=SR, n_fft=512, hop_length=128,
        n_filters=80, n_lfcc=80,
    ).numpy()[0]
    infer = audio_to_lfcc(
        y, sample_rate=SR, n_fft=512, hop_length=128, n_filters=80, n_lfcc=80,
        add_channel_dim=False,
    )
    assert train.shape == (375, 80)
    assert float(np.max(np.abs(train - infer))) < 1e-4


def test_prepare_audio_lfcc_shape_and_switch():
    """LFCC preserva (375,80,1) e difere do log-mel."""
    from app.domain.services.detection.audio_preprocessing import (
        prepare_audio_for_model,
    )

    y = _tone()
    lfcc = prepare_audio_for_model(
        y, input_type="spectrogram", input_shape=(375, 80, 1),
        sample_rate=SR, feature_frontend="lfcc", n_lfcc=80,
    )
    logmel = prepare_audio_for_model(
        y, input_type="spectrogram", input_shape=(375, 80, 1),
        sample_rate=SR, feature_frontend="logmel",
    )
    assert lfcc.shape == (375, 80, 1)
    assert logmel.shape == (375, 80, 1)
    assert np.all(np.isfinite(lfcc))
    # de fato é um front-end diferente
    assert float(np.mean(np.abs(lfcc - logmel))) > 0.01


def test_prepare_audio_default_frontend_is_logmel():
    """Sem `feature_frontend` (modelo legado), o default deve ser log-mel.

    Verifica que a saída default é IDÊNTICA ao log-mel explícito (mesmo caminho
    de normalização) e DIFERENTE do LFCC.
    """
    from app.domain.services.detection.audio_preprocessing import (
        prepare_audio_for_model,
    )

    y = _tone()
    default = prepare_audio_for_model(
        y, input_type="spectrogram", input_shape=(375, 80, 1), sample_rate=SR,
    )
    explicit_logmel = prepare_audio_for_model(
        y, input_type="spectrogram", input_shape=(375, 80, 1), sample_rate=SR,
        feature_frontend="logmel",
    )
    explicit_lfcc = prepare_audio_for_model(
        y, input_type="spectrogram", input_shape=(375, 80, 1), sample_rate=SR,
        feature_frontend="lfcc", n_lfcc=80,
    )
    assert default.shape == (375, 80, 1)
    # default == logmel explícito (default É log-mel)
    assert float(np.max(np.abs(default - explicit_logmel))) < 1e-6
    # e difere do LFCC
    assert float(np.mean(np.abs(default - explicit_lfcc))) > 0.01


@pytest.mark.parametrize("algo", [1, 2, 3, 4, 5])
def test_rawboost_finite_and_shape(algo):
    import tensorflow as tf

    from app.domain.models.training.rawboost import rawboost_tf

    x = tf.random.normal((4, T)) * 0.3
    y = rawboost_tf(x, sr=SR, algo=algo, p=1.0)
    assert tuple(y.shape) == (4, T)
    assert bool(tf.reduce_all(tf.math.is_finite(y)))


def test_rawboost_identity_and_active():
    import tensorflow as tf

    from app.domain.models.training.rawboost import rawboost_tf

    x = tf.random.normal((2, T)) * 0.3
    # algo=0 → identidade
    assert bool(tf.reduce_all(tf.equal(rawboost_tf(x, algo=0, p=1.0), x)))
    # algo=4 com p=1 → de fato altera o sinal
    y = rawboost_tf(x, algo=4, p=1.0)
    assert float(tf.reduce_mean(tf.abs(y - x))) > 0.0
