"""Testes P1: SpecAugment + fine-tuning parcial dos SSL (WavLM/HuBERT)."""

from __future__ import annotations

import numpy as np
import pytest


# ───────────────────────── SpecAugment ─────────────────────────

@pytest.mark.parametrize("shape", [(8, 375, 80), (8, 375, 80, 1)])
def test_spec_augment_shape_and_finite(shape):
    import tensorflow as tf

    from app.domain.models.training.spec_augment import spec_augment_tf

    x = tf.random.normal(shape)
    y = spec_augment_tf(x, p=1.0)
    assert tuple(y.shape) == shape
    assert bool(tf.reduce_all(tf.math.is_finite(y)))


def test_spec_augment_identity_and_active():
    import tensorflow as tf

    from app.domain.models.training.spec_augment import spec_augment_tf

    x = tf.random.normal((4, 375, 80, 1))
    # p=0 → identidade
    assert bool(tf.reduce_all(tf.equal(spec_augment_tf(x, p=0.0), x)))
    # p=1 → mascara (gera zeros que não existiam)
    y = spec_augment_tf(x, p=1.0)
    zeros_before = int(tf.reduce_sum(tf.cast(tf.equal(x, 0.0), tf.int32)))
    zeros_after = int(tf.reduce_sum(tf.cast(tf.equal(y, 0.0), tf.int32)))
    assert zeros_after > zeros_before


# ───────────────── SSL: helper de trainability ─────────────────

class _FakeLayer:
    def __init__(self):
        self.trainable = True


class _FakeEncoder:
    def __init__(self, n):
        self.layers = [_FakeLayer() for _ in range(n)]


class _FakeBackbone:
    """Emula o setter recursivo de `trainable` do Keras."""

    def __init__(self, n):
        self.wavlm = type("X", (), {})()
        self.wavlm.encoder = _FakeEncoder(n)
        self._trainable = True

    @property
    def trainable(self):
        return self._trainable

    @trainable.setter
    def trainable(self, v):
        self._trainable = v
        for lyr in self.wavlm.encoder.layers:
            lyr.trainable = v


def test_ssl_partial_unfreeze_last_n():
    from app.domain.models.architectures.ssl_utils import (
        find_encoder_layers,
        set_ssl_backbone_trainability,
    )

    bb = _FakeBackbone(12)
    assert len(find_encoder_layers(bb)) == 12

    mode = set_ssl_backbone_trainability(bb, freeze_weights=True, n_trainable_layers=3)
    assert "3/12" in mode
    flags = [lyr.trainable for lyr in bb.wavlm.encoder.layers]
    assert flags[:9] == [False] * 9  # primeiras 9 congeladas
    assert flags[9:] == [True] * 3   # últimas 3 treináveis


def test_ssl_freeze_all_and_full_trainable():
    from app.domain.models.architectures.ssl_utils import (
        set_ssl_backbone_trainability,
    )

    bb = _FakeBackbone(12)
    set_ssl_backbone_trainability(bb, freeze_weights=True, n_trainable_layers=0)
    assert bb.trainable is False

    bb2 = _FakeBackbone(12)
    set_ssl_backbone_trainability(bb2, freeze_weights=False, n_trainable_layers=0)
    assert bb2.trainable is True


def test_ssl_unrecognized_structure_is_safe():
    from app.domain.models.architectures.ssl_utils import (
        find_encoder_layers,
        set_ssl_backbone_trainability,
    )

    class _Opaque:
        trainable = True

    bb = _Opaque()
    assert find_encoder_layers(bb) is None
    # não deve levantar exceção
    mode = set_ssl_backbone_trainability(bb, freeze_weights=False, n_trainable_layers=3)
    assert isinstance(mode, str)


# ───────────── SSL: build + config (caminho CNN fallback) ─────────────
# Sem `transformers` instalado, WavLM/HuBERT usam o extrator CNN; o objetivo
# aqui é garantir que os novos parâmetros fluem e o modelo constrói/treina.

def test_wavlm_build_with_finetune_param():
    from app.domain.models.architectures import wavlm

    model = wavlm.create_model(
        input_shape=(48000, 1), num_classes=2, n_trainable_layers=3
    )
    assert model is not None
    assert len(model.trainable_weights) > 0


def test_hubert_build_with_finetune_param():
    from app.domain.models.architectures import hubert

    model = hubert.create_model(
        input_shape=(48000, 1), num_classes=2, n_trainable_layers=3
    )
    assert model is not None
    assert len(model.trainable_weights) > 0
