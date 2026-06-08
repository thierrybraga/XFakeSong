"""Cobertura de `app/domain/services/detection/utils.py` (numpy + tf.config).

`pad_or_truncate`/`prepare_batch_for_model` são numpy puro; `get_available_devices`
consulta `tf.config` (mockado para o ramo de GPU).
"""

import numpy as np

import app.domain.services.detection.utils as det_utils
from app.domain.services.detection.utils import (
    get_available_devices,
    pad_or_truncate,
    prepare_batch_for_model,
)


def test_pad_or_truncate_pads_1d():
    out = pad_or_truncate(np.array([1.0, 2, 3]), 5)
    assert out.shape == (5,)
    assert np.allclose(out[3:], 0)


def test_pad_or_truncate_truncates_1d():
    out = pad_or_truncate(np.array([1.0, 2, 3, 4, 5]), 3)
    assert out.tolist() == [1, 2, 3]


def test_pad_or_truncate_equal_returns_same_values():
    out = pad_or_truncate(np.array([1.0, 2, 3]), 3)
    assert out.tolist() == [1, 2, 3]


def test_pad_or_truncate_along_axis1():
    out = pad_or_truncate(np.ones((2, 3)), 5, axis=1)
    assert out.shape == (2, 5)


def test_prepare_batch_empty_returns_empty():
    out = prepare_batch_for_model([])
    assert out.size == 0


def test_prepare_batch_target_shape_none_aligns_to_max_first_dim():
    feats = [np.ones((3, 4)), np.ones((5, 4))]
    out = prepare_batch_for_model(feats)
    assert out.shape == (2, 5, 4)


def test_prepare_batch_with_explicit_target_shape():
    feats = [np.ones((10, 4)), np.ones((2, 4))]
    out = prepare_batch_for_model(feats, target_shape=(6, 4))
    assert out.shape == (2, 6, 4)


def test_get_available_devices_always_has_cpu():
    assert "CPU" in get_available_devices()


def test_get_available_devices_lists_gpus(monkeypatch):
    class _Dev:  # stub de device físico
        pass

    monkeypatch.setattr(
        det_utils.tf.config,
        "list_physical_devices",
        lambda kind: [_Dev(), _Dev()] if kind == "GPU" else [],
    )
    assert get_available_devices() == ["CPU", "GPU:0", "GPU:1"]


def test_get_available_devices_handles_tf_error(monkeypatch):
    def _boom(_kind):
        raise RuntimeError("tf indisponível")

    monkeypatch.setattr(det_utils.tf.config, "list_physical_devices", _boom)
    # degrada graciosamente para apenas CPU
    assert get_available_devices() == ["CPU"]
