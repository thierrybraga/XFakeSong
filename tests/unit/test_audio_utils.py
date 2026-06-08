"""Cobertura de `app/core/utils/audio_utils.py` (numpy puro)."""

import logging

import numpy as np

from app.core.utils.audio_utils import (
    normalize_audio,
    pad_or_truncate,
    preprocess_legacy,
)


def test_normalize_audio_scales_to_target_db():
    sig = np.sin(np.linspace(0, 10, 1000)).astype(np.float32)
    out = normalize_audio(sig, target_db=-20.0)
    rms = float(np.sqrt(np.mean(out ** 2)))
    expected = 10 ** (-20.0 / 20)  # 0.1
    assert np.isclose(rms, expected, rtol=0.05)


def test_normalize_audio_silence_returned_unchanged():
    sig = np.zeros(128, dtype=np.float32)
    out = normalize_audio(sig)
    assert np.array_equal(out, sig)


def test_pad_or_truncate_pads_shorter():
    out = pad_or_truncate(np.array([1.0, 2, 3]), 5)
    assert out.shape == (5,)
    assert out[-1] == 0


def test_pad_or_truncate_truncates_longer():
    out = pad_or_truncate(np.arange(10.0), 4)
    assert out.tolist() == [0, 1, 2, 3]


def test_preprocess_legacy_is_passthrough_and_warns(caplog):
    x = np.array([1.0, 2, 3])
    with caplog.at_level(logging.WARNING):
        out = preprocess_legacy(x)
    assert np.array_equal(out, x)
    assert any("DEPRECATED" in r.message for r in caplog.records)
