"""Regressão: rede de segurança de reamostragem no FeaturePreparer.

Garante que áudio em taxa != treino (ex.: 44.1 kHz de upload) seja reamostrado
para a taxa do front-end (16 kHz) ANTES do STFT/SincConv — caso contrário o
pré-processamento sairia silenciosamente errado.
"""

from __future__ import annotations

import numpy as np


def _audio(sr, seconds=3.0, freq=150.0):
    from app.core.interfaces.audio import AudioData

    n = int(sr * seconds)
    y = (0.3 * np.sin(2 * np.pi * freq * np.arange(n) / sr)).astype("float32")
    return AudioData(samples=y, sample_rate=sr, duration=float(seconds))


def test_ensure_sample_rate_resamples():
    from app.domain.services.detection.feature_preparer import FeaturePreparer

    ad = _audio(44100)  # 132300 amostras
    out = FeaturePreparer._ensure_sample_rate(ad, 16000)
    assert out.sample_rate == 16000
    # 3 s @ 16 kHz ≈ 48000 amostras (tolerância do resampler)
    assert abs(len(out.samples) - 48000) <= 4
    assert abs(out.duration - 3.0) < 0.05
    assert np.all(np.isfinite(out.samples))


def test_ensure_sample_rate_idempotent():
    from app.domain.services.detection.feature_preparer import FeaturePreparer

    ad = _audio(16000)
    out = FeaturePreparer._ensure_sample_rate(ad, 16000)
    assert out is ad  # mesma instância — sem trabalho desnecessário


def test_prepare_input_resamples_for_raw_model():
    """Áudio 44.1 kHz → modelo raw deve receber exatamente (48000, 1)."""
    from app.core.interfaces.audio import AudioData  # noqa: F401
    from app.domain.models.architectures import rawnet2
    from app.domain.services.detection.feature_preparer import FeaturePreparer
    from app.domain.services.detection.model_loader import ModelInfo
    from app.domain.services.feature_extraction_service import (
        AudioFeatureExtractionService,
    )

    model = rawnet2.create_model(input_shape=(48000, 1), num_classes=2)
    mi = ModelInfo(
        name="rn2", architecture="rawnet2", model=model, scaler=None,
        input_shape=(48000, 1), model_type="tensorflow",
        input_contract={"input_type": "raw_audio", "sample_rate": 16000},
    )
    fp = FeaturePreparer(AudioFeatureExtractionService())
    prep = fp.prepare_input(_audio(44100), mi, None)
    assert prep["status"] == "ok"
    assert tuple(prep["features"].shape) == (48000, 1)


def test_prepare_input_resamples_for_spectrogram_model():
    """Áudio 44.1 kHz → modelo de espectrograma deve receber (375, 80, 1)."""
    from app.domain.models.architectures import multiscale_cnn
    from app.domain.services.detection.feature_preparer import FeaturePreparer
    from app.domain.services.detection.model_loader import ModelInfo
    from app.domain.services.feature_extraction_service import (
        AudioFeatureExtractionService,
    )

    model = multiscale_cnn.create_model(input_shape=(375, 80, 1), num_classes=2)
    mi = ModelInfo(
        name="msc", architecture="multiscale_cnn", model=model, scaler=None,
        input_shape=(375, 80, 1), model_type="tensorflow",
        input_contract={
            "input_type": "spectrogram", "sample_rate": 16000,
            "n_fft": 512, "hop_length": 128, "n_mels": 80,
            "feature_frontend": "lfcc", "n_lfcc": 80,
        },
    )
    fp = FeaturePreparer(AudioFeatureExtractionService())
    prep = fp.prepare_input(_audio(44100), mi, None)
    assert prep["status"] == "ok"
    assert tuple(prep["features"].shape) == (375, 80, 1)
