"""Paridade treino↔inferência do front-end do benchmark.

Garante que ``app/domain/features/benchmark_frontend`` (usado pela inferência
via ``FeaturePreparer``) e ``benchmarks/data.py`` (usado no treino) produzem
EXATAMENTE os mesmos tensores — a propriedade que dá validade às métricas do
benchmark quando o modelo é servido pelo app.
"""

from __future__ import annotations

import numpy as np
import pytest

from app.domain.features import benchmark_frontend as bf


@pytest.fixture(scope="module")
def raw_batch() -> np.ndarray:
    rng = np.random.default_rng(42)
    return (rng.normal(size=(3, 80000)) * 0.25).astype("float32")


class TestParityWithBenchmarkData:
    """As funções delegadas de benchmarks/data.py devem ser bit-idênticas."""

    def test_raw_parity(self, raw_batch):
        from benchmarks import data as bd

        ref = bd._to_raw_audio(raw_batch, {"target_sequence_length": 16000})
        new = bf.raw_audio_batch(raw_batch, target_len=16000)
        np.testing.assert_array_equal(ref, new)
        # single == batch[i]
        np.testing.assert_array_equal(
            bf.raw_audio_single(raw_batch[0], 16000), ref[0]
        )

    def test_logmel_parity(self, raw_batch):
        pytest.importorskip("librosa")
        from benchmarks import data as bd

        req = {"sample_rate": 16000, "feature_dim": 80, "min_sequence_length": 100}
        ref = bd._raw_audio_to_logmel(raw_batch, req)
        new = bf.log_mel_batch(raw_batch, 16000, 80, 100)
        np.testing.assert_array_equal(ref, new)
        assert ref.shape == (3, 100, 80)
        np.testing.assert_array_equal(bf.log_mel_single(raw_batch[0]), ref[0])

    def test_tabular_parity_and_contract(self, raw_batch):
        pytest.importorskip("librosa")
        from app.core.xai.tabular import N_FEATURES
        from benchmarks import data as bd

        ref = bd._to_tabular_features(raw_batch)
        new = bf.tabular_features_batch(raw_batch)
        np.testing.assert_array_equal(ref, new)
        assert ref.shape == (3, N_FEATURES)
        np.testing.assert_array_equal(
            bf.tabular_features_single(raw_batch[0]), ref[0]
        )

    def test_short_clip_is_tiled_not_padded(self):
        short = np.ones((1, 4000), dtype="float32")
        fitted = bf.fit_length_tile(short, 16000)
        assert fitted.shape == (1, 16000)
        # tile repete o sinal (nada de zeros)
        assert float(np.abs(fitted).min()) > 0.0


class TestPrepareSingleDispatch:
    def test_raw_frontend(self):
        y = np.random.default_rng(0).normal(size=24000).astype("float32")
        out = bf.prepare_single(y, bf.FRONTEND_RAW, target_sequence_length=16000)
        assert out.shape == (16000, 1)
        # z-score por amostra
        assert abs(float(out.mean())) < 1e-4
        assert abs(float(out.std()) - 1.0) < 1e-3

    def test_logmel_frontend_with_channel(self):
        pytest.importorskip("librosa")
        y = np.random.default_rng(1).normal(size=80000).astype("float32")
        out = bf.prepare_single(
            y, bf.FRONTEND_LOGMEL, feature_dim=80, time_steps=100,
            add_channel_dim=True,
        )
        assert out.shape == (100, 80, 1)

    def test_unknown_frontend_raises(self):
        with pytest.raises(ValueError, match="feature_frontend"):
            bf.prepare_single(np.zeros(100, dtype="float32"), "outro")


class TestFeaturePreparerBenchmarkPath:
    """O FeaturePreparer roteia contratos benchmark_* para a fonte única."""

    @staticmethod
    def _make_audio(n=80000):
        from app.core.interfaces.audio import AudioData

        rng = np.random.default_rng(7)
        samples = (rng.normal(size=n) * 0.2).astype("float32")
        return AudioData(
            samples=samples, sample_rate=16000,
            duration=n / 16000.0, channels=1,
        ), samples

    @staticmethod
    def _make_preparer():
        from unittest.mock import MagicMock

        from app.domain.services.detection.feature_preparer import (
            FeaturePreparer,
        )

        return FeaturePreparer(feature_service=MagicMock())

    def test_logmel_contract_matches_shared_frontend(self):
        pytest.importorskip("librosa")
        from unittest.mock import MagicMock

        audio, samples = self._make_audio()
        preparer = self._make_preparer()
        model_info = MagicMock()
        model_info.input_shape = (100, 80)
        model_info.input_contract = {
            "feature_frontend": bf.FRONTEND_LOGMEL,
            "sample_rate": 16000,
            "feature_dim": 80,
            "time_steps": 100,
            "source_samples": 80000,
        }
        result = preparer.prepare_input(audio, model_info, arch_info=None)
        assert result["status"] == "ok"
        np.testing.assert_array_equal(
            result["features"], bf.log_mel_single(samples)
        )
        assert result["metadata"]["feature_frontend"] == bf.FRONTEND_LOGMEL

    def test_tabular_contract_strict_dimension(self):
        pytest.importorskip("librosa")
        from unittest.mock import MagicMock

        audio, _ = self._make_audio()
        preparer = self._make_preparer()
        model_info = MagicMock()
        model_info.input_shape = (64,)  # dimensão ERRADA de propósito
        model_info.input_contract = {
            "feature_frontend": bf.FRONTEND_TABULAR,
            "sample_rate": 16000,
        }
        result = preparer.prepare_input(audio, model_info, arch_info=None)
        assert result["status"] == "error"
        assert "63" in result["error"] and "64" in result["error"]
