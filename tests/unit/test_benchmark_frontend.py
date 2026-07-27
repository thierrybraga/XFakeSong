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
    """Clipe MAIS LONGO que a janela canônica — exercita as políticas de corte."""
    rng = np.random.default_rng(42)
    return (rng.normal(size=(3, 80000)) * 0.25).astype("float32")


@pytest.fixture(scope="module")
def canonical_batch() -> np.ndarray:
    """Clipe exatamente na janela-fonte canônica.

    As versões single-sample de log-mel e tabular ajustam o clipe a
    `DEFAULT_SOURCE_SAMPLES` antes de extrair, para que o hop dinâmico e as
    estatísticas sejam os mesmos do treino. A paridade com `benchmarks/data.py`
    — que recebe o lote já na janela — só é comparável nessa mesma duração.
    """
    rng = np.random.default_rng(42)
    return (rng.normal(size=(3, bf.DEFAULT_SOURCE_SAMPLES)) * 0.25).astype("float32")


class TestParityWithBenchmarkData:
    """As funções delegadas de benchmarks/data.py devem ser bit-idênticas."""

    def test_raw_parity(self, raw_batch):
        from benchmarks import data as bd

        ref = bd._to_raw_audio(raw_batch, {"target_sequence_length": 16000})
        new = bf.raw_audio_batch(raw_batch, target_len=16000)
        np.testing.assert_array_equal(ref, new)
        # single == batch[i]
        np.testing.assert_array_equal(bf.raw_audio_single(raw_batch[0], 16000), ref[0])

    def test_logmel_parity(self, canonical_batch):
        pytest.importorskip("librosa")
        from benchmarks import data as bd

        req = {"sample_rate": 16000, "feature_dim": 80, "min_sequence_length": 100}
        ref = bd._raw_audio_to_logmel(canonical_batch, req)
        new = bf.log_mel_batch(canonical_batch, 16000, 80, 100)
        np.testing.assert_array_equal(ref, new)
        assert ref.shape == (3, 100, 80)
        np.testing.assert_array_equal(bf.log_mel_single(canonical_batch[0]), ref[0])

    def test_tabular_parity_and_contract(self, canonical_batch):
        pytest.importorskip("librosa")
        from app.domain.xai.tabular import N_FEATURES
        from benchmarks import data as bd

        ref = bd._to_tabular_features(canonical_batch)
        new = bf.tabular_features_batch(canonical_batch)
        np.testing.assert_array_equal(ref, new)
        assert ref.shape == (3, N_FEATURES)
        np.testing.assert_array_equal(
            bf.tabular_features_single(canonical_batch[0]), ref[0]
        )

    def test_short_clip_is_tiled_not_padded(self):
        short = np.ones((1, 4000), dtype="float32")
        fitted = bf.fit_length_tile(short, 16000)
        assert fitted.shape == (1, 16000)
        # tile repete o sinal (nada de zeros)
        assert float(np.abs(fitted).min()) > 0.0


class TestRawCropPolicy:
    def test_default_window_matches_paper_protocol(self):
        assert bf.DEFAULT_RAW_TARGET == 48000

    def test_random_crop_is_seeded_per_sample(self, raw_batch):
        first = bf.raw_audio_batch(
            raw_batch, target_len=48000, crop_strategy="random", seed=7
        )
        second = bf.raw_audio_batch(
            raw_batch, target_len=48000, crop_strategy="random", seed=7
        )
        center = bf.raw_audio_batch(raw_batch, target_len=48000)
        np.testing.assert_array_equal(first, second)
        assert not np.array_equal(first, center)

    def test_multicrop_start_center_end(self, raw_batch):
        crops = bf.raw_audio_multicrop_batch(raw_batch, target_len=48000, num_crops=3)
        assert crops.shape == (3, 3, 48000, 1)
        assert np.all(np.isfinite(crops))
        assert np.allclose(crops.mean(axis=(2, 3)), 0.0, atol=1e-4)
        assert np.allclose(crops.std(axis=(2, 3)), 1.0, atol=1e-3)


class TestPrepareSingleDispatch:
    def test_raw_frontend(self):
        y = np.random.default_rng(0).normal(size=24000).astype("float32")
        out = bf.prepare_single(y, bf.FRONTEND_RAW, target_sequence_length=16000)
        assert out.shape == (16000, 1)
        # z-score por amostra
        assert abs(float(out.mean())) < 1e-4
        assert abs(float(out.std()) - 1.0) < 1e-3

    def test_raw_frontend_multicrop(self):
        y = np.random.default_rng(3).normal(size=80000).astype("float32")
        out = bf.prepare_single(
            y,
            bf.FRONTEND_RAW,
            target_sequence_length=48000,
            raw_num_crops=3,
        )
        assert out.shape == (3, 48000, 1)

    def test_logmel_frontend_with_channel(self):
        pytest.importorskip("librosa")
        y = np.random.default_rng(1).normal(size=80000).astype("float32")
        out = bf.prepare_single(
            y,
            bf.FRONTEND_LOGMEL,
            feature_dim=80,
            time_steps=100,
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
        from app.core.contracts.audio import AudioData

        rng = np.random.default_rng(7)
        samples = (rng.normal(size=n) * 0.2).astype("float32")
        return (
            AudioData(
                samples=samples,
                sample_rate=16000,
                duration=n / 16000.0,
                channels=1,
            ),
            samples,
        )

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

        from app.utils.silero_vad import apply_agc

        audio, samples = self._make_audio()
        preparer = self._make_preparer()
        model_info = MagicMock()
        model_info.input_shape = (100, 80)
        model_info.input_contract = {
            "feature_frontend": bf.FRONTEND_LOGMEL,
            "sample_rate": 16000,
            "feature_dim": 80,
            "time_steps": 100,
            "source_samples": 48000,
        }
        result = preparer.prepare_input(audio, model_info, arch_info=None)
        assert result["status"] == "ok"
        # BUG FIX (Limitação (viii) do TCC): o FeaturePreparer agora aplica a
        # mesma AGC por RMS/LUFS da construção do corpus (Eq. 5) antes de
        # delegar ao front-end compartilhado — não mais os `samples` crus.
        # Numericamente inconsequente aqui (dB ref-max + z-score por amostra
        # são invariantes a reescala linear, a menos de ruído de ponto
        # flutuante ~1e-6 no termo epsilon do log), mas a referência precisa
        # espelhar o passo de AGC para a igualdade se manter exata.
        np.testing.assert_array_equal(
            result["features"], bf.log_mel_single(apply_agc(samples))
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
