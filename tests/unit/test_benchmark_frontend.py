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
        """O benchmark prepara o v2; o v1 segue disponível para artefatos antigos."""
        pytest.importorskip("librosa")
        from app.domain.xai.tabular import N_FEATURES, N_FEATURES_V2
        from benchmarks import data as bd

        ref = bd._to_tabular_features(canonical_batch)
        new = bf.tabular_features_v2_batch(canonical_batch)
        np.testing.assert_array_equal(ref, new)
        assert ref.shape == (3, N_FEATURES_V2)
        np.testing.assert_array_equal(
            bf.tabular_features_v2_single(canonical_batch[0]), ref[0]
        )

        v1 = bf.tabular_features_batch(canonical_batch)
        assert v1.shape == (3, N_FEATURES)
        np.testing.assert_array_equal(
            bf.tabular_features_single(canonical_batch[0]), v1[0]
        )

    def test_v2_e_superset_estrito_do_v1(self, canonical_batch):
        """As 63 primeiras colunas do v2 SÃO o v1, na mesma ordem.

        Sendo superset, qualquer diferença de desempenho entre os dois é
        atribuível ao bloco LFCC ou à dimensionalidade — nunca à remoção de um
        descritor que o v1 tinha.
        """
        pytest.importorskip("librosa")
        from app.domain.xai.tabular import N_FEATURES

        v1 = bf.tabular_features_batch(canonical_batch)
        v2 = bf.tabular_features_v2_batch(canonical_batch)
        np.testing.assert_array_equal(v2[:, :N_FEATURES], v1)
        assert np.isfinite(v2).all()

    def test_largura_do_vetor_e_verificada(self, canonical_batch, monkeypatch):
        """Sem librosa o v1 caía de 63 para 37 colunas EM SILÊNCIO.

        `N_TABULAR_FEATURES` estava declarado desde sempre e não era
        referenciado em lugar nenhum; um modelo treinado assim declararia 37 no
        contrato e passaria por válido.
        """
        pytest.importorskip("librosa")
        import builtins

        real_import = builtins.__import__

        def sem_librosa(name, *args, **kwargs):
            if name == "librosa":
                raise ImportError("librosa indisponível (simulado)")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", sem_librosa)
        with pytest.raises(ImportError):
            bf.tabular_features_batch(canonical_batch)

    def test_rasta_degradado_em_todas_as_amostras_falha(
        self, canonical_batch, monkeypatch
    ):
        """Zeros por amostra eram silenciosos; falha total é erro de ambiente.

        26 dos 63 descritores podiam sair constantes num lote inteiro sem uma
        linha de log.
        """
        from app.domain.features.extractors.cepstral.components import plp

        def sempre_falha(*_a, **_k):
            raise ValueError("extrator quebrado (simulado)")

        monkeypatch.setattr(plp, "extract_rasta_plp_features", sempre_falha)
        with pytest.raises(RuntimeError, match="RASTA-PLP falhou em TODAS"):
            bf._rasta_plp_stats(canonical_batch)

    def test_rasta_degradado_em_parte_do_lote_avisa(
        self, canonical_batch, monkeypatch, caplog
    ):
        from app.domain.features.extractors.cepstral.components import plp

        real = plp.extract_rasta_plp_features
        chamadas = {"n": 0}

        def falha_na_primeira(*a, **k):
            chamadas["n"] += 1
            if chamadas["n"] == 1:
                raise ValueError("amostra ruim (simulada)")
            return real(*a, **k)

        monkeypatch.setattr(plp, "extract_rasta_plp_features", falha_na_primeira)
        with caplog.at_level("WARNING"):
            out = bf._rasta_plp_stats(canonical_batch)
        assert out.shape == (26, len(canonical_batch))
        assert "RASTA-PLP degradado a zeros em 1 de 3" in caplog.text

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
        # BUG FIX (Limitação (viii) do TCC): o FeaturePreparer aplica a mesma
        # AGC por RMS da construção do corpus (Eq. 5) antes de delegar ao
        # front-end compartilhado — não mais os `samples` crus.
        # Numericamente inconsequente aqui (dB ref-max + z-score por amostra
        # são invariantes a reescala linear, a menos de ruído de ponto
        # flutuante ~1e-6 no termo epsilon do log), mas a referência precisa
        # espelhar o passo de AGC para a igualdade se manter exata.
        #
        # O ALVO é o do corpus (−26 dBFS), não o default de −23 LUFS do módulo:
        # essa divergência de 3 dB deslocava em 1,41x todas as colunas lineares
        # em amplitude do vetor tabular do SVM/RandomForest. Aqui a referência
        # precisa usar o mesmo alvo da produção para a igualdade exata valer.
        np.testing.assert_array_equal(
            result["features"],
            bf.log_mel_single(
                apply_agc(samples, target_lufs=bf.BAND_CORRECTION_RMS_DBFS)
            ),
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
