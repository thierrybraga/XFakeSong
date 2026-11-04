"""Medidas forenses do domínio, sem matplotlib.

SUJEITO: `app.domain.services.forensic_measures`, o lado de cálculo que saiu
de `forensic_visualization.py` (P1-7). Cada medida é verificada contra um
sinal cujo valor se conhece de antemão — tom puro, ruído, silêncio — em vez
de contra a figura que a exibe.
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("librosa")

from app.domain.services import forensic_measures as fm  # noqa: E402

SR = 16000


def _tom(freq: float, dur: float = 1.0) -> np.ndarray:
    t = np.arange(int(SR * dur)) / SR
    return (0.4 * np.sin(2 * np.pi * freq * t)).astype("float32")


def _ruido(dur: float = 1.0, semente: int = 0) -> np.ndarray:
    rng = np.random.default_rng(semente)
    return (0.4 * rng.standard_normal(int(SR * dur))).astype("float32")


def test_hnr_separa_tom_de_ruido():
    tempos, hnr_tom = fm.hnr_contour(_tom(200.0), SR)
    _, hnr_ruido = fm.hnr_contour(_ruido(), SR)

    assert len(tempos) == len(hnr_tom) > 0
    assert np.all((hnr_tom >= -10) & (hnr_tom <= 40))
    # Ordem, não escala: a autocorrelação é a enviesada (sem dividir pela
    # da janela), então um quadro de 25 ms limita o HNR a ~10,6 dB e o tom
    # puro de 200 Hz fica em 6,0 dB — ver o registro da Fase 4 (P1-7).
    assert np.median(hnr_tom) > 0 > np.median(hnr_ruido)


def test_pitch_encontra_a_fundamental_do_tom():
    pitch = fm.pitch_analysis(_tom(220.0), SR)

    assert pitch.f0_mean == pytest.approx(220.0, rel=0.02)
    assert pitch.f0_detrended is not None and pitch.vibrato_extent >= 0
    assert len(pitch.times) == len(pitch.f0) == len(pitch.voiced_mask)


def test_silencio_nao_tem_pitch_nem_jitter():
    silencio = np.zeros(SR, dtype="float32")

    pitch = fm.pitch_analysis(silencio, SR)
    assert pitch.f0_mean is None and pitch.f0_detrended is None
    assert fm.jitter_shimmer(silencio, SR) is None


def test_jitter_e_shimmer_alinham_tempo_e_valor():
    medida = fm.jitter_shimmer(_tom(180.0), SR)

    assert medida is not None
    assert len(medida.jitter_times) == len(medida.jitter)
    assert len(medida.shimmer_times) == len(medida.shimmer)
    # tom estável: perturbação pequena, nunca negativa
    assert 0 <= medida.jitter_mean < 1.04
    assert 0 <= medida.shimmer_mean < 3.81


def test_formantes_e_lfcc_tem_a_forma_esperada():
    sinal = _tom(300.0) + 0.5 * _tom(1200.0)
    tempos, trilhas = fm.formant_tracks(sinal, SR)

    assert list(trilhas) == ["F1", "F2", "F3", "F4"]
    assert all(len(t) == len(tempos) for t in trilhas.values())
    assert fm.linear_cepstrum(sinal).shape[0] == 20


def test_det_e_varredura_de_limiar():
    rotulos = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    separados = np.array([0.1, 0.2, 0.25, 0.28, 0.7, 0.75, 0.8, 0.9])

    fpr, fnr, eer = fm.det_curve(rotulos, separados)
    assert len(fpr) == len(fnr) and 0.0 <= eer < 0.01

    varredura = fm.threshold_sweep(rotulos, separados)
    assert len(varredura.thresholds) == len(varredura.f1) == 91
    # F1 = 1 em todo limiar acima de 0,28 até 0,70; o primeiro vence o empate
    assert varredura.best_threshold == pytest.approx(0.29)
    assert max(varredura.f1) == 1.0


def test_radar_mede_oito_eixos_normalizados():
    tom = fm.radar_axes(_tom(220.0, dur=2.0), SR)
    ruido = fm.radar_axes(_ruido(dur=2.0), SR)

    assert len(fm.RADAR_AXES) == len(fm.RADAR_RANGES) == len(tom) == 8
    assert np.all((tom >= 0) & (tom <= 1)) and np.all((ruido >= 0) & (ruido <= 1))
    eixo = fm.RADAR_AXES.index("Phase")
    assert tom[eixo] < ruido[eixo]  # fase desdobrada separa tom de ruído


def test_resumo_do_lote():
    assert fm.batch_report([]) == {"total": 0, "summary": "Sem resultados"}

    resumo = fm.batch_report(
        [
            {
                "filename": "a.wav",
                "is_fake": True,
                "confidence": 0.9,
                "model_name": "m",
            },
            {"filename": "b.wav", "is_fake": False, "confidence": 0.5},
        ]
    )
    assert resumo["total_analyzed"] == 2 and resumo["fake_percentage"] == 50.0
    assert resumo["high_confidence_fakes"] == 1
    assert resumo["low_confidence_detections"] == 1
    assert resumo["per_file"][1]["model"] == "unknown"
