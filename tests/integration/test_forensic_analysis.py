"""Aba Investigar: análise completa, análise em lote e relatório CSV.

SUJEITO: os handlers de `app.interfaces.gradio.tabs.forensic_analysis`, que
só tinham cobertura por inspeção de fonte. Fixa o que a aba devolve antes de a
plotagem sair de `domain/services` (P1-7): 13 figuras + resumo, a pontuação por
segmento vinda do detector, o radar de oito eixos e o lote. O detector é falso
e determinístico; nenhum modelo de `data/models` é carregado.

O detector falso separa de propósito `confidence` (a do VEREDITO: `p_real` quando
o veredito é real) de `probabilities["fake"]`, como o serviço real faz. A aba
plotava a primeira como se fosse a segunda, e áudio real aparecia como fake.
"""

from __future__ import annotations

import csv
from types import SimpleNamespace

import numpy as np
import pytest

pytest.importorskip("gradio")
pytest.importorskip("librosa")
soundfile = pytest.importorskip("soundfile")

from app.domain.services.detection.operating_point import fake_score  # noqa: E402
from app.interfaces.gradio.tabs import forensic_analysis as fa  # noqa: E402

SR = 16000
LIMIAR = 0.55  # de propósito diferente de 0,5: o limiar é por modelo


def _voz(f0: float, segundos: float, semente: int) -> np.ndarray:
    t = np.arange(int(SR * segundos)) / SR
    vibrato = f0 * (1 + 0.02 * np.sin(2 * np.pi * 5 * t))
    fase = 2 * np.pi * np.cumsum(vibrato) / SR
    onda = sum(np.sin(k * fase) / k for k in range(1, 6))
    onda += 0.01 * np.random.default_rng(semente).standard_normal(len(t))
    return (0.3 * onda / np.abs(onda).max()).astype("float32")


def _p_fake(trecho: np.ndarray) -> float:
    return round(float(np.clip(np.sqrt(np.mean(trecho**2)) * 4.0, 0.0, 1.0)), 6)


class _Detector:
    """`p_fake` = energia do trecho; o veredito sai do limiar do modelo."""

    def detect_single(self, audio):
        p_fake = _p_fake(np.asarray(audio.samples))
        is_fake = p_fake >= LIMIAR
        return SimpleNamespace(
            status=SimpleNamespace(name="SUCCESS"),
            data=SimpleNamespace(
                # Como no Predictor: a confiança é a do veredito.
                confidence=p_fake if is_fake else 1.0 - p_fake,
                probabilities={"fake": p_fake, "real": 1.0 - p_fake},
                is_fake=is_fake,
                model_name="detector_falso",
                metadata={"classification_threshold": LIMIAR},
            ),
        )


@pytest.fixture(scope="module")
def audio():
    # duas falas separadas por meio segundo de silêncio
    return np.concatenate(
        [_voz(140, 1.5, 1), np.zeros(SR // 2, "float32"), _voz(210, 1.5, 2)]
    )


@pytest.fixture(scope="module")
def wav(audio, tmp_path_factory):
    caminho = tmp_path_factory.mktemp("investigar") / "voz.wav"
    soundfile.write(str(caminho), audio, SR)
    return caminho


@pytest.fixture(scope="module")
def analise(wav):
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(fa, "_get_detection_service", lambda: _Detector())
        return fa.run_forensic_analysis(str(wav))


def _titulos(figura):
    return [ax.get_title() for ax in figura.axes if ax.get_title()]


def test_analise_completa_devolve_as_treze_figuras_e_o_resumo(analise):
    assert len(analise) == 14
    figuras, resumo = analise[:13], analise[13]
    assert all(f is not None for f in figuras)
    assert [_titulos(f)[0] for f in figuras] == [
        "Mel Spectrogram",
        "Espectro de Fase (STFT)",
        "Cromograma",
        "Envelope Espectral & Formantes",
        "Taxa de Cruzamento por Zero (ZCR)",
        "Centroide Espectral",
        "Contorno de Energia (RMS)",
        "Harmonic-to-Noise Ratio (HNR) Temporal",
        _titulos(figuras[8])[0],  # "Jitter Local (x.xx%)"
        "Contorno de pitch (F0) com confiança",
        "Importância relativa das características",
        "Heatmap de Anomalias sobre Espectrograma",
        "Pontuação de voz sintética por segmento",
    ]
    assert _titulos(figuras[0]) == [
        "Mel Spectrogram",
        "STFT Magnitude (dB)",
        "Constant-Q Transform",
        "LFCC (Linear Frequency Cepstral Coefficients)",
    ]
    jitter, shimmer = _titulos(figuras[8])
    assert jitter.startswith("Jitter Local (") and shimmer.startswith("Shimmer Local (")
    assert "**Duração:** 3.50s" in resumo and "**Amostras:** 56,000" in resumo


def test_formantes_saem_em_quatro_trilhas(analise):
    linhas = analise[3].axes[0].lines
    assert [ln.get_label() for ln in linhas] == ["F1", "F2", "F3", "F4"]
    f1 = np.asarray(linhas[0].get_ydata(), dtype=float)
    f2 = np.asarray(linhas[1].get_ydata(), dtype=float)
    validos = ~np.isnan(f1) & ~np.isnan(f2)
    assert validos.any() and np.all(f1[validos] <= f2[validos])  # ordenadas


def test_linha_do_tempo_usa_a_pontuacao_centrada_no_limiar_do_modelo(analise, audio):
    seg, salto = SR, SR // 2
    inicios = range(0, len(audio) - seg, salto)
    brutos = [_p_fake(audio[i:i + seg]) for i in inicios]
    esperado = [fake_score(p, LIMIAR) for p in brutos]
    tempos = [(i + seg / 2) / SR for i in inicios]
    assert min(brutos) < LIMIAR, "a fixture precisa ter trecho com veredito real"

    ax = analise[12].axes[0]
    np.testing.assert_allclose(ax.lines[0].get_ydata(), esperado)
    np.testing.assert_allclose(ax.lines[0].get_xdata(), tempos)
    # O limiar do modelo já está na escala: o limite é o mesmo para todo modelo.
    np.testing.assert_allclose(ax.lines[1].get_ydata(), [0.5, 0.5])
    assert "50%" in ax.lines[1].get_label()
    assert ax.get_ylabel() == "Pontuação de voz sintética"


def test_mapa_de_anomalias_pinta_a_pontuacao_e_nao_a_confianca(analise, audio):
    from app.interfaces.gradio.presenters.forensic_plots import ForensicTheme

    seg, salto = SR, SR // 2
    esperado = [
        fake_score(_p_fake(audio[i:i + seg]), LIMIAR)
        for i in range(0, len(audio) - seg, salto)
    ]
    faixas = analise[11].axes[0].patches
    assert len(faixas) == len(esperado)  # uma faixa por segmento
    mapa = ForensicTheme.get_anomaly_cmap()
    for faixa, pontuacao in zip(faixas, esperado):
        np.testing.assert_allclose(faixa.get_facecolor()[:3], mapa(pontuacao)[:3])


def test_audio_real_com_veredito_real_nao_aparece_acima_do_limiar():
    """O defeito: `p_fake` 0,02 virava 0,98 no gráfico, rotulado de fake."""
    audio = np.zeros(SR * 3, dtype="float32")  # energia zero: p_fake 0, veredito real

    _, pontos, limiar = fa._pontuacao_por_segmento(_Detector(), audio, SR)

    assert limiar == 0.5
    assert len(pontos) > 0 and np.all(pontos < limiar)


def test_trecho_que_falha_fica_de_fora_sem_valor_inventado():
    """Plotava 0,5: um número que ninguém mediu, de um lado ou outro do limiar."""

    class _Falha:
        def __init__(self):
            self.chamadas = 0

        def detect_single(self, audio):
            self.chamadas += 1
            if self.chamadas % 2 == 0:
                raise RuntimeError("segmento ruim")
            return _Detector().detect_single(audio)

    servico = _Falha()
    audio = np.zeros(SR * 3, dtype="float32")
    tempos, pontos, _ = fa._pontuacao_por_segmento(servico, audio, SR)

    assert len(tempos) == len(pontos) == (servico.chamadas + 1) // 2
    assert 0.5 not in pontos.tolist()


def test_segmentos_sem_resposta_devolvem_series_vazias_e_limiar_neutro():
    class _SemSucesso:
        def detect_single(self, audio):
            return SimpleNamespace(status=SimpleNamespace(name="ERROR"), data=None)

    audio = np.zeros(SR * 3, dtype="float32")
    tempos, pontos, limiar = fa._pontuacao_por_segmento(_SemSucesso(), audio, SR)

    assert len(tempos) == len(pontos) == 0
    assert limiar == 0.5


def test_radar_tem_oito_eixos_medidos_e_normalizados(analise):
    ax = analise[10].axes[0]
    assert [t.get_text() for t in ax.get_xticklabels()] == [
        "Spectral", "Cepstral", "Temporal", "Prosodic",
        "Phase", "Formant", "HNR", "Jitter/Shimmer",
    ]
    valores = np.asarray(ax.lines[0].get_ydata(), dtype=float)[:8]
    assert np.all((valores >= 0) & (valores <= 1))
    assert len(set(np.round(valores, 6))) > 1  # nenhum eixo fabricado


def test_sem_modelo_so_a_explicabilidade_fica_vazia(wav, monkeypatch):
    monkeypatch.setattr(fa, "_get_detection_service", lambda: None)
    r = fa.run_forensic_analysis(str(wav))
    assert all(f is not None for f in r[:10])
    assert r[10:13] == (None, None, None)


def test_sem_arquivo_devolve_aviso():
    r = fa.run_forensic_analysis(None)
    assert r == (None,) * 13 + ("Nenhum arquivo selecionado.",)


def test_lote_resume_e_exporta_csv(wav, tmp_path, monkeypatch):
    monkeypatch.setattr(fa, "_get_detection_service", lambda: _Detector())
    monkeypatch.setenv("GRADIO_TEMP_DIR", str(tmp_path / "gradio"))
    baixa = tmp_path / "baixa.wav"
    soundfile.write(str(baixa), _voz(180, 1.0, 9) * 0.2, SR)

    pizza, histograma, tabela, resumo = fa.run_batch_analysis(
        [str(wav), str(baixa), str(tmp_path / "nao_existe.wav")]
    )
    assert list(tabela.columns) == ["Arquivo", "Resultado", "Confiança", "Modelo", "Duração"]
    assert tabela["Arquivo"].tolist() == ["voz.wav", "baixa.wav", "nao_existe.wav"]
    assert tabela["Resultado"].tolist() == ["FAKE", "REAL", "REAL"]
    assert tabela["Modelo"].tolist()[2].startswith("error: ")
    assert "**Total analisado:** 3" in resumo and "**Fake:** 1 (33.3%)" in resumo
    assert _titulos(pizza) == ["Distribuição de resultados (total: 3)"]
    # O histograma é da pontuação, com o limite em 50%, e só com o que foi medido:
    # o arquivo que falhou tem confiança 0 na tabela, mas nenhuma pontuação.
    ax = histograma.axes[0]
    assert _titulos(histograma) == ["Distribuição da pontuação de voz sintética"]
    assert ax.get_xlabel() == "Pontuação de voz sintética"
    assert sum(barra.get_height() for barra in ax.patches) == 2
    np.testing.assert_allclose(ax.lines[0].get_xdata(), [0.5, 0.5])

    caminho = fa.export_batch_report([str(wav), str(baixa)])
    with open(caminho, encoding="utf-8", newline="") as fh:
        linhas = list(csv.reader(fh))
    assert linhas[0] == ["Arquivo", "Resultado", "Confiança", "Modelo", "Duração(s)"]
    assert [r[:2] for r in linhas[1:3]] == [["voz.wav", "FAKE"], ["baixa.wav", "REAL"]]
    resumo_csv = {r[0]: r[1] for r in linhas[4:] if len(r) == 2}
    assert resumo_csv["Total"] == "2" and resumo_csv["% Fake"] == "50.00%"
