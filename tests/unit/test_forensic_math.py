"""Correção matemática das medidas exibidas na análise forense.

Uma ferramenta forense não pode mostrar número que não mediu. Três defeitos
foram encontrados na revisão de 2026-07-28:

1. três dos oito eixos do radar eram as constantes 0.5, 0.4 e 0.3, rotuladas
   "Formant", "HNR" e "Jitter/Shimmer";
2. a descontinuidade de fase usava `np.angle` sem `unwrap`, medindo o salto de
   wrap em (-pi, pi] em vez da descontinuidade;
3. o jitter (RAP/PPQ) concatenava trechos vozeados separados por silêncio,
   transformando a junção artificial em perturbação.
"""

from __future__ import annotations

import numpy as np
import pytest

librosa = pytest.importorskip("librosa")

SR = 16000


def _tom(freq: float, dur: float = 1.0, sr: int = SR) -> np.ndarray:
    t = np.arange(int(sr * dur)) / sr
    return (0.4 * np.sin(2 * np.pi * freq * t)).astype("float32")


# ───────────── jitter: contiguidade dos trechos vozeados ─────────────

def test_jitter_nao_inventa_perturbacao_entre_trechos_vozeados():
    """F0 constante nos dois lados de uma pausa: jitter real é ZERO."""
    from app.domain.features.extractors.voice_quality.components.perturbation import (
        compute_ppq,
        compute_rap,
    )

    f0 = np.concatenate([np.full(20, 200.0), np.zeros(20), np.full(20, 120.0)])

    assert compute_rap(None, f0, SR) == pytest.approx(0.0, abs=1e-9), (
        "a juncao entre trechos vozeados voltou a virar perturbacao"
    )
    assert compute_ppq(None, f0, SR) == pytest.approx(0.0, abs=1e-9)


def test_jitter_continua_medindo_perturbacao_real():
    """A correção não pode zerar a sensibilidade."""
    from app.domain.features.extractors.voice_quality.components.perturbation import (
        compute_rap,
    )

    rng = np.random.default_rng(0)
    perturbado = 200.0 * (1 + 0.02 * rng.normal(size=60))
    constante = np.full(60, 200.0)

    assert compute_rap(None, perturbado, SR) > 5e-3
    assert compute_rap(None, constante, SR) == pytest.approx(0.0, abs=1e-9)


def test_trechos_vozeados_preservam_a_contiguidade():
    from app.domain.features.extractors.voice_quality.components.utils import (
        voiced_runs,
    )

    f0 = np.concatenate([np.full(5, 200.0), np.zeros(3), np.full(4, 100.0)])
    trechos = voiced_runs(f0, SR)

    assert len(trechos) == 2
    assert len(trechos[0]) == 5 and len(trechos[1]) == 4
    np.testing.assert_allclose(trechos[0], 1 / 200.0)
    np.testing.assert_allclose(trechos[1], 1 / 100.0)


# ───────────── fase: unwrap antes de diferenciar ─────────────

def test_fase_desdobrada_separa_sinal_limpo_de_ruido():
    """Sem `unwrap`, tom puro e ruído davam valores próximos."""
    tom = _tom(440.0)
    ruido = (0.4 * np.random.default_rng(0).normal(size=SR)).astype("float32")

    def descontinuidade(sinal, desdobrar):
        fase = np.angle(librosa.stft(sinal))
        if desdobrar:
            fase = np.unwrap(fase, axis=1)
        return float(np.mean(np.abs(np.diff(fase, axis=1))))

    assert descontinuidade(tom, True) < descontinuidade(ruido, True), (
        "a medida precisa separar sinal harmonico de ruido"
    )
    # e desdobrar tem de reduzir o valor do sinal limpo
    assert descontinuidade(tom, True) < descontinuidade(tom, False)


def test_aba_forense_desdobra_a_fase():
    import inspect

    from app.interfaces.gradio.tabs import forensic_analysis

    fonte = inspect.getsource(forensic_analysis)
    assert "np.unwrap(np.angle(D), axis=1)" in fonte


# ───────────── radar: nenhum eixo constante ─────────────

def test_estabilidade_de_formante_varia_com_o_audio():
    """Era a constante 0.5."""
    from app.interfaces.gradio.tabs.forensic_analysis import (
        _estabilidade_de_formante,
    )

    t = np.arange(2 * SR) / SR
    fixo = _tom(300.0, dur=2.0)
    varredura = (
        0.4 * np.sin(2 * np.pi * np.cumsum(np.linspace(300, 1200, len(t))) / SR)
    ).astype("float32")

    v_fixo = _estabilidade_de_formante(fixo, SR)
    v_varredura = _estabilidade_de_formante(varredura, SR)

    assert v_fixo != v_varredura, "o eixo voltou a ser constante"
    assert v_varredura > v_fixo, (
        "uma varredura de formante tem de ser MENOS estavel que um tom fixo"
    )


def test_radar_nao_tem_mais_valores_fabricados():
    import inspect

    from app.interfaces.gradio.tabs import forensic_analysis

    fonte = inspect.getsource(forensic_analysis)
    assert "# placeholder" not in fonte, (
        "voltou a existir eixo fabricado no grafico forense"
    )
    for medida in ("compute_nhr", "compute_rap", "_estabilidade_de_formante"):
        assert medida in fonte, f"{medida} nao esta sendo medido"


def test_radar_normaliza_por_eixo_e_nao_pelo_maximo_global():
    """Grandezas de unidades diferentes no mesmo gráfico."""
    from app.domain.services.forensic_visualization import (
        AudioForensicVisualizer,
    )

    visualizador = AudioForensicVisualizer()
    # ja normalizado por eixo pelo chamador: nao pode ser reescalado
    valores = np.array([0.2, 0.5, 0.1, 0.3])
    figura = visualizador.plot_feature_importance_radar(
        ["a", "b", "c", "d"], valores
    )
    linha = figura.axes[0].lines[0]
    plotados = np.asarray(linha.get_ydata())[:4]
    np.testing.assert_allclose(plotados, valores, atol=1e-9)

    # fora de [0, 1] o reescalonamento continua acontecendo
    figura2 = visualizador.plot_feature_importance_radar(
        ["a", "b"], np.array([10.0, 50.0])
    )
    plotados2 = np.asarray(figura2.axes[0].lines[0].get_ydata())[:2]
    assert plotados2.max() == pytest.approx(1.0)
