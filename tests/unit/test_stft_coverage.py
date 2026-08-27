"""Nenhuma amostra do sinal pode ficar invisível ao log-mel.

O salto entre quadros é imposto pelo contrato da arquitetura
(`ceil(T / time_steps)`), mas a janela de análise era a constante 512 — e as
duas eram independentes no código. Com 100 quadros em 3 s o salto fica em 480:
janelas consecutivas se sobrepunham em 32 amostras e o taper de Hann é ~0 nas
duas pontas.

Medido antes da correção: o envelope de soma-e-sobreposição ia de 1,0 a
EXATAMENTE 0, com **27% do sinal** em regiões de peso desprezível. E um clique
de 1 ms era 9× mais ou menos visível conforme onde caísse (razão mín/máx 0,11)
— sendo transiente justamente a pista de síntese que a tarefa procura.
"""

from __future__ import annotations

import numpy as np
import pytest

librosa = pytest.importorskip("librosa")

from app.domain.features.benchmark_frontend import (  # noqa: E402
    MIN_STFT_OVERLAP,
    resolve_n_fft,
)

SR = 16000
JANELA_FONTE = 48000


def _envelope_de_cobertura(n_fft: int, hop: int, total: int = JANELA_FONTE):
    """Peso com que cada amostra participa da análise (soma de Hann²)."""
    janela = np.hanning(n_fft) ** 2
    envelope = np.zeros(total)
    for inicio in range(0, total - n_fft, hop):
        envelope[inicio:inicio + n_fft] += janela
    return envelope[n_fft:total - n_fft]  # descarta as bordas do sinal


@pytest.mark.parametrize(
    "time_steps,declarado",
    [
        (100, None),   # Conformer, CCT, Res2Net, Sonic Sleuth, EfficientNet
        (300, 400),    # AST: janela de 25 ms vem do artigo
        (150, None),
        (200, None),
    ],
)
def test_nenhuma_amostra_fica_invisivel(time_steps, declarado):
    hop = max(64, int(np.ceil(JANELA_FONTE / time_steps)))
    n_fft = resolve_n_fft(hop, declarado)

    envelope = _envelope_de_cobertura(n_fft, hop)
    invisivel = float((envelope < 0.05 * envelope.max()).mean())

    assert invisivel == 0.0, (
        f"time_steps={time_steps} n_fft={n_fft} hop={hop}: {invisivel:.1%} do "
        "sinal cai em regiao de peso desprezivel"
    )


def test_sobreposicao_minima_e_respeitada_quando_derivada():
    for time_steps in (50, 100, 150, 200, 300):
        hop = max(64, int(np.ceil(JANELA_FONTE / time_steps)))
        n_fft = resolve_n_fft(hop, None)
        sobreposicao = (n_fft - hop) / n_fft
        assert sobreposicao >= MIN_STFT_OVERLAP, (
            f"time_steps={time_steps}: sobreposicao {sobreposicao:.1%} abaixo "
            f"do minimo {MIN_STFT_OVERLAP:.0%}"
        )


def test_contrato_da_arquitetura_vence_a_derivacao():
    """O AST especifica 25 ms por definição do artigo (Gong et al., 2021)."""
    assert resolve_n_fft(160, 400) == 400
    assert resolve_n_fft(480, 400) == 400
    # sem declaração, deriva
    assert resolve_n_fft(480, None) == 1024
    assert resolve_n_fft(160, None) == 512  # piso do projeto


def test_transiente_e_detectado_independentemente_da_posicao():
    """O sintoma que motivou a correção, medido de novo."""
    hop = 480

    def visibilidade(n_fft, posicao_relativa):
        t = np.arange(JANELA_FONTE) / SR
        base = 0.3 * np.sin(2 * np.pi * 220 * t)
        com_clique = base.copy()
        p = int(posicao_relativa * JANELA_FONTE)
        com_clique[p:p + 16] += 0.9
        S = np.abs(librosa.stft(com_clique, n_fft=n_fft, hop_length=hop))
        S0 = np.abs(librosa.stft(base, n_fft=n_fft, hop_length=hop))
        return float(np.sum(np.abs(S - S0)) / (np.sum(S0) + 1e-9))

    posicoes = np.linspace(0.30, 0.34, 6)
    derivado = resolve_n_fft(hop, None)

    valores = [visibilidade(derivado, p) for p in posicoes]
    razao = min(valores) / max(valores)
    assert razao > 0.7, (
        f"com n_fft={derivado} a deteccao do transiente ainda depende muito da "
        f"posicao (min/max = {razao:.2f})"
    )

    # e a janela antiga continua sendo ruim — o teste falharia se alguem a
    # restaurasse achando que o problema era outro
    antigos = [visibilidade(512, p) for p in posicoes]
    assert min(antigos) / max(antigos) < razao


def test_frontend_produz_a_forma_do_contrato():
    from app.domain.features.benchmark_frontend import log_mel_batch

    X = np.random.default_rng(0).normal(0, 0.05, (3, JANELA_FONTE)).astype("float32")

    grupo = log_mel_batch(X, time_steps=100, feature_dim=80)
    assert grupo.shape == (3, 100, 80) and np.isfinite(grupo).all()

    ast = log_mel_batch(X, time_steps=300, feature_dim=128, n_fft=400)
    assert ast.shape == (3, 300, 128) and np.isfinite(ast).all()
