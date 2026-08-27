"""Correção de banda: remove a assinatura de reamostragem do corpus pareado.

SUJEITO: `benchmark_frontend.apply_band_correction` e o parâmetro
`BenchmarkConfig.band_correction_hz` que a aciona.

MOTIVAÇÃO (2026-08-19). O corpus pareado junta duas fontes com taxa de origem
distinta -- CETUC em 16 kHz nativo na classe bonafide, `unfake/fake_voices` em
24 kHz reamostrado para 16 kHz na classe spoof. A taxa de origem prediz a
classe com 100% de acurácia (49.264 arquivos de cada lado, sem exceção,
conferido no manifesto do build), e o filtro anti-aliasing do `soxr_hq` deixa
assinatura acima de 7,5 kHz.

Medido no conjunto de teste: energia relativa em 7,9--8,0 kHz de 4,3e-06 na
bonafide contra 4,3e-10 na spoof. Uma regressão logística sobre 40 energias de
banda alcança AUC 1,0000; com a correção, 0,84. A banda isolada sai de 0,9810
para 0,5432.

O atalho vivia em 100 Hz de espectro, e nenhum modelo precisava aprender nada
sobre vocoders para explorá-lo.
"""
from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("scipy")

from app.domain.features.benchmark_frontend import (  # noqa: E402
    BAND_CORRECTION_HZ,
    apply_band_correction,
)

SR = 16000


def _tom(freq: float, n: int = 16000) -> np.ndarray:
    t = np.arange(n) / SR
    return (0.5 * np.sin(2 * np.pi * freq * t)).astype("float32")


def _energia_em(x: np.ndarray, lo: float, hi: float) -> float:
    """Energia ABSOLUTA na faixa.

    Fracao da energia total nao serve para medir atenuacao: removido o tom, o
    total despenca junto e o residuo continua concentrado na mesma faixa --
    a fracao permanece alta enquanto a energia caiu ordens de grandeza.
    """
    E = np.abs(np.fft.rfft(x)) ** 2
    f = np.fft.rfftfreq(len(x), 1 / SR)
    return float(E[(f >= lo) & (f <= hi)].sum())


def test_atenua_acima_do_corte():
    """Um tom a 7,9 kHz -- dentro da faixa contaminada -- tem de ser removido."""
    x = _tom(7900)[None, :]
    antes = _energia_em(x[0], 7600, 8000)
    # `renormalize_rms_dbfs=None` isola o FILTRO: com a renormalizacao ligada,
    # um tom puro na faixa removida tem o residuo amplificado de volta a -26
    # dBFS, e a energia absoluta deixa de medir atenuacao.
    filtrado = apply_band_correction(x, renormalize_rms_dbfs=None)
    depois = _energia_em(filtrado[0], 7600, 8000)

    atenuacao_db = 10 * np.log10(max(depois, 1e-30) / max(antes, 1e-30))
    assert atenuacao_db < -40, (
        f"atenuacao insuficiente acima do corte: {atenuacao_db:.1f} dB"
    )


def test_preserva_a_banda_de_fala():
    """A faixa que carrega fala não pode ser tocada de forma apreciável."""
    x = _tom(1000)[None, :]
    antes = _energia_em(x[0], 900, 1100)
    depois = _energia_em(
        apply_band_correction(x, renormalize_rms_dbfs=None)[0], 900, 1100
    )

    assert depois > 0.95 * antes, (
        f"o passa-baixas atenuou 1 kHz: {antes:.4e} -> {depois:.4e}"
    )


def test_preserva_forma_e_finitude():
    """O runner passa (N, 48000, 1); a forma tem de voltar igual."""
    rng = np.random.default_rng(0)
    x = rng.standard_normal((6, 4800, 1)).astype("float32")
    y = apply_band_correction(x)

    assert y.shape == x.shape
    assert y.dtype == np.float32
    assert np.isfinite(y).all()


def test_e_uniforme_entre_as_classes():
    """Aplicar a UMA classe só reintroduziria o problema com outro sinal.

    O contrato é que a função não conhece rótulo: ela transforma o que
    recebe. Quem garante a uniformidade é o runner, que a aplica às três
    partições inteiras antes de qualquer separação por classe.
    """
    rng = np.random.default_rng(1)
    lote = rng.standard_normal((4, 4800)).astype("float32")
    saida = apply_band_correction(lote)

    for i in range(len(lote)):
        individual = apply_band_correction(lote[i][None, :])[0]
        assert np.allclose(saida[i], individual, atol=1e-5), (
            "o resultado depende do lote — a correção precisa ser por amostra"
        )


def test_corte_padrao_e_o_medido():
    """7.500 Hz é o menor corte que remove o artefato sem descartar sinal.

    A varredura mediu o penhasco inteiramente entre 8,0 e 7,5 kHz
    (AUC 1,0000 -> 0,8317) e estabilidade abaixo (7,0 kHz -> 0,8293).
    """
    assert BAND_CORRECTION_HZ == 7500


def test_config_declara_o_parametro():
    """Sem campo declarado, o TrainingService/BenchmarkConfig descartaria em
    silêncio -- o mesmo defeito que o `checkpoint_monitor` teve."""
    import dataclasses

    from benchmarks.config import BenchmarkConfig

    campos = {f.name for f in dataclasses.fields(BenchmarkConfig)}
    assert "band_correction_hz" in campos
    # Default desligado preserva a reprodutibilidade dos artefatos anteriores.
    assert BenchmarkConfig(dataset_path="x").band_correction_hz is None


def test_runner_aplica_antes_da_auditoria():
    """A ordem importa: corrige o SINAL, depois o AWGN (canal), e a auditoria
    precisa descrever o dado realmente usado."""
    import inspect

    from benchmarks import runner

    fonte = inspect.getsource(runner.run_benchmark)
    i_corr = fonte.find("band_correction_hz")
    i_aud = fonte.find("_audit_split_overlap")
    assert i_corr > 0, "a correção de banda não é aplicada em run_benchmark"
    assert i_corr < i_aud, (
        "a correção precisa vir ANTES da auditoria de sobreposição"
    )


# ─── DC e nível ────────────────────────────────────────────────────────────
#
# Segundo atalho, medido em 2026-08-19: o offset DC separa as classes com AUC
# 0,7387 (bonafide 2,04e-03 contra spoof 1,38e-04, razão 14,7x). Não é
# artefato de síntese — é de captura.


def test_remove_o_offset_dc():
    rng = np.random.default_rng(3)
    x = (rng.standard_normal((4, 4800)).astype("float32") * 0.1) + 0.05

    y = apply_band_correction(x)

    assert np.abs(y.mean(axis=1)).max() < 1e-6, "offset DC sobreviveu"


def test_renormaliza_o_rms_depois_do_dc():
    """A ordem ingênua quebra: remover DC sozinho DESIGUALA o RMS.

    A normalização do build foi feita sobre o sinal COM DC. Como a classe
    bonafide tinha 14,7x mais DC, tirá-lo desequaliza o RMS que estava
    equalizado — o atalho de RMS sobe de 0,5126 para 0,7503. A renormalização
    logo atrás devolve ao acaso (0,5106).
    """
    rng = np.random.default_rng(4)
    # dois lotes com DC bem diferente, como as duas classes do corpus
    a = (rng.standard_normal((3, 4800)).astype("float32") * 0.05) + 0.02
    b = (rng.standard_normal((3, 4800)).astype("float32") * 0.05) + 0.0005

    ra = np.sqrt(np.mean(apply_band_correction(a) ** 2, axis=1))
    rb = np.sqrt(np.mean(apply_band_correction(b) ** 2, axis=1))

    alvo = 10 ** (-26 / 20)
    assert np.allclose(ra, alvo, rtol=0.02), f"RMS do lote A: {ra}"
    assert np.allclose(rb, alvo, rtol=0.02), f"RMS do lote B: {rb}"


def test_preserva_o_fator_de_crista():
    """Compressão de faixa dinâmica É artefato de síntese previsto no texto.

    A correção não pode apagá-lo: o ganho aplicado no build mostra a
    assimetria de origem (bonafide +3,3 dB com desvio 6,6; spoof -8,7 dB com
    desvio 0,8), e o fator de crista sobrevive em AUC 0,697 de propósito.
    """
    rng = np.random.default_rng(5)
    x = rng.standard_normal((3, 4800)).astype("float32") * 0.05
    x[:, ::500] *= 8.0  # picos esparsos -> fator de crista alto

    def crista(a):
        return np.max(np.abs(a), axis=1) / (np.sqrt(np.mean(a ** 2, axis=1)) + 1e-12)

    antes, depois = crista(x), crista(apply_band_correction(x))
    assert np.allclose(antes, depois, rtol=0.15), (
        f"o fator de crista foi alterado: {antes} -> {depois}"
    )


# ─── repasse pelo runner sequencial ────────────────────────────────────────


def test_runner_sequencial_repassa_as_flags():
    """A bateria roda pelo `run_models_sequential`, não pelo `run_benchmark`.

    Sem o repasse, a bateria inteira rodaria SEM correção e com o monitor
    antigo, em silêncio — e o operador só descobriria ao comparar os
    artefatos. É a família de defeito que este projeto já pagou duas vezes:
    o `checkpoint_monitor` que não chegava ao treinador (2026-08-17) e os
    grids regularizados sem chamador (2026-08-09).
    """
    from pathlib import Path

    fonte = (
        Path(__file__).resolve().parents[2]
        / "scripts/benchmark/run_models_sequential.py"
    ).read_text(encoding="utf-8")

    for flag in ("--band-correction-hz", "--checkpoint-monitor"):
        assert flag in fonte, f"{flag} não é declarada no runner sequencial"
    for chave in ("band_correction_hz", "checkpoint_monitor"):
        assert f'getattr(args, "{chave}"' in fonte, (
            f"{chave} declarada mas não repassada ao comando"
        )


def test_ramo_ssl_tambem_recebe_a_correcao():
    """WavLM e HuBERT rodam por um runner PyTorch separado, e o ramo deles
    faz `return` ANTES do trecho comum onde as flags são acrescentadas.

    Sem esta linha, os dois treinariam COM o atalho de reamostragem enquanto
    os outros nove treinariam sem — e a tabela ficaria incomparável sem que
    nada avisasse. `--checkpoint-monitor` NÃO entra nesse ramo: o runner SSL
    tem laço de treino próprio em PyTorch, sem o ModelCheckpoint do Keras que
    a opção controla; passá-la seria config morto.
    """
    import argparse
    import inspect
    import re
    from pathlib import Path as _P

    from scripts.benchmark.run_models_sequential import _build_command

    fonte = inspect.getsource(_build_command)
    attrs = set(re.findall(r"args\.(\w+)", fonte)) | set(
        re.findall(r'getattr\(args, "(\w+)"', fonte)
    )
    base = {a: "" for a in attrs}
    base.update(
        dataset="data/datasets/benchmark_dataset_15k.npz", epochs=2,
        snr=[20], train_aug_snr=[20], seed=42, batch_size=8,
        device_profile="cpu", latency_runs=1, train_noise_copies=1,
        waveform_noise_batch_size=8, ssl_train_batch_size=4,
        ssl_feature_batch_size=4, waveform_train_augmentation=True,
        band_correction_hz=7500.0, checkpoint_monitor="val_eer",
        codec_eval=None, cross_generator=None, group_split=False,
        academic_protocol=False, source_shortcut_limit=None,
    )
    args = argparse.Namespace(**base)

    for modelo in ("WavLM Original", "HuBERT Original"):
        cmd = [str(c) for c in _build_command(args, modelo, _P("/tmp/x"))]
        assert "wavlm_original" in " ".join(cmd), f"{modelo} não foi ao runner SSL"
        assert "--band-correction-hz" in cmd, (
            f"{modelo} treinaria COM o atalho enquanto os outros nove não"
        )
        assert "--checkpoint-monitor" not in cmd, (
            "o runner SSL não tem ModelCheckpoint do Keras — a flag seria "
            "config morto"
        )

    for modelo in ("AASIST", "Hybrid CNN-Transformer", "SVM"):
        cmd = [str(c) for c in _build_command(args, modelo, _P("/tmp/x"))]
        assert "--band-correction-hz" in cmd
        assert "--checkpoint-monitor" in cmd


def test_funciona_com_sinal_curto():
    """`filtfilt` exige o sinal maior que `padlen` (3× a ordem = 765 aqui).

    As janelas do protocolo têm 48.000 amostras, mas `extract_window` é
    chamada também com entradas curtas — e o teste de contrato
    `test_window_is_a_pure_center_crop_when_long_enough` usa 40 amostras.
    Sem o `padlen` adaptativo a função levantava
    "The length of the input vector x must be greater than padlen".
    """
    rng = np.random.default_rng(6)
    for n in (40, 100, 800, 4800):
        x = rng.standard_normal((2, n)).astype("float32")
        y = apply_band_correction(x)
        assert y.shape == x.shape, f"forma quebrou em n={n}"
        assert np.isfinite(y).all(), f"saída não finita em n={n}"


def test_corte_zero_desliga_a_correcao_em_vez_de_zerar_o_sinal():
    """`--band-correction-hz 0` tem que DESLIGAR, nao filtrar em 0 Hz.

    O contrato e implicito: `benchmarks/runner.py` decide por VERACIDADE
    (`if getattr(cfg, "band_correction_hz", None)`), e 0.0 e falso. Trocar
    isso por `is not None` -- refatoracao que parece inofensiva e ate mais
    correta -- faria um passa-baixas de 0 Hz ZERAR todo o corpus, e o treino
    seguiria em silencio sobre ruido numerico. A ablacao documentada em
    `train_by_family.py --band-correction-hz 0` depende deste comportamento.
    """
    from benchmarks.config import BenchmarkConfig

    for valor, esperado_ativo in ((0, False), (0.0, False), (None, False),
                                  (7500.0, True)):
        cfg = BenchmarkConfig(band_correction_hz=valor)
        assert bool(getattr(cfg, "band_correction_hz", None)) is esperado_ativo, (
            f"band_correction_hz={valor!r} deveria "
            f"{'ativar' if esperado_ativo else 'desligar'} a correcao"
        )

    # E o filtro em si nunca deve ser chamado com corte 0: se for, e erro
    # ruidoso, nao um corpus zerado.
    X = np.random.default_rng(0).standard_normal((4, 2048)).astype("float32")
    with pytest.raises(Exception):
        apply_band_correction(X, cutoff_hz=0.0)
