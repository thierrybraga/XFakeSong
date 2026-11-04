"""Medida da fusão: pontuação coerente com a inferência e medida em locutor não visto.

SUJEITO: `benchmarks/fusion.py`.
"""

from __future__ import annotations

import numpy as np
import pytest

from app.domain.services.detection.operating_point import fake_score
from benchmarks.fusion import (
    centered_scores,
    compare_fusions,
    eer_and_threshold,
    held_out_scores,
    top_k_chooser,
)


@pytest.mark.parametrize("limiar", [0.047, 0.5, 0.941])
def test_pontuacao_vetorial_e_a_mesma_da_inferencia(limiar):
    p = np.linspace(0.0, 1.0, 101)
    np.testing.assert_allclose(
        centered_scores(p, limiar), [fake_score(v, limiar) for v in p]
    )


def test_eer_de_separacao_perfeita_e_zero():
    y = np.array([0, 0, 1, 1])
    eer, limiar = eer_and_threshold(y, np.array([0.1, 0.2, 0.8, 0.9]))
    assert eer == 0.0
    assert 0.2 < limiar <= 0.8


def _cenario():
    """Dois modelos com erros independentes e limiares opostos, e um só de ruído."""
    rng = np.random.default_rng(0)
    n = 1200
    y = np.tile([0, 1], n // 2)
    locutores = np.repeat([f"L{i}" for i in range(6)], n // 6)
    a = 1.0 / (1.0 + np.exp(-(3.0 * (2 * y - 1) + rng.normal(0, 2.5, n) + 3.0)))
    b = 1.0 / (1.0 + np.exp(-(3.0 * (2 * y - 1) + rng.normal(0, 2.5, n) - 3.0)))
    return {"a": a, "b": b, "ruido": rng.random(n)}, y, locutores


def _linha(linhas, regra, condicao="clean"):
    return next(r for r in linhas if r["rule"] == regra and r["condition"] == condicao)


def test_fusao_de_erros_independentes_supera_cada_modelo():
    p, y, locutores = _cenario()
    linhas = compare_fusions(p, y, locutores, top_ks=(1, 2))
    fusao = _linha(linhas, "top-2")
    assert fusao["eer"] < _linha(linhas, "model:a")["eer"]
    assert fusao["eer"] < _linha(linhas, "model:b")["eer"]
    assert fusao["eer"] < _linha(linhas, "top-1")["eer"]
    assert fusao["chosen"] == [{"models": ["a", "b"], "rounds": 6}]


def test_limiares_opostos_nao_atrapalham_a_decisao_em_meio():
    """Com P(fake) bruta, a média ficaria longe do ponto de decisão de cada um."""
    p, y, locutores = _cenario()
    linhas = compare_fusions(p, y, locutores, fixed_sets=[("a", "b")])
    assert _linha(linhas, "set:a+b")["accuracy_at_center"] > 0.85
    bruta = (np.mean([p["a"], p["b"]], axis=0) >= 0.5) == y
    assert bruta.mean() <= _linha(linhas, "set:a+b")["accuracy_at_center"]


def test_locutor_de_fora_nao_entra_no_limiar():
    """Um locutor com escala própria não pode ajustar o limiar que o mede."""
    y = np.tile([0, 1], 200)
    locutores = np.repeat(["L0", "L1", "L2", "L3"], 100)
    rng = np.random.default_rng(1)
    p = np.clip(0.2 + 0.4 * y + rng.normal(0, 0.05, 400), 0.01, 0.99)
    p[locutores == "L3"] = np.clip(p[locutores == "L3"] + 0.35, 0.01, 0.99)
    linhas = compare_fusions({"p": p}, y, locutores, top_ks=())
    # O deslocamento só aparece fora da amostra: o real de L3 passa do limiar dos outros.
    assert _linha(linhas, "model:p")["accuracy_at_center"] < 0.9


def test_regra_escolhe_olhando_so_os_locutores_de_ajuste():
    y = np.tile([0, 1], 100)
    locutores = np.repeat(["L0", "L1"], 100)
    bom_em_l0 = np.where(locutores == "L0", y.astype(float), 1.0 - y)
    bom_em_l1 = np.where(locutores == "L1", y.astype(float), 1.0 - y)
    calibracao = {"bom_em_l0": bom_em_l0, "bom_em_l1": bom_em_l1}
    escolher = top_k_chooser(calibracao, y, 1, calibracao)
    assert escolher(locutores == "L0") == ("bom_em_l0",)
    assert escolher(locutores == "L1") == ("bom_em_l1",)
    _, escolhas = held_out_scores(calibracao, calibracao, y, locutores, escolher)
    assert escolhas == {("bom_em_l0",): 1, ("bom_em_l1",): 1}


def test_ruido_mede_no_alvo_com_limiar_da_validacao_limpa():
    p, y, locutores = _cenario()
    ruidoso = {"a": np.clip(p["a"] * 0.5, 0.0, 1.0), "b": p["b"]}
    linhas = compare_fusions(
        p, y, locutores, {"10 dB": ruidoso}, fixed_sets=[("a", "b"), ("a", "ruido")]
    )
    assert _linha(linhas, "model:a", "10 dB")["accuracy_at_center"] < (
        _linha(linhas, "model:a")["accuracy_at_center"]
    )
    regras = {r["rule"] for r in linhas if r["condition"] == "10 dB"}
    # Sem medida de "ruido" nessa condição, nada que dependa dele aparece nela.
    assert "set:a+b" in regras
    assert "set:a+ruido" not in regras
    assert "model:ruido" not in regras
    assert "top-3" not in regras
