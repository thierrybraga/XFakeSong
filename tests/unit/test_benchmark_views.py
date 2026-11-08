"""Formatação dos números da aba Benchmark.

SUJEITO: `app/interfaces/gradio/presenters/benchmark_views.py`. A taxa de
aprendizado tem formatador próprio: com duas casas fixas, o LR 1e-3 do Res2Net
aparecia como `0,00`.
"""

from __future__ import annotations

import pytest

from app.interfaces.gradio.presenters.benchmark_views import lr, num, pct


@pytest.mark.parametrize(
    ("valor", "esperado"),
    [(1e-3, "0,001"), (3e-4, "0,0003"), (1e-5, "1e-05"), (None, "—")],
)
def test_lr_nao_arredonda_a_taxa(valor, esperado):
    assert lr(valor) == esperado


def test_num_e_pct_em_portugues():
    assert num(0.98439, 4) == "0,9844"
    assert num(17_600_000, 0) == "17.600.000"
    assert pct(0.0116) == "1,16%"
    assert pct("x") == "—"
