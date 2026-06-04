"""Testes dos gráficos de busca de hiperparâmetros (Optuna).

Usam um `study` duck-typed, então rodam SEM optuna instalado: a convergência é
verificada, e o painel de importância cai para a nota de fallback.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")


class _FakeDir:
    def __init__(self, name):
        self.name = name


class _FakeTrial:
    def __init__(self, value):
        self.value = value


class _FakeStudy:
    def __init__(self, values, maximize=True):
        self.trials = [_FakeTrial(v) for v in values]
        self.direction = _FakeDir("MAXIMIZE" if maximize else "MINIMIZE")


def _close(fig):
    from app.interfaces.gradio.utils.plotting import close_fig

    close_fig(fig)


def test_convergence_best_is_monotonic_maximize():
    from app.interfaces.gradio.utils.tuning_charts import render_tuning_figure

    fig = render_tuning_figure(_FakeStudy([0.6, 0.55, 0.8, 0.7]))
    conv_ax = fig.axes[0]
    lines = conv_ax.get_lines()
    assert lines, "deve haver curvas no painel de convergência"
    best_ys = list(lines[-1].get_ydata())  # 'melhor acumulado'
    assert best_ys == sorted(best_ys), "melhor acumulado deve ser monotônico ↑"
    assert best_ys[-1] == 0.8
    assert conv_ax.get_xlabel() == "Trial"
    _close(fig)


def test_convergence_minimize_direction():
    from app.interfaces.gradio.utils.tuning_charts import render_tuning_figure

    fig = render_tuning_figure(_FakeStudy([0.9, 0.7, 0.8, 0.5], maximize=False))
    best_ys = list(fig.axes[0].get_lines()[-1].get_ydata())
    assert best_ys == sorted(best_ys, reverse=True), "minimize → monotônico ↓"
    assert best_ys[-1] == 0.5
    _close(fig)


def test_importance_fallback_without_optuna():
    from app.interfaces.gradio.utils.tuning_charts import render_tuning_figure

    fig = render_tuning_figure(_FakeStudy([0.5, 0.6]))
    assert len(fig.axes) == 2  # convergência + importância (com nota de fallback)
    _close(fig)


def test_empty_study_no_crash():
    from app.interfaces.gradio.utils.tuning_charts import render_tuning_figure

    fig = render_tuning_figure(_FakeStudy([]))
    assert len(fig.axes) == 2
    _close(fig)


def test_run_auto_tuning_degrades_without_optuna():
    from app.interfaces.gradio.tabs.optimization import run_auto_tuning

    status, best, fig = run_auto_tuning(
        "MultiscaleCNN", "/caminho/inexistente.npz", 5, "val_accuracy", 3
    )
    # optuna ausente neste ambiente → mensagem clara, sem best params nem figura
    assert "optuna" in status.lower()
    assert best == {} and fig is None
