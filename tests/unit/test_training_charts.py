"""Testes da figura de curvas de treino do wizard (_history_figure).

Trava as melhorias da revisão dos gráficos de treinamento:
- eixo X em épocas 1-based (antes o plot final começava em 0);
- eixo Y de accuracy fixado em [0, 1.02];
- rótulo de eixo "Época";
- curva de validação só quando há dados (sem legenda/linha fantasma).
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")


def _hist_fig(*args):
    from app.interfaces.gradio.tabs.training_wizard import _history_figure

    return _history_figure(*args)


def _close(fig):
    from app.interfaces.gradio.tabs.training_wizard import close_fig

    close_fig(fig)


def test_epoch_axis_is_one_based_and_labeled():
    fig = _hist_fig([0.7, 0.5, 0.4], [0.8, 0.6, 0.5], [0.5, 0.7, 0.8], [0.4, 0.6, 0.7])
    loss_ax, acc_ax = fig.axes
    xticks = [int(t) for t in loss_ax.get_xticks() if float(t).is_integer()]
    assert min(xticks) >= 1, "eixo X de épocas deve começar em 1, não 0"
    assert 3 in xticks
    assert loss_ax.get_xlabel() == "Época"
    assert acc_ax.get_xlabel() == "Época"
    _close(fig)


def test_accuracy_axis_bounded_zero_one():
    fig = _hist_fig([0.7], [0.7], [0.5], [0.5])
    _loss_ax, acc_ax = fig.axes
    lo, hi = acc_ax.get_ylim()
    assert lo == 0.0 and 1.0 <= hi <= 1.1
    _close(fig)


def test_no_validation_no_ghost_legend():
    # val_loss/val_acc só com None → não deve criar curva/legenda "validação"
    fig = _hist_fig([0.7, 0.5], [None, None], [0.5, 0.6], [None, None])
    leg = fig.axes[0].get_legend()
    labels = [t.get_text() for t in leg.get_texts()] if leg else []
    assert labels == ["treino"]
    _close(fig)


def test_validation_curve_present_when_data():
    fig = _hist_fig([0.7, 0.5], [0.8, 0.6], [0.5, 0.6], [0.45, 0.55])
    leg = fig.axes[0].get_legend()
    labels = sorted(t.get_text() for t in leg.get_texts())
    assert labels == ["treino", "validação"]
    _close(fig)


def test_single_epoch_does_not_crash():
    fig = _hist_fig([0.6], [0.7], [0.55], [0.5])
    assert len(fig.axes) == 2
    _close(fig)
