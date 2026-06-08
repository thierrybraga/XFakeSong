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


def test_history_figure_from_binary_accuracy_history():
    from app.interfaces.gradio.tabs.training_wizard import (
        _history_figure_from_history,
    )

    history = {
        "loss": [0.7, 0.5],
        "val_loss": [0.8, 0.6],
        "binary_accuracy": [0.55, 0.75],
        "val_binary_accuracy": [0.5, 0.7],
    }
    fig = _history_figure_from_history(history, "binary_accuracy")
    loss_ax, acc_ax = fig.axes
    assert len(loss_ax.lines) == 2
    assert len(acc_ax.lines) == 2
    _close(fig)


def test_history_figure_from_sparse_accuracy_history():
    from app.interfaces.gradio.tabs.training_wizard import (
        _history_figure_from_history,
    )

    history = {
        "loss": [0.9, 0.4],
        "val_loss": [1.0, 0.5],
        "sparse_categorical_accuracy": [0.45, 0.8],
        "val_sparse_categorical_accuracy": [0.4, 0.72],
    }
    fig = _history_figure_from_history(history)
    loss_ax, acc_ax = fig.axes
    assert len(loss_ax.lines) == 2
    assert len(acc_ax.lines) == 2
    _close(fig)


def test_history_figure_from_history_without_validation_accuracy():
    from app.interfaces.gradio.tabs.training_wizard import (
        _history_figure_from_history,
    )

    history = {
        "loss": [0.9, 0.4],
        "accuracy": [0.45, 0.8],
    }
    fig = _history_figure_from_history(history)
    loss_ax, acc_ax = fig.axes
    assert len(loss_ax.lines) == 1
    assert len(acc_ax.lines) == 1
    _close(fig)


def test_history_confusion_figure_has_confusion_matrix_axis():
    from app.interfaces.gradio.tabs.training_wizard import (
        _history_confusion_figure_from_history,
    )

    history = {
        "loss": [0.9, 0.4],
        "val_loss": [1.0, 0.5],
        "accuracy": [0.45, 0.8],
        "val_accuracy": [0.4, 0.72],
    }
    fig = _history_confusion_figure_from_history(
        history,
        y_true=[0, 0, 1, 1],
        y_pred=[0, 1, 1, 1],
    )
    assert len(fig.axes) == 3
    assert fig.axes[2].get_title() == "Matriz de Confusão"
    labels = [txt.get_text() for txt in fig.axes[2].texts]
    assert labels == ["1", "1", "0", "2"]
    _close(fig)


def test_prediction_labels_support_sigmoid_outputs():
    from app.interfaces.gradio.tabs.training_wizard import (
        _prediction_labels_and_scores,
    )

    labels, scores = _prediction_labels_and_scores([[0.1], [0.7], [0.49], [0.5]])
    assert labels.tolist() == [0, 1, 0, 1]
    assert [round(float(s), 2) for s in scores] == [0.1, 0.7, 0.49, 0.5]


def test_prediction_labels_support_binary_logits():
    from app.interfaces.gradio.tabs.training_wizard import (
        _prediction_labels_and_scores,
    )

    labels, scores = _prediction_labels_and_scores([[-2.0], [2.0]])
    assert labels.tolist() == [0, 1]
    assert round(float(scores[0]), 2) == 0.12
    assert round(float(scores[1]), 2) == 0.88


def test_training_eval_figures_return_all_plots():
    from app.interfaces.gradio.tabs.training_wizard import _training_eval_figures

    figs = _training_eval_figures(
        y_true=[0, 0, 1, 1],
        y_pred=[0, 1, 1, 1],
        y_scores=[0.1, 0.55, 0.7, 0.9],
        lr_history=[1e-3, 5e-4],
    )
    assert set(figs) == {"roc", "cm", "pr", "det", "threshold", "class_acc", "lr"}
    assert figs["cm"].axes[0].get_title() == "Matriz de Confusão"
    assert figs["class_acc"].axes[0].get_title() == "Acurácia por Classe"
    for fig in figs.values():
        assert fig.axes
        _close(fig)


def test_training_eval_figures_single_class_do_not_crash():
    from app.interfaces.gradio.tabs.training_wizard import _training_eval_figures

    figs = _training_eval_figures(
        y_true=[0, 0, 0],
        y_pred=[0, 0, 1],
        y_scores=[0.1, 0.2, 0.8],
        lr_history=[],
    )
    assert figs["roc"].axes[0].get_title() == "Curva ROC"
    assert figs["det"].axes[0].get_title() == "Curva DET / EER"
    assert figs["threshold"].axes[0].get_title() == "Otimização de Threshold"
    for fig in figs.values():
        assert fig.axes
        _close(fig)
