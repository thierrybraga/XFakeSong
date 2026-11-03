"""Wizard de Treinamento — UI Fase 2.

Substitui as 5 sub-tabs aninhadas do training.py original por um workflow
linear de 4 steps. Cada step é um gr.Group cuja visibilidade é controlada
pelo estado atual, dando ao usuário uma sensação clara de progressão.

Steps:
    1. Dataset       — escolher pasta + validação automática de classes
    2. Modelo        — escolher arquitetura + variante (com cards informativos)
    3. Hiperparâmetros — formulário organizado (basic + advanced)
    4. Treinar       — execução com progress visual + logs em tempo real

Backend INTACTO: reutiliza `create_model_by_name` do registry e a mesma
lógica de binarização de labels (BUG.Training.4).
"""

from __future__ import annotations

import logging
import threading as _threading
from pathlib import Path
from typing import List

import gradio as gr

from app.domain.services.interactive_training import (
    CLASSICAL_FEATURE_TYPES,
    ClassicalCompleted,
    ClassicalDatasetInsufficient,
    ClassicalExtractionProgress,
    ClassicalFitting,
    ClassicalStarted,
    ClassicalTooFewFeatures,
    DatasetLoading,
    DatasetReady,
    EpochCompleted,
    FitCompleted,
    FitHeartbeat,
    FitStarted,
    SmokeTestFailed,
    TrainedModel,
    TrainingFailed,
    save_trained_model,
)
from app.domain.services.interactive_training import (
    REAL_CLASS_PATTERN as _REAL_PATTERNS,
)
from app.domain.services.interactive_training import (
    history_series as _history_series,
)
from app.domain.services.interactive_training import (
    normalize_true_labels as _normalize_true_labels,
)
from app.domain.services.interactive_training import (
    resolve_accuracy_history_keys as _resolve_accuracy_history_keys,
)
from app.interfaces.gradio.utils.components import ui_safe
from app.interfaces.gradio.utils.notifications import (
    CommonErrors,
    notify_from_actionable,
    notify_success,
)
from app.interfaces.gradio.utils.plotting import (
    PLOT_ACCENT,
    PLOT_DANGER,
    close_fig,
    new_figure,
    safe_tight_layout,
    style_ax,
)
from app.interfaces.gradio.utils.training_wizard_presenter import (
    get_model_catalog as _get_model_catalog,
)
from app.interfaces.gradio.utils.training_wizard_presenter import (
    render_model_cards_html as _render_model_cards_html,
)
from app.interfaces.gradio.utils.training_wizard_presenter import (
    step_visibility as _step_visibility,
)
from app.interfaces.gradio.utils.training_wizard_presenter import (
    stepper_html as _stepper_html,
)

logger = logging.getLogger("gradio_training_wizard")

# Holder do ÚLTIMO modelo treinado com sucesso nesta sessão do servidor.
# O modelo Keras é criado dentro do gerador do treino e seria coletado pelo GC
# ao terminar — guardamos a referência aqui para o botão "Salvar" do Step 4
# conseguir persisti-lo. (App local single-user: holder de slot único.)
_LAST_TRAINED: TrainedModel | None = None


def _save_trained_model(name: str) -> tuple[bool, str]:
    """Persiste o último modelo treinado no diretório de modelos do app.

    A gravação (.keras/.pkl + _config.json com o input_contract) é do domínio,
    em ``interactive_training.save_trained_model``; aqui ficam a mensagem e o
    recarregamento do DetectionService. O diretório é o de
    ``OperationalPaths`` — o mesmo que o ModelLoader lê.

    Returns (ok, mensagem_markdown).
    """
    if _LAST_TRAINED is None or _LAST_TRAINED.model is None:
        return False, (
            "⚠️ Nenhum modelo treinado nesta sessão. "
            "Conclua um treino no Step 4 antes de salvar."
        )

    try:
        from app.core.bootstrap import OperationalPaths

        salvo = save_trained_model(
            _LAST_TRAINED, name, OperationalPaths.resolve().models
        )
    except FileExistsError as e:
        return False, (
            f"⚠️ Já existe um modelo chamado **{e.args[0]}**. "
            "Escolha outro nome para não sobrescrever."
        )
    except Exception as e:
        logger.error(f"Falha ao salvar modelo: {e}", exc_info=True)
        return False, f"❌ Erro ao salvar: {e}"

    # Recarrega os modelos no DetectionService para uso imediato em Detectar.
    reload_note = " Reinicie ou abra a aba Detectar para carregá-lo."
    try:
        from app.dependencies import get_detection_service

        get_detection_service().reload_models()
        reload_note = " Já disponível na aba **🎯 Detectar**."
    except Exception as e:
        logger.debug(f"Reload do DetectionService pulado: {e}")

    return True, (
        f"✅ Modelo salvo como **{salvo.slug}** "
        f"(`{salvo.model_path.name}`, {salvo.size_mb:.1f} MB).{reload_note}"
    )


# =====================================================================
# Helpers de feedback ao vivo durante o treinamento (Step 4)
# =====================================================================


def _fmt_secs(s: float) -> str:
    """Formata segundos como mm:ss (ou h:mm:ss se > 1h)."""
    s = int(max(0, s))
    h, rem = divmod(s, 3600)
    m, sec = divmod(rem, 60)
    if h:
        return f"{h}:{m:02d}:{sec:02d}"
    return f"{m:02d}:{sec:02d}"


def _metric_card(label: str, value, kind: str = "neutral", fmt: str = "{:.4f}") -> str:
    """Card compacto de uma métrica (loss/acc) com cor semântica."""
    import math

    if value is None:
        shown, state = "—", "muted"
    elif isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        shown, state = "NaN", "bad"
    else:
        shown = fmt.format(value)
        state = kind if kind in {"good", "bad", "accent", "neutral"} else "neutral"
    return (
        f'<div class="tl-card">'
        f'<div class="tl-card-label">{label}</div>'
        f'<div class="tl-card-value tl-card-value-{state}">{shown}</div>'
        f"</div>"
    )


def _history_figure(train_loss, val_loss, train_acc, val_acc):
    """Figura canônica das curvas de treino (Loss | Accuracy).

    Usada tanto no plot AO VIVO quanto no FINAL, para garantir consistência:
    - Eixo X em épocas 1-based ("Época"), com ticks inteiros em treinos curtos.
    - Curva de validação só aparece quando há dados (evita legenda 'val' vazia
      e linha fantasma).
    - Eixo Y de accuracy fixado em [0, 1.02] — a acurácia é uma fração, então a
      auto-escala fazia variações minúsculas parecerem enormes.

    Retorna a `Figure` (o chamador fecha com close_fig para evitar leak).
    """

    def _clean(seq):
        return [v for v in (seq or []) if v is not None]

    tl, vl = _clean(train_loss), _clean(val_loss)
    ta, va = _clean(train_acc), _clean(val_acc)

    fig, ax = new_figure(1, 2, figsize=(12, 4))

    # ── Loss ──
    style_ax(ax[0], fig, "Perda")
    ax[0].plot(
        range(1, len(tl) + 1), tl, label="treino", color=PLOT_ACCENT, marker="o", ms=3
    )
    if vl:
        ax[0].plot(
            range(1, len(vl) + 1),
            vl,
            label="validação",
            color=PLOT_DANGER,
            marker="o",
            ms=3,
        )
    ax[0].set_xlabel("Época")
    if 1 <= len(tl) <= 20:
        ax[0].set_xticks(list(range(1, len(tl) + 1)))
    ax[0].legend()

    # ── Accuracy ──
    style_ax(ax[1], fig, "Acurácia")
    ax[1].plot(
        range(1, len(ta) + 1), ta, label="treino", color=PLOT_ACCENT, marker="o", ms=3
    )
    if va:
        ax[1].plot(
            range(1, len(va) + 1),
            va,
            label="validação",
            color=PLOT_DANGER,
            marker="o",
            ms=3,
        )
    ax[1].set_xlabel("Época")
    ax[1].set_ylim(0.0, 1.02)
    if 1 <= len(ta) <= 20:
        ax[1].set_xticks(list(range(1, len(ta) + 1)))
    ax[1].legend()

    safe_tight_layout(fig)
    return fig


def _history_figure_from_history(history, preferred_train_acc_key: str = None):
    """Cria o gráfico canônico diretamente de um dict/History do Keras."""
    train_acc_key, val_acc_key = _resolve_accuracy_history_keys(
        history, preferred_train_acc_key
    )
    return _history_figure(
        _history_series(history, "loss"),
        _history_series(history, "val_loss"),
        _history_series(history, train_acc_key) if train_acc_key else [],
        _history_series(history, val_acc_key) if val_acc_key else [],
    )


def _plot_confusion_matrix_on_axis(ax, fig, y_true, y_pred, title="Matriz de Confusão"):
    """Renderiza matriz de confusão binária em eixo existente."""
    import numpy as np
    from sklearn.metrics import confusion_matrix

    cm = confusion_matrix(
        _normalize_true_labels(y_true),
        _normalize_true_labels(y_pred),
        labels=[0, 1],
    )
    style_ax(ax, fig, title)
    ax.grid(False)
    ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["real", "fake"])
    ax.set_yticklabels(["real", "fake"])
    ax.set_xlabel("Predito")
    ax.set_ylabel("Verdadeiro")
    cm_max = int(np.max(cm)) if cm.size and np.max(cm) > 0 else 1
    for row in range(2):
        for col in range(2):
            color = "#f1f5f9" if cm[row, col] > cm_max / 2 else "#0f172a"
            ax.text(
                col,
                row,
                str(int(cm[row, col])),
                ha="center",
                va="center",
                color=color,
                fontweight="bold",
            )
    return cm


def _message_figure(title: str, message: str):
    """Figura fallback para avaliação indisponível sem quebrar a UI."""

    fig, ax = new_figure(figsize=(8, 5))
    style_ax(ax, fig, title)
    ax.text(
        0.5,
        0.5,
        message,
        ha="center",
        va="center",
        transform=ax.transAxes,
        color="#f1f5f9",
        wrap=True,
    )
    ax.set_xticks([])
    ax.set_yticks([])
    safe_tight_layout(fig)
    return fig


def _roc_figure(y_true, y_scores):
    import numpy as np
    from sklearn.metrics import auc, roc_curve

    y_true = _normalize_true_labels(y_true)
    y_scores = np.asarray(y_scores, dtype="float32").reshape(-1)
    if len(np.unique(y_true)) < 2:
        return _message_figure(
            "Curva ROC", "ROC indisponível: validação tem uma única classe."
        )
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)
    fig, ax = new_figure(figsize=(8, 5))
    style_ax(ax, fig, "Curva ROC")
    ax.plot(fpr, tpr, color=PLOT_ACCENT, lw=2, label=f"AUC = {roc_auc:.3f}")
    ax.plot([0, 1], [0, 1], color="#334155", lw=1.5, linestyle="--")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("Taxa de falsos positivos")
    ax.set_ylabel("Taxa de verdadeiros positivos")
    ax.legend()
    safe_tight_layout(fig)
    return fig


def _precision_recall_figure(y_true, y_scores):
    import numpy as np
    from sklearn.metrics import auc, precision_recall_curve

    y_true = _normalize_true_labels(y_true)
    y_scores = np.asarray(y_scores, dtype="float32").reshape(-1)
    if len(np.unique(y_true)) < 2:
        return _message_figure(
            "Curva precisão-revocação",
            "Curva precisão-revocação indisponível: validação tem uma única classe.",
        )
    precision, recall, _ = precision_recall_curve(y_true, y_scores)
    pr_auc = auc(recall, precision)
    fig, ax = new_figure(figsize=(8, 5))
    style_ax(ax, fig, "Curva precisão-revocação")
    ax.plot(recall, precision, color=PLOT_ACCENT, lw=2, label=f"AUC = {pr_auc:.3f}")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("Revocação")
    ax.set_ylabel("Precisão")
    ax.legend()
    safe_tight_layout(fig)
    return fig


def _det_figure(y_true, y_scores):
    import numpy as np

    y_true = _normalize_true_labels(y_true)
    if len(np.unique(y_true)) < 2:
        return _message_figure(
            "Curva DET / EER", "DET indisponível: validação tem uma única classe."
        )
    try:
        from app.interfaces.gradio.presenters.forensic_plots import (
            TrainingAnalyticsVisualizer,
        )

        fig, _eer = TrainingAnalyticsVisualizer().plot_det_curve_eer(
            y_true, np.asarray(y_scores, dtype="float32").reshape(-1)
        )
        return fig
    except Exception as exc:
        return _message_figure("Curva DET / EER", f"Falha ao gerar DET/EER:\n{exc}")


def _threshold_figure(y_true, y_scores):
    import numpy as np

    y_true = _normalize_true_labels(y_true)
    if len(np.unique(y_true)) < 2:
        return _message_figure(
            "Otimização do limiar de decisão",
            "Limiar indisponível: validação tem uma única classe.",
        )
    try:
        from app.interfaces.gradio.presenters.forensic_plots import (
            TrainingAnalyticsVisualizer,
        )

        return TrainingAnalyticsVisualizer().plot_threshold_optimization(
            y_true, np.asarray(y_scores, dtype="float32").reshape(-1)
        )
    except Exception as exc:
        return _message_figure(
            "Otimização do limiar de decisão",
            f"Falha ao calcular o limiar de decisão:\n{exc}",
        )


def _class_accuracy_figure(y_true, y_pred):
    import numpy as np

    y_true = _normalize_true_labels(y_true)
    y_pred = _normalize_true_labels(y_pred)
    scores = []
    for cls in (0, 1):
        mask = y_true == cls
        scores.append(float(np.mean(y_pred[mask] == cls)) if np.any(mask) else 0.0)
    fig, ax = new_figure(figsize=(7, 5))
    style_ax(ax, fig, "Acurácia por Classe")
    ax.bar(["real", "fake"], scores, color=[PLOT_ACCENT, PLOT_DANGER])
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Acurácia")
    for i, score in enumerate(scores):
        ax.text(i, min(1.0, score + 0.03), f"{score:.3f}", ha="center")
    safe_tight_layout(fig)
    return fig


def _standalone_confusion_figure(y_true, y_pred):

    fig, ax = new_figure(figsize=(7, 5))
    _plot_confusion_matrix_on_axis(ax, fig, y_true, y_pred, "Matriz de Confusão")
    safe_tight_layout(fig)
    return fig


def _lr_schedule_figure(lr_history):
    if not lr_history:
        return _message_figure(
            "Evolução da taxa de aprendizado",
            "Learning-rate schedule indisponível: nenhuma época concluída.",
        )
    try:
        from app.interfaces.gradio.presenters.forensic_plots import (
            TrainingAnalyticsVisualizer,
        )

        return TrainingAnalyticsVisualizer().plot_lr_schedule(lr_history)
    except Exception as exc:
        return _message_figure(
            "Evolução da taxa de aprendizado",
            f"Falha ao gerar o gráfico da taxa de aprendizado:\n{exc}",
        )


def _training_eval_figures(y_true, y_pred, y_scores, lr_history=None):
    """Conjunto de gráficos de avaliação pós-treino."""
    return {
        "roc": _roc_figure(y_true, y_scores),
        "cm": _standalone_confusion_figure(y_true, y_pred),
        "pr": _precision_recall_figure(y_true, y_scores),
        "det": _det_figure(y_true, y_scores),
        "threshold": _threshold_figure(y_true, y_scores),
        "class_acc": _class_accuracy_figure(y_true, y_pred),
        "lr": _lr_schedule_figure(lr_history or []),
    }


def _history_confusion_figure_from_history(
    history,
    y_true,
    y_pred,
    preferred_train_acc_key: str = None,
):
    """Figura final canônica: Loss, Accuracy e Matriz de Confusão."""

    train_acc_key, val_acc_key = _resolve_accuracy_history_keys(
        history, preferred_train_acc_key
    )
    tl = [v for v in _history_series(history, "loss") if v is not None]
    vl = [v for v in _history_series(history, "val_loss") if v is not None]
    ta = [
        v
        for v in (_history_series(history, train_acc_key) if train_acc_key else [])
        if v is not None
    ]
    va = [
        v
        for v in (_history_series(history, val_acc_key) if val_acc_key else [])
        if v is not None
    ]

    fig, ax = new_figure(1, 3, figsize=(16, 4))

    style_ax(ax[0], fig, "Perda")
    ax[0].plot(
        range(1, len(tl) + 1), tl, label="treino", color=PLOT_ACCENT, marker="o", ms=3
    )
    if vl:
        ax[0].plot(
            range(1, len(vl) + 1),
            vl,
            label="validação",
            color=PLOT_DANGER,
            marker="o",
            ms=3,
        )
    ax[0].set_xlabel("Época")
    if 1 <= len(tl) <= 20:
        ax[0].set_xticks(list(range(1, len(tl) + 1)))
    ax[0].legend()

    style_ax(ax[1], fig, "Acurácia")
    ax[1].plot(
        range(1, len(ta) + 1), ta, label="treino", color=PLOT_ACCENT, marker="o", ms=3
    )
    if va:
        ax[1].plot(
            range(1, len(va) + 1),
            va,
            label="validação",
            color=PLOT_DANGER,
            marker="o",
            ms=3,
        )
    ax[1].set_xlabel("Época")
    ax[1].set_ylim(0.0, 1.02)
    if 1 <= len(ta) <= 20:
        ax[1].set_xticks(list(range(1, len(ta) + 1)))
    ax[1].legend()

    _plot_confusion_matrix_on_axis(ax[2], fig, y_true, y_pred, "Matriz de Confusão")
    safe_tight_layout(fig)
    return fig


#: Pedido de interrupção do treino em andamento.
#:
#: Um treino do assistente pode levar horas — o RawGAT-ST chega a ~54 h de GPU
#: no orçamento de 100 épocas. Sem isto, quem clicasse "Treinar" por engano
#: ficava sem saída pela interface: a thread do `fit` é daemon e só morre com o
#: processo. O Keras não expõe cancelamento; o caminho suportado é um callback
#: marcar `model.stop_training` ao fim de uma época.
_PEDIDO_DE_PARADA = _threading.Event()


def _estimativa_de_treino(arch: str, epochs: int, n_amostras: int) -> str:
    """Custo esperado ANTES de começar, na linguagem do usuário.

    Reaproveita as horas medidas em `benchmarks.planning`, as mesmas que
    derivam o timeout por modelo do benchmark. Saber de antemão que uma
    configuração leva dias é o que evita descobrir isso na terceira hora.
    """
    try:
        from benchmarks.planning import (
            _REFERENCE_EPOCHS,
            _REFERENCE_FIT_SAMPLES,
            EXPECTED_TRAINING_HOURS,
            _compact,
        )

        custo = EXPECTED_TRAINING_HOURS.get(_compact(arch))
        if not custo:
            return ""
        try:
            import tensorflow as tf

            perfil = "gpu" if tf.config.list_physical_devices("GPU") else "cpu"
        except Exception:  # noqa: BLE001
            perfil = "cpu"

        horas = custo[perfil]
        horas *= max(1, int(epochs)) / _REFERENCE_EPOCHS
        horas *= max(1, int(n_amostras)) / _REFERENCE_FIT_SAMPLES
        if horas < 1 / 60:
            return ""
        texto = _fmt_secs(horas * 3600)
        aviso = " — considere reduzir as épocas" if horas > 6 else ""
        # separador de milhar em pt-BR, sem tocar na vírgula da frase
        amostras = f"{n_amostras:,}".replace(",", ".")
        return (
            f"Estimativa: <b>{texto}</b> em {perfil.upper()} "
            f"({epochs} épocas, {amostras} amostras){aviso}."
        )
    except Exception:  # noqa: BLE001 — estimativa e opcional
        return ""


def _train_status_html(
    arch: str,
    device: str,
    epoch: int,
    total: int,
    hist: dict,
    elapsed_s: float,
    phase: str = "running",
    note: str = "",
) -> str:
    """Painel HTML rico de progresso do treino ao vivo.

    phase: "preparing" | "running" | "done" | "error"
    hist: {"loss":[...], "acc":[...], "val_loss":[...], "val_acc":[...]}
    """
    import math

    def _last(key):
        seq = hist.get(key) or []
        return seq[-1] if seq else None

    loss, acc = _last("loss"), _last("acc")
    vloss, vacc = _last("val_loss"), _last("val_acc")

    pct = (epoch / total * 100.0) if total else 0.0
    pct = max(0.0, min(100.0, pct))

    # ETA: extrapola pelo tempo médio por época concluída
    eta_txt = "—"
    if phase == "running" and epoch > 0 and epoch < total and elapsed_s > 0:
        per_epoch = elapsed_s / epoch
        eta_txt = _fmt_secs(per_epoch * (total - epoch))

    # Estado/Cabeçalho
    if phase == "preparing":
        head_icon, head_txt = "⏳", "Preparando…"
        pct_disp = 4.0  # barra indeterminada-ish
    elif phase == "done":
        head_icon, head_txt = "✓", "Concluído"
        pct_disp = 100.0
    elif phase == "error":
        head_icon, head_txt = "✗", "Falhou"
        pct_disp = 100.0
    else:  # running
        head_icon, head_txt = (
            "▶",
            f"Treinando · época {epoch}/{total}",
        )
        pct_disp = pct

    # cor das métricas
    acc_kind = (
        "good"
        if (isinstance(acc, float) and not math.isnan(acc) and acc >= 0.5)
        else "neutral"
    )
    vacc_kind = (
        "good"
        if (isinstance(vacc, float) and not math.isnan(vacc) and vacc >= 0.5)
        else "neutral"
    )
    loss_kind = (
        "bad"
        if (isinstance(loss, float) and (math.isnan(loss) or math.isinf(loss)))
        else "neutral"
    )
    vloss_kind = (
        "bad"
        if (isinstance(vloss, float) and (math.isnan(vloss) or math.isinf(vloss)))
        else "accent"
    )

    # SINAL DE SOBREAJUSTE, ao vivo.
    #
    # O painel mostrava loss/acc de treino e validação lado a lado, mas cabia
    # ao usuário perceber a divergência olhando quatro números por época. É a
    # informação mais acionável durante um treino longo: quando a val_loss para
    # de melhorar e a de treino continua caindo, o resto das épocas só piora o
    # modelo — e, com orçamento fixo de época, isso pode custar horas.
    #
    # Critério: épocas decorridas desde o melhor val_loss. Só a partir de 5
    # épocas, para não alarmar com a oscilação normal do início.
    vloss_hist = [
        v for v in (hist.get("val_loss") or [])
        if isinstance(v, (int, float)) and not math.isnan(v)
    ]
    aviso_overfit = ""
    if phase == "running" and len(vloss_hist) >= 5:
        melhor = min(vloss_hist)
        desde_o_melhor = len(vloss_hist) - 1 - vloss_hist.index(melhor)
        if desde_o_melhor >= 5:
            aviso_overfit = (
                f'<div class="tl-note" style="color:#f59e0b">'
                f"⚠ val_loss sem melhorar há {desde_o_melhor} épocas "
                f"(melhor: {melhor:.4f}). O melhor checkpoint já está salvo; "
                f"interromper agora não perde o melhor modelo.</div>"
            )

    note_html = f'<div class="tl-note">{note}</div>' if note else ""
    note_html += aviso_overfit

    return f"""
    <div class="train-live train-live-{phase}">
      <div class="tl-head">
        <span class="tl-head-title tl-head-title-{phase}">{head_icon} {head_txt}</span>
        <span class="tl-head-meta">{arch} · {device} · ⏱ {_fmt_secs(elapsed_s)}
          {("· ETA " + eta_txt) if eta_txt != "—" else ""}</span>
      </div>
      <progress class="tl-progress tl-progress-{phase}" value="{pct_disp:.1f}" max="100"></progress>
      <div class="tl-cards">
        {_metric_card("loss", loss, loss_kind)}
        {_metric_card("accuracy", acc, acc_kind)}
        {_metric_card("val_loss", vloss, vloss_kind)}
        {_metric_card("val_accuracy", vacc, vacc_kind)}
      </div>
      {note_html}
    </div>
    """


# =====================================================================
# Helpers de dados (validação)
# =====================================================================


def _scan_dataset(path_str: str) -> dict:
    """Inspeciona um diretório de dataset.

    Returns:
        {
            "ok": bool,
            "message": str (mensagem amigável),
            "class_names": list[str],
            "binary_map": dict[int, int],  # idx_dir -> {0 real, 1 fake}
            "file_count": int,
            "files_per_class": dict[str, int],
            "n_real": int,
            "n_fake": int,
        }
    """
    out = {
        "ok": False,
        "message": "",
        "class_names": [],
        "binary_map": {},
        "file_count": 0,
        "files_per_class": {},
        "n_real": 0,
        "n_fake": 0,
    }
    p = Path(path_str)
    if not p.exists():
        out["message"] = f"❌ Pasta `{path_str}` não existe."
        return out
    if not p.is_dir():
        out["message"] = f"❌ `{path_str}` não é um diretório."
        return out

    subdirs = sorted([d.name for d in p.iterdir() if d.is_dir()])
    if not subdirs:
        out["message"] = (
            f"❌ Pasta vazia ou sem subdiretórios de classes.\n\n"
            f"Estrutura esperada:\n"
            f"```\n{path_str}/\n  ├─ real/\n  │   ├─ audio1.wav\n  │   └─ ...\n"
            f"  └─ fake/\n      ├─ audio2.wav\n      └─ ...\n```"
        )
        return out

    # Conta arquivos .wav por subdir
    files_per_class = {}
    binary_map = {}
    n_real = n_fake = 0
    for idx, name in enumerate(subdirs):
        wavs = list((p / name).glob("*.wav"))
        files_per_class[name] = len(wavs)
        is_real = bool(_REAL_PATTERNS.match(name.strip()))
        binary_map[idx] = 0 if is_real else 1
        if is_real:
            n_real += files_per_class[name]
        else:
            n_fake += files_per_class[name]

    total = sum(files_per_class.values())
    if total == 0:
        out["message"] = "❌ Nenhum arquivo .wav encontrado nos subdiretórios."
        return out

    real_dirs = [n for n in subdirs if _REAL_PATTERNS.match(n.strip())]
    fake_dirs = [n for n in subdirs if not _REAL_PATTERNS.match(n.strip())]

    if not real_dirs:
        out["message"] = (
            f"❌ Nenhum subdiretório de áudios REAIS encontrado.\n\n"
            f"Subdiretórios devem começar com: `real`, `bonafide`, `genuine`, "
            f"`original`, `authentic` ou `natural`.\n\n"
            f"Encontrei: {subdirs}"
        )
        return out
    if not fake_dirs:
        out["message"] = (
            f"❌ Nenhum subdiretório de áudios FAKE encontrado.\n\n"
            f"Apenas subdiretórios de áudios reais detectados: {real_dirs}"
        )
        return out

    out.update(
        {
            "ok": True,
            "class_names": subdirs,
            "binary_map": binary_map,
            "file_count": total,
            "files_per_class": files_per_class,
            "n_real": n_real,
            "n_fake": n_fake,
        }
    )

    # Mensagem detalhada
    ratio = n_real / max(n_fake, 1)
    if ratio < 0.5 or ratio > 2:
        balance = f"⚠️ Desbalanceado ({ratio:.2f}:1) — class weighting será aplicado"
    else:
        balance = f"✓ Balanceado ({ratio:.2f}:1)"

    detail = "\n".join(
        f"  • {name} ({'real' if binary_map[idx] == 0 else 'fake'}): "
        f"{files_per_class[name]} arquivos"
        for idx, name in enumerate(subdirs)
    )
    out["message"] = (
        f"✓ **Dataset válido** — {total} arquivos em {len(subdirs)} subdiretórios\n\n"
        f"{detail}\n\n"
        f"**Total**: {n_real} real / {n_fake} fake — {balance}"
    )
    return out


# =====================================================================
# Step 4: training execution
# =====================================================================


def _training_error_panel(error: Exception) -> str:
    return (
        '<div class="train-live train-live-error">'
        '<div class="tl-head"><span class="tl-head-title tl-head-title-error">'
        "✗ Treinamento falhou</span></div>"
        '<div class="tl-note tl-note-error">'
        f"{str(error)[:300]}</div></div>"
    )


def _fmt_metric(v) -> str:
    return "nan" if (v is None or (isinstance(v, float) and v != v)) else f"{v:.4f}"


def _classical_figure(ev: "ClassicalCompleted"):
    """Matriz de confusão + (RF) importância das features ou distribuição."""
    import numpy as np

    cm = ev.confusion
    feat_names = ev.trained.feature_names
    fig, ax = new_figure(1, 2, figsize=(12, 4))
    style_ax(ax[0], fig, "Matriz de Confusão (val)")
    ax[0].imshow(cm, cmap="Blues")
    ax[0].set_xticks([0, 1])
    ax[0].set_yticks([0, 1])
    ax[0].set_xticklabels(["real", "fake"])
    ax[0].set_yticklabels(["real", "fake"])
    ax[0].set_xlabel("Predito")
    ax[0].set_ylabel("Verdadeiro")
    cm_max = int(cm.max()) if cm.size and cm.max() > 0 else 1
    for r in range(2):
        for c in range(2):
            # Contraste por célula: texto CLARO nas células escuras (contagem
            # alta) e ESCURO nas claras (contagem baixa). Antes era sempre
            # claro → o número ficava invisível nas células de baixa contagem
            # do colormap Blues.
            txt_color = "#f1f5f9" if cm[r, c] > cm_max / 2 else "#0f172a"
            ax[0].text(
                c,
                r,
                str(cm[r, c]),
                ha="center",
                va="center",
                color=txt_color,
                fontweight="bold",
            )
    if ev.feature_importances is not None:
        style_ax(ax[1], fig, "Top-15 features (importância)")
        imp = ev.feature_importances
        idx = np.argsort(imp)[::-1][:15]
        ax[1].barh(range(len(idx))[::-1], imp[idx], color=PLOT_ACCENT)
        labels = [
            (feat_names[j] if feat_names and j < len(feat_names) else f"f{j}")
            for j in idx
        ]
        ax[1].set_yticks(range(len(idx))[::-1])
        ax[1].set_yticklabels(labels, fontsize=7)
    else:
        style_ax(ax[1], fig, "Distribuição de classes (treino)")
        ax[1].bar(
            ["real", "fake"],
            ev.train_class_counts,
            color=[PLOT_ACCENT, PLOT_DANGER],
        )
    safe_tight_layout(fig)
    return fig


def _eval_outputs(eval_figs: dict) -> tuple:
    return (
        eval_figs["roc"],
        eval_figs["cm"],
        eval_figs["pr"],
        eval_figs["det"],
        eval_figs["threshold"],
        eval_figs["class_acc"],
        eval_figs["lr"],
    )


_DONE_NOTE = (
    "✓ Treino concluído — dê um nome e clique em "
    "<b>💾 Salvar Modelo</b> abaixo para usá-lo na detecção."
)


def _run_training(
    arch: str,
    dataset_path: str,
    epochs: int,
    batch_size: int,
    lr: float,
    use_class_weighting: bool,
    progress=gr.Progress(),  # noqa: B008 - o Gradio injeta o rastreador por este default
):
    """Roda o treino do domínio e traduz cada evento nas saídas da etapa 4.

    O pipeline (dados, front-end, treino, avaliação) está em
    ``app.domain.services.interactive_training``; aqui ficam painel, log,
    gráficos e notificações. Cada ``yield`` é ``(painel, log, gráfico)`` ou,
    no fim, os mesmos três seguidos dos 7 gráficos de avaliação. A parada vem
    do botão "Interromper" por ``_PEDIDO_DE_PARADA``.
    """
    global _LAST_TRAINED
    from app.dependencies import get_interactive_training_service

    eventos = get_interactive_training_service().train(
        arch,
        dataset_path,
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        use_class_weighting=use_class_weighting,
        stop_event=_PEDIDO_DE_PARADA,
        on_progress=lambda fracao, descricao: progress(fracao, desc=descricao),
    )

    header_lines: List[str] = []
    epoch_lines: List[str] = []
    live_fig = None
    device = ""
    for ev in eventos:
        if isinstance(ev, DatasetLoading):
            yield (
                "Carregando dataset...",
                f"[treino v3: float32+clipvalue] Iniciando treino: {ev.arch}, "
                f"{ev.epochs} épocas, batch={ev.batch_size}",
                None,
            )
        elif isinstance(ev, DatasetReady):
            yield (
                "Criando modelo...",
                f"Input type: {ev.input_type} | shape: {ev.input_shape}\n"
                f"Mapeamento detectado: {len(ev.class_names)} subdirs -> binário\n"
                f"{', '.join(f'{n}->{ev.binary_map[i]}' for i, n in enumerate(ev.class_names))}",
                None,
            )
        elif isinstance(ev, SmokeTestFailed):
            yield (
                "Smoke test falhou",
                f"❌ Treinamento abortado ANTES da época 1:\n\n{ev.message}",
                None,
            )
        elif isinstance(ev, FitStarted):
            device = ev.device
            header_lines = [
                f"Modelo: {ev.arch} ({ev.n_params:,} params)",
                f"Treino: {ev.n_train_batches} batches · "
                f"Validação: {ev.n_val_batches} batches",
                f"Class weight: {ev.class_weight or 'desabilitado'} · "
                f"Device: {ev.device}",
                "─" * 48,
            ]
            yield (
                _train_status_html(
                    ev.arch,
                    ev.device,
                    0,
                    ev.epochs,
                    ev.history,
                    0.0,
                    phase="preparing",
                    note="Compilando o grafo e iniciando a 1ª época "
                    "(pode levar alguns segundos)…",
                ),
                "\n".join(header_lines + ["Iniciando treinamento…"]),
                None,
            )
        elif isinstance(ev, FitHeartbeat):
            if ev.epochs_done > 0:
                yield (
                    _train_status_html(
                        arch,
                        device,
                        ev.epochs_done,
                        ev.epochs,
                        ev.history,
                        ev.elapsed,
                        phase="running",
                    ),
                    "\n".join(header_lines + epoch_lines),
                    gr.update(),  # não recria o plot no heartbeat
                )
            else:
                # 1ª época ainda em andamento — relógio vivo + aviso
                yield (
                    _train_status_html(
                        arch,
                        device,
                        0,
                        ev.epochs,
                        ev.history,
                        ev.elapsed,
                        phase="preparing",
                        note="Treinando a 1ª época… na CPU pode levar "
                        "vários minutos (áudio bruto). O relógio acima "
                        "confirma que está rodando.",
                    ),
                    "\n".join(
                        header_lines
                        + [f"⏱ {_fmt_secs(ev.elapsed)} — 1ª época em andamento…"]
                    ),
                    gr.update(),
                )
        elif isinstance(ev, EpochCompleted):
            m = ev.metrics
            epoch_lines.append(
                f"Época {ev.epoch:>3}/{ev.epochs} — loss {_fmt_metric(m.get('loss'))} · "
                f"acc {_fmt_metric(m.get('acc'))} · "
                f"val_loss {_fmt_metric(m.get('val_loss'))} · "
                f"val_acc {_fmt_metric(m.get('val_acc'))}"
            )
            # Plot ao vivo (fecha o anterior p/ evitar leak de memória)
            close_fig(live_fig)
            h = ev.history
            live_fig = _history_figure(h["loss"], h["val_loss"], h["acc"], h["val_acc"])
            yield (
                _train_status_html(
                    arch, device, ev.epoch, ev.epochs, h, ev.elapsed, phase="running"
                ),
                "\n".join(header_lines + epoch_lines),
                live_fig,
            )
        elif isinstance(ev, FitCompleted):
            eval_figs = _training_eval_figures(
                ev.y_true, ev.y_pred, ev.y_score, ev.lr_history
            )
            close_fig(live_fig)
            fig = _history_confusion_figure_from_history(
                ev.history, ev.y_true, ev.y_pred, ev.train_acc_key
            )
            progress(1.0, desc="Concluído!")
            notify_success(
                f"Treino de {arch} concluído",
                message=f"val_acc={ev.final_acc:.4f}, val_loss={ev.final_loss:.4f}",
            )
            # Guarda o modelo treinado para o botão "Salvar" do Step 4.
            _LAST_TRAINED = ev.trained
            yield (
                _train_status_html(
                    arch,
                    ev.device,
                    ev.epochs,
                    ev.epochs,
                    ev.live_history,
                    ev.elapsed,
                    phase="done",
                    note=_DONE_NOTE,
                ),
                "\n".join(
                    header_lines
                    + epoch_lines
                    + [
                        "─" * 48,
                        f"✓ Treino concluído — val_accuracy={ev.final_acc:.4f}, "
                        f"val_loss={ev.final_loss:.4f}",
                        "Matriz de confusão gerada com o conjunto de validação.",
                    ]
                ),
                fig,
                *_eval_outputs(eval_figs),
            )
        elif isinstance(ev, ClassicalStarted):
            yield (
                _train_status_html(
                    ev.arch,
                    "CPU",
                    0,
                    1,
                    {},
                    0.0,
                    phase="preparing",
                    note="Coletando arquivos do dataset…",
                ),
                f"[clássico] Iniciando treino {ev.arch} em {ev.dataset_path}",
                None,
            )
        elif isinstance(ev, ClassicalDatasetInsufficient):
            yield (
                _train_status_html(
                    ev.arch,
                    "CPU",
                    0,
                    1,
                    {},
                    ev.elapsed,
                    phase="error",
                    note="Dataset insuficiente — precisa de áudios "
                    "em subpastas real/ e fake/.",
                ),
                f"❌ Encontrados {ev.n_real} real + {ev.n_fake} fake "
                f"({ev.total} total). Mínimo: 5 por classe.",
                None,
            )
        elif isinstance(ev, ClassicalExtractionProgress):
            yield (
                _train_status_html(
                    ev.arch,
                    "CPU",
                    0,
                    1,
                    {},
                    ev.elapsed,
                    phase="preparing",
                    note=f"Extraindo features tabulares: {ev.done}/{ev.total} "
                    f"arquivos ({ev.valid} OK, {ev.skipped} pulados)…",
                ),
                f"[clássico] Extração: {ev.done}/{ev.total} · válidos={ev.valid} · "
                f"pulados={ev.skipped}",
                None,
            )
        elif isinstance(ev, ClassicalTooFewFeatures):
            yield (
                _train_status_html(
                    ev.arch,
                    "CPU",
                    0,
                    1,
                    {},
                    ev.elapsed,
                    phase="error",
                    note="Poucas features válidas.",
                ),
                f"❌ Só {ev.valid} amostras válidas após extração. "
                "Verifique se os áudios não estão corrompidos.",
                None,
            )
        elif isinstance(ev, ClassicalFitting):
            yield (
                _train_status_html(
                    ev.arch,
                    "CPU",
                    0,
                    1,
                    {},
                    ev.elapsed,
                    phase="running",
                    note=f"Treinando {ev.arch} em {ev.n_samples} amostras × "
                    f"{ev.n_features} features…",
                ),
                f"[clássico] Matriz: ({ev.n_samples}, {ev.n_features}) · "
                f"treinando {ev.arch}…",
                None,
            )
        elif isinstance(ev, ClassicalCompleted):
            nice = ev.trained.arch
            eval_figs = _training_eval_figures(ev.y_val, ev.y_pred, ev.y_score, [])
            fig = _classical_figure(ev)
            _LAST_TRAINED = ev.trained
            notify_success(
                f"Treino de {nice} concluído", message=f"val_accuracy={ev.val_acc:.4f}"
            )
            progress(1.0, desc="Concluído!")
            yield (
                _train_status_html(
                    nice,
                    "CPU",
                    1,
                    1,
                    {"acc": [ev.val_acc], "val_acc": [ev.val_acc]},
                    ev.elapsed,
                    phase="done",
                    note=_DONE_NOTE,
                ),
                "\n".join(
                    [
                        f"✓ {nice} treinado",
                        f"Amostras: {ev.n_samples} ({ev.n_real} real + "
                        f"{ev.n_fake} fake, {ev.skipped} pulados)",
                        f"Características por amostra: {ev.n_features} "
                        f"({'+'.join(CLASSICAL_FEATURE_TYPES)})",
                        f"Val accuracy: {ev.val_acc:.4f}",
                        f"Matriz de confusão [real,fake]:\n{ev.confusion}",
                    ]
                ),
                fig,
                *_eval_outputs(eval_figs),
            )
        elif isinstance(ev, TrainingFailed):
            # Notify.5: erro acionável com hint específico
            notify_from_actionable(CommonErrors.training_failed(str(ev.error)))
            yield (
                _training_error_panel(ev.error),
                f"❌ Erro no treino clássico:\n{ev.error}"
                if ev.classical
                else f"❌ Erro durante o treino:\n\n{ev.error}\n\n"
                "(Stack trace completo nos logs do servidor.)",
                None,
            )


# =====================================================================
# Tab builder
# =====================================================================


def create_training_wizard_tab():
    """Constrói o wizard de treinamento (Tab principal)."""
    with gr.Tab("🪄 Assistente", id="tab_train_wizard"):
        from app.interfaces.gradio.utils.components import page_header

        page_header(
            "🪄",
            "Assistente de Treinamento",
            "Treine um modelo em 4 passos guiados: dataset, modelo, "
            "hiperparâmetros e execução.",
        )

        # State compartilhado entre steps
        current_step = gr.State(1)
        scan_result = gr.State({})  # dict do _scan_dataset

        # Stepper visual no topo
        stepper = gr.HTML(_stepper_html(1), elem_id="wizard_stepper")

        # ───────────── Step 1: Dataset ─────────────
        with gr.Group(visible=True) as group_s1:
            gr.Markdown("### Etapa 1: escolha o dataset")
            gr.Markdown(
                "Indique uma pasta com **uma subpasta por classe**. Subpastas começando "
                "com `real`, `bonafide`, `genuine`, `original`, `authentic` ou `natural` "
                "contam como voz real. As demais (`spoof`, `fake` etc.) contam como "
                "sintética."
            )

            with gr.Row(elem_classes="action-row"):
                dataset_path_s1 = gr.Textbox(
                    label="Caminho do dataset",
                    value="data/datasets",
                    placeholder="ex: /data/asvspoof2019",
                    scale=4,
                    min_width=240,
                )
                scan_btn = gr.Button("Validar dataset", variant="secondary", scale=1)

            scan_output = gr.Markdown("*Valide o dataset para continuar.*")

            with gr.Row(elem_classes="action-row wizard-actions"):
                next_s1_btn = gr.Button(
                    "Próximo →",
                    variant="primary",
                    interactive=False,
                )

        # ───────────── Step 2: Modelo ─────────────
        with gr.Group(visible=False) as group_s2:
            gr.Markdown("### Etapa 2: escolha o modelo")

            arch_select = gr.Dropdown(
                choices=[m["name"] for m in _get_model_catalog()],
                label="Arquitetura",
                value="AASIST",
                interactive=True,
                info=(
                    "Modelos com grafos (AASIST) tendem a ser mais precisos, o Sonic "
                    "Sleuth é o mais leve e o Ensemble, o mais robusto."
                ),
            )

            with gr.Accordion("Comparar arquiteturas", open=False):
                cards_html = gr.HTML(_render_model_cards_html())

            with gr.Row(elem_classes="action-row wizard-actions"):
                back_s2_btn = gr.Button("← Voltar")
                next_s2_btn = gr.Button("Próximo →", variant="primary")

        # ───────────── Step 3: Hiperparâmetros ─────────────
        # Saíram desta tela as opções que não tinham efeito: SWA e mixup chegavam ao
        # handler e paravam ali, e a calibração de temperatura e o limiar EER eram
        # caixas soltas, sem ligação com o treino interativo.
        with gr.Group(visible=False) as group_s3:
            gr.Markdown("### Etapa 3: ajuste o treino")
            gr.Markdown("Os valores sugeridos funcionam bem na maioria dos casos.")

            with gr.Row(elem_classes="responsive-grid"):
                epochs_s3 = gr.Slider(
                    1,
                    200,
                    value=50,
                    step=1,
                    label="Épocas",
                    info=(
                        "Mais épocas podem melhorar a acurácia, mas aumentam "
                        "o tempo de treino."
                    ),
                )
                batch_s3 = gr.Slider(
                    4,
                    128,
                    value=16,
                    step=4,
                    label="Tamanho do lote",
                    info=(
                        "Lotes maiores podem acelerar o treino, mas exigem "
                        "mais memória RAM ou VRAM."
                    ),
                )

            lr_s3 = gr.Number(
                value=0.0008,
                label="Taxa de aprendizado",
                precision=5,
                info="Padrão: 8e-4. Arquiteturas com atenção/grafos "
                "(AASIST, Conformer, Transformers) são auto-limitadas a "
                "1e-4 para estabilidade numérica.",
            )

            use_cw = gr.Checkbox(
                True,
                label="Compensar desbalanceamento entre classes",
                info="Dá mais peso à classe com menos amostras. Recomendado.",
            )

            estimativa_html = gr.HTML(
                "", elem_classes="wizard-estimate"
            )

            with gr.Row(elem_classes="action-row wizard-actions"):
                back_s3_btn = gr.Button("← Voltar")
                next_s3_btn = gr.Button(
                    "Iniciar treinamento →",
                    variant="primary",
                )

        # ───────────── Step 4: Treinar ─────────────
        with gr.Group(visible=False) as group_s4:
            with gr.Row(elem_classes="action-row wizard-actions"):
                gr.Markdown("### Etapa 4: treinamento")
                parar_btn = gr.Button(
                    "⏹ Interromper",
                    variant="stop",
                    scale=0,
                    min_width=140,
                )

            # Painel de métricas ao vivo (progresso, loss/acc, ETA) — ocupa
            # toda a largura para máxima visibilidade durante o treino.
            status_box = gr.HTML(
                '<div class="train-live"><div class="tl-note">'
                "Aguardando início do treinamento…</div></div>"
            )

            with gr.Row(elem_classes="responsive-grid"):
                with gr.Column(scale=1, min_width=320):
                    history_plot = gr.Plot(label="Perda, acurácia e matriz de confusão")
                with gr.Column(scale=1, min_width=320):
                    logs_box = gr.TextArea(
                        label="Log por época",
                        interactive=False,
                        lines=16,
                        max_lines=40,
                    )

            # Avaliação e salvamento só existem depois que o treino termina: antes
            # eram sete gráficos vazios e um botão sem nada para salvar.
            with gr.Accordion(
                "Avaliação do treinamento", open=True, visible=False
            ) as eval_section:
                with gr.Row(elem_classes="responsive-grid responsive-grid-3 plot-grid"):
                    eval_roc_plot = gr.Plot(label="Curva ROC")
                    eval_cm_plot = gr.Plot(label="Matriz de confusão")
                    eval_pr_plot = gr.Plot(label="Curva precisão-revocação")
                with gr.Row(elem_classes="responsive-grid responsive-grid-3 plot-grid"):
                    eval_det_plot = gr.Plot(label="Curva DET / EER")
                    eval_threshold_plot = gr.Plot(
                        label="Otimização do limiar de decisão"
                    )
                    eval_class_acc_plot = gr.Plot(label="Acurácia por classe")
                eval_lr_plot = gr.Plot(label="Evolução da taxa de aprendizado")

            # ── Salvar modelo treinado ──────────────────────────────────
            with gr.Group(visible=False) as save_section:
                gr.Markdown(
                    "#### Salvar modelo\n"
                    "Dê um nome e salve para usar na aba **🎯 Detectar**. "
                    "O pré-processamento do treino é gravado junto, para a inferência "
                    "usar exatamente as mesmas características."
                )
                with gr.Row(elem_classes="action-row"):
                    save_name = gr.Textbox(
                        label="Nome do modelo",
                        placeholder="ex: aasist_ptbr_v1",
                        scale=3,
                        min_width=220,
                    )
                    save_btn = gr.Button(
                        "Salvar modelo",
                        variant="primary",
                        scale=1,
                    )
                save_status = gr.Markdown("")

            with gr.Row(elem_classes="action-row wizard-actions"):
                back_s4_btn = gr.Button("← Novo treino")

        # ────────────────────────── Event handlers ──────────────────────────

        @ui_safe("Falha ao varrer o dataset")
        def on_scan(path):
            # Feedback imediato: escanear pode varrer milhares de arquivos.
            # 1º yield mostra estado "validando" + desabilita o avançar;
            # 2º yield entrega o resultado.
            yield (
                gr.update(),
                "⏳ Validando dataset… (varrendo arquivos)",
                gr.update(interactive=False),
            )
            res = _scan_dataset(path)
            yield (
                res,
                res["message"],
                gr.update(interactive=res["ok"]),
            )

        scan_btn.click(
            fn=on_scan,
            inputs=[dataset_path_s1],
            outputs=[scan_result, scan_output, next_s1_btn],
        )

        def goto_step(n):
            return (n, _stepper_html(n), *_step_visibility(n))

        next_s1_btn.click(
            fn=lambda: goto_step(2),
            inputs=[],
            outputs=[current_step, stepper, group_s1, group_s2, group_s3, group_s4],
        )
        back_s2_btn.click(
            fn=lambda: goto_step(1),
            inputs=[],
            outputs=[current_step, stepper, group_s1, group_s2, group_s3, group_s4],
        )
        next_s2_btn.click(
            fn=lambda: goto_step(3),
            inputs=[],
            outputs=[current_step, stepper, group_s1, group_s2, group_s3, group_s4],
        )
        back_s3_btn.click(
            fn=lambda: goto_step(2),
            inputs=[],
            outputs=[current_step, stepper, group_s1, group_s2, group_s3, group_s4],
        )
        back_s4_btn.click(
            fn=lambda: goto_step(1),
            inputs=[],
            outputs=[current_step, stepper, group_s1, group_s2, group_s3, group_s4],
        ).then(
            fn=lambda: (gr.update(visible=False), gr.update(visible=False)),
            outputs=[eval_section, save_section],
            show_progress="hidden",
        )

        # Salvar modelo treinado (Step 4)
        @ui_safe("Falha ao salvar o modelo")
        def on_save_model(name):
            # Feedback imediato + desabilita o botão durante a escrita
            yield "⏳ Salvando modelo…", gr.update(interactive=False)
            ok, msg = _save_trained_model(name)
            if ok:
                notify_success("Modelo salvo", message=msg)
            # reabilita o botão (permite renomear/tentar de novo)
            yield msg, gr.update(interactive=True)

        save_btn.click(
            fn=on_save_model,
            inputs=[save_name],
            outputs=[save_status, save_btn],
        )

        # Atualiza preview de cards quando seleciona arquitetura
        arch_select.change(
            fn=_render_model_cards_html,
            inputs=[arch_select],
            outputs=[cards_html],
        )


        @ui_safe("Falha ao interromper")
        def pedir_parada():
            """Sinaliza a interrupcao; o treino para no fim da epoca atual."""
            _PEDIDO_DE_PARADA.set()
            return gr.update(
                value="⏹ Interrompendo…", interactive=False
            )

        parar_btn.click(fn=pedir_parada, outputs=[parar_btn])

        # Estimativa de custo: atualiza quando muda a arquitetura ou as epocas.
        # Ligada DEPOIS da definicao do handler — `fn=` e avaliado na ligacao.

        @ui_safe("Falha ao estimar o custo")
        def atualizar_estimativa(scan_state, arch, epochs):
            """Custo esperado ANTES de comecar."""
            n = int((scan_state or {}).get("total_samples") or 0)
            if not (arch and n):
                return gr.update(value="")
            texto = _estimativa_de_treino(arch, int(epochs or 0), n)
            if not texto:
                return gr.update(value="")
            return gr.update(
                value=f'<div class="xf-callout xf-callout-info">{texto}</div>'
            )

        for _controle in (arch_select, epochs_s3):
            _controle.change(
                fn=atualizar_estimativa,
                inputs=[scan_result, arch_select, epochs_s3],
                outputs=[estimativa_html],
            )

        # Step 3 → Step 4 + dispara treinamento
        @ui_safe("Falha ao iniciar o treino")
        def start_training(
            scan_state,
            arch,
            epochs,
            batch,
            lr,
            use_cw_val,
        ):
            # Vai para Step 4
            updates_step = [4, _stepper_html(4), *_step_visibility(4)]

            # Streamed do _run_training
            for update in _run_training(
                arch=arch,
                dataset_path=(scan_state or {}).get("class_names")
                and (scan_state.get("path") or "data/datasets")
                or "data/datasets",
                epochs=int(epochs),
                batch_size=int(batch),
                lr=float(lr),
                use_class_weighting=bool(use_cw_val),
            ):
                status, logs, plot, *eval_plots = update
                while len(eval_plots) < 7:
                    eval_plots.append(gr.update())
                yield (*updates_step, status, logs, plot, *eval_plots[:7])
            # o treino terminou (concluido ou interrompido): libera o botao
            _PEDIDO_DE_PARADA.clear()

        # Captura o path do dataset junto com o resultado do scan
        @ui_safe("Falha ao anexar o caminho")
        def attach_path_to_scan(scan_state, path):
            if isinstance(scan_state, dict) and scan_state.get("ok"):
                scan_state = {**scan_state, "path": path}
            return scan_state

        # Pequeno hack: quando entra no Step 4, atualizamos o scan_state com o path
        next_s3_btn.click(
            fn=attach_path_to_scan,
            inputs=[scan_result, dataset_path_s1],
            outputs=[scan_result],
        ).then(
            fn=start_training,
            inputs=[
                scan_result,
                arch_select,
                epochs_s3,
                batch_s3,
                lr_s3,
                use_cw,
            ],
            outputs=[
                current_step,
                stepper,
                group_s1,
                group_s2,
                group_s3,
                group_s4,
                status_box,
                logs_box,
                history_plot,
                eval_roc_plot,
                eval_cm_plot,
                eval_pr_plot,
                eval_det_plot,
                eval_threshold_plot,
                eval_class_acc_plot,
                eval_lr_plot,
            ],
        ).success(
            fn=lambda: (gr.update(visible=True), gr.update(visible=True)),
            outputs=[eval_section, save_section],
            show_progress="hidden",
        )
