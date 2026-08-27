"""Plotting helpers compartilhados entre tabs Gradio (FE.8).

Centraliza:
- Paleta de cores dark consistente
- `style_ax()` para aplicar tema dark em qualquer matplotlib axis
- `close_fig()` para fechar figura após retornar (evita memory leak FE.2)
- `make_figure()` context manager que garante close automático

Por que isto importa:
- Cada tab duplicava as constantes `_PLOT_BG`, `_PLOT_FACE`, etc.
- matplotlib em servidor Gradio threaded acumula figures se não fechar
- `matplotlib.use('Agg')` é OBRIGATÓRIO em ambiente sem display (Docker, headless)
"""

from __future__ import annotations

import logging
import os
from contextlib import contextmanager
from typing import Iterator, Optional

# CRÍTICO: usar backend não-interativo ANTES de qualquer import de pyplot.
# Em ambiente Gradio (threaded, Docker headless), o backend default TkAgg
# falha com "main thread is not in main loop". Agg renderiza para buffer
# de bytes — exatamente o que precisamos para gr.Plot().
import matplotlib
if not os.environ.get("MPLBACKEND"):
    matplotlib.use("Agg", force=True)

import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402

logger = logging.getLogger(__name__)


# =====================================================================
# Paleta dark (alinhada com CSS de gradio_app.py)
# =====================================================================

PLOT_BG = "#0f172a"
PLOT_FACE = "#1e293b"
PLOT_TEXT = "#f1f5f9"
PLOT_TEXT_MUTED = "#94a3b8"
PLOT_GRID = "#334155"
PLOT_ACCENT = "#3b82f6"
PLOT_ACCENT2 = "#06b6d4"
PLOT_SUCCESS = "#10b981"
PLOT_WARNING = "#f59e0b"
PLOT_DANGER = "#ef4444"

# Aliases legados (compatibilidade com código antigo)
_PLOT_BG = PLOT_BG
_PLOT_FACE = PLOT_FACE
_PLOT_TEXT = PLOT_TEXT
_PLOT_GRID = PLOT_GRID
_PLOT_ACCENT = PLOT_ACCENT
_PLOT_ACCENT2 = PLOT_ACCENT2
_PLOT_DANGER = PLOT_DANGER


# =====================================================================
# Estilização de plots
# =====================================================================

def style_ax(ax, fig: Figure, title: str = "", xlabel: str = "", ylabel: str = "") -> None:
    """Aplica estilo dark consistente a um par (ax, fig).

    Args:
        ax: matplotlib Axes
        fig: matplotlib Figure (necessário para `fig.patch`)
        title: título do plot (opcional)
        xlabel/ylabel: rótulos dos eixos (opcional)
    """
    try:
        fig.patch.set_facecolor(PLOT_BG)
        ax.set_facecolor(PLOT_FACE)
        if title:
            ax.set_title(title, color=PLOT_TEXT, fontweight="600", fontsize=12, pad=10)
        if xlabel:
            ax.set_xlabel(xlabel)
        if ylabel:
            ax.set_ylabel(ylabel)
        ax.tick_params(colors=PLOT_TEXT, labelsize=9)
        for lbl in (ax.xaxis.label, ax.yaxis.label):
            lbl.set_color(PLOT_TEXT)
            lbl.set_fontsize(10)
        for spine in ax.spines.values():
            spine.set_color(PLOT_GRID)
        ax.grid(True, color=PLOT_GRID, alpha=0.3, linewidth=0.5)
    except Exception as e:
        logger.debug(f"style_ax falhou: {e}")


# Alias legado
_style_ax = style_ax


# =====================================================================
# Gestão de memória (FE.2: leak fix)
# =====================================================================

def new_figure(nrows: int = 1, ncols: int = 1, *, figsize: tuple = (10, 4)):
    """Cria ``(fig, ax)`` FORA do registro global do pyplot.

    A assinatura espelha a de `plt.subplots(nrows, ncols, figsize=...)` de
    propósito: é substituta direta nas chamadas existentes, inclusive as que
    passam linhas/colunas posicionalmente.

    Esta é a forma correta de gerar gráficos num servidor Gradio, e substitui
    `plt.subplots` em todo o pacote.

    `plt.subplots` registra a figura no gerenciador global do pyplot, que
    mantém a referência viva até alguém chamar `plt.close`. Num handler que
    devolve a figura para `gr.Plot`, ninguém chama — e o processo, que fica
    dias no ar, acumula uma figura por interação. Havia 22 chamadas assim, e
    `features.py`/`dataset_management.py` não fechavam nenhuma.

    `Figure()` direto não entra em registro nenhum: quando o Gradio termina de
    serializar e o handler sai de escopo, o coletor de lixo leva a figura. Não
    há o que fechar, e portanto não há como esquecer de fechar.
    """
    fig = Figure(figsize=figsize)
    axes = fig.subplots(nrows, ncols)
    return fig, axes


def close_fig(fig: Optional[Figure]) -> None:
    """Fecha a figura e libera memória. Idempotente.

    Só é necessária para figuras vindas de `plt.subplots`. Com `new_figure`
    não há registro global e a chamada é inócua — mantida por compatibilidade
    com chamadores existentes.
    """
    if fig is None:
        return
    try:
        plt.close(fig)
    except Exception as e:
        logger.debug(f"close_fig falhou: {e}")


@contextmanager
def make_figure(
    figsize: tuple = (10, 4),
    title: str = "",
    xlabel: str = "",
    ylabel: str = "",
) -> Iterator[tuple]:
    """Context manager que cria ``(fig, ax)`` com o tema escuro aplicado.

    Uso:
        with make_figure(figsize=(10, 4), title="Forma de Onda") as (fig, ax):
            ax.plot(x, y, color=PLOT_ACCENT)
            return fig

    A figura vem de `new_figure` — fora do registro do pyplot —, então
    devolvê-la ao Gradio não vaza e não exige `close_fig` depois. A versão
    anterior usava `plt.subplots` e tinha um `finally` VAZIO, apesar de o
    exemplo do docstring prometer fechamento automático.
    """
    fig, ax = new_figure(figsize=figsize)
    style_ax(ax, fig, title=title, xlabel=xlabel, ylabel=ylabel)
    yield fig, ax


def safe_tight_layout(fig: Figure) -> None:
    """tight_layout que não-quebra em axes com problemas (colorbar, etc.)."""
    try:
        fig.tight_layout()
    except Exception as e:
        logger.debug(f"tight_layout falhou (não-crítico): {e}")


# =====================================================================
# Helpers de serviços (singletons thread-safe — FE.3)
# =====================================================================

import threading

_service_locks: dict = {}
_service_locks_lock = threading.Lock()


def get_service_lock(name: str) -> threading.Lock:
    """Retorna um Lock per-service-name. Cria sob demanda."""
    with _service_locks_lock:
        if name not in _service_locks:
            _service_locks[name] = threading.Lock()
        return _service_locks[name]


# =====================================================================
# UX feedback helpers — DELEGAM para notifications.py (módulo unificado)
# =====================================================================
# Mantidos aqui por backwards compat — código antigo que faz
# `from ...plotting import notify_error` continua funcionando.
#
# O sistema agora tem 5 níveis (success/info/warning/error/critical),
# histórico e erros acionáveis. Veja: app/interfaces/gradio/utils/notifications.py

from app.interfaces.gradio.utils.notifications import (  # noqa: E402, F401
    notify_error,
    notify_info,
    notify_success,
    notify_warning,
)


def confirm_destructive(
    confirm_state: bool,
    item_label: str,
    *,
    requires_double_click: bool = True,
) -> tuple:
    """Padrão de confirmação para ações destrutivas (delete, reset, etc.).

    UI Fase 3 (UI.12). Implementa o padrão "double-click confirm":
    1ª chamada → mostra warning + retorna `False` (não executa)
    2ª chamada → executa e reseta state → retorna `True`

    Uso típico em event handlers:

        confirm = gr.State(False)
        delete_btn.click(
            fn=lambda c: confirm_destructive(c, "modelo XYZ"),
            inputs=[confirm],
            outputs=[confirm, status],
        ).then(
            fn=actually_delete_if_confirmed,
            inputs=[confirm],
            ...
        )

    Args:
        confirm_state: estado atual (False = aguardando 1ª confirmação)
        item_label: descrição amigável do item (ex: "modelo AASIST_v1")
        requires_double_click: se True, exige 2 cliques. Se False, executa direto.

    Returns:
        (next_state, message): novo state + mensagem para o usuário
    """
    if not requires_double_click:
        return (False, f"✓ {item_label} removido")

    if not confirm_state:
        # 1ª chamada — pede confirmação
        msg = (
            f"⚠ Tem certeza que deseja remover **{item_label}**?\n"
            f"Esta ação não pode ser desfeita. Clique novamente para confirmar."
        )
        notify_warning(f"Confirme: clique novamente para remover {item_label}")
        return (True, msg)
    else:
        # 2ª chamada — executa
        notify_success(f"✓ {item_label} removido")
        return (False, f"✓ {item_label} removido com sucesso")
