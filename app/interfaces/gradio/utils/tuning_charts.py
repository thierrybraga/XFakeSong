"""Gráficos de resultado da busca de hiperparâmetros (Optuna).

Renderiza, a partir de um `optuna.Study`, dois painéis:
- **Convergência**: valor de cada trial + melhor valor acumulado por trial.
- **Importância**: importância relativa de cada hiperparâmetro (fANOVA do Optuna).

O helper é tolerante: opera por duck-typing sobre `study` (apenas `trials`,
`direction`), e a importância degrada para uma nota quando `optuna` não está
instalado ou há trials insuficientes — então a função é testável sem optuna.
"""

from __future__ import annotations

from typing import Any, List

from app.interfaces.gradio.utils.plotting import (
    PLOT_ACCENT,
    PLOT_DANGER,
    PLOT_TEXT_MUTED,
    safe_tight_layout,
    style_ax,
)


def _completed_values(study: Any) -> List[float]:
    """Extrai os valores (objetivo) dos trials concluídos, em ordem."""
    out: List[float] = []
    for t in getattr(study, "trials", []) or []:
        v = getattr(t, "value", None)
        if v is not None:
            try:
                out.append(float(v))
            except (TypeError, ValueError):
                continue
    return out


def _is_maximize(study: Any) -> bool:
    """True se o estudo maximiza (default), via study.direction.name."""
    try:
        return str(getattr(study.direction, "name", "MAXIMIZE")).upper() == "MAXIMIZE"
    except Exception:
        return True


def render_tuning_figure(study: Any):
    """Constrói a figura (Convergência | Importância) a partir do `study`.

    Retorna a `matplotlib.figure.Figure` (o chamador fecha com close_fig).
    """
    import matplotlib.pyplot as plt

    values = _completed_values(study)
    maximize = _is_maximize(study)

    # Melhor valor acumulado (curva monotônica de convergência)
    best_so_far: List[float] = []
    cur = None
    for v in values:
        cur = v if cur is None else (max(cur, v) if maximize else min(cur, v))
        best_so_far.append(cur)

    fig, ax = plt.subplots(1, 2, figsize=(12, 4))

    # ── Painel 1: Convergência ──
    style_ax(ax[0], fig, "Convergência da busca")
    if values:
        xs = list(range(1, len(values) + 1))
        ax[0].plot(
            xs, values, color=PLOT_DANGER, linestyle=":", marker=".",
            ms=7, label="trial",
        )
        ax[0].plot(
            xs, best_so_far, color=PLOT_ACCENT, marker="o", ms=4,
            label="melhor acumulado",
        )
        if 1 <= len(xs) <= 20:
            ax[0].set_xticks(xs)
        ax[0].legend()
    else:
        ax[0].text(
            0.5, 0.5, "Sem trials concluídos", ha="center", va="center",
            transform=ax[0].transAxes, color=PLOT_TEXT_MUTED,
        )
    ax[0].set_xlabel("Trial")
    ax[0].set_ylabel("Métrica objetivo")

    # ── Painel 2: Importância dos hiperparâmetros ──
    style_ax(ax[1], fig, "Importância dos hiperparâmetros")
    importances = None
    try:
        import optuna  # noqa: PLC0415

        importances = optuna.importance.get_param_importances(study)
    except Exception:
        importances = None

    if importances:
        # Maior no topo do barh → inverte a ordem
        names = list(importances.keys())[::-1]
        vals = [float(importances[k]) for k in names]
        ax[1].barh(range(len(names)), vals, color=PLOT_ACCENT)
        ax[1].set_yticks(range(len(names)))
        ax[1].set_yticklabels(names, fontsize=8)
        ax[1].set_xlabel("Importância relativa")
    else:
        ax[1].text(
            0.5, 0.5,
            "Importância indisponível\n(requer optuna e ≥2 trials concluídos)",
            ha="center", va="center", transform=ax[1].transAxes,
            color=PLOT_TEXT_MUTED, fontsize=9,
        )
        ax[1].set_xticks([])
        ax[1].set_yticks([])

    safe_tight_layout(fig)
    return fig


__all__ = ["render_tuning_figure"]
