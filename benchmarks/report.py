"""Geração dos artefatos do benchmark: JSON, CSV, Markdown, LaTeX e figuras."""

from __future__ import annotations

import csv
import json
import logging
import re
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

logger = logging.getLogger("benchmark")


# ───────────────────────── formatação (pt-BR) ─────────────────────────

def _pct(x, d: int = 2) -> str:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "---"
    return f"{x * 100:.{d}f}".replace(".", ",") + r"\%"


def _num(x, d: int = 3) -> str:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "---"
    return f"{x:.{d}f}".replace(".", ",")


def _intbr(x) -> str:
    if x is None:
        return "---"
    return f"{int(x):,}".replace(",", ".")


def _conv(flag) -> str:
    return (r"\textcolor{successgreen}{Sim}" if flag
            else r"\textcolor{dangerred}{Não}")


def _train_budget_label(r: Dict[str, Any]) -> str:
    if r.get("type") == "classical":
        fit = r.get("fit_strategy") or (r.get("training_config") or {}).get("fit_strategy") or {}
        if fit.get("kind") == "grid_search_cv_then_refit":
            n_fits = fit.get("n_fits")
            cv = fit.get("cv")
            candidates = fit.get("n_candidates")
            if n_fits:
                return f"CV {n_fits}+fit"
            if cv and candidates:
                return f"CV {cv}x{candidates}+fit"
            return "CV+fit"
        return "fit"
    epochs = r.get("epochs")
    return str(epochs) if epochs is not None else "--"


# ───────────────────────── tabelas LaTeX ─────────────────────────

def _ok_items(results) -> List:
    arch = results["architectures"]
    return [(name, r) for name, r in arch.items() if r.get("status") == "ok"]


def _slug(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_") or "model"


def _table_resultados(results) -> str:
    rows = []
    for name, r in results["architectures"].items():
        if r.get("status") != "ok":
            rows.append(f"{name} & \\multicolumn{{7}}{{c}}{{"
                        f"\\textit{{falhou: {r.get('error','?')[:40]}}}}} \\\\")
            continue
        c, e = r["clean"], r["efficiency"]
        rows.append(
            f"{name} & {_pct(c.get('accuracy'))} & {_pct(c.get('eer'))} & "
            f"{_num(c.get('auc_roc'))} & {_num(c.get('min_tdcf'),4)} & "
            f"{_num(e.get('latency_ms'),1)} & {_train_budget_label(r)} & "
            f"{_conv(r.get('converged'))} \\\\"
        )
    body = "\n".join(rows)
    return (
        "\\begin{table}[H]\n\\centering\n"
        "\\caption{Desempenho das arquiteturas (conjunto de teste, "
        f"{results['dataset']['n_test']} amostras). Gerado pelo benchmark.}}\n"
        "\\label{tab:bench_resultados}\n\\small\\singlespacing\n"
        "\\begin{tabular}{lccccccc}\n\\toprule\n"
        "\\textbf{Arquitetura} & \\textbf{Acur.} & \\textbf{EER} & "
        "\\textbf{AUC-ROC} & \\textbf{min-tDCF} & \\textbf{Lat.\\,(ms)} & "
        "\\textbf{Treino} & \\textbf{Conv.?} \\\\\n\\midrule\n"
        f"{body}\n\\bottomrule\n\\end{{tabular}}\n\\end{{table}}\n"
    )


def _table_eficiencia(results) -> str:
    rows = []
    for name, r in _ok_items(results):
        e = r["efficiency"]
        rows.append(
            f"{name} & {_intbr(e.get('params'))} & {_num(e.get('size_mb'),1)} & "
            f"{_num(e.get('latency_ms'),1)} & {_conv(r.get('converged'))} \\\\"
        )
    body = "\n".join(rows)
    return (
        "\\begin{table}[H]\n\\centering\n"
        "\\caption{Eficiência computacional (medido na máquina do benchmark).}\n"
        "\\label{tab:bench_eficiencia}\n\\small\\singlespacing\n"
        "\\begin{tabular}{lcccc}\n\\toprule\n"
        "\\textbf{Arquitetura} & \\textbf{Parâmetros} & \\textbf{Memória (MB)} & "
        "\\textbf{Latência (ms)} & \\textbf{Conv.?} \\\\\n\\midrule\n"
        f"{body}\n\\bottomrule\n\\end{{tabular}}\n\\end{{table}}\n"
    )


def _table_robustez(results) -> str:
    snrs = results["config"]["snr_levels_db"]
    conv = [(n, r) for n, r in _ok_items(results) if r.get("converged")]
    head_snr = "".join(
        f"\\multicolumn{{2}}{{c}}{{\\textbf{{SNR {s}\\,dB}}}} & " for s in snrs
    ).rstrip("& ")
    cmid = "\\cmidrule(lr){2-3}" + "".join(
        f"\\cmidrule(lr){{{4 + 2 * i}-{5 + 2 * i}}}" for i in range(len(snrs))
    )
    sub = "& Acur. & EER " * (len(snrs) + 1)
    rows = []
    for name, r in conv:
        cells = [_pct(r["clean"].get("accuracy")), _pct(r["clean"].get("eer"))]
        for s in snrs:
            rob = r["robustness"].get(str(s), {})
            cells += [_pct(rob.get("accuracy")), _pct(rob.get("eer"))]
        rows.append(f"{name} & " + " & ".join(cells) + " \\\\")
    cols = "l" + "cc" * (len(snrs) + 1)
    n_cols = 1 + 2 * (len(snrs) + 1)
    body = "\n".join(rows) or (
        f"\\multicolumn{{{n_cols}}}{{c}}{{(nenhum modelo convergente)}}\\\\"
    )
    return (
        "\\begin{table}[H]\n\\centering\n"
        "\\caption{Robustez sob ruído AWGN (acurácia e EER por SNR). "
        "Ruído aplicado no espaço de entrada do modelo.}\n"
        "\\label{tab:bench_robustez}\n\\small\\singlespacing\n"
        f"\\begin{{tabular}}{{{cols}}}\n\\toprule\n"
        f"\\multirow{{2}}{{*}}{{\\textbf{{Arquitetura}}}} & "
        f"\\multicolumn{{2}}{{c}}{{\\textbf{{Limpo}}}} & {head_snr} \\\\\n"
        f"{cmid}\n {sub}\\\\\n\\midrule\n"
        f"{body}\n\\bottomrule\n\\end{{tabular}}\n\\end{{table}}\n"
    )


# ───────────────────────── figuras ─────────────────────────

FIG_DPI = 300  # qualidade de impressão/TCC (era 150)


def _series_styles(n: int) -> List[dict]:
    """Estilos distintos para até ~40 séries: cor (tab20) + linestyle + marker.

    Resolve a ambiguidade do ciclo default do matplotlib (10 cores) quando há
    várias arquiteturas no mesmo eixo (ROC, robustez, convergência).
    """
    import matplotlib.pyplot as plt

    colors = list(plt.get_cmap("tab20").colors)
    linestyles = ["-", "--", "-.", ":"]
    markers = ["o", "s", "^", "D", "v", "P", "X", "*"]
    return [
        {
            "color": colors[i % len(colors)],
            "linestyle": linestyles[(i // len(colors)) % len(linestyles)],
            "marker": markers[i % len(markers)],
        }
        for i in range(n)
    ]


_STYLE_APPLIED = False


def _apply_style() -> None:
    """Centraliza a tipografia/qualidade das figuras (rcParams) — uma vez.

    Evita repetir tamanhos de fonte/dpi por figura e garante consistência. Não
    mexe em `axes.grid` global (quebraria as matrizes de confusão via imshow);
    o grid continua sendo ligado explicitamente onde faz sentido.
    """
    global _STYLE_APPLIED
    if _STYLE_APPLIED:
        return
    import matplotlib as mpl

    mpl.rcParams.update({
        "savefig.dpi": FIG_DPI,
        "savefig.bbox": "tight",
        "font.size": 10,
        "axes.titlesize": 12,
        "axes.labelsize": 10,
        "legend.fontsize": 8,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "axes.axisbelow": True,  # grid atrás dos dados
        "lines.antialiased": True,
    })
    _STYLE_APPLIED = True


def _save_fig(fig, path: Path) -> None:
    """Salva em PNG (300 dpi) e PDF vetorial (mesmo nome) para o LaTeX.

    `bbox_inches='tight'` garante que legendas movidas para fora do eixo não
    sejam cortadas.
    """
    fig.savefig(path, dpi=FIG_DPI, bbox_inches="tight")
    try:
        fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    except Exception as e:  # noqa: BLE001
        logger.debug("PDF de %s falhou: %s", path.name, e)


def _legend_outside(ax, **kwargs) -> None:
    """Legenda à direita, fora da área de plotagem (não cobre os dados)."""
    ax.legend(loc="center left", bbox_to_anchor=(1.01, 0.5), fontsize=7,
              borderaxespad=0.0, **kwargs)


def _fig_roc(results, out: Path) -> None:
    import matplotlib.pyplot as plt
    from sklearn.metrics import roc_curve

    y = np.asarray(results["dataset"].get("y_test", []))
    items = [(n, r) for n, r in _ok_items(results) if r.get("scores_clean")]
    if y.size == 0 or not items or len(np.unique(y)) < 2:
        _write_placeholder_figure(
            out / "roc.png",
            "Curvas ROC",
            "Dados insuficientes para plotar ROC. Verifique y_test e scores "
            "limpos das arquiteturas.",
        )
        return
    # Ordena por AUC desc → legenda legível (melhores no topo; NaN ao fim).
    def _auc_key(item):
        a = item[1]["clean"].get("auc_roc")
        return a if isinstance(a, (int, float)) and np.isfinite(a) else -1.0

    items.sort(key=_auc_key, reverse=True)
    styles = _series_styles(len(items))
    fig, ax = plt.subplots(figsize=(6.4, 5))
    plotted = 0
    curves = []  # (fpr, tpr, style) p/ replotar no zoom inset
    for (name, r), st in zip(items, styles):
        s = np.asarray(r["scores_clean"], dtype=float)
        if len(s) != len(y) or not np.isfinite(s).all():
            continue
        fpr, tpr, _ = roc_curve(y, s)
        auc = r["clean"].get("auc_roc")
        ax.plot(fpr, tpr, lw=1.6, color=st["color"], linestyle=st["linestyle"],
                label=f"{name} (AUC={_num(auc).replace(chr(92),'')})")
        curves.append((fpr, tpr, st))
        plotted += 1
    if plotted == 0:
        plt.close(fig)
        _write_placeholder_figure(
            out / "roc.png",
            "Curvas ROC",
            "Scores encontrados, mas nenhum vetor era compatível com y_test "
            "ou continha apenas valores finitos.",
        )
        return
    ax.plot([0, 1], [0, 1], "--", color="#94a3b8", lw=1)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal")
    # Zoom inset no canto superior-esquerdo (curvas de AUC alto se aglomeram lá).
    # Posicionado no quadrante inferior-direito do eixo p/ não cobrir as curvas.
    try:
        axins = ax.inset_axes([0.46, 0.08, 0.5, 0.5])
        for fpr, tpr, st in curves:
            axins.plot(fpr, tpr, lw=1.1, color=st["color"], linestyle=st["linestyle"])
        axins.plot([0, 1], [0, 1], "--", color="#94a3b8", lw=0.8)
        axins.set_xlim(0.0, 0.2)
        axins.set_ylim(0.8, 1.0)
        axins.set_title("zoom (canto sup.-esq.)", fontsize=7)
        axins.tick_params(labelsize=6)
        axins.grid(alpha=0.3)
        ax.indicate_inset_zoom(axins, edgecolor="#64748b", alpha=0.6)
    except Exception as e:  # noqa: BLE001
        logger.debug("zoom inset da ROC falhou: %s", e)
    ax.set_xlabel("Taxa de falsos positivos (FPR)")
    ax.set_ylabel("Taxa de verdadeiros positivos (TPR)")
    ax.set_title("Curvas ROC (conjunto de teste limpo)")
    _legend_outside(ax)
    ax.grid(alpha=0.3)
    _save_fig(fig, out / "roc.png")
    plt.close(fig)


def _fig_robustez(results, out: Path) -> None:
    import matplotlib.pyplot as plt

    snrs = results["config"]["snr_levels_db"]
    conv = [(n, r) for n, r in _ok_items(results) if r.get("converged")]
    if not conv:
        _write_placeholder_figure(
            out / "robustez.png",
            "Robustez sob ruído AWGN",
            "Nenhuma arquitetura convergente com métricas de robustez foi "
            "registrada nesta execução.",
        )
        return
    xs = ["Limpo"] + [f"{s} dB" for s in snrs]
    styles = _series_styles(len(conv))
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    plotted = 0
    lowest = 100.0
    for (name, r), st in zip(conv, styles):
        ys = [r["clean"].get("accuracy", np.nan)]
        ys += [(r.get("robustness") or {}).get(str(s), {}).get("accuracy", np.nan)
               for s in snrs]
        values = np.asarray(ys, dtype=float)
        if not np.isfinite(values).any():
            continue
        ax.plot(xs, values * 100, lw=1.8, color=st["color"],
                linestyle=st["linestyle"], marker=st["marker"], ms=5, label=name)
        lowest = min(lowest, float(np.nanmin(values)) * 100)
        plotted += 1
    if plotted == 0:
        plt.close(fig)
        _write_placeholder_figure(
            out / "robustez.png",
            "Robustez sob ruído AWGN",
            "Arquiteturas convergentes encontradas, mas sem métricas finitas "
            "de acurácia para plotagem.",
        )
        return
    ax.axhline(50, ls="--", color="#94a3b8", lw=1, label="acaso (50%)")
    ax.set_ylabel("Acurácia (%)")
    ax.set_xlabel("Condição de ruído (SNR decrescente →)")
    ax.set_title("Robustez sob ruído AWGN")
    # ylim dinâmico: não corta modelos que degradam abaixo de 40%.
    ax.set_ylim(min(40.0, lowest - 5), 102)
    _legend_outside(ax)
    ax.grid(alpha=0.3)
    _save_fig(fig, out / "robustez.png")
    plt.close(fig)


def _fig_eficiencia(results, out: Path) -> None:
    import matplotlib.pyplot as plt

    items = [(n, r) for n, r in _ok_items(results)
             if r["efficiency"].get("latency_ms") is not None]
    if not items:
        _write_placeholder_figure(
            out / "eficiencia.png",
            "Eficiência computacional",
            "Nenhuma arquitetura registrou latência válida para plotagem.",
        )
        return
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    plotted = 0
    seen_conv = {True: False, False: False}
    # Marcador (○ converge / ✕ não) além da cor → acessível a daltônicos.
    for i, (name, r) in enumerate(items):
        lat = r["efficiency"]["latency_ms"]
        acc = r["clean"].get("accuracy", np.nan)
        acc = float(acc) * 100 if acc is not None else np.nan
        if not np.isfinite(acc) or lat is None or lat <= 0:
            continue
        converged = bool(r.get("converged"))
        color = "#10b981" if converged else "#ef4444"
        marker = "o" if converged else "X"
        ax.scatter(lat, acc, s=90, color=color, marker=marker,
                   edgecolor="#1e293b", zorder=3,
                   label=("convergiu" if converged else "não convergiu")
                   if not seen_conv[converged] else None)
        seen_conv[converged] = True
        # Offset alternado p/ reduzir sobreposição de rótulos.
        dy = 6 if i % 2 == 0 else -10
        ax.annotate(name, (lat, acc), fontsize=7, xytext=(5, dy),
                    textcoords="offset points")
        plotted += 1
    if plotted == 0:
        plt.close(fig)
        _write_placeholder_figure(
            out / "eficiencia.png",
            "Eficiência computacional",
            "Latências encontradas, mas nenhuma arquitetura registrou acurácia "
            "finita para plotagem.",
        )
        return
    # Latência varia em ordens de grandeza (clássico ~ms vs SSL ~centenas) →
    # escala log evita o amontoado à esquerda.
    ax.set_xscale("log")
    ax.set_xlabel("Latência (ms/amostra, escala log)")
    ax.set_ylabel("Acurácia (%)")
    ax.set_title("Eficiência × desempenho")
    ax.legend(fontsize=8, loc="lower left")
    ax.grid(alpha=0.3, which="both")
    _save_fig(fig, out / "eficiencia.png")
    plt.close(fig)


def _write_placeholder_figure(path: Path, title: str, message: str) -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 3.6))
    ax.axis("off")
    ax.set_title(title)
    ax.text(
        0.5,
        0.5,
        message,
        ha="center",
        va="center",
        wrap=True,
        fontsize=11,
        color="#334155",
    )
    fig.tight_layout()
    _save_fig(fig, path)
    plt.close(fig)


def _fig_convergencia(results, out: Path) -> None:
    import matplotlib.pyplot as plt

    items = [(n, r) for n, r in _ok_items(results) if r.get("history")]
    if not items:
        _write_placeholder_figure(
            out / "convergencia.png",
            "Convergência do treino",
            "Nenhuma arquitetura desta execução registrou histórico por época. "
            "Isso é esperado para modelos clássicos ou execuções sem treino neural.",
        )
        return

    def _val_accuracy(history: Dict[str, Any]):
        candidates = (
            "val_accuracy",
            "val_binary_accuracy",
            "val_categorical_accuracy",
            "val_sparse_categorical_accuracy",
            "val_acc",
        )
        for key in candidates:
            values = history.get(key)
            if values:
                return values
        for key, values in history.items():
            if key.startswith("val_") and (
                "acc" in key.lower() or "accuracy" in key.lower()
            ):
                return values
        return None

    styles = _series_styles(len(items))
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    plotted = 0
    lowest = 100.0
    for (name, r), st in zip(items, styles):
        h = r["history"]
        va = _val_accuracy(h)
        if not va:
            continue
        ax.plot(range(1, len(va) + 1), [v * 100 for v in va],
                marker=st["marker"], ms=3, lw=1.6, color=st["color"],
                linestyle=st["linestyle"], label=name)
        lowest = min(lowest, min(v * 100 for v in va))
        plotted += 1
    if plotted == 0:
        plt.close(fig)
        _write_placeholder_figure(
            out / "convergencia.png",
            "Convergência do treino",
            "Históricos encontrados, mas sem série de acurácia de validação "
            "compatível para plotagem.",
        )
        return
    ax.set_xlabel("Época")
    ax.set_ylabel("Acurácia de validação (%)")
    ax.set_title("Convergência do treino")
    ax.set_ylim(min(40.0, lowest - 5), 102)
    _legend_outside(ax)
    ax.grid(alpha=0.3)
    _save_fig(fig, out / "convergencia.png")
    plt.close(fig)


def _fig_confusion_matrices(results, out: Path) -> None:
    import math
    import matplotlib.pyplot as plt
    from sklearn.metrics import confusion_matrix

    y = np.asarray(results["dataset"].get("y_test", []), dtype=int)
    items = [(n, r) for n, r in _ok_items(results) if r.get("scores_clean")]
    items = [
        (n, r)
        for n, r in items
        if len(r["scores_clean"]) == len(y)
        and np.isfinite(np.asarray(r["scores_clean"], dtype=float)).all()
    ]
    if y.size == 0 or not items:
        _write_placeholder_figure(
            out / "confusion_matrices.png",
            "Matrizes de confusão",
            "Dados insuficientes para gerar matrizes de confusão. Verifique "
            "y_test e scores limpos das arquiteturas.",
        )
        return

    n_cols = min(3, len(items))
    n_rows = int(math.ceil(len(items) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4.2 * n_cols, 3.7 * n_rows))
    axes = np.asarray(axes).reshape(-1)

    for ax, (name, r) in zip(axes, items):
        scores = np.asarray(r["scores_clean"], dtype=float)
        y_pred = (scores >= 0.5).astype(int)
        cm = confusion_matrix(y, y_pred, labels=[0, 1])
        ax.imshow(cm, cmap="Blues")
        ax.set_title(name)
        ax.set_xticks([0, 1], labels=["real", "fake"])
        ax.set_yticks([0, 1], labels=["real", "fake"])
        ax.set_xlabel("Predito")
        ax.set_ylabel("Verdadeiro")
        cm_max = int(np.max(cm)) if cm.size and np.max(cm) > 0 else 1
        for row in range(2):
            for col in range(2):
                color = "white" if cm[row, col] > cm_max / 2 else "#0f172a"
                ax.text(
                    col,
                    row,
                    str(int(cm[row, col])),
                    ha="center",
                    va="center",
                    color=color,
                    fontweight="bold",
                )

    for ax in axes[len(items):]:
        ax.axis("off")

    fig.suptitle("Matrizes de confusão no conjunto de teste limpo", y=1.02)
    fig.tight_layout()
    _save_fig(fig, out / "confusion_matrices.png")
    plt.close(fig)


def _fig_score_distributions(results, out: Path) -> None:
    import matplotlib.pyplot as plt

    y = np.asarray(results["dataset"].get("y_test", []), dtype=int)
    items = [(n, r) for n, r in _ok_items(results) if r.get("scores_clean")]
    items = [
        (n, r)
        for n, r in items
        if len(r["scores_clean"]) == len(y)
        and np.isfinite(np.asarray(r["scores_clean"], dtype=float)).all()
    ]
    if y.size == 0 or not items:
        _write_placeholder_figure(
            out / "score_distributions.png",
            "Distribuição dos scores",
            "Dados insuficientes para plotar distribuição de scores. Verifique "
            "y_test e scores limpos das arquiteturas.",
        )
        return

    fig, ax = plt.subplots(figsize=(7, 4.2))
    positions = []
    labels = []
    data = []
    pos = 1
    for name, r in items:
        scores = np.asarray(r["scores_clean"], dtype=float)
        data.extend([scores[y == 0], scores[y == 1]])
        positions.extend([pos, pos + 0.35])
        labels.append((pos + 0.175, name))
        pos += 1.1

    bp = ax.boxplot(data, positions=positions, widths=0.28, patch_artist=True)
    for idx, box in enumerate(bp["boxes"]):
        box.set_facecolor("#60a5fa" if idx % 2 == 0 else "#f97316")
        box.set_alpha(0.65)
    ax.axhline(0.5, color="#334155", linestyle="--", linewidth=1)
    ax.set_xticks([p for p, _ in labels], [n for _, n in labels], rotation=20, ha="right")
    ax.set_ylabel("p(fake)")
    ax.set_title("Distribuição dos scores por classe real")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(
        handles=[
            plt.Line2D([0], [0], color="#60a5fa", lw=8, label="Real"),
            plt.Line2D([0], [0], color="#f97316", lw=8, label="Fake"),
        ],
        loc="best",
    )
    fig.tight_layout()
    _save_fig(fig, out / "score_distributions.png")
    plt.close(fig)


def _fig_single_confusion(name: str, y_true, scores, out: Path) -> None:
    import matplotlib.pyplot as plt
    from sklearn.metrics import confusion_matrix

    y = np.asarray(y_true, dtype=int)
    s = np.asarray(scores, dtype=float)
    if y.size == 0 or len(s) != len(y) or not np.isfinite(s).all():
        _write_placeholder_figure(
            out / "confusion_matrix.png",
            f"Matriz de confusão — {name}",
            "Dados insuficientes para gerar a matriz de confusão desta "
            "arquitetura.",
        )
        return
    cm = confusion_matrix(y, (s >= 0.5).astype(int), labels=[0, 1])
    fig, ax = plt.subplots(figsize=(4.5, 4.0))
    ax.imshow(cm, cmap="Blues")
    ax.set_title(f"Matriz de confusão — {name}")
    ax.set_xticks([0, 1], labels=["real", "fake"])
    ax.set_yticks([0, 1], labels=["real", "fake"])
    ax.set_xlabel("Predito")
    ax.set_ylabel("Verdadeiro")
    cm_max = int(np.max(cm)) if cm.size and np.max(cm) > 0 else 1
    for row in range(2):
        for col in range(2):
            color = "white" if cm[row, col] > cm_max / 2 else "#0f172a"
            ax.text(
                col,
                row,
                str(int(cm[row, col])),
                ha="center",
                va="center",
                color=color,
                fontweight="bold",
            )
    fig.tight_layout()
    _save_fig(fig, out / "confusion_matrix.png")
    plt.close(fig)


def _fig_single_roc(name: str, y_true, scores, out: Path) -> None:
    import matplotlib.pyplot as plt
    from sklearn.metrics import roc_curve

    y = np.asarray(y_true, dtype=int)
    s = np.asarray(scores, dtype=float)
    if (
        y.size == 0
        or len(s) != len(y)
        or len(np.unique(y)) < 2
        or not np.isfinite(s).all()
    ):
        _write_placeholder_figure(
            out / "roc.png",
            f"ROC — {name}",
            "Dados insuficientes para gerar a curva ROC desta arquitetura.",
        )
        return
    fpr, tpr, _ = roc_curve(y, s)
    fig, ax = plt.subplots(figsize=(5.0, 4.2))
    from sklearn.metrics import roc_auc_score
    try:
        auc = roc_auc_score(y, s)
    except Exception:  # noqa: BLE001
        auc = None
    ax.plot(fpr, tpr, lw=1.8,
            label=f"ROC (AUC={_num(auc).replace(chr(92), '')})")
    ax.plot([0, 1], [0, 1], "--", color="#94a3b8", lw=1)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal")
    ax.set_xlabel("FPR")
    ax.set_ylabel("TPR")
    ax.set_title(f"ROC — {name}")
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    _save_fig(fig, out / "roc.png")
    plt.close(fig)


def _fig_single_score_distribution(name: str, y_true, scores, out: Path) -> None:
    import matplotlib.pyplot as plt

    y = np.asarray(y_true, dtype=int)
    s = np.asarray(scores, dtype=float)
    if y.size == 0 or len(s) != len(y) or not np.isfinite(s).all():
        _write_placeholder_figure(
            out / "score_distribution.png",
            f"Scores por classe — {name}",
            "Dados insuficientes para plotar a distribuição de scores desta "
            "arquitetura.",
        )
        return
    fig, ax = plt.subplots(figsize=(5.0, 4.2))
    ax.boxplot(
        [s[y == 0], s[y == 1]],
        tick_labels=["real", "fake"],
        widths=0.45,
        patch_artist=True,
    )
    ax.axhline(0.5, color="#334155", linestyle="--", linewidth=1)
    ax.set_ylabel("p(fake)")
    ax.set_title(f"Scores por classe — {name}")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    _save_fig(fig, out / "score_distribution.png")
    plt.close(fig)


def _fig_single_convergence(name: str, history: Dict[str, Any], out: Path) -> None:
    import matplotlib.pyplot as plt

    if not history:
        _write_placeholder_figure(
            out / "convergence.png",
            f"Convergência — {name}",
            "Histórico por época não disponível para esta arquitetura. "
            "Modelos clássicos não possuem curva de loss/acurácia por época.",
        )
        return
    loss = history.get("loss") or []
    val_loss = history.get("val_loss") or []
    acc = (
        history.get("accuracy")
        or history.get("binary_accuracy")
        or history.get("categorical_accuracy")
        or history.get("sparse_categorical_accuracy")
        or history.get("acc")
        or []
    )
    val_acc = (
        history.get("val_accuracy")
        or history.get("val_binary_accuracy")
        or history.get("val_categorical_accuracy")
        or history.get("val_sparse_categorical_accuracy")
        or history.get("val_acc")
        or []
    )
    if not loss and not val_loss and not acc and not val_acc:
        _write_placeholder_figure(
            out / "convergence.png",
            f"Convergência — {name}",
            "Histórico de treino salvo, mas sem séries de loss/acurácia "
            "compatíveis para plotagem.",
        )
        return
    fig, axes = plt.subplots(1, 2, figsize=(8.0, 3.5))
    if loss:
        axes[0].plot(range(1, len(loss) + 1), loss, marker="o", label="train")
    if val_loss:
        axes[0].plot(range(1, len(val_loss) + 1), val_loss, marker="o", label="val")
    axes[0].set_title("Loss")
    axes[0].set_xlabel("Época")
    axes[0].grid(alpha=0.3)
    if axes[0].lines:
        axes[0].legend(fontsize=8)
    if acc:
        axes[1].plot(range(1, len(acc) + 1), acc, marker="o", label="train")
    if val_acc:
        axes[1].plot(range(1, len(val_acc) + 1), val_acc, marker="o", label="val")
    axes[1].set_title("Acurácia")
    axes[1].set_xlabel("Época")
    axes[1].grid(alpha=0.3)
    if axes[1].lines:
        axes[1].legend(fontsize=8)
    fig.suptitle(f"Convergência — {name}")
    fig.tight_layout()
    _save_fig(fig, out / "convergence.png")
    plt.close(fig)


# ───────────────────────── CSV / Markdown ─────────────────────────

def _write_csv(results, path: Path) -> None:
    snrs = results["config"]["snr_levels_db"]
    cols = ["arquitetura", "status", "tipo", "convergiu", "accuracy",
            "precision", "recall", "f1", "auc_roc", "eer", "min_tdcf",
            "params", "size_mb", "latency_ms"]
    for s in snrs:
        cols += [f"acc_snr{s}", f"eer_snr{s}"]
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(cols)
        for name, r in results["architectures"].items():
            if r.get("status") != "ok":
                w.writerow([name, r.get("status"), "", "", "", "", "", "", "",
                            "", "", "", "", ""] + [""] * (2 * len(snrs)))
                continue
            c, e = r["clean"], r["efficiency"]
            row = [name, "ok", r.get("type"), r.get("converged"),
                   c.get("accuracy"), c.get("precision"), c.get("recall"),
                   c.get("f1"), c.get("auc_roc"), c.get("eer"),
                   c.get("min_tdcf"), e.get("params"), e.get("size_mb"),
                   e.get("latency_ms")]
            for s in snrs:
                rob = r["robustness"].get(str(s), {})
                row += [rob.get("accuracy"), rob.get("eer")]
            w.writerow(row)


def _write_predictions_csv(results, path: Path) -> None:
    y = np.asarray(results["dataset"].get("y_test", []), dtype=int)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["architecture", "sample_index", "y_true", "p_fake", "y_pred", "correct"])
        if y.size == 0:
            return
        for name, r in results["architectures"].items():
            if r.get("status") != "ok" or not r.get("scores_clean"):
                continue
            scores = np.asarray(r["scores_clean"], dtype=float)
            if len(scores) != len(y) or not np.isfinite(scores).all():
                continue
            pred = (scores >= 0.5).astype(int)
            for idx, (yt, score, yp) in enumerate(zip(y, scores, pred)):
                w.writerow([
                    name,
                    idx,
                    int(yt),
                    round(float(score), 6),
                    int(yp),
                    bool(int(yt) == int(yp)),
                ])


def _write_arch_predictions_csv(name: str, r: Dict[str, Any], y_true, path: Path) -> None:
    y = np.asarray(y_true, dtype=int)
    scores = np.asarray(r.get("scores_clean") or [], dtype=float)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["sample_index", "y_true", "p_fake", "y_pred", "correct"])
        if y.size == 0 or len(scores) != len(y) or not np.isfinite(scores).all():
            return
        pred = (scores >= 0.5).astype(int)
        for idx, (yt, score, yp) in enumerate(zip(y, scores, pred)):
            w.writerow([idx, int(yt), round(float(score), 6), int(yp), bool(int(yt) == int(yp))])


def _write_arch_robustness_csv(r: Dict[str, Any], path: Path) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["condition", "accuracy", "precision", "recall", "f1", "auc_roc", "eer", "min_tdcf"])
        clean = r.get("clean") or {}
        w.writerow([
            "clean",
            clean.get("accuracy"),
            clean.get("precision"),
            clean.get("recall"),
            clean.get("f1"),
            clean.get("auc_roc"),
            clean.get("eer"),
            clean.get("min_tdcf"),
        ])
        for snr, metrics in (r.get("robustness") or {}).items():
            w.writerow([
                f"snr_{snr}db",
                metrics.get("accuracy"),
                metrics.get("precision"),
                metrics.get("recall"),
                metrics.get("f1"),
                metrics.get("auc_roc"),
                metrics.get("eer"),
                metrics.get("min_tdcf"),
            ])


def _write_arch_summary(name: str, r: Dict[str, Any], path: Path) -> None:
    if r.get("status") != "ok":
        path.write_text(f"# {name}\n\nStatus: erro\n\n{r.get('error', '')}\n", encoding="utf-8")
        return
    clean = r.get("clean") or {}
    eff = r.get("efficiency") or {}
    lines = [
        f"# {name}",
        "",
        f"- Status: {r.get('status')}",
        f"- Tipo: {r.get('type')}",
        f"- Convergiu: {r.get('converged')}",
        f"- Acurácia: {clean.get('accuracy')}",
        f"- EER: {clean.get('eer')}",
        f"- AUC-ROC: {clean.get('auc_roc')}",
        f"- min-tDCF: {clean.get('min_tdcf')}",
        f"- Latência ms/amostra: {eff.get('latency_ms')}",
        f"- Parâmetros: {eff.get('params')}",
        f"- Shape preparado: {(r.get('input_preparation') or {}).get('prepared_shape')}",
        "",
        "Arquivos nesta pasta: `metrics.json`, `predictions_clean.csv`, "
        "`robustness.csv`, `hyperparameter_tuning.*` quando aplicável, "
        "`models/*` e figuras individuais.",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def _write_per_architecture(results: Dict[str, Any], out: Path) -> None:
    root = out / "architectures"
    root.mkdir(parents=True, exist_ok=True)
    y = np.asarray(results["dataset"].get("y_test", []), dtype=int)
    for name, r in results["architectures"].items():
        arch_out = root / _slug(name)
        arch_out.mkdir(parents=True, exist_ok=True)
        arch_payload = {
            "architecture": name,
            "dataset": results.get("dataset"),
            "environment": results.get("environment"),
            **r,
        }
        (arch_out / "metrics.json").write_text(
            json.dumps(arch_payload, indent=2, ensure_ascii=False, default=str),
            encoding="utf-8",
        )
        _write_arch_summary(name, r, arch_out / "summary.md")
        if r.get("status") != "ok":
            continue
        _write_arch_predictions_csv(name, r, y, arch_out / "predictions_clean.csv")
        _write_arch_robustness_csv(r, arch_out / "robustness.csv")
        scores = r.get("scores_clean") or []
        _fig_single_confusion(name, y, scores, arch_out)
        _fig_single_roc(name, y, scores, arch_out)
        _fig_single_score_distribution(name, y, scores, arch_out)
        _fig_single_convergence(name, r.get("history") or {}, arch_out)


def _write_summary(results, path: Path) -> None:
    env = results["environment"]
    ds = results["dataset"]
    lines = [
        "# Benchmark XFakeSong — resumo\n",
        f"- Ambiente: {env.get('platform')} · Python {env.get('python')} · "
        f"TF {env.get('tensorflow')} · GPU={env.get('gpu')}",
        f"- Dispositivo: {env.get('device')}",
        f"- Dataset: **{ds['name']}** — {ds['n_total']} amostras "
        f"(teste held-out: {ds['n_test']} → {ds['balance_test']})",
        "",
        "| Arquitetura | Status | Conv. | Acur. | EER | AUC | min-tDCF | "
        "Lat.(ms) | Params |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for name, r in results["architectures"].items():
        if r.get("status") != "ok":
            lines.append(f"| {name} | ❌ {r.get('error','erro')[:30]} | | | | "
                         "| | | |")
            continue
        c, e = r["clean"], r["efficiency"]
        conv = "✅" if r.get("converged") else "⚠️"
        lines.append(
            f"| {name} | ok | {conv} | {_pct(c.get('accuracy'))} | "
            f"{_pct(c.get('eer'))} | {_num(c.get('auc_roc'))} | "
            f"{_num(c.get('min_tdcf'), 4)} | "
            f"{e.get('latency_ms')} | {e.get('params')} |"
        )
    if "api" in results:
        api = results["api"]
        lines += ["", f"## API ({api.get('status','?')})"]
        for ep in api.get("endpoints", []):
            lines.append(f"- `{ep.get('method')} {ep.get('path')}` → "
                         f"{ep.get('status_code')} ({ep.get('latency_ms')} ms)")
    path.write_text("\n".join(lines), encoding="utf-8")


def _write_tcc_report(results: Dict[str, Any], path: Path) -> None:
    """Relatório Markdown completo para anexar ao TCC.

    O relatório usa caminhos relativos, então continua navegável quando a pasta
    de resultados é movida inteira.
    """
    env = results.get("environment", {})
    ds = results.get("dataset", {})
    cfg = results.get("config", {})
    snrs = cfg.get("snr_levels_db", [])
    lines = [
        "# Relatório de Benchmark para TCC - XFakeSong",
        "",
        "## 1. Dataset",
        "",
        f"- Nome: `{ds.get('name')}`",
        f"- Total de amostras exportadas: `{ds.get('n_total')}`",
        f"- Amostras no teste held-out: `{ds.get('n_test')}`",
        f"- Shape de entrada bruto: `{ds.get('input_shape')}`",
        f"- Balanceamento no teste: `{ds.get('balance_test')}`",
        f"- Caminho de origem: `{ds.get('source') or ds.get('metadata', {}).get('source', '')}`",
        "",
        "## 2. Ambiente de Execução",
        "",
        f"- Plataforma: `{env.get('platform')}`",
        f"- Python: `{env.get('python')}`",
        f"- TensorFlow: `{env.get('tensorflow')}`",
        f"- GPU ativa: `{env.get('gpu')}`",
        f"- Dispositivo: `{env.get('device')}`",
        "",
        "## 3. Configuração Global do Benchmark",
        "",
        f"- Arquiteturas: `{', '.join(cfg.get('architectures', []))}`",
        f"- Épocas por arquitetura neural: `{cfg.get('epochs')}`",
        f"- Batch size: `{cfg.get('batch_size')}`",
        f"- Semente: `{cfg.get('seed')}`",
        f"- Testes de robustez SNR: `{snrs}`",
        f"- Medições de latência por arquitetura: `{cfg.get('latency_runs')}`",
        f"- API probe: `{cfg.get('run_api_probe')}`",
        "",
        "## 4. Resultados Numéricos",
        "",
        "| Arquitetura | Status | Acurácia | AUC-ROC | EER | min-tDCF | Latência ms | Params |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name, r in results.get("architectures", {}).items():
        if r.get("status") != "ok":
            lines.append(
                f"| {name} | erro |  |  |  |  |  |  |"
            )
            continue
        clean = r.get("clean") or {}
        eff = r.get("efficiency") or {}
        lines.append(
            f"| {name} | ok | {_num(clean.get('accuracy'), 4)} | "
            f"{_num(clean.get('auc_roc'), 4)} | {_num(clean.get('eer'), 4)} | "
            f"{_num(clean.get('min_tdcf'), 4)} | "
            f"{_num(eff.get('latency_ms'), 2)} | {eff.get('params')} |"
        )

    lines += [
        "",
        "## 5. Gráficos Agregados",
        "",
        "![Curvas ROC](figures/roc.png)",
        "",
        "![Matrizes de confusão](figures/confusion_matrices.png)",
        "",
        "![Distribuição de scores](figures/score_distributions.png)",
        "",
        "![Eficiência](figures/eficiencia.png)",
        "",
        "![Convergência](figures/convergencia.png)",
        "",
        "![Robustez](figures/robustez.png)",
        "",
        "## 6. Hiperparâmetros e Artefatos por Arquitetura",
        "",
    ]

    for name, r in results.get("architectures", {}).items():
        slug = _slug(name)
        lines += [
            f"### {name}",
            "",
            f"- Status: `{r.get('status')}`",
            f"- Tipo: `{r.get('type')}`",
            f"- Shape preparado: `{r.get('input_shape')}`",
            f"- Treino executado: `{_train_budget_label(r)}`",
            f"- Tempo total: `{r.get('wall_time_s')}` s",
            f"- Artefato do modelo: `{r.get('model_artifact')}`",
            "",
            "Hiperparâmetros/configuração de treino:",
            "",
            "```json",
            json.dumps(r.get("training_config") or {}, indent=2, ensure_ascii=False, default=str),
            "```",
            "",
        ]
        if r.get("status") == "ok":
            tuning = r.get("hyperparameter_tuning") or {}
            lines += [
                f"![Matriz de confusão — {name}](architectures/{slug}/confusion_matrix.png)",
                "",
                f"![ROC — {name}](architectures/{slug}/roc.png)",
                "",
                f"![Scores — {name}](architectures/{slug}/score_distribution.png)",
                "",
                f"![Convergência — {name}](architectures/{slug}/convergence.png)",
                "",
                f"- Métricas completas: `architectures/{slug}/metrics.json`",
                f"- Predições limpas: `architectures/{slug}/predictions_clean.csv`",
                f"- Robustez: `architectures/{slug}/robustness.csv`",
                *(
                    [
                        f"- Tuning de hiperparâmetros: `architectures/{slug}/hyperparameter_tuning.json`",
                        f"- Candidatos do grid: `architectures/{slug}/hyperparameter_tuning.csv`",
                    ]
                    if tuning.get("enabled")
                    else []
                ),
                "",
            ]
        else:
            lines += [f"- Erro: `{r.get('error')}`", ""]

    if results.get("api"):
        api = results["api"]
        lines += [
            "## 7. Verificação da API",
            "",
            f"- Status: `{api.get('status')}`",
            f"- Endpoints 2xx: `{api.get('n_2xx')}/{api.get('n_probed')}`",
            f"- Latência mediana: `{api.get('latency_ms_median')}` ms",
            "",
        ]
        for endpoint in api.get("endpoints", []):
            lines.append(
                f"- `{endpoint.get('method')} {endpoint.get('path')}` -> "
                f"{endpoint.get('status_code')} ({endpoint.get('latency_ms')} ms)"
            )

    path.write_text("\n".join(lines), encoding="utf-8")


# ───────────────────────── orquestração ─────────────────────────

def write_all(results: Dict[str, Any], output_dir: str) -> None:
    """Grava todos os artefatos do benchmark em `output_dir`."""
    _apply_style()  # tipografia/qualidade consistente em todas as figuras
    out = Path(output_dir)
    (out / "tables").mkdir(parents=True, exist_ok=True)
    (out / "figures").mkdir(parents=True, exist_ok=True)

    (out / "results.json").write_text(
        json.dumps(results, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )
    _write_csv(results, out / "results.csv")
    _write_predictions_csv(results, out / "predictions_clean.csv")
    _write_summary(results, out / "summary.md")
    _write_per_architecture(results, out)

    (out / "tables" / "tab_resultados.tex").write_text(
        _table_resultados(results), encoding="utf-8")
    (out / "tables" / "tab_eficiencia.tex").write_text(
        _table_eficiencia(results), encoding="utf-8")
    (out / "tables" / "tab_robustez.tex").write_text(
        _table_robustez(results), encoding="utf-8")

    for fig_fn in (
        _fig_roc,
        _fig_robustez,
        _fig_eficiencia,
        _fig_convergencia,
        _fig_confusion_matrices,
        _fig_score_distributions,
    ):
        try:
            fig_fn(results, out / "figures")
        except Exception as e:  # noqa: BLE001
            logger.warning("Figura %s falhou: %s", fig_fn.__name__, e)

    _write_tcc_report(results, out / "tcc_report.md")
    logger.info("Artefatos do benchmark gravados em: %s", out.resolve())
