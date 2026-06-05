"""Geração dos artefatos do benchmark: JSON, CSV, Markdown, LaTeX e figuras."""

from __future__ import annotations

import csv
import json
import logging
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


# ───────────────────────── tabelas LaTeX ─────────────────────────

def _ok_items(results) -> List:
    arch = results["architectures"]
    return [(name, r) for name, r in arch.items() if r.get("status") == "ok"]


def _table_resultados(results) -> str:
    rows = []
    for name, r in results["architectures"].items():
        if r.get("status") != "ok":
            rows.append(f"{name} & \\multicolumn{{6}}{{c}}{{"
                        f"\\textit{{falhou: {r.get('error','?')[:40]}}}}} \\\\")
            continue
        c, e = r["clean"], r["efficiency"]
        rows.append(
            f"{name} & {_pct(c.get('accuracy'))} & {_pct(c.get('eer'))} & "
            f"{_num(c.get('auc_roc'))} & {_num(c.get('min_tdcf'),4)} & "
            f"{_num(e.get('latency_ms'),1)} & {r.get('epochs','--')} & "
            f"{_conv(r.get('converged'))} \\\\"
        )
    body = "\n".join(rows)
    return (
        "\\begin{table}[H]\n\\centering\n"
        "\\caption{Desempenho das arquiteturas (conjunto de teste, "
        f"{results['dataset']['n_test']} amostras). Gerado pelo benchmark.}}\n"
        "\\label{tab:bench_resultados}\n\\small\\singlespacing\n"
        "\\begin{tabular}{lcccccc}\n\\toprule\n"
        "\\textbf{Arquitetura} & \\textbf{Acur.} & \\textbf{EER} & "
        "\\textbf{AUC-ROC} & \\textbf{min-tDCF} & \\textbf{Lat.\\,(ms)} & "
        "\\textbf{Époc.} & \\textbf{Conv.?} \\\\\n\\midrule\n"
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
    body = "\n".join(rows) or "\\multicolumn{9}{c}{(nenhum modelo convergente)}\\\\"
    cols = "l" + "cc" * (len(snrs) + 1)
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

def _fig_roc(results, out: Path) -> None:
    import matplotlib.pyplot as plt
    from sklearn.metrics import roc_curve

    y = np.asarray(results["dataset"].get("y_test", []))
    items = [(n, r) for n, r in _ok_items(results) if r.get("scores_clean")]
    if y.size == 0 or not items or len(np.unique(y)) < 2:
        return
    fig, ax = plt.subplots(figsize=(6, 5))
    for name, r in items:
        s = np.asarray(r["scores_clean"], dtype=float)
        if len(s) != len(y):
            continue
        fpr, tpr, _ = roc_curve(y, s)
        auc = r["clean"].get("auc_roc")
        ax.plot(fpr, tpr, lw=1.8,
                label=f"{name} (AUC={_num(auc).replace(chr(92),'')})")
    ax.plot([0, 1], [0, 1], "--", color="#94a3b8", lw=1)
    ax.set_xlabel("Taxa de falsos positivos (FPR)")
    ax.set_ylabel("Taxa de verdadeiros positivos (TPR)")
    ax.set_title("Curvas ROC (conjunto de teste limpo)")
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out / "roc.png", dpi=150)
    plt.close(fig)


def _fig_robustez(results, out: Path) -> None:
    import matplotlib.pyplot as plt

    snrs = results["config"]["snr_levels_db"]
    conv = [(n, r) for n, r in _ok_items(results) if r.get("converged")]
    if not conv:
        return
    xs = ["Limpo"] + [f"{s} dB" for s in snrs]
    fig, ax = plt.subplots(figsize=(7, 4.2))
    for name, r in conv:
        ys = [r["clean"].get("accuracy", np.nan)]
        ys += [r["robustness"].get(str(s), {}).get("accuracy", np.nan)
               for s in snrs]
        ax.plot(xs, [v * 100 for v in ys], marker="o", lw=1.8, label=name)
    ax.axhline(50, ls="--", color="#94a3b8", lw=1, label="acaso (50%)")
    ax.set_ylabel("Acurácia (%)")
    ax.set_xlabel("Condição de ruído (SNR decrescente →)")
    ax.set_title("Robustez sob ruído AWGN")
    ax.set_ylim(40, 102)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out / "robustez.png", dpi=150)
    plt.close(fig)


def _fig_eficiencia(results, out: Path) -> None:
    import matplotlib.pyplot as plt

    items = [(n, r) for n, r in _ok_items(results)
             if r["efficiency"].get("latency_ms") is not None]
    if not items:
        return
    fig, ax = plt.subplots(figsize=(7, 4.2))
    for name, r in items:
        lat = r["efficiency"]["latency_ms"]
        acc = r["clean"].get("accuracy", np.nan) * 100
        color = "#10b981" if r.get("converged") else "#ef4444"
        ax.scatter(lat, acc, s=80, color=color, edgecolor="#1e293b", zorder=3)
        ax.annotate(name, (lat, acc), fontsize=7, xytext=(4, 4),
                    textcoords="offset points")
    ax.set_xlabel("Latência (ms/amostra)")
    ax.set_ylabel("Acurácia (%)")
    ax.set_title("Eficiência × desempenho (verde=convergiu)")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out / "eficiencia.png", dpi=150)
    plt.close(fig)


def _fig_convergencia(results, out: Path) -> None:
    import matplotlib.pyplot as plt

    items = [(n, r) for n, r in _ok_items(results) if r.get("history")]
    if not items:
        return
    fig, ax = plt.subplots(figsize=(7, 4.2))
    for name, r in items:
        h = r["history"]
        va = h.get("val_accuracy") or h.get("val_acc")
        if not va:
            continue
        ax.plot(range(1, len(va) + 1), [v * 100 for v in va],
                marker="o", ms=3, lw=1.6, label=name)
    ax.set_xlabel("Época")
    ax.set_ylabel("Acurácia de validação (%)")
    ax.set_title("Convergência do treino")
    ax.set_ylim(40, 102)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out / "convergencia.png", dpi=150)
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
            f"| {name} | ok | {conv} | {c.get('accuracy',0)*100:.2f}% | "
            f"{c.get('eer',float('nan'))*100:.2f}% | {c.get('auc_roc',0):.3f} | "
            f"{c.get('min_tdcf',float('nan')):.4f} | "
            f"{e.get('latency_ms')} | {e.get('params')} |"
        )
    if "api" in results:
        api = results["api"]
        lines += ["", f"## API ({api.get('status','?')})"]
        for ep in api.get("endpoints", []):
            lines.append(f"- `{ep.get('method')} {ep.get('path')}` → "
                         f"{ep.get('status_code')} ({ep.get('latency_ms')} ms)")
    path.write_text("\n".join(lines), encoding="utf-8")


# ───────────────────────── orquestração ─────────────────────────

def write_all(results: Dict[str, Any], output_dir: str) -> None:
    """Grava todos os artefatos do benchmark em `output_dir`."""
    out = Path(output_dir)
    (out / "tables").mkdir(parents=True, exist_ok=True)
    (out / "figures").mkdir(parents=True, exist_ok=True)

    (out / "results.json").write_text(
        json.dumps(results, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )
    _write_csv(results, out / "results.csv")
    _write_summary(results, out / "summary.md")

    (out / "tables" / "tab_resultados.tex").write_text(
        _table_resultados(results), encoding="utf-8")
    (out / "tables" / "tab_eficiencia.tex").write_text(
        _table_eficiencia(results), encoding="utf-8")
    (out / "tables" / "tab_robustez.tex").write_text(
        _table_robustez(results), encoding="utf-8")

    for fig_fn in (_fig_roc, _fig_robustez, _fig_eficiencia, _fig_convergencia):
        try:
            fig_fn(results, out / "figures")
        except Exception as e:  # noqa: BLE001
            logger.warning("Figura %s falhou: %s", fig_fn.__name__, e)

    logger.info("Artefatos do benchmark gravados em: %s", out.resolve())
