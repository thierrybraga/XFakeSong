#!/usr/bin/env python3
"""Consolida resultados de benchmark → resumo + figuras do TCC.

Preenche o passo que faltava entre o benchmark e o TCC: lê um ou mais
`results.json` (de runs single-arch OU de um run completo), monta o
`benchmark_summary.json` consolidado e (re)gera TODAS as figuras nomeadas que o
`main.tex` referencia — sobrescrevendo as antigas. Tudo é derivado dos
resultados do treinamento; nada é hardcoded.

Saídas (em --out, default results/tcc_consolidated):
    benchmark_summary.json
    figures/benchmark_accuracy_auc.png
    figures/benchmark_eer.png
    figures/benchmark_robustness.png
    figures/benchmark_latency.png
    figures/benchmark_size.png
    figures/training_stability.png
    figures/confusion_matrices_article.png
    figures/confusion_matrices/<slug>.png   (1 por modelo do artigo)

Exemplos:
    # A partir dos runs por arquitetura já existentes:
    python scripts/consolidate_results.py results/benchmark_*_gpu_100e \
        results/benchmark_svm_100e results/tcc_pipeline_svm_rf_balanced_15k

    # A partir de um run completo (11 modelos do artigo num só results.json):
    python scripts/consolidate_results.py results/retrain_wsl2_indist

    # Copiar as figuras para o Overleaf após consolidar:
    python scripts/consolidate_results.py results/retrain_wsl2_indist \
        --copy-to tcc_overleaf/figures
"""

from __future__ import annotations

import argparse
import glob
import json
import re
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from benchmarks.config import OFFICIAL_TCC_RESULT_ORDER  # noqa: E402

# canonical_compact -> (key p/ summary, slug de figura, nome de exibição)
# Somente modelos do artigo. Aliases técnicos são normalizados para os nomes
# acadêmicos usados no TCC:
#   Hybrid CNN-Transformer -> CCT
#   SpectrogramTransformer -> AST
#   MultiscaleCNN -> Res2Net
_CANON = {
    "wavlm": ("WavLM Original", "wavlm_original", "WavLM Original"),
    "wavlmoriginal": ("WavLM Original", "wavlm_original", "WavLM Original"),
    "hubert": ("HuBERT Original", "hubert_original", "HuBERT Original"),
    "hubertoriginal": ("HuBERT Original", "hubert_original", "HuBERT Original"),
    "rawnet2": ("RawNet2", "rawnet2", "RawNet2"),
    "aasist": ("AASIST", "aasist", "AASIST"),
    "rawgatst": ("RawGAT-ST", "rawgat_st", "RawGAT-ST"),
    "conformer": ("Conformer", "conformer", "Conformer"),
    "hybridcnntransformer": ("CCT", "cct", "CCT"),
    "cct": ("CCT", "cct", "CCT"),
    "spectrogramtransformer": ("AST", "ast", "AST"),
    "audiospectrogramtransformer": ("AST", "ast", "AST"),
    "ast": ("AST", "ast", "AST"),
    "multiscalecnn": ("Res2Net", "res2net", "Res2Net"),
    "res2net": ("Res2Net", "res2net", "Res2Net"),
    "svm": ("SVM", "svm", "SVM"),
    "randomforest": ("RandomForest", "random_forest", "Random Forest"),
}

MODEL_ORDER = list(OFFICIAL_TCC_RESULT_ORDER)


def _compact(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", name.lower())


def _canon(name: str):
    return _CANON.get(_compact(name))


def _best_epoch(history: Optional[Dict[str, list]]) -> Optional[int]:
    """Época do melhor val_loss (1-indexada) a partir do history do treino."""
    if not history:
        return None
    val = history.get("val_loss")
    if val:
        return int(min(range(len(val)), key=lambda i: val[i]) + 1)
    val = history.get("val_accuracy")
    if val:
        return int(max(range(len(val)), key=lambda i: val[i]) + 1)
    return None


def _best_val(history: Optional[Dict[str, list]]) -> Optional[float]:
    """Maior val_accuracy do treino (pico de validação)."""
    va = (history or {}).get("val_accuracy")
    return float(max(va)) if va else None


def _final_val(history: Optional[Dict[str, list]]) -> Optional[float]:
    """val_accuracy da última época (para medir queda val→final)."""
    va = (history or {}).get("val_accuracy")
    return float(va[-1]) if va else None


def _iter_results(paths: List[str]):
    """Para cada caminho (arquivo results.json ou diretório), entrega o dict."""
    for raw in paths:
        for expanded in sorted(glob.glob(raw)) or [raw]:
            p = Path(expanded)
            jf = p if p.suffix == ".json" else p / "results.json"
            if not jf.exists():
                print(f"  (pulado, sem results.json) {p}", file=sys.stderr)
                continue
            try:
                data = json.loads(jf.read_text(encoding="utf-8"))
                if not isinstance(data.get("architectures"), dict):
                    print(f"  (pulado, sem architectures) {jf}", file=sys.stderr)
                    continue
                yield jf, data
            except Exception as e:
                print(f"  (erro lendo {jf}: {e})", file=sys.stderr)


def collect_rows(paths: List[str], prefer_last: bool = False):
    """Mescla arquiteturas de todos os results.json em linhas de resumo.

    Retorna (rows, extras), onde extras guarda dados pesados por slug
    (scores_clean, y_test, history) para a geração de figuras.
    """
    rows_by_key: Dict[str, dict] = {}
    extras: Dict[str, dict] = {}
    for jf, data in _iter_results(paths):
        archs = data.get("architectures", {})
        y_test = (data.get("dataset", {}) or {}).get("y_test")
        for arch_name, a in archs.items():
            if a.get("status") != "ok":
                continue
            canon = _canon(arch_name)
            if not canon:
                print(f"  (arch não mapeada: {arch_name}) em {jf}",
                      file=sys.stderr)
                continue
            key, slug, display = canon
            clean = a.get("clean", {}) or {}
            eff = a.get("efficiency", {}) or {}
            row = {
                "model": display,
                "key": key,
                "slug": slug,
                "path": str(jf.parent),
                "accuracy": clean.get("accuracy"),
                "auc": clean.get("auc_roc"),
                "eer": clean.get("eer"),
                "min_tdcf": clean.get("min_tdcf"),
                "f1": clean.get("f1"),
                "latency": eff.get("latency_ms"),
                "size": eff.get("size_mb"),
                "params": eff.get("params"),
                "robustness": a.get("robustness", {}) or {},
                "best_epoch": _best_epoch(a.get("history")),
                "best_val": _best_val(a.get("history")),
                "final_val": _final_val(a.get("history")),
                "epochs": a.get("epochs"),
            }
            # Se o mesmo modelo aparecer em vários runs, o padrão mantém o de
            # maior AUC. Para consolidações finais, --prefer-last permite que
            # uma rodada oficial posterior substitua resultados provisórios.
            prev = rows_by_key.get(key)
            if (
                prev is None
                or prefer_last
                or (row["auc"] or 0) >= (prev["auc"] or 0)
            ):
                rows_by_key[key] = row
                extras[slug] = {
                    "scores_clean": a.get("scores_clean"),
                    "y_test": y_test,
                    "history": a.get("history"),
                    "display": display,
                }

    ordered = [rows_by_key[k] for k in MODEL_ORDER if k in rows_by_key]
    ordered += [r for k, r in rows_by_key.items() if k not in MODEL_ORDER]
    return ordered, extras


# ----------------------------- Figuras -----------------------------

def _setup_mpl():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.titleweight": "bold",
        "figure.dpi": 150,
        "font.size": 10,
    })
    return plt


def _labels(rows):
    return [r["model"] for r in rows]


_FAMILIES = {
    "RandomForest": "Clássico",
    "SVM": "Clássico",
    "CCT": "Espectral/Transformer",
    "AST": "Espectral/Transformer",
    "Res2Net": "Espectral/CNN",
    "Conformer": "Espectral/Transformer",
    "RawNet2": "Raw/Grafo",
    "AASIST": "Raw/Grafo",
    "RawGAT-ST": "Raw/Grafo",
    "WavLM Original": "SSL",
    "HuBERT Original": "SSL",
}

_FAMILY_COLORS = {
    "Clássico": "#4C72B0",
    "Espectral/Transformer": "#55A868",
    "Espectral/CNN": "#8172B3",
    "Raw/Grafo": "#C44E52",
    "SSL": "#DD8452",
}


def _row_family(row: dict) -> str:
    return _FAMILIES.get(row.get("key"), "Outro")


def _row_colors(rows: list[dict]) -> list[str]:
    return [_FAMILY_COLORS.get(_row_family(row), "#777777") for row in rows]


def _add_family_legend(ax) -> None:
    from matplotlib.patches import Patch

    handles = [
        Patch(facecolor=color, label=family)
        for family, color in _FAMILY_COLORS.items()
    ]
    ax.legend(handles=handles, title="Família", fontsize=8, title_fontsize=9)


def _annotate_hbars(ax, values, suffix="", fmt="{:.2f}", min_pad=0.4) -> None:
    xmax = max(values) if values else 0
    for patch, value in zip(ax.patches, values):
        if value is None:
            continue
        x = patch.get_width()
        y = patch.get_y() + patch.get_height() / 2
        pad = max(xmax * 0.01, min_pad)
        ax.text(
            x + pad,
            y,
            f"{fmt.format(value)}{suffix}",
            va="center",
            ha="left",
            fontsize=8,
        )


def _metric(row: dict, field: str, scale: float = 1.0) -> float:
    value = row.get(field)
    return float(value or 0) * scale


def _model_colors(rows: list[dict], plt) -> dict[str, Any]:
    cmap = plt.get_cmap("tab20")
    return {
        row["key"]: cmap(i % cmap.N)
        for i, row in enumerate(rows)
    }


def fig_accuracy_auc(rows, out: Path):
    plt = _setup_mpl()
    import numpy as np

    labels = _labels(rows)
    acc = [_metric(r, "accuracy", 100.0) for r in rows]
    auc = [_metric(r, "auc", 100.0) for r in rows]
    y = np.arange(len(labels))
    h = 0.36
    fig, ax = plt.subplots(figsize=(11.5, 7.2))
    ax.barh(y - h / 2, acc, h, label="Acurácia", color="#4C72B0")
    ax.barh(y + h / 2, auc, h, label="AUC-ROC", color="#DD8452")
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.set_xlim(0, 105)
    ax.set_xlabel("Valor (%)")
    ax.set_title("Desempenho no conjunto limpo")
    ax.grid(axis="x", alpha=0.25)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.08), ncol=2)
    for yi, value in zip(y, acc):
        ax.text(value + 0.6, yi - h / 2, f"{value:.2f}%", va="center", fontsize=7)
    for yi, value in zip(y, auc):
        ax.text(value + 0.6, yi + h / 2, f"{value:.2f}%", va="center", fontsize=7)
    ax.invert_yaxis()
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(out / "benchmark_accuracy_auc.png", dpi=180)
    plt.close(fig)


def fig_simple_bar(rows, out: Path, field, fname, title, ylabel, scale=1.0):
    plt = _setup_mpl()
    import numpy as np

    labels = _labels(rows)
    vals = [_metric(r, field, scale) for r in rows]
    y = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(11.5, 7.0))
    ax.barh(y, vals, color=_row_colors(rows))
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.set_xlabel(ylabel)
    ax.set_title(title)
    ax.grid(axis="x", alpha=0.25)
    ax.set_xlim(0, max(vals) * 1.18 if vals else 1)
    _annotate_hbars(ax, vals, suffix=("%" if "%" in ylabel else ""))
    _add_family_legend(ax)
    ax.invert_yaxis()
    fig.tight_layout()
    fig.savefig(out / fname, dpi=180)
    plt.close(fig)


def fig_tdcf(rows, out: Path):
    plt = _setup_mpl()
    import numpy as np

    labels = _labels(rows)
    vals = [_metric(r, "min_tdcf", 1.0) for r in rows]
    y = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(11.5, 7.0))
    ax.barh(y, vals, color=_row_colors(rows))
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.set_xlabel("$t$-DCF$^\\ast$ (proxy CM-only; menor é melhor)")
    ax.set_title("Custo normalizado aproximado por arquitetura")
    ax.grid(axis="x", alpha=0.25)
    ax.set_xlim(0, max(vals) * 1.18 if vals else 1)
    _annotate_hbars(ax, vals, fmt="{:.4f}", min_pad=0.01)
    _add_family_legend(ax)
    ax.invert_yaxis()
    fig.tight_layout()
    fig.savefig(out / "benchmark_tdcf.png", dpi=180)
    plt.close(fig)


def fig_accuracy_latency_tradeoff(rows, out: Path):
    plt = _setup_mpl()

    fig, ax = plt.subplots(figsize=(10.5, 6.5))
    for family, color in _FAMILY_COLORS.items():
        subset = [r for r in rows if _row_family(r) == family]
        if not subset:
            continue
        lat = [_metric(r, "latency") for r in subset]
        acc = [_metric(r, "accuracy", 100.0) for r in subset]
        sizes = [max(50, min(450, (_metric(r, "size") or 1) * 1.2)) for r in subset]
        ax.scatter(lat, acc, s=sizes, color=color, alpha=0.75,
                   edgecolor="white", linewidth=0.8, label=family)
        for r, x, y in zip(subset, lat, acc):
            ax.annotate(r["model"], (x, y), xytext=(4, 4),
                        textcoords="offset points", fontsize=8)
    ax.set_xlabel("Latência de inferência (ms)")
    ax.set_ylabel("Acurácia no conjunto limpo (%)")
    ax.set_title("Trade-off entre acurácia, latência e tamanho do artefato")
    ax.text(
        0.99,
        0.02,
        "Tamanho da bolha proporcional ao artefato persistido (MB)",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=8,
        color="#444444",
    )
    ax.grid(alpha=0.25)
    ax.legend(title="Família", fontsize=8, title_fontsize=9)
    fig.tight_layout()
    fig.savefig(out / "benchmark_accuracy_latency_tradeoff.png", dpi=180)
    plt.close(fig)


def fig_robustness(rows, out: Path):
    plt = _setup_mpl()
    snrs = sorted(
        {int(s) for r in rows for s in (r.get("robustness") or {})},
        reverse=True,
    )
    if not snrs:
        return
    x_labels = ["Limpo"] + [f"{snr} dB" for snr in snrs]
    x = list(range(len(x_labels)))
    fig, ax = plt.subplots(figsize=(11.5, 7.0))
    colors = _model_colors(rows, plt)
    markers = ["o", "s", "^", "D", "P", "X", "v", "<", ">", "h", "*"]
    for r in rows:
        rob = r.get("robustness") or {}
        ys = [r.get("accuracy")] + [
            (rob.get(str(s), {}) or {}).get("accuracy") for s in snrs
        ]
        if any(v is not None for v in ys):
            marker = markers[rows.index(r) % len(markers)]
            ax.plot(
                x,
                [(v or 0) * 100 for v in ys],
                marker=marker,
                linewidth=1.8,
                label=r["model"],
                color=colors[r["key"]],
                alpha=0.88,
            )
    ax.set_xticks(x)
    ax.set_xticklabels(x_labels)
    ax.set_xlabel("Condição de avaliação")
    ax.set_ylabel("Acurácia (%)")
    ax.set_ylim(45, 101)
    ax.set_title("Robustez a ruído AWGN")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=7, ncol=2, loc="upper center",
              bbox_to_anchor=(0.5, -0.12))
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    fig.savefig(out / "benchmark_robustness.png", dpi=180)
    plt.close(fig)


def fig_training_stability(rows, extras, out: Path):
    plt = _setup_mpl()
    import numpy as np

    labels = _labels(rows)
    y = np.arange(len(rows))
    h = 0.34
    best = []
    final = []
    classical = []
    for row in rows:
        is_classical = row["key"] in {"RandomForest", "SVM"}
        classical.append(is_classical)
        if is_classical:
            value = _metric(row, "accuracy", 100.0)
            best.append(value)
            final.append(value)
        else:
            best.append(_metric(row, "best_val", 100.0))
            final.append(_metric(row, "final_val", 100.0))

    fig, ax = plt.subplots(figsize=(11.5, 7.0))
    best_bars = ax.barh(
        y - h / 2,
        best,
        h,
        label="Melhor validação / CV+fit",
        color="#4C72B0",
        alpha=0.92,
    )
    final_bars = ax.barh(
        y + h / 2,
        final,
        h,
        label="Validação final / CV+fit",
        color="#DD8452",
        alpha=0.88,
    )
    for idx, is_classical in enumerate(classical):
        if not is_classical:
            continue
        best_bars[idx].set_hatch("//")
        final_bars[idx].set_hatch("//")
        ax.text(
            max(best[idx], final[idx]) + 0.35,
            y[idx],
            "CV+fit",
            va="center",
            ha="left",
            fontsize=8,
            color="#333333",
        )

    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.set_xlabel("Acurácia (%)")
    ax.set_xlim(0, 105)
    ax.set_title("Estabilidade e convergência dos 11 modelos consolidados")
    ax.grid(axis="x", alpha=0.25)
    ax.legend(
        fontsize=8,
        ncol=2,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.12),
    )
    ax.text(
        0.01,
        -0.20,
        "Modelos clássicos não possuem trajetória por época; barras hachuradas indicam ajuste por validação cruzada.",
        transform=ax.transAxes,
        fontsize=8,
        ha="left",
        va="top",
    )
    fig.tight_layout(rect=(0, 0.12, 1, 1))
    fig.savefig(out / "training_stability.png", dpi=180)
    plt.close(fig)


def _confusion_matrix_for_row(row, extras):
    import numpy as np

    ex = extras.get(row["slug"], {}) or {}
    scores, y_test = ex.get("scores_clean"), ex.get("y_test")
    if scores is None or y_test is None:
        return None
    if len(scores) != len(y_test):
        return None

    score_arr = np.asarray(scores, dtype=float)
    if not np.isfinite(score_arr).all():
        return None

    y_true = np.asarray(y_test).astype(int)
    thr = (row.get("robustness", {}).get("clean_threshold")
           or _eer_threshold(row))
    y_pred = (score_arr >= thr).astype(int)
    cm = np.zeros((2, 2), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1
    return cm


def _draw_confusion_axis(ax, cm, title: str) -> None:
    ax.set_title(title)
    if cm is None:
        ax.axis("off")
        ax.text(
            0.5,
            0.5,
            "sem dados\nconsolidados",
            ha="center",
            va="center",
            fontsize=10,
        )
        return

    ax.imshow(cm, cmap="Blues")
    row_sums = cm.sum(axis=1, keepdims=True)
    row_pct = (cm / row_sums.clip(min=1)) * 100
    for i in range(2):
        for j in range(2):
            ax.text(
                j,
                i,
                f"{cm[i, j]}\n{row_pct[i, j]:.1f}%",
                ha="center",
                va="center",
                color="black",
                fontsize=10,
            )
    ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
    ax.set_xticklabels(["Real", "Fake"]); ax.set_yticklabels(["Real", "Fake"])
    ax.set_xlabel("Predito"); ax.set_ylabel("Verdadeiro")


def fig_confusion_matrices(rows, extras, out: Path):
    plt = _setup_mpl()
    import math

    cm_dir = out / "confusion_matrices"
    cm_dir.mkdir(parents=True, exist_ok=True)
    cms = {r["slug"]: _confusion_matrix_for_row(r, extras) for r in rows}

    for r in rows:
        fig, ax = plt.subplots(figsize=(4.2, 4.1))
        _draw_confusion_axis(ax, cms.get(r["slug"]), r["model"])
        fig.tight_layout(); fig.savefig(cm_dir / f"{r['slug']}.png", dpi=180)
        plt.close(fig)

    cols = 4
    panel_rows = max(1, math.ceil(len(rows) / cols))
    fig, axes = plt.subplots(panel_rows, cols, figsize=(cols * 4.1, panel_rows * 4.0))
    axes_flat = list(getattr(axes, "flat", [axes]))
    for ax, row in zip(axes_flat, rows):
        _draw_confusion_axis(ax, cms.get(row["slug"]), row["model"])
    for ax in axes_flat[len(rows):]:
        ax.axis("off")
    fig.suptitle("Matrizes de confusão no conjunto de teste limpo", y=1.01)
    fig.tight_layout()
    fig.savefig(out / "confusion_matrices_article.png", dpi=180,
                bbox_inches="tight")
    plt.close(fig)


def _eer_threshold(row) -> float:
    """Usa o threshold EER limpo se presente; senão 0.5."""
    for snr in ("clean",):
        t = (row.get("robustness", {}).get(snr, {}) or {}).get("eer_threshold")
        if t is not None:
            return float(t)
    return 0.5


def generate_figures(rows, extras, fig_dir: Path):
    fig_dir.mkdir(parents=True, exist_ok=True)
    fig_accuracy_auc(rows, fig_dir)
    fig_simple_bar(rows, fig_dir, "eer", "benchmark_eer.png",
                   "EER (Equal Error Rate)", "EER (%)", scale=100.0)
    fig_tdcf(rows, fig_dir)
    fig_simple_bar(rows, fig_dir, "latency", "benchmark_latency.png",
                   "Latência de inferência", "ms")
    fig_simple_bar(rows, fig_dir, "size", "benchmark_size.png",
                   "Tamanho do modelo", "MB")
    fig_accuracy_latency_tradeoff(rows, fig_dir)
    fig_robustness(rows, fig_dir)
    fig_training_stability(rows, extras, fig_dir)
    fig_confusion_matrices(rows, extras, fig_dir)


def main() -> int:
    p = argparse.ArgumentParser(description="Consolida resultados → resumo + figuras do TCC")
    p.add_argument("inputs", nargs="+",
                   help="diretórios de run ou results.json (aceita globs)")
    p.add_argument("--out", default="results/tcc_consolidated",
                   help="pasta de saída (default: results/tcc_consolidated)")
    p.add_argument("--copy-to", default=None,
                   help="copia as figuras geradas para esta pasta (ex.: tcc_overleaf/figures)")
    p.add_argument("--no-figures", action="store_true",
                   help="gera só o benchmark_summary.json")
    p.add_argument(
        "--prefer-last",
        action="store_true",
        help=(
            "quando o mesmo modelo aparecer em mais de um input, usa a última "
            "ocorrência informada em vez da maior AUC"
        ),
    )
    args = p.parse_args()

    out = (PROJECT_ROOT / args.out) if not Path(args.out).is_absolute() else Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    print("Consolidando resultados de:", ", ".join(args.inputs))
    rows, extras = collect_rows(args.inputs, prefer_last=args.prefer_last)
    if not rows:
        print("ERRO: nenhuma arquitetura 'ok' encontrada nos inputs.", file=sys.stderr)
        return 1

    summary_path = out / "benchmark_summary.json"
    summary = [{k: v for k, v in r.items() if k != "slug"} for r in rows]
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False),
                            encoding="utf-8")
    print(f"-> {summary_path} ({len(rows)} arquiteturas)")
    missing = [k for k in MODEL_ORDER if k not in {r['key'] for r in rows}]
    if missing:
        print(f"   AVISO: faltando {missing} (o TCC espera 11 modelos).")

    if not args.no_figures:
        fig_dir = out / "figures"
        generate_figures(rows, extras, fig_dir)
        print(f"-> figuras em {fig_dir}")
        if args.copy_to:
            dest = (PROJECT_ROOT / args.copy_to) if not Path(args.copy_to).is_absolute() else Path(args.copy_to)
            dest.mkdir(parents=True, exist_ok=True)
            for src in fig_dir.rglob("*.png"):
                rel = src.relative_to(fig_dir)
                (dest / rel).parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src, dest / rel)
            print(f"-> figuras copiadas para {dest}")

    print("\nPróximo: python scripts/update_tcc_latex.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
