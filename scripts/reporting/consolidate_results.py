#!/usr/bin/env python3
"""Consolida resultados de benchmark → resumo + figuras do TCC.

Preenche o passo que faltava entre o benchmark e o TCC: lê um ou mais
`results.json` (de runs single-arch OU de um run completo), monta o
`benchmark_summary.json` consolidado e (re)gera TODAS as figuras nomeadas que o
`main.tex` referencia — sobrescrevendo as antigas. Tudo é derivado dos
resultados do treinamento; nada é hardcoded.

Saídas (em --out, default data/results/paper/consolidated):
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
    python scripts/reporting/consolidate_results.py data/results/benchmark_*_gpu_100e \
        data/results/benchmark_svm_100e data/results/tcc_pipeline_svm_rf_balanced_15k

    # A partir de um run completo (11 modelos do artigo num só results.json):
    python scripts/reporting/consolidate_results.py data/results/retrain_wsl2_indist

    # Copiar as figuras para o artigo após consolidar:
    python scripts/reporting/consolidate_results.py data/results/retrain_wsl2_indist \
        --copy-to data/results/paper/figures
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

PROJECT_ROOT = Path(__file__).resolve().parents[2]
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


def _results_files(p: Path) -> List[Path]:
    """Localiza os `results.json` de um caminho de entrada.

    Um run de `run_models_sequential.py` grava UM subdiretório por modelo
    (`<run>/<slug>/results.json`) e nenhum `results.json` na raiz. Até
    2026-08-09 esta função só olhava a raiz, então o comando documentado no
    `data/results/paper/README.md` e no checklist de promoção —
    `consolidate_results.py data/results/clean_benchmark_15k` — saía com
    "nenhuma arquitetura 'ok' encontrada" e exigia um glob que a documentação
    não menciona.
    """
    if p.suffix == ".json":
        return [p] if p.exists() else []
    direct = p / "results.json"
    if direct.exists():
        return [direct]
    return sorted(p.glob("*/results.json"))


def _iter_results(paths: List[str]):
    """Para cada caminho (arquivo results.json ou diretório), entrega o dict."""
    for raw in paths:
        for expanded in sorted(glob.glob(raw)) or [raw]:
            found = _results_files(Path(expanded))
            if not found:
                print(f"  (pulado, sem results.json) {expanded}", file=sys.stderr)
                continue
            for jf in found:
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
            dataset_info = data.get("dataset", {}) or {}
            run_config = data.get("config", {}) or {}
            training_config = a.get("training_config", {}) or {}
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
                # A latência de runtimes diferentes não é comparável entre si
                # (Keras/TF x PyTorch x sklearn). Propagado para que a figura de
                # tradeoff marque a diferença em vez de sugerir uma escala só.
                "latency_runtime": (
                    (eff.get("latency_profile") or {}).get("runtime") or "unknown"
                ),
                "latency_profile": eff.get("latency_profile") or {},
                "size": eff.get("size_mb"),
                "params": eff.get("params"),
                "training_stability": a.get("training_stability") or {},
                "robustness": a.get("robustness", {}) or {},
                # Pior locutor do teste. O protocolo é speaker-disjoint (11
                # locutores no teste, nenhum visto no treino), então esta é a
                # leitura de generalização — o agregado esconde, por exemplo,
                # os 74,2% do RawNet2 em M026 dentro de 95,88% médios.
                "grouped_clean": a.get("grouped_clean") or {},
                "worst_speaker_accuracy": (
                    ((a.get("grouped_clean") or {}).get("speaker") or {})
                    .get("worst_group_accuracy")
                ),
                "n_speakers": (
                    ((a.get("grouped_clean") or {}).get("speaker") or {})
                    .get("n_groups")
                ),
                "best_epoch": _best_epoch(a.get("history")),
                "best_val": _best_val(a.get("history")),
                "final_val": _final_val(a.get("history")),
                "epochs": a.get("epochs"),
                "epochs_budget": training_config.get("epochs_budget", training_config.get("epochs")),
                "training_config": training_config,
                "noise_protocol": (
                    a.get("noise_protocol")
                    or training_config.get("noise_protocol")
                    or a.get("input_preparation")
                ),
                "input_preparation": a.get("input_preparation"),
                "decision_threshold": run_config.get("decision_threshold", 0.5),
                "seed": run_config.get("seed"),
                "dataset": {
                    "name": dataset_info.get("name"),
                    "source": dataset_info.get("source"),
                    "n_total": dataset_info.get("n_total"),
                    "n_test": dataset_info.get("n_test"),
                    "split_source": dataset_info.get("split_source"),
                    "split_overlap_audit": dataset_info.get("split_overlap_audit"),
                    "provenance_overlap_audit": dataset_info.get("provenance_overlap_audit"),
                    # Identidade do conjunto de teste. A comparação pareada só
                    # é válida se for a MESMA em todos os modelos — as variantes
                    # de 15k e 40k têm fingerprints distintos e não podem cair
                    # na mesma consolidação.
                    "test_split_sha256": dataset_info.get("test_split_sha256"),
                },
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
                    # Unidade de reamostragem do teste, para o teste PAREADO
                    # entre modelos. Runs anteriores a 2026-08-09 não gravam a
                    # chave; nesse caso a comparação cai para amostra e declara.
                    "test_cluster_ids": dataset_info.get("test_cluster_ids"),
                    # Segunda unidade de reamostragem. A frase (183 clusters) é
                    # mais fina que o LOCUTOR (11), e é o locutor que casa com a
                    # alegação speaker-disjoint — reamostrar frases trata frases
                    # do mesmo locutor como independentes.
                    "test_speaker_ids": dataset_info.get("test_speaker_ids"),
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


#: Marcador por runtime da medição de latência. O eixo x mistura pilhas de
#: execução diferentes — Keras/TF, PyTorch e scikit-learn —, e a diferença entre
#: elas é da mesma ordem da diferença entre arquiteturas. Colorir por família e
#: marcar por runtime deixa o confundidor visível em vez de implícito.
_RUNTIME_MARKERS = {
    "keras": ("o", "Keras/TF"),
    "pytorch": ("s", "PyTorch"),
    "sklearn": ("^", "scikit-learn"),
    "unknown": ("X", "não declarado"),
}


def fig_accuracy_latency_tradeoff(rows, out: Path):
    plt = _setup_mpl()

    fig, ax = plt.subplots(figsize=(10.5, 6.5))
    seen_runtimes: list[str] = []
    for family, color in _FAMILY_COLORS.items():
        subset = [r for r in rows if _row_family(r) == family]
        if not subset:
            continue
        labelled = False
        for runtime, (marker, _) in _RUNTIME_MARKERS.items():
            group = [
                r
                for r in subset
                if (r.get("latency_runtime") or "unknown") == runtime
            ]
            if not group:
                continue
            if runtime not in seen_runtimes:
                seen_runtimes.append(runtime)
            lat = [_metric(r, "latency") for r in group]
            acc = [_metric(r, "accuracy", 100.0) for r in group]
            sizes = [
                max(50, min(450, (_metric(r, "size") or 1) * 1.2)) for r in group
            ]
            ax.scatter(
                lat, acc, s=sizes, color=color, alpha=0.75, marker=marker,
                edgecolor="white", linewidth=0.8,
                label=family if not labelled else None,
            )
            labelled = True
            for r, x, y in zip(group, lat, acc):
                ax.annotate(r["model"], (x, y), xytext=(4, 4),
                            textcoords="offset points", fontsize=8)
    ax.set_xlabel("Latência de inferência (ms)")
    ax.set_ylabel("Acurácia no conjunto limpo (%)")
    ax.set_title("Trade-off entre acurácia, latência e tamanho do artefato")
    caveat = "Tamanho da bolha proporcional ao artefato persistido (MB)"
    if len(seen_runtimes) > 1:
        nomes = ", ".join(_RUNTIME_MARKERS[rt][1] for rt in seen_runtimes)
        caveat += (
            f"\nLatências medidas em runtimes distintos ({nomes}) — "
            "comparáveis DENTRO de cada marcador, não entre eles"
        )
    ax.text(
        0.99,
        0.02,
        caveat,
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=8,
        color="#444444",
    )
    ax.grid(alpha=0.25)
    handles, labels = ax.get_legend_handles_labels()
    family_legend = ax.legend(
        handles, labels, title="Família", fontsize=8, title_fontsize=9,
        loc="lower left",
    )
    if len(seen_runtimes) > 1:
        ax.add_artist(family_legend)
        from matplotlib.lines import Line2D

        ax.legend(
            handles=[
                Line2D(
                    [], [], color="#666666", linestyle="none",
                    marker=_RUNTIME_MARKERS[rt][0], markersize=7,
                    label=_RUNTIME_MARKERS[rt][1],
                )
                for rt in seen_runtimes
            ],
            title="Runtime da medição",
            fontsize=8,
            title_fontsize=9,
            loc="upper left",
        )
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
    thr = row.get("decision_threshold", 0.5)
    try:
        thr = float(thr)
    except (TypeError, ValueError):
        thr = 0.5
    if not np.isfinite(thr):
        thr = 0.5
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


def build_significance_report(rows, extras, n_bootstrap: int = 1000):
    """Comparações pareadas entre todos os modelos com scores disponíveis.

    IC 95% individuais que se sobrepõem NÃO decidem diferença quando os modelos
    são avaliados nas mesmas amostras — é o caso de Conformer x Hybrid
    CNN-Transformer no `clean_benchmark_15k`. Ver benchmarks/significance.py.
    """
    from benchmarks.significance import compare_models

    # Comparar modelos avaliados em conjuntos de teste DIFERENTES é o erro que
    # este relatório mais convida — as variantes de 15k e 40k do dataset têm
    # fingerprints distintos e os números não são misturáveis. Sem esta guarda,
    # a saída pareceria válida.
    fingerprints = {
        (r.get("dataset") or {}).get("test_split_sha256")
        for r in rows
        if (r.get("dataset") or {}).get("test_split_sha256")
    }
    if len(fingerprints) > 1:
        return {
            "status": "skipped",
            "reason": (
                "os modelos vêm de conjuntos de teste diferentes "
                f"({len(fingerprints)} fingerprints distintos) — a comparação "
                "pareada exige as MESMAS amostras nos dois lados"
            ),
            "test_split_sha256": sorted(fingerprints),
        }

    y_test = None
    cluster_ids = None
    speaker_ids = None
    models = {}
    for row in rows:
        extra = extras.get(row["slug"]) or {}
        scores = extra.get("scores_clean")
        if not scores:
            continue
        if y_test is None:
            y_test = extra.get("y_test")
            cluster_ids = extra.get("test_cluster_ids")
            speaker_ids = extra.get("test_speaker_ids")
        if y_test is None or len(scores) != len(y_test):
            continue
        models[row["model"]] = {"scores": scores}

    if len(models) < 2 or y_test is None:
        return {
            "status": "skipped",
            "reason": "menos de dois modelos com scores alinhados ao y_test",
        }
    if cluster_ids is not None and len(cluster_ids) != len(y_test):
        cluster_ids = None
    if speaker_ids is not None and len(speaker_ids) != len(y_test):
        speaker_ids = None

    threshold = rows[0].get("decision_threshold", 0.5)

    def _compare(ids):
        return compare_models(
            y_test,
            models,
            threshold=threshold,
            cluster_ids=ids,
            n_bootstrap=n_bootstrap,
        )

    # A unidade principal segue sendo a FRASE, por continuidade com o que já
    # estava publicado. A de LOCUTOR entra ao lado porque é ela que casa com a
    # alegação do protocolo: o teste é speaker-disjoint, e reamostrar frases
    # trata frases do mesmo locutor como independentes — o IC sai estreito
    # demais. No `clean_benchmark_15k` a troca de unidade transforma três
    # separações em empate (Conformer x MultiscaleCNN, MultiscaleCNN x RawNet2
    # e RawGAT-ST x SVM), então a diferença NÃO é cosmética.
    report = _compare(cluster_ids)
    report["protocol"]["unit_detail"] = (
        "cluster = frase (text_id)" if cluster_ids is not None else "amostra"
    )
    if cluster_ids is None:
        report["protocol"]["warning"] = (
            "sem test_cluster_ids no results.json: reamostragem por AMOSTRA. "
            "Amostras da mesma frase não são independentes, então o p-valor é "
            "otimista. Runs a partir de 2026-08-09 gravam a chave."
        )

    if speaker_ids is not None:
        by_speaker = _compare(speaker_ids)
        by_speaker["protocol"]["unit_detail"] = "cluster = locutor (speaker_id)"
        report["by_speaker"] = by_speaker
        report["protocol"]["speaker_unit_available"] = True
        report["protocol"]["n_speakers"] = len(set(map(str, speaker_ids)))
        report["protocol"]["note_speaker_unit"] = (
            "`by_speaker` reamostra LOCUTORES, a unidade que corresponde à "
            "alegação de generalização do protocolo speaker-disjoint. São menos "
            "unidades que frases, então os IC são mais largos — e é o veredito "
            "conservador que deve valer para qualquer afirmação sobre locutores "
            "não vistos."
        )
    else:
        report["protocol"]["speaker_unit_available"] = False
        report["protocol"]["note_speaker_unit"] = (
            "sem test_speaker_ids no results.json: só a unidade de frase está "
            "disponível. Runs a partir de 2026-08-09 gravam a chave; para runs "
            "anteriores, scripts/reporting/backfill_artifact_metadata.py a deriva "
            "do .npz."
        )
    return report


def main() -> int:
    p = argparse.ArgumentParser(description="Consolida resultados → resumo + figuras do TCC")
    p.add_argument("inputs", nargs="+",
                   help="diretórios de run ou results.json (aceita globs)")
    p.add_argument("--out", default="data/results/paper/consolidated",
                   help="pasta de saída (default: data/results/paper/consolidated)")
    p.add_argument("--copy-to", default="data/results/paper/figures",
                   help="copia as figuras para o artigo (default: data/results/paper/figures)")
    p.add_argument("--no-figures", action="store_true",
                   help="gera só o benchmark_summary.json")
    p.add_argument(
        "--no-significance",
        action="store_true",
        help=(
            "pula as comparações pareadas entre modelos (McNemar exato + "
            "bootstrap pareado). O default é gerá-las: IC individuais "
            "sobrepostos não decidem diferença em avaliação pareada"
        ),
    )
    p.add_argument(
        "--significance-bootstrap",
        type=int,
        default=5000,
        help=(
            "reamostragens do bootstrap pareado (default: 5000). O menor "
            "p-valor expressável é 2/(n+1), e Holm multiplica esse piso pelo "
            "número de comparações: com 11 modelos são 55 pares, e 1.000 "
            "reamostragens travariam todo p ajustado em 0,11"
        ),
    )
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

    # `stable is False` cobre colapso e divergência. `unstable_oscillation`
    # mantém `stable: True` de propósito (o artefato serve), mas precisa
    # aparecer: é o padrão do RawGAT-ST, cuja época selecionada depende do ruído
    # da val_loss — e o `selection_gap` diz quanto isso custou.
    unstable = [
        (r["model"], (r.get("training_stability") or {}).get("status"))
        for r in rows
        if (r.get("training_stability") or {}).get("stable") is False
        or (r.get("training_stability") or {}).get("status") == "unstable_oscillation"
    ]
    if unstable:
        print("   AVISO: treino instável em " + ", ".join(
            f"{m} ({s})" for m, s in unstable
        ) + " — ver training_stability no metrics.json.")

    costly = [
        (r["model"], (r.get("training_stability") or {}).get("selection_gap"))
        for r in rows
        if isinstance(
            (r.get("training_stability") or {}).get("selection_gap"), (int, float)
        )
        and (r.get("training_stability") or {})["selection_gap"] <= -0.01
    ]
    if costly:
        print("   AVISO: o checkpoint de menor val_loss não é o de melhor "
              "monitor em " + ", ".join(f"{m} ({g:+.4f})" for m, g in costly) +
              " — seleção mantida por protocolo, ver selection_gap.")

    no_speaker = [
        r["model"] for r in rows if not (r.get("grouped_clean") or {}).get("speaker")
    ]
    if no_speaker:
        print("   AVISO: sem grouped_clean por locutor em " +
              ", ".join(no_speaker) + " — a coluna de pior locutor fica vazia.")

    if not args.no_significance:
        significance = build_significance_report(
            rows, extras, n_bootstrap=args.significance_bootstrap
        )
        sig_path = out / "benchmark_significance.json"
        sig_path.write_text(
            json.dumps(significance, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        pairs = significance.get("pairs", []) or []
        n_pairs = len(pairs)
        print(f"-> {sig_path} ({n_pairs} comparações pareadas)")
        if significance.get("protocol", {}).get("warning"):
            print(f"   AVISO: {significance['protocol']['warning']}")

        # Pares que falharam contam como "comparações" na linha acima, então a
        # contagem sozinha não distingue um arquivo completo de um degradado.
        # Foi assim que uma rodada sem scikit-learn instalado gravou os 55 pares
        # apenas com McNemar, cada um carregando `status: error`, enquanto o
        # console anunciava "55 comparações pareadas" — e o arquivo seguiu para
        # as tabelas do artigo sem que nada denunciasse a falta do bootstrap.
        falhos = [p for p in pairs if p.get("status") == "error"]
        if falhos:
            motivos = sorted({str(p.get("error") or "?") for p in falhos})
            print(
                f"   ATENÇÃO: {len(falhos)}/{n_pairs} comparações FALHARAM e "
                f"ficaram sem bootstrap pareado — {'; '.join(motivos[:3])}"
            )
            print(
                "   O arquivo está INCOMPLETO: corrija a causa e reconsolide "
                "antes de usá-lo no artigo."
            )

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

    print("\nPróximo: python scripts/reporting/update_tcc_latex.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
