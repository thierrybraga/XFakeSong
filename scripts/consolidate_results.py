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
    figures/confusion_matrices/<slug>.png   (1 por arquitetura)

Exemplos:
    # A partir dos runs por arquitetura já existentes:
    python scripts/consolidate_results.py results/benchmark_*_gpu_100e \
        results/benchmark_svm_100e results/tcc_pipeline_svm_rf_balanced_15k

    # A partir de um run completo (14 arq. num só results.json):
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

# canonical_compact -> (key p/ summary, slug de figura, nome de exibição)
_CANON = {
    "wavlm": ("WavLM Original", "wavlm_original", "WavLM Original"),
    "wavlmoriginal": ("WavLM Original", "wavlm_original", "WavLM Original"),
    "hubert": ("HuBERT Original", "hubert_original", "HuBERT Original"),
    "hubertoriginal": ("HuBERT Original", "hubert_original", "HuBERT Original"),
    "rawnet2": ("RawNet2", "rawnet2", "RawNet2"),
    "sonicsleuth": ("Sonic Sleuth", "sonic_sleuth", "Sonic Sleuth"),
    "aasist": ("AASIST", "aasist", "AASIST"),
    "rawgatst": ("RawGAT-ST", "rawgat_st", "RawGAT-ST"),
    "conformer": ("Conformer", "conformer", "Conformer"),
    "hybridcnntransformer": ("Hybrid CNN-Transformer", "hybrid_cnn_transformer",
                             "Hybrid CNN-Transformer"),
    "spectrogramtransformer": ("SpectrogramTransformer",
                               "spectrogram_transformer",
                               "Spectrogram Transformer"),
    "efficientnetlstm": ("EfficientNet-LSTM", "efficientnet_lstm",
                         "EfficientNet-LSTM"),
    "multiscalecnn": ("MultiscaleCNN", "multiscalecnn", "MultiscaleCNN"),
    "ensemble": ("Ensemble", "ensemble", "Ensemble"),
    "svm": ("SVM", "svm", "SVM"),
    "randomforest": ("RandomForest", "random_forest", "Random Forest"),
}

MODEL_ORDER = [
    "WavLM Original", "HuBERT Original", "RawNet2", "Sonic Sleuth", "AASIST",
    "RawGAT-ST", "Conformer", "Hybrid CNN-Transformer", "SpectrogramTransformer",
    "EfficientNet-LSTM", "MultiscaleCNN", "Ensemble", "SVM", "RandomForest",
]


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
                yield jf, json.loads(jf.read_text(encoding="utf-8"))
            except Exception as e:
                print(f"  (erro lendo {jf}: {e})", file=sys.stderr)


def collect_rows(paths: List[str]):
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
            # Se o mesmo modelo aparecer em vários runs, mantém o de maior AUC.
            prev = rows_by_key.get(key)
            if prev is None or (row["auc"] or 0) >= (prev["auc"] or 0):
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
    return plt


def _labels(rows):
    return [r["model"] for r in rows]


def fig_accuracy_auc(rows, out: Path):
    plt = _setup_mpl()
    import numpy as np
    labels = _labels(rows)
    acc = [(r["accuracy"] or 0) * 100 for r in rows]
    auc = [(r["auc"] or 0) * 100 for r in rows]
    x = np.arange(len(labels)); w = 0.4
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.bar(x - w / 2, acc, w, label="Acurácia (%)")
    ax.bar(x + w / 2, auc, w, label="AUC-ROC (%)")
    ax.set_xticks(x); ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_ylim(0, 100); ax.set_ylabel("%"); ax.legend()
    ax.set_title("Acurácia e AUC-ROC (conjunto limpo)")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout(); fig.savefig(out / "benchmark_accuracy_auc.png", dpi=150)
    plt.close(fig)


def fig_simple_bar(rows, out: Path, field, fname, title, ylabel, scale=1.0):
    plt = _setup_mpl()
    import numpy as np
    labels = _labels(rows)
    vals = [(r.get(field) or 0) * scale for r in rows]
    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.bar(x, vals, color="#4C72B0")
    ax.set_xticks(x); ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_ylabel(ylabel); ax.set_title(title); ax.grid(axis="y", alpha=0.3)
    fig.tight_layout(); fig.savefig(out / fname, dpi=150); plt.close(fig)


def fig_robustness(rows, out: Path):
    plt = _setup_mpl()
    snrs = sorted({int(s) for r in rows for s in (r.get("robustness") or {})},
                  reverse=True)
    if not snrs:
        return
    fig, ax = plt.subplots(figsize=(11, 6))
    for r in rows:
        rob = r.get("robustness") or {}
        ys = [(rob.get(str(s), {}) or {}).get("accuracy") for s in snrs]
        if any(v is not None for v in ys):
            ax.plot(snrs, [(v or 0) * 100 for v in ys], marker="o",
                    label=r["model"])
    ax.set_xlabel("SNR (dB)"); ax.set_ylabel("Acurácia (%)")
    ax.set_title("Robustez a ruído (AWGN)"); ax.invert_xaxis()
    ax.grid(alpha=0.3); ax.legend(fontsize=7, ncol=2)
    fig.tight_layout(); fig.savefig(out / "benchmark_robustness.png", dpi=150)
    plt.close(fig)


def fig_training_stability(rows, extras, out: Path):
    plt = _setup_mpl()
    fig, ax = plt.subplots(figsize=(11, 6))
    plotted = False
    for r in rows:
        h = (extras.get(r["slug"], {}) or {}).get("history") or {}
        va = h.get("val_accuracy")
        if va:
            ax.plot(range(1, len(va) + 1), [v * 100 for v in va],
                    label=r["model"], alpha=0.8)
            plotted = True
    if not plotted:
        plt.close(fig); return
    ax.set_xlabel("Época"); ax.set_ylabel("Acurácia de validação (%)")
    ax.set_title("Estabilidade de treinamento (val_accuracy)")
    ax.grid(alpha=0.3); ax.legend(fontsize=7, ncol=2)
    fig.tight_layout(); fig.savefig(out / "training_stability.png", dpi=150)
    plt.close(fig)


def fig_confusion_matrices(rows, extras, out: Path):
    plt = _setup_mpl()
    import numpy as np
    cm_dir = out / "confusion_matrices"
    cm_dir.mkdir(parents=True, exist_ok=True)
    for r in rows:
        ex = extras.get(r["slug"], {}) or {}
        scores, y_test = ex.get("scores_clean"), ex.get("y_test")
        if not scores or not y_test or len(scores) != len(y_test):
            continue
        y_true = np.asarray(y_test).astype(int)
        thr = (r.get("robustness", {}).get("clean_threshold")
               or _eer_threshold(r))
        y_pred = (np.asarray(scores) >= thr).astype(int)
        cm = np.zeros((2, 2), dtype=int)
        for t, p in zip(y_true, y_pred):
            cm[t, p] += 1
        fig, ax = plt.subplots(figsize=(4, 4))
        ax.imshow(cm, cmap="Blues")
        for i in range(2):
            for j in range(2):
                ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                        color="black", fontsize=12)
        ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
        ax.set_xticklabels(["Real", "Fake"]); ax.set_yticklabels(["Real", "Fake"])
        ax.set_xlabel("Predito"); ax.set_ylabel("Verdadeiro")
        ax.set_title(r["model"])
        fig.tight_layout(); fig.savefig(cm_dir / f"{r['slug']}.png", dpi=150)
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
    fig_simple_bar(rows, fig_dir, "latency", "benchmark_latency.png",
                   "Latência de inferência", "ms")
    fig_simple_bar(rows, fig_dir, "size", "benchmark_size.png",
                   "Tamanho do modelo", "MB")
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
    args = p.parse_args()

    out = (PROJECT_ROOT / args.out) if not Path(args.out).is_absolute() else Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    print("Consolidando resultados de:", ", ".join(args.inputs))
    rows, extras = collect_rows(args.inputs)
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
        print(f"   AVISO: faltando {missing} (o TCC espera 14).")

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
