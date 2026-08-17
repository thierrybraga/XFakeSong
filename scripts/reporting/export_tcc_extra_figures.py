"""Gera figuras extras do TCC a partir dos artefatos promovidos.

Figuras geradas em ``data/results/paper/figures/``:

- ``benchmark_det_curves.png``: curvas DET (FNR x FPR, eixos probit) dos 11
  modelos consolidados, calculadas a partir de ``predictions_clean.csv``.
- ``score_distributions_gat.png``: histogramas da pontuacao ``p_fake`` por
  classe verdadeira para AASIST e RawGAT-ST, evidenciando o formato das
  distribuicoes de score (saturacao induzida pela cabeca AM-Softmax).

Uso:
    python scripts/reporting/export_tcc_extra_figures.py
"""

from __future__ import annotations

import csv
import logging
from pathlib import Path
from statistics import NormalDist
from typing import List, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[2]

# Fonte das predicoes: o DIRETORIO DO RUN, nao `benchmark_final/`.
#
# Apontava para `data/models/benchmark_final/<slug>/results/`, um layout que a
# promocao atual nao produz (`sync_completed_benchmark_artifacts.py` grava
# `results_copied: false`) — o script quebrava com FileNotFoundError. O run e a
# fonte canonica das predicoes; `benchmark_final/` guarda o modelo empacotado.
#
# As figuras que este script gerava foram descartadas em 2026-08-09 porque
# vinham do dataset com atalho de fonte (`benchmark_audio_raw_balanced_15k`).
# O script em si nao tinha defeito: apontado para o run limpo, reproduz as duas
# figuras a partir do protocolo valido.
DEFAULT_RESULTS_ROOT = ROOT / "data" / "results" / "clean_benchmark_15k"
RESULTS_ROOT = DEFAULT_RESULTS_ROOT
FIGURES_DIR = ROOT / "data/results/paper" / "figures"

MODELS: List[Tuple[str, str]] = [
    ("randomforest", "Random Forest"),
    ("svm", "SVM"),
    ("hybrid_cnn_transformer", "CCT"),
    ("spectrogramtransformer", "AST"),
    ("multiscalecnn", "Res2Net"),
    ("conformer", "Conformer"),
    ("rawnet2", "RawNet2"),
    ("aasist", "AASIST"),
    ("rawgat_st", "RawGAT-ST"),
    ("wavlm_original", "WavLM Original"),
    ("hubert_original", "HuBERT Original"),
]


def load_predictions(slug: str) -> Tuple[np.ndarray, np.ndarray]:
    """Carrega (y_true, score de ORDENACAO) de predictions_clean.csv.

    Prefere `ranking_score` quando a coluna vem preenchida. Nos classicos, a
    calibracao isotonica colapsa a margem em degraus (52 no SVM deste run), e
    curva DET e distribuicao de score descrevem ORDENACAO — desenha-las sobre a
    probabilidade calibrada mostraria uma escada que nao corresponde ao EER
    publicado. Onde a coluna esta vazia, `p_fake` ja E o score de ordenacao.
    """
    path = RESULTS_ROOT / slug / "predictions_clean.csv"
    y_true: List[int] = []
    scores: List[float] = []
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            y_true.append(int(row["y_true"]))
            bruto = (row.get("ranking_score") or "").strip()
            scores.append(float(bruto) if bruto else float(row["p_fake"]))
    return np.asarray(y_true), np.asarray(scores)


def load_p_fake(slug: str) -> Tuple[np.ndarray, np.ndarray]:
    """Carrega (y_true, p_fake) — a probabilidade, para o histograma."""
    path = RESULTS_ROOT / slug / "predictions_clean.csv"
    y_true: List[int] = []
    p_fake: List[float] = []
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            y_true.append(int(row["y_true"]))
            p_fake.append(float(row["p_fake"]))
    return np.asarray(y_true), np.asarray(p_fake)


def det_points(y_true: np.ndarray, scores: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """FPR/FNR ordenados por limiar (classe positiva = spoof)."""
    order = np.argsort(-scores)
    y = y_true[order]
    n_pos = int(y.sum())
    n_neg = len(y) - n_pos
    tp = np.cumsum(y)
    fp = np.cumsum(1 - y)
    fpr = fp / max(n_neg, 1)
    fnr = 1.0 - tp / max(n_pos, 1)
    return fpr, fnr


def _probit(values: np.ndarray) -> np.ndarray:
    nd = NormalDist()
    clipped = np.clip(values, 1e-4, 1 - 1e-4)
    return np.asarray([nd.inv_cdf(float(v)) for v in clipped])


def export_det_curves() -> Path:
    """Curvas DET dos 11 modelos em eixos probit (padrao NIST/ASVspoof)."""
    ticks = np.asarray([0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.10, 0.20, 0.40])
    tick_labels = ["0,1", "0,2", "0,5", "1", "2", "5", "10", "20", "40"]
    cmap = plt.get_cmap("tab20")

    fig, ax = plt.subplots(figsize=(8.2, 6.4))
    for idx, (slug, label) in enumerate(MODELS):
        y_true, scores = load_predictions(slug)
        fpr, fnr = det_points(y_true, scores)
        mask = (fpr > 0) & (fnr > 0)
        ax.plot(
            _probit(fpr[mask]),
            _probit(fnr[mask]),
            label=label,
            color=cmap(idx % 20),
            linewidth=1.6,
        )

    probit_ticks = _probit(ticks)
    ax.set_xticks(probit_ticks)
    ax.set_xticklabels(tick_labels)
    ax.set_yticks(probit_ticks)
    ax.set_yticklabels(tick_labels)
    ax.set_xlim(_probit(np.asarray([0.001]))[0], _probit(np.asarray([0.40]))[0])
    ax.set_ylim(_probit(np.asarray([0.001]))[0], _probit(np.asarray([0.40]))[0])
    ax.plot(ax.get_xlim(), ax.get_ylim(), color="gray", linewidth=0.8, linestyle=":")
    ax.set_xlabel("Taxa de falso-alarme, $P_{fa}$ (%)")
    ax.set_ylabel("Taxa de perda, $P_{miss}$ (%)")
    ax.set_title("Curvas DET no conjunto de teste limpo")
    ax.grid(True, which="both", linewidth=0.4, alpha=0.5)
    ax.legend(fontsize=8, ncol=2, loc="upper right", framealpha=0.9)
    fig.tight_layout()

    out = FIGURES_DIR / "benchmark_det_curves.png"
    fig.savefig(out, dpi=200)
    plt.close(fig)
    logger.info("DET salvo em %s", out)
    return out


def export_score_distributions() -> Path:
    """Histogramas de p_fake por classe para AASIST e RawGAT-ST."""
    pair = [("aasist", "AASIST"), ("rawgat_st", "RawGAT-ST")]
    bins = np.linspace(0.0, 1.0, 41)

    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.2), sharey=True)
    for ax, (slug, label) in zip(axes, pair):
        # `p_fake` explicito: o eixo e a PROBABILIDADE em [0, 1] e os bins
        # cobrem esse intervalo. Nestes dois modelos as duas leituras coincidem
        # (nao ha calibracao separada), mas depender disso deixaria o
        # histograma errado no dia em que alguem apontar a funcao para um
        # classico, cujo score de ordenacao e uma margem centrada em zero.
        y_true, scores = load_p_fake(slug)
        ax.hist(
            scores[y_true == 0], bins=bins, alpha=0.65,
            label="bonafide", color="#27ae60",
        )
        ax.hist(
            scores[y_true == 1], bins=bins, alpha=0.65,
            label="spoof", color="#c0392b",
        )
        extremes = float(np.mean((scores < 0.05) | (scores > 0.95))) * 100.0
        ax.set_title(f"{label} ({extremes:.0f}% dos scores nos extremos)")
        ax.set_xlabel("Pontuação $p_{fake}$")
        ax.set_yscale("log")
        ax.grid(True, linewidth=0.4, alpha=0.5)
        ax.legend(fontsize=9)
    axes[0].set_ylabel("Nº de amostras (escala log)")
    fig.tight_layout()

    out = FIGURES_DIR / "score_distributions_gat.png"
    fig.savefig(out, dpi=200)
    plt.close(fig)
    logger.info("Distribuições salvas em %s", out)
    return out


def main() -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    export_det_curves()
    export_score_distributions()


if __name__ == "__main__":
    main()
