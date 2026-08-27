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
    figures/benchmark_det_curves.png
    figures/benchmark_robustness.png
    figures/benchmark_robustness_threshold_free.png
    figures/score_distributions_gat.png
    figures/benchmark_size.png
    figures/training_stability.png
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


#: Direção de otimização por métrica de validação. `min` para erro e perda,
#: `max` para acerto — errar isto reporta a PIOR época como a melhor.
_DIRECAO_MONITOR = {
    "val_eer": min, "val_loss": min, "loss": min,
    "val_accuracy": max, "val_acc": max, "val_auc": max, "accuracy": max,
}


def _serie_de_selecao(
    history: Optional[Dict[str, list]], monitor: Optional[str]
) -> tuple[Optional[str], Optional[list]]:
    """Série do history que REALMENTE selecionou o checkpoint, e seu nome.

    Preferência ao `monitor` declarado no run; se ele não estiver no history
    (caso do runner SSL, que treina em PyTorch e grava outras chaves), cai para
    `val_loss` e depois `val_accuracy` — nessa ordem, que é a do próprio
    `checkpoint_selection: minimum_clean_validation_loss` daquele runner.
    """
    if not history:
        return None, None
    candidatos = [monitor] if monitor else []
    candidatos += ["val_eer", "val_loss", "val_accuracy"]
    for nome in candidatos:
        if not nome:
            continue
        serie = history.get(nome)
        if serie:
            return nome, serie
    return None, None


def _best_epoch(
    history: Optional[Dict[str, list]], monitor: Optional[str] = None
) -> Optional[int]:
    """Época (1-indexada) do checkpoint EFETIVAMENTE restaurado.

    CORREÇÃO 2026-08-21. Esta função calculava sempre pelo `val_loss`, mas o
    protocolo passou a selecionar por `val_eer` (`--checkpoint-monitor val_eer`
    no compose, propagado até o `ModelCheckpoint` do trainer). A época
    reportada, portanto, não era a época avaliada: no run corrigido do CCT o
    mínimo de `val_eer` cai na época 47 e o de `val_loss` em outra, e a tabela
    do TCC anunciava a segunda enquanto os pesos vinham da primeira.

    Deriva do `history` em vez de exigir um campo novo no artefato: a regra é a
    mesma do `ModelCheckpoint` (primeiro ótimo da série monitorada), então o
    valor vale também para os runs já concluídos, sem retreinar nada.
    """
    nome, serie = _serie_de_selecao(history, monitor)
    if not serie:
        return None
    escolha = _DIRECAO_MONITOR.get(nome, min)
    return int(escolha(range(len(serie)), key=lambda i: serie[i]) + 1)


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
                # IC 95% de bootstrap POR CLUSTER, já calculado por
                # `benchmarks/evaluate.py::_bootstrap_cis` e gravado no
                # results.json. Ele era descartado aqui, e o TCC compensava com
                # um IC de Wilson escrito à mão no main.tex — calculado por
                # AMOSTRA, que subestima o intervalo quando as amostras de um
                # mesmo locutor não são independentes. Propagar permite a
                # tabela usar o intervalo correto.
                "accuracy_ci95_low": clean.get("accuracy_ci95_low"),
                "accuracy_ci95_high": clean.get("accuracy_ci95_high"),
                "eer_ci95_low": clean.get("eer_ci95_low"),
                "eer_ci95_high": clean.get("eer_ci95_high"),
                "auc_ci95_low": clean.get("auc_roc_ci95_low"),
                "auc_ci95_high": clean.get("auc_roc_ci95_high"),
                "bootstrap_unit": clean.get("bootstrap_unit"),
                "bootstrap_samples": clean.get("bootstrap_samples"),
                # Critério de convergência medido no run. Sem ele a coluna
                # "Status" da tabela do artigo não tem o que reportar e cai em
                # literal — ver `update_tcc_latex.build_results_table`.
                "converged": a.get("converged"),
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
                "best_epoch": _best_epoch(
                    a.get("history"), run_config.get("checkpoint_monitor")
                ),
                # QUAL criterio selecionou a epoca. Precisa ir para a tabela: as
                # 9 entradas Keras selecionam por `val_eer` e as 2 SSL por perda
                # de validacao (runner PyTorch com laco proprio). Sem declarar,
                # a coluna de epoca compara escolhas feitas sob criterios
                # diferentes como se fossem a mesma coisa.
                "selection_monitor": _serie_de_selecao(
                    a.get("history"), run_config.get("checkpoint_monitor")
                )[0],
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
            # DESEMPATE ENTRE RUNS DO MESMO MODELO.
            #
            # O critério anterior era `row["auc"] >= prev["auc"]`: com o mesmo
            # modelo em dois runs, ficava o que foi melhor NO CONJUNTO DE TESTE.
            # Isso é seleção de modelo sobre o teste — exatamente o que o resto
            # do protocolo (teste selado, limiar fixo, EER-threshold marcado
            # como oráculo) existe para impedir. Um run colapsado que por acaso
            # pontuou alto vencia um retreino saudável.
            #
            # Agora a duplicata é um ERRO que o operador precisa resolver
            # declarando qual run vale (`--prefer-last`, que mantém o último
            # caminho lido). Nenhuma métrica de teste participa da escolha.
            prev = rows_by_key.get(key)
            if prev is not None and not prefer_last:
                raise SystemExit(
                    f"'{display}' aparece em mais de um run:\n"
                    f"  {prev['path']}\n  {row['path']}\n"
                    "Escolher entre eles por métrica de teste seria seleção de "
                    "modelo sobre o teste. Passe --prefer-last para que o "
                    "último caminho listado prevaleça, ou consolide um run só."
                )
            if prev is None or prefer_last:
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

_FIGURE_DPI = 300

def _setup_mpl():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.titleweight": "bold",
        "figure.dpi": 150,
        "savefig.dpi": _FIGURE_DPI,
        "savefig.bbox": "tight",
        "font.size": 10,
    })
    return plt


def _labels(rows):
    # Os dois classificadores clássicos tiveram o frontend revisto depois de
    # uma inspeção exploratória do teste. A marca precisa acompanhar TODAS as
    # figuras; deixar a ressalva apenas no corpo do artigo transforma um
    # resultado exploratório em confirmatório quando a figura circula sozinha.
    return [
        f"{r['model']} (expl.)"
        if r.get("key") in {"RandomForest", "SVM"}
        else r["model"]
        for r in rows
    ]


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


def _plot_limit(values, fallback: float = 1.0) -> float:
    """Limite superior do eixo ignorando ausentes (``NaN``).

    `max()` sobre uma lista com `NaN` devolve `NaN` e quebra o `set_xlim`.
    """
    import math

    finitos = [v for v in values if v is not None and math.isfinite(v)]
    if not finitos:
        return fallback
    return max(max(finitos) * 1.18, fallback * 1e-9)


def _annotate_hbars(ax, values, suffix="", fmt="{:.2f}", min_pad=0.4) -> None:
    """Anota cada barra; métrica ausente vira "n/d", não um número.

    Uma métrica que não foi gravada chega aqui como `NaN` (ver `_metric`).
    Formatá-la publicaria "nan" na figura do artigo; pior, antes de `_metric`
    devolver `NaN` ela chegava como 0,0 e virava a melhor barra do gráfico.
    """
    import math

    finitos = [v for v in values if v is not None and math.isfinite(v)]
    xmax = max(finitos) if finitos else 0
    for patch, value in zip(ax.patches, values):
        if value is None:
            continue
        y = patch.get_y() + patch.get_height() / 2
        pad = max(xmax * 0.01, min_pad)
        if not math.isfinite(value):
            ax.text(pad, y, "n/d", va="center", ha="left",
                    fontsize=8, style="italic", color="0.45")
            continue
        ax.text(
            patch.get_width() + pad,
            y,
            f"{fmt.format(value)}{suffix}",
            va="center",
            ha="left",
            fontsize=8,
        )


def _metric(row: dict, field: str, scale: float = 1.0) -> float:
    """Valor da métrica para plotagem; ``NaN`` quando ausente.

    Devolvia ``float(value or 0)``, e isso publicava campo AUSENTE como 0,0.
    Em EER e min t-DCF zero é o MELHOR valor possível: um modelo sem a métrica
    gravada aparecia como a melhor barra do gráfico. `NaN` faz o matplotlib
    pular o ponto, e as funções de barra anotam "n/d".
    """
    value = row.get(field)
    if value is None:
        return float("nan")
    try:
        return float(value) * scale
    except (TypeError, ValueError):
        return float("nan")


def _model_colors(rows: list[dict], plt) -> dict[str, Any]:
    cmap = plt.get_cmap("tab20")
    return {
        row["key"]: cmap(i % cmap.N)
        for i, row in enumerate(rows)
    }


def fig_accuracy_auc(rows, out: Path):
    plt = _setup_mpl()
    import math
    import numpy as np

    labels = _labels(rows)
    acc = [_metric(r, "accuracy", 100.0) for r in rows]
    auc = [_metric(r, "auc", 100.0) for r in rows]
    y = np.arange(len(labels))
    h = 0.36
    # As figuras são inseridas aproximadamente na largura do texto (6,3 in).
    # Produzi-las já nessa escala evita que rótulos e legendas percam quase
    # metade do corpo tipográfico quando o LaTeX reduz imagens largas.
    def _xerr(field: str, values: list[float]) -> np.ndarray:
        lows = []
        highs = []
        for row, value in zip(rows, values):
            low = row.get(f"{field}_ci95_low")
            high = row.get(f"{field}_ci95_high")
            if low is None or high is None or not math.isfinite(value):
                lows.append(0.0)
                highs.append(0.0)
            else:
                lows.append(max(0.0, value - float(low) * 100.0))
                highs.append(max(0.0, float(high) * 100.0 - value))
        return np.asarray([lows, highs])

    fig, ax = plt.subplots(figsize=(6.8, 4.7))
    ax.errorbar(
        acc,
        y - h / 2,
        xerr=_xerr("accuracy", acc),
        fmt="o",
        color="#4C72B0",
        ecolor="#4C72B0",
        capsize=2.5,
        linewidth=1.1,
        markersize=5,
        label="Acurácia",
    )
    ax.errorbar(
        auc,
        y + h / 2,
        xerr=_xerr("auc", auc),
        fmt="s",
        color="#DD8452",
        ecolor="#DD8452",
        capsize=2.5,
        linewidth=1.1,
        markersize=4.5,
        label="AUC-ROC",
    )
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ci_lows = [
        float(row.get(field) or value / 100.0) * 100.0
        for row, value in zip(rows, acc)
        for field in ("accuracy_ci95_low",)
    ] + [
        float(row.get(field) or value / 100.0) * 100.0
        for row, value in zip(rows, auc)
        for field in ("auc_ci95_low",)
    ]
    lower = max(0.0, math.floor(min(ci_lows) - 1.0))
    ax.set_xlim(lower, 100.6)
    ax.set_xlabel("Valor (%)")
    ax.set_title("Desempenho limpo e IC 95% por locutor")
    ax.grid(axis="x", alpha=0.25)
    for yi, value in zip(y, acc):
        ax.annotate(
            f"{value:.2f}", (value, yi - h / 2), xytext=(-5, -1),
            textcoords="offset points", ha="right", va="center",
            fontsize=6.7, color="#294f80",
        )
    for yi, value in zip(y, auc):
        ax.annotate(
            f"{value:.2f}", (value, yi + h / 2), xytext=(-5, 1),
            textcoords="offset points", ha="right", va="center",
            fontsize=6.7, color="#9a4f25",
        )
    ax.invert_yaxis()
    fig.legend(
        *ax.get_legend_handles_labels(),
        loc="lower center",
        bbox_to_anchor=(0.5, 0.005),
        ncol=2,
        frameon=False,
    )
    fig.text(
        0.5,
        0.012,
        f"Eixo ampliado a partir de {lower:.0f}%; barras: IC 95% por bootstrap de locutores.",
        ha="center",
        fontsize=7.2,
        color="#444444",
    )
    fig.tight_layout(rect=(0, 0.13, 1, 1))
    fig.savefig(out / "benchmark_accuracy_auc.png", dpi=_FIGURE_DPI)
    plt.close(fig)


def fig_simple_bar(
    rows,
    out: Path,
    field,
    fname,
    title,
    ylabel,
    scale=1.0,
    group_field=None,
    figsize=(6.5, 3.96),
):
    plt = _setup_mpl()
    import numpy as np

    labels = _labels(rows)
    if group_field:
        # Anexa o runtime ao rótulo e avisa na figura: valores medidos em
        # runtimes diferentes não são comparáveis entre si.
        runtimes = [str(r.get(group_field) or "?") for r in rows]
        labels = [f"{lbl}  [{rt}]" for lbl, rt in zip(labels, runtimes)]
        distintos = sorted(set(runtimes))
        if len(distintos) > 1:
            title = f"{title} — runtimes distintos, não comparáveis entre si"
    vals = [_metric(r, field, scale) for r in rows]
    y = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=figsize)
    ax.barh(y, vals, color=_row_colors(rows))
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.set_xlabel(ylabel)
    ax.set_title(title)
    ax.grid(axis="x", alpha=0.25)
    ax.set_xlim(0, _plot_limit(vals))
    _annotate_hbars(ax, vals, suffix=("%" if "%" in ylabel else ""))
    _add_family_legend(ax)
    ax.invert_yaxis()
    fig.tight_layout()
    fig.savefig(out / fname, dpi=_FIGURE_DPI)
    plt.close(fig)


def fig_tdcf(rows, out: Path):
    plt = _setup_mpl()
    import numpy as np

    labels = _labels(rows)
    vals = [_metric(r, "min_tdcf", 1.0) for r in rows]
    y = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(6.8, 4.45))
    ax.barh(y, vals, color=_row_colors(rows))
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.set_xlabel("$t$-DCF$^\\ast$ (proxy CM-only; menor é melhor)")
    ax.set_title("Custo normalizado aproximado por arquitetura")
    ax.grid(axis="x", alpha=0.25)
    ax.set_xlim(0, _plot_limit(vals))
    _annotate_hbars(ax, vals, fmt="{:.4f}", min_pad=0.01)
    _add_family_legend(ax)
    ax.invert_yaxis()
    fig.tight_layout()
    fig.savefig(out / "benchmark_tdcf.png", dpi=_FIGURE_DPI)
    plt.close(fig)


def fig_model_size(rows, out: Path):
    """Tamanho em escala log, para tornar visíveis artefatos de 1,9 a 362 MB."""
    plt = _setup_mpl()
    import numpy as np

    labels = _labels(rows)
    values = [_metric(row, "size") for row in rows]
    y = np.arange(len(rows))
    fig, ax = plt.subplots(figsize=(6.6, 4.35))
    ax.scatter(values, y, s=48, color=_row_colors(rows), edgecolor="white", linewidth=0.7)
    for yi, value in zip(y, values):
        ax.annotate(
            f"{value:.2f} MB", (value, yi), xytext=(5, 0),
            textcoords="offset points", va="center", fontsize=7.5,
        )
    ax.set_xscale("log")
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.set_xlabel("Tamanho do artefato em disco (MB; escala logarítmica)")
    ax.set_title("Armazenamento dos artefatos serializados")
    ax.grid(axis="x", which="both", alpha=0.25)
    ax.invert_yaxis()
    _add_family_legend(ax)
    fig.tight_layout()
    fig.savefig(out / "benchmark_size.png", dpi=_FIGURE_DPI)
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

    runtimes = [
        runtime for runtime in ("sklearn", "keras", "pytorch", "unknown")
        if any((row.get("latency_runtime") or "unknown") == runtime for row in rows)
    ]
    fig, axes = plt.subplots(
        len(runtimes), 1, figsize=(6.8, 1.95 * len(runtimes) + 0.65),
        sharey=True, squeeze=False,
    )
    axes = axes[:, 0]
    for axis, runtime in zip(axes, runtimes):
        group = [
            row for row in rows
            if (row.get("latency_runtime") or "unknown") == runtime
        ]
        for row in group:
            x = _metric(row, "latency")
            y = _metric(row, "accuracy", 100.0)
            size = max(55, min(430, (_metric(row, "size") or 1) * 1.2))
            axis.scatter(
                [x], [y], s=size,
                color=_FAMILY_COLORS.get(_row_family(row), "#777777"),
                alpha=0.78, edgecolor="white", linewidth=0.8,
            )
            label = row["model"] + (
                " (expl.)" if row.get("key") in {"RandomForest", "SVM"} else ""
            )
            axis.annotate(
                label, (x, y), xytext=(5, 4), textcoords="offset points",
                fontsize=7.4,
            )
        axis.set_title(_RUNTIME_MARKERS[runtime][1], fontsize=9.2, pad=3)
        axis.set_xlabel("Latência do forward (ms; escala própria do painel)", fontsize=8.5)
        axis.grid(alpha=0.25)
        axis.set_ylim(84, 101)
    fig.supylabel("Acurácia limpa (%)", x=0.015)
    fig.suptitle(
        "Acurácia, latência e tamanho - comparação somente dentro de runtime",
        fontsize=10.5,
        fontweight="bold",
        y=0.995,
    )
    fig.text(
        0.5,
        0.012,
        "Área da bolha: tamanho do artefato (MB). Escalas x independentes; não há ranking global de latência.",
        ha="center",
        va="bottom",
        fontsize=7.5,
        color="#444444",
    )
    fig.subplots_adjust(left=0.11, right=0.98, top=0.94, bottom=0.08, hspace=0.55)
    fig.savefig(out / "benchmark_accuracy_latency_tradeoff.png", dpi=_FIGURE_DPI)
    plt.close(fig)


def fig_robustness(rows, out: Path):
    """Desenha curvas AWGN em dois painéis para preservar a legibilidade.

    O recorte visual é feito somente pela acurácia limpa: não altera os dados,
    as escalas ou o ordenamento discutido no artigo. Com onze linhas em um
    único eixo, a legenda ocupava pouco espaço e tornava a Figura 3 difícil de
    ler na impressão.
    """
    plt = _setup_mpl()
    snrs = sorted(
        {int(s) for r in rows for s in (r.get("robustness") or {})},
        reverse=True,
    )
    if not snrs:
        return
    x_labels = ["Limpo"] + [f"{snr} dB" for snr in snrs]
    x = list(range(len(x_labels)))
    ordered = sorted(rows, key=lambda row: row.get("accuracy") or 0, reverse=True)
    groups = [
        ("Cinco maiores acurácias limpas", ordered[:5]),
        ("Demais arquiteturas", ordered[5:]),
    ]
    fig, axes = plt.subplots(
        2, 1, figsize=(6.5, 5.15), sharex=True, sharey=True
    )
    colors = _model_colors(rows, plt)
    markers = ["o", "s", "^", "D", "P", "X", "v", "<", ">", "h", "*"]
    marker_by_key = {row["key"]: markers[index % len(markers)] for index, row in enumerate(rows)}
    for axis, (title, group) in zip(axes, groups):
        for row in group:
            robustness = row.get("robustness") or {}
            ys = [row.get("accuracy")] + [
                (robustness.get(str(s), {}) or {}).get("accuracy") for s in snrs
            ]
            if any(value is not None for value in ys):
                axis.plot(
                    x,
                    [
                        float("nan") if value is None else float(value) * 100
                        for value in ys
                    ],
                    marker=marker_by_key[row["key"]],
                    linewidth=2.0,
                    markersize=5.5,
                    label=row["model"],
                    color=colors[row["key"]],
                    alpha=0.92,
                )
        axis.set_title(title, fontsize=10, pad=5)
        axis.set_ylim(45, 101)
        axis.grid(alpha=0.25)
        handles, labels = axis.get_legend_handles_labels()
        if handles:
            axis.legend(
                handles,
                labels,
                fontsize=8,
                ncol=3,
                loc="lower left",
                frameon=True,
            )
    axes[-1].set_xticks(x)
    axes[-1].set_xticklabels(x_labels)
    axes[-1].set_xlabel("Condição de avaliação")
    fig.supylabel("Acurácia (%)")
    # Sem título interno: a legenda/caption do LaTeX já nomeia a figura. A
    # remoção evita a sobreposição que ocorria entre o suptitle e o primeiro
    # painel no PDF reduzido.
    fig.subplots_adjust(left=0.10, right=0.98, top=0.98, bottom=0.09, hspace=0.36)
    fig.savefig(out / "benchmark_robustness.png", dpi=_FIGURE_DPI)
    plt.close(fig)


def fig_robustness_threshold_free(rows, out: Path):
    """AUC-ROC e EER sob AWGN, sem dependência do limiar fixo de 0,5.

    A figura principal de robustez usa acurácia porque ela responde diretamente
    à QP2 no ponto de operação comum. Sozinha, porém, ela mistura perda de
    separabilidade com deriva de calibração. Este painel complementar mostra as
    duas métricas de ordenação disponíveis em todos os SNR e impede que uma
    queda de acurácia seja interpretada automaticamente como colapso do
    detector.
    """
    plt = _setup_mpl()

    snrs = sorted(
        {int(s) for row in rows for s in (row.get("robustness") or {})},
        reverse=True,
    )
    if not snrs:
        return

    x_labels = ["Limpo"] + [f"{snr} dB" for snr in snrs]
    x = list(range(len(x_labels)))
    colors = _model_colors(rows, plt)
    markers = ["o", "s", "^", "D", "P", "X", "v", "<", ">", "h", "*"]
    marker_by_key = {
        row["key"]: markers[index % len(markers)]
        for index, row in enumerate(rows)
    }

    fig, axes = plt.subplots(2, 1, figsize=(6.5, 6.15), sharex=True)
    definitions = (
        ("auc", "auc_roc", "AUC-ROC", (50, 101)),
        ("eer", "eer", "EER (%)", None),
    )
    for axis, (clean_field, noisy_field, ylabel, ylim) in zip(axes, definitions):
        for row in rows:
            robustness = row.get("robustness") or {}
            values = [row.get(clean_field)] + [
                (robustness.get(str(snr), {}) or {}).get(noisy_field)
                for snr in snrs
            ]
            ys = [
                float("nan") if value is None else float(value) * 100
                for value in values
            ]
            if all(value != value for value in ys):
                continue
            axis.plot(
                x,
                ys,
                marker=marker_by_key[row["key"]],
                linewidth=1.7,
                markersize=5.0,
                label=row["model"],
                color=colors[row["key"]],
                alpha=0.92,
            )
        axis.set_ylabel(ylabel)
        if ylim is not None:
            axis.set_ylim(*ylim)
        axis.grid(alpha=0.25)

    axes[-1].set_xticks(x)
    axes[-1].set_xticklabels(x_labels)
    axes[-1].set_xlabel("Condição de avaliação")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.005),
        ncol=3,
        fontsize=7.5,
        frameon=False,
    )
    fig.subplots_adjust(left=0.11, right=0.98, top=0.98, bottom=0.20, hspace=0.25)
    fig.savefig(
        out / "benchmark_robustness_threshold_free.png",
        dpi=_FIGURE_DPI,
    )
    plt.close(fig)


def _det_points(y_true, scores):
    """Retorna FPR/FNR por limiar, com ``spoof`` como classe positiva."""
    import numpy as np

    order = np.argsort(-np.asarray(scores, dtype=float))
    y = np.asarray(y_true, dtype=int)[order]
    n_pos = int(y.sum())
    n_neg = len(y) - n_pos
    true_positives = np.cumsum(y)
    false_positives = np.cumsum(1 - y)
    fpr = false_positives / max(n_neg, 1)
    fnr = 1.0 - true_positives / max(n_pos, 1)
    return fpr, fnr


def _probit(values):
    """Transforma probabilidades em quantis normais para o eixo DET."""
    from statistics import NormalDist

    import numpy as np

    normal = NormalDist()
    clipped = np.clip(np.asarray(values, dtype=float), 1e-4, 1 - 1e-4)
    return np.asarray([normal.inv_cdf(float(value)) for value in clipped])


def fig_det_curves(rows, extras, out: Path):
    """Curvas DET limpas, derivadas dos mesmos escores da tabela canônica."""
    plt = _setup_mpl()
    import numpy as np

    ticks = np.asarray([0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.10, 0.20, 0.40])
    tick_labels = ["0,1", "0,2", "0,5", "1", "2", "5", "10", "20", "40"]
    colors = plt.get_cmap("tab20")
    fig, ax = plt.subplots(figsize=(6.8, 5.55))
    plotted = 0
    for index, row in enumerate(rows):
        extra = extras.get(row["slug"], {}) or {}
        scores = extra.get("scores_clean")
        y_true = extra.get("y_test")
        if scores is None or y_true is None or len(scores) != len(y_true):
            continue
        fpr, fnr = _det_points(y_true, scores)
        mask = (fpr > 0) & (fnr > 0)
        ax.plot(
            _probit(fpr[mask]),
            _probit(fnr[mask]),
            label=row["model"],
            color=colors(index % 20),
            linewidth=1.55,
        )
        plotted += 1
    if not plotted:
        plt.close(fig)
        return

    probit_ticks = _probit(ticks)
    ax.set_xticks(probit_ticks)
    ax.set_xticklabels(tick_labels)
    ax.set_yticks(probit_ticks)
    ax.set_yticklabels(tick_labels)
    lower, upper = _probit(np.asarray([0.001, 0.40]))
    ax.set_xlim(lower, upper)
    ax.set_ylim(lower, upper)
    ax.plot(ax.get_xlim(), ax.get_ylim(), color="gray", linewidth=0.8, linestyle=":")
    ax.set_xlabel("FPR — bonafide rejeitado (%)")
    ax.set_ylabel("FNR — spoof aceito (%)")
    ax.set_title("Curvas DET no conjunto de teste limpo")
    ax.grid(True, linewidth=0.4, alpha=0.5)
    ax.legend(
        fontsize=7.2,
        ncol=4,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.13),
        frameon=False,
    )
    fig.subplots_adjust(left=0.13, right=0.98, top=0.92, bottom=0.27)
    fig.savefig(out / "benchmark_det_curves.png", dpi=_FIGURE_DPI)
    plt.close(fig)


def fig_score_distributions_gat(rows, extras, out: Path):
    """Distribuições dos escores GAT, sem atribuir causalidade à topologia."""
    plt = _setup_mpl()
    import numpy as np

    requested = (("aasist", "AASIST"), ("rawgat_st", "RawGAT-ST"))
    available = []
    for slug, label in requested:
        extra = extras.get(slug, {}) or {}
        scores = extra.get("scores_clean")
        y_true = extra.get("y_test")
        if scores is not None and y_true is not None and len(scores) == len(y_true):
            available.append((label, np.asarray(y_true, dtype=int), np.asarray(scores)))
    if len(available) != len(requested):
        return

    bins = np.linspace(0.0, 1.0, 41)
    fig, axes = plt.subplots(1, 2, figsize=(6.8, 3.45), sharey=True)
    for axis, (label, y_true, scores) in zip(axes, available):
        axis.hist(
            scores[y_true == 0],
            bins=bins,
            alpha=0.65,
            label="bonafide",
            color="#27ae60",
        )
        axis.hist(
            scores[y_true == 1],
            bins=bins,
            alpha=0.65,
            label="spoof",
            color="#c0392b",
        )
        extremes = float(np.mean((scores < 0.05) | (scores > 0.95))) * 100.0
        axis.set_title(f"{label}\n({extremes:.0f}% dos escores nos extremos)", fontsize=9)
        axis.set_xlabel("Pontuação $p_{fake}$")
        axis.set_yscale("log")
        axis.grid(True, linewidth=0.4, alpha=0.5)
    axes[0].set_ylabel("Nº de amostras (escala log)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.005),
        ncol=2,
        frameon=False,
        fontsize=8.5,
    )
    fig.tight_layout(rect=(0, 0.13, 1, 1))
    fig.savefig(out / "score_distributions_gat.png", dpi=_FIGURE_DPI)
    plt.close(fig)


def fig_training_stability(rows, extras, out: Path):
    plt = _setup_mpl()
    import numpy as np

    neural = [row for row in rows if row["key"] not in {"RandomForest", "SVM"}]
    fig, axes = plt.subplots(3, 3, figsize=(7.1, 7.0), sharex=True)
    for axis, row in zip(axes.flat, neural):
        history = (extras.get(row["slug"]) or {}).get("history") or {}
        values = history.get("val_accuracy") or []
        epochs = np.arange(1, len(values) + 1)
        values_pct = np.asarray(values, dtype=float) * 100.0
        axis.plot(epochs, values_pct, color="#4C72B0", linewidth=1.25)
        selected = row.get("best_epoch")
        if selected is not None and 1 <= int(selected) <= len(values_pct):
            axis.axvline(int(selected), color="#C44E52", linestyle="--", linewidth=1.0)
            axis.scatter(
                [int(selected)], [values_pct[int(selected) - 1]],
                color="#C44E52", s=18, zorder=4,
            )
        status = (row.get("training_stability") or {}).get("status") or "unknown"
        status_label = {
            "stable": "estável",
            "unstable_oscillation": "oscilação",
            "collapsed": "colapso",
        }.get(status, status)
        axis.set_title(f"{row['model']} - {status_label}", fontsize=8.4, pad=3)
        if len(values_pct):
            lower = max(45.0, float(np.nanmin(values_pct)) - 4.0)
            axis.set_ylim(lower, 101.0)
        axis.grid(alpha=0.22)
        axis.tick_params(labelsize=7.5)
    for axis in axes[-1, :]:
        axis.set_xlabel("Época", fontsize=8)
    for axis in axes[:, 0]:
        axis.set_ylabel("Val. acur. (%)", fontsize=8)
    from matplotlib.lines import Line2D

    fig.legend(
        handles=[
            Line2D([], [], color="#4C72B0", label="acurácia de validação"),
            Line2D([], [], color="#C44E52", linestyle="--", marker="o",
                   markersize=4, label="checkpoint selecionado por val_loss"),
        ],
        fontsize=8,
        ncol=2,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.012),
        frameon=False,
    )
    fig.text(
        0.5,
        0.047,
        "Execução única por arquitetura; curvas não quantificam variância entre sementes.",
        fontsize=7.2,
        ha="center",
        color="#444444",
    )
    fig.subplots_adjust(left=0.085, right=0.985, top=0.97, bottom=0.105, hspace=0.43, wspace=0.25)
    fig.savefig(out / "training_stability.png", dpi=_FIGURE_DPI)
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

    cm_dir = out / "confusion_matrices"
    cm_dir.mkdir(parents=True, exist_ok=True)
    cms = {r["slug"]: _confusion_matrix_for_row(r, extras) for r in rows}

    for r in rows:
        fig, ax = plt.subplots(figsize=(3.1, 3.03))
        _draw_confusion_axis(ax, cms.get(r["slug"]), r["model"])
        fig.tight_layout()
        fig.savefig(cm_dir / f"{r['slug']}.png", dpi=_FIGURE_DPI)
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
    # Saídas antigas que duplicavam visualizações já usadas no artigo. A barra
    # de latência repetia tabela e diagrama de compromisso; o painel composto
    # repetia as matrizes individuais do apêndice.
    for obsolete in ("benchmark_latency.png", "confusion_matrices_article.png"):
        (fig_dir / obsolete).unlink(missing_ok=True)
    fig_accuracy_auc(rows, fig_dir)
    fig_simple_bar(
        rows,
        fig_dir,
        "eer",
        "benchmark_eer.png",
        "EER (Equal Error Rate)",
        "EER (%)",
        scale=100.0,
        figsize=(6.5, 4.80),
    )
    fig_tdcf(rows, fig_dir)
    fig_model_size(rows, fig_dir)
    fig_accuracy_latency_tradeoff(rows, fig_dir)
    fig_robustness(rows, fig_dir)
    fig_robustness_threshold_free(rows, fig_dir)
    fig_det_curves(rows, extras, fig_dir)
    fig_score_distributions_gat(rows, extras, fig_dir)
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
        "--allow-mixed-test-sets",
        action="store_true",
        help=(
            "consolida runs com conjuntos de teste DIFERENTES (variantes do "
            "dataset). Fora de uma ablação deliberada isto produz tabela e "
            "figuras inválidas: as métricas não são comparáveis entre si."
        ),
    )
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

    # CONJUNTOS DE TESTE DIFERENTES NÃO SE CONSOLIDAM.
    #
    # As variantes do dataset (completo e 15k) têm partições de teste
    # distintas, então suas métricas não são comparáveis. Esta checagem existia
    # apenas dentro de `build_significance_report`, que só protege o relatório
    # pareado: o `benchmark_summary.json` e TODAS as figuras do artigo
    # (benchmark_eer.png, benchmark_accuracy_auc.png, benchmark_robustness.png)
    # eram geradas antes dela, comparando modelos avaliados em conjuntos
    # diferentes na mesma barra.
    impressoes = {
        (r.get("dataset") or {}).get("test_split_sha256")
        for r in rows
        if (r.get("dataset") or {}).get("test_split_sha256")
    }
    if len(impressoes) > 1 and not args.allow_mixed_test_sets:
        detalhe = "\n".join(
            f"  {r['model']:<18} {((r.get('dataset') or {}).get('test_split_sha256') or '?')[:16]}"
            f"  {r['path']}"
            for r in rows
        )
        print(
            "ERRO: os runs informados usam CONJUNTOS DE TESTE diferentes "
            f"({len(impressoes)} impressões distintas).\n{detalhe}\n"
            "Métricas de partições diferentes não são comparáveis e não podem "
            "ir para a mesma tabela ou figura. Consolide uma variante por vez, "
            "ou passe --allow-mixed-test-sets se souber o que está fazendo.",
            file=sys.stderr,
        )
        return 2

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
            for obsolete in ("benchmark_latency.png", "confusion_matrices_article.png"):
                (dest / obsolete).unlink(missing_ok=True)
            for src in fig_dir.rglob("*.png"):
                rel = src.relative_to(fig_dir)
                (dest / rel).parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src, dest / rel)
            print(f"-> figuras copiadas para {dest}")

    print("\nPróximo: python scripts/reporting/update_tcc_latex.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
