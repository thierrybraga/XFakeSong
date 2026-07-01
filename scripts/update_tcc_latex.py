"""Generate the benchmark LaTeX fragment for the TCC article scope.

This script is intentionally conservative: it only writes the benchmark
fragment (`tabelas_benchmark.tex`) and only includes the official models fixed
in `benchmarks.config.OFFICIAL_TCC_MODEL_MANIFEST`. The previous full-thesis rewrite mode was
removed because it could reintroduce legacy 14-model prose.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmarks.config import (  # noqa: E402
    OFFICIAL_TCC_DISPLAY_NAMES,
    OFFICIAL_TCC_RESULT_ORDER,
)

SUMMARY = Path("results/tcc_consolidated/benchmark_summary.json")
OUTPUT = Path("tcc_overleaf/tabelas_benchmark.tex")
FIGURES_DIR = "figures"

MODEL_ORDER = list(OFFICIAL_TCC_RESULT_ORDER)
DISPLAY_NAME = dict(OFFICIAL_TCC_DISPLAY_NAMES)

KEY_ALIAS = {
    "Hybrid CNN-Transformer": "CCT",
    "SpectrogramTransformer": "AST",
    "Audio Spectrogram Transformer": "AST",
    "MultiscaleCNN": "Res2Net",
    "Random Forest": "RandomForest",
    "WavLM": "WavLM Original",
    "HuBERT": "HuBERT Original",
}

BEST_EPOCH = {
    "RawNet2": 55,
    "AASIST": 51,
    "RawGAT-ST": 3,
    "Conformer": 17,
    "CCT": 30,
    "AST": 50,
    "Res2Net": 25,
}


def pct(value: float | None) -> str:
    if value is None:
        return "--"
    return f"{value * 100:.2f}".replace(".", ",") + r"\%"


def num(value: float | None, digits: int = 4) -> str:
    if value is None:
        return "--"
    return f"{value:.{digits}f}".replace(".", ",")


def ms(value: float | None) -> str:
    if value is None:
        return "--"
    return f"{value:.2f}".replace(".", ",")


def integer(value: int | None) -> str:
    if value is None:
        return "--"
    return f"{value:,}".replace(",", ".")


def canonical_key(row: dict) -> str | None:
    raw_key = row.get("key") or row.get("model")
    if not raw_key:
        return None
    key = KEY_ALIAS.get(str(raw_key), str(raw_key))
    return key if key in DISPLAY_NAME else None


def model_rows(data: list[dict]) -> list[dict]:
    by_key: dict[str, dict] = {}
    skipped: list[str] = []

    for source_row in data:
        row = dict(source_row)
        key = canonical_key(row)
        if key is None:
            skipped.append(str(row.get("key") or row.get("model") or "<sem chave>"))
            continue
        row["key"] = key
        by_key[key] = row

    missing = [key for key in MODEL_ORDER if key not in by_key]
    if missing:
        print(f"AVISO: modelos ausentes no summary: {missing}", file=sys.stderr)
    if skipped:
        print(f"AVISO: modelos fora do recorte do artigo ignorados: {skipped}",
              file=sys.stderr)

    return [by_key[key] for key in MODEL_ORDER if key in by_key]


def tdcf_value(row: dict) -> float | None:
    return (
        row.get("min_tdcf")
        or row.get("min_tDCF")
        or row.get("tdcf")
        or row.get("t_dcf")
    )


def build_results_table(rows: list[dict]) -> str:
    table_rows = []
    for row in rows:
        key = row["key"]
        training = "CV+fit" if key in {"SVM", "RandomForest"} else "100"
        table_rows.append(
            "        "
            + " & ".join(
                [
                    DISPLAY_NAME[key],
                    pct(row.get("accuracy")),
                    pct(row.get("eer")),
                    num(tdcf_value(row), 4),
                    num(row.get("auc"), 3),
                    pct(row.get("f1")),
                    ms(row.get("latency")),
                    training,
                    r"\textcolor{successgreen}{OK}",
                ]
            )
            + r" \\"
        )
    return "\n".join(table_rows)


def build_efficiency_table(rows: list[dict]) -> str:
    table_rows = []
    for row in rows:
        key = row["key"]
        table_rows.append(
            "        "
            + " & ".join(
                [
                    DISPLAY_NAME[key],
                    integer(row.get("params")),
                    ms(row.get("size")),
                    ms(row.get("latency")),
                    pct(row.get("accuracy")),
                    pct(row.get("eer")),
                ]
            )
            + r" \\"
        )
    return "\n".join(table_rows)


def build_robustness_table(rows: list[dict]) -> str:
    table_rows = []
    for row in rows:
        key = row["key"]
        robustness = row.get("robustness") or {}
        table_rows.append(
            "        "
            + " & ".join(
                [
                    DISPLAY_NAME[key],
                    pct(row.get("accuracy")),
                    pct((robustness.get("30") or {}).get("accuracy")),
                    pct((robustness.get("20") or {}).get("accuracy")),
                    pct((robustness.get("10") or {}).get("accuracy")),
                ]
            )
            + r" \\"
        )
    return "\n".join(table_rows)


def stability_note(key: str, drop: float | None) -> str:
    if key == "RawGAT-ST":
        return "convergência precoce"
    if key == "CCT" and drop is not None and drop >= 0.03:
        return "flutuação moderada"
    return "estável"


def build_stability_table(rows: list[dict]) -> str:
    table_rows = []
    for row in rows:
        key = row["key"]
        if key in {"SVM", "RandomForest"}:
            table_rows.append(
                "        "
                + " & ".join(
                    [
                        DISPLAY_NAME[key],
                        "--",
                        "CV+fit",
                        "--",
                        "--",
                        "não aplicável (modelo clássico)",
                    ]
                )
                + r" \\"
            )
            continue
        best_epoch = row.get("best_epoch") or BEST_EPOCH.get(key)
        drop = None
        if row.get("best_val") is not None and row.get("final_val") is not None:
            drop = max(0.0, row["best_val"] - row["final_val"])
        table_rows.append(
            "        "
            + " & ".join(
                [
                    DISPLAY_NAME[key],
                    pct(row.get("best_val")),
                    str(best_epoch) if best_epoch is not None else "--",
                    pct(row.get("final_val")),
                    pct(drop),
                    stability_note(key, drop),
                ]
            )
            + r" \\"
        )
    return "\n".join(table_rows)


def tables_fragment(
    results_table: str,
    efficiency_table: str,
    robustness_table: str,
    stability_table: str,
    figures_dir: str,
) -> str:
    fd = figures_dir.replace("\\", "/")
    return rf"""% ====================================================================
% Tabelas e figuras de benchmark -- GERADO automaticamente.
% Recorte oficial: Random Forest, SVM, CCT, AST, Res2Net, Conformer,
% RawNet2, AASIST, RawGAT-ST, WavLM Original e HuBERT Original.
% Nao editar a mao; regenerar com:
%   python scripts/consolidate_results.py <runs...> --prefer-last --copy-to tcc_overleaf/figures
%   python scripts/update_tcc_latex.py
% ====================================================================

\begin{{table}}[ht]
\centering
\caption{{Resultados consolidados no conjunto de teste limpo.}}
\label{{tab:resultados_consolidados}}
\resizebox{{\textwidth}}{{!}}{{%
\begin{{tabular}}{{lcccccccc}}
\hline
Modelo & Acur. & EER & $t$-DCF$^\ast$ & AUC & F1 & Lat.\,(ms) & Treino & Status \\
\hline
{results_table}
\hline
\end{{tabular}}
}}
\end{{table}}

\noindent\footnotesize{{$t$-DCF$^\ast$: proxy interno de custo normalizado,
sem acoplamento a um sistema ASV oficial.}}\normalsize

\begin{{table}}[ht]
\centering
\caption{{Eficiência computacional dos modelos consolidados.}}
\label{{tab:eficiencia_modelos}}
\resizebox{{\textwidth}}{{!}}{{%
\begin{{tabular}}{{lccccc}}
\hline
Modelo & Parâmetros & Tam.\,(MB) & Lat.\,(ms) & Acur. & EER \\
\hline
{efficiency_table}
\hline
\end{{tabular}}
}}
\end{{table}}

\begin{{table}}[ht]
\centering
\caption{{Robustez a ruído AWGN por modelo consolidado.}}
\label{{tab:robustez_awgn}}
\resizebox{{\textwidth}}{{!}}{{%
\begin{{tabular}}{{lcccc}}
\hline
Modelo & Limpo & 30\,dB & 20\,dB & 10\,dB \\
\hline
{robustness_table}
\hline
\end{{tabular}}
}}
\end{{table}}

\begin{{table}}[ht]
\centering
\caption{{Estabilidade de treinamento (validação).}}
\label{{tab:estabilidade_treinamento}}
\resizebox{{\textwidth}}{{!}}{{%
\begin{{tabular}}{{lccccc}}
\hline
Modelo & Val.\,pico & Época & Val.\,final & Queda & Nota \\
\hline
{stability_table}
\hline
\end{{tabular}}
}}
\end{{table}}

\begin{{figure}}[ht]\centering
\includegraphics[width=\textwidth]{{{fd}/benchmark_accuracy_auc.png}}
\caption{{Acurácia e AUC-ROC no conjunto de teste limpo.}}
\label{{fig:benchmark_accuracy_auc}}
\end{{figure}}

\begin{{figure}}[ht]\centering
\includegraphics[width=\textwidth]{{{fd}/benchmark_robustness.png}}
\caption{{Robustez a ruído (acurácia vs.\ SNR).}}
\label{{fig:benchmark_robustness}}
\end{{figure}}

\begin{{figure}}[ht]\centering
\includegraphics[width=0.92\textwidth]{{{fd}/benchmark_eer.png}}
\caption{{Taxa de Erro Igual (EER) por arquitetura.}}
\label{{fig:benchmark_eer}}
\end{{figure}}

\begin{{figure}}[ht]\centering
\includegraphics[width=0.92\textwidth]{{{fd}/benchmark_tdcf.png}}
\caption{{$t$-DCF$^\ast$ por arquitetura.}}
\label{{fig:benchmark_tdcf}}
\end{{figure}}

\begin{{figure}}[ht]\centering
\includegraphics[width=0.95\textwidth]{{{fd}/benchmark_latency.png}}
\includegraphics[width=0.95\textwidth]{{{fd}/benchmark_size.png}}
\caption{{Latência e tamanho dos modelos.}}
\label{{fig:benchmark_latency_size}}
\end{{figure}}

\begin{{figure}}[ht]\centering
\includegraphics[width=0.95\textwidth]{{{fd}/benchmark_accuracy_latency_tradeoff.png}}
\caption{{Trade-off entre acurácia, latência e tamanho dos artefatos.}}
\label{{fig:benchmark_accuracy_latency_tradeoff}}
\end{{figure}}

\begin{{figure}}[ht]\centering
\includegraphics[width=0.95\textwidth]{{{fd}/training_stability.png}}
\caption{{Resumo de estabilidade e convergência dos onze modelos consolidados. Modelos clássicos são indicados como ajuste por validação cruzada, sem trajetória temporal por época.}}
\label{{fig:training_stability}}
\end{{figure}}

\begin{{figure}}[ht]\centering
\includegraphics[width=\textwidth]{{{fd}/confusion_matrices_article.png}}
\caption{{Matrizes de confusão dos onze modelos avaliados no artigo.}}
\label{{fig:confusion_matrices_article}}
\end{{figure}}
"""


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Gera tabelas_benchmark.tex com o manifesto oficial do artigo."
        )
    )
    parser.add_argument("--summary", default=str(SUMMARY),
                        help="benchmark_summary.json consolidado")
    parser.add_argument("--output", default=str(OUTPUT),
                        help="arquivo .tex de saída")
    parser.add_argument("--figures-dir", default=FIGURES_DIR,
                        help="diretório das figuras visto pelo main.tex")
    parser.add_argument("--source", default=None,
                        help="aceito por compatibilidade; não é reescrito")
    parser.add_argument("--in-place", action="store_true",
                        help="desativado: este script não altera main.tex")
    parser.add_argument("--full-rewrite", action="store_true",
                        help="desativado: modo legado removido por segurança")
    args = parser.parse_args()

    if args.full_rewrite or args.in_place:
        sys.exit(
            "ERRO: --full-rewrite/--in-place foram desativados. "
            "Atualize apenas tabelas_benchmark.tex e mantenha main.tex revisado."
        )

    summary = Path(args.summary)
    if not summary.exists():
        sys.exit(
            f"ERRO: summary não encontrado: {summary}. "
            "Rode antes scripts/consolidate_results.py."
        )

    rows = model_rows(json.loads(summary.read_text(encoding="utf-8")))
    fragment = tables_fragment(
        results_table=build_results_table(rows),
        efficiency_table=build_efficiency_table(rows),
        robustness_table=build_robustness_table(rows),
        stability_table=build_stability_table(rows),
        figures_dir=args.figures_dir,
    )

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(fragment, encoding="utf-8")
    print(f"Fragmento escrito: {output.resolve()}")


if __name__ == "__main__":
    main()
