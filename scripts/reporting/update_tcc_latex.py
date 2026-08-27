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

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmarks.config import (  # noqa: E402
    OFFICIAL_TCC_DISPLAY_NAMES,
    OFFICIAL_TCC_RESULT_ORDER,
)

SUMMARY = ROOT / "data" / "results" / "paper" / "consolidated" / "benchmark_summary.json"
BLIND_HOLDOUT = (
    ROOT / "data" / "results" / "paper" / "consolidated" /
    "p18_avaliacao_cega.json"
)
OUTPUT = ROOT / "data" / "results" / "paper" / "tabelas_benchmark.tex"
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



def pct(value: float | None) -> str:
    if value is None:
        return "--"
    return f"{value * 100:.2f}".replace(".", ",") + r"\%"


def pct_ci(row: dict, field: str) -> str:
    """Estimativa e IC 95% por cluster, em duas linhas na mesma célula."""
    value = row.get(field)
    low = row.get(f"{field}_ci95_low")
    high = row.get(f"{field}_ci95_high")
    if value is None:
        return "--"
    if low is None or high is None:
        return pct(value)
    low_text = f"{float(low) * 100:.2f}".replace(".", ",")
    high_text = f"{float(high) * 100:.2f}".replace(".", ",")
    return (
        rf"\shortstack{{{pct(value)}\\"
        rf"{{\scriptsize [{low_text}; {high_text}]\%}}}}"
    )


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


def model_name(key: str) -> str:
    """Marca os resultados clássicos como exploratórios em toda tabela."""
    label = DISPLAY_NAME[key]
    if key in {"SVM", "RandomForest"}:
        return label + r"\textsuperscript{E}"
    return label


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



def academic_protocol_issues(rows: list[dict]) -> list[str]:
    """Valida os controles que tornam as tabelas comparáveis."""

    issues: list[str] = []
    for row in rows:
        key = row["key"]
        dataset = row.get("dataset") or {}
        noise = row.get("noise_protocol") or {}
        if dataset.get("n_total") != 15000:
            issues.append(f"{key}: dataset n_total={dataset.get('n_total')}, esperado 15000")
        overlap = dataset.get("split_overlap_audit") or {}
        if overlap.get("passed") is not True:
            issues.append(f"{key}: auditoria de duplicatas ausente ou reprovada")
        if noise.get("evaluation_domain") != "waveform":
            issues.append(f"{key}: AWGN não registrado no domínio waveform")
        if noise.get("frontend_after_noise") is not True:
            issues.append(f"{key}: frontend_after_noise não confirmado")
        threshold = row.get("decision_threshold")
        if threshold is None or abs(float(threshold) - 0.5) > 1e-9:
            issues.append(f"{key}: limiar={threshold}, esperado 0.5")
        if key not in {"SVM", "RandomForest"} and row.get("epochs") != 100:
            issues.append(f"{key}: épocas executadas={row.get('epochs')}, esperado 100")
    return issues
def tdcf_value(row: dict) -> float | None:
    """min t-DCF sob qualquer um dos aliases históricos da chave.

    Resolução por PRESENÇA, não por cadeia ``or``: em anti-spoofing o melhor
    valor possível do min t-DCF é exatamente 0,0, que é falsy. Com a cadeia
    anterior um modelo de separação perfeita caía nos aliases inexistentes,
    devolvia ``None`` e era publicado como "--" — o melhor resultado da tabela
    lido como métrica ausente.
    """
    for chave in ("min_tdcf", "min_tDCF", "tdcf", "t_dcf"):
        valor = row.get(chave)
        if valor is not None:
            return valor
    return None


def build_results_table(rows: list[dict]) -> str:
    table_rows = []
    for row in rows:
        key = row["key"]
        # TREINO: a MELHOR EPOCA sobre o orcamento, nao so o orcamento.
        #
        # A coluna trazia apenas as epocas treinadas (100 para todo neural, por
        # `fixed_epoch_budget`), o que nao informa nada: o protocolo restaura o
        # checkpoint da melhor epoca segundo `val_loss`, e
        # os pesos avaliados sao os DAQUELA epoca, nao os da centesima. Escrever
        # "100" sugeria que o modelo publicado e o do fim do treino.
        #
        # Formato "47/100": epoca selecionada sobre o orcamento. O leitor ve de
        # imediato quanto do orcamento foi util e se o modelo parou de melhorar
        # cedo — o CCT seleciona na 47 de 100, e essa distancia e informacao
        # sobre sobreajuste que a tabela escondia.
        if key in {"SVM", "RandomForest"}:
            training = "CV+fit"
        else:
            epochs = row.get("epochs")
            melhor = row.get("best_epoch")
            orcamento = str(int(epochs)) if epochs else "--"
            training = f"{int(melhor)}/{orcamento}" if melhor else orcamento
        # A coluna Status era o literal `\textcolor{successgreen}{OK}`: nenhum
        # valor de `converged` a fazia imprimir outra coisa, então um modelo
        # que REPROVASSE o critério de convergência saía "OK" verde na tabela
        # do artigo. Agora ela reporta o critério medido, como a tabela de
        # robustez de `report.py::_conv` já fazia.
        convergiu = row.get("converged")
        if convergiu is None:
            status = r"\textcolor{mediumgray}{n/d}"
        elif convergiu:
            status = r"\textcolor{successgreen}{OK}"
        else:
            status = r"\textcolor{dangerred}{não}"
        table_rows.append(
            "        "
            + " & ".join(
                [
                    model_name(key),
                    pct_ci(row, "accuracy"),
                    pct_ci(row, "eer"),
                    num(tdcf_value(row), 4),
                    num(row.get("auc"), 3),
                    pct(row.get("f1")),
                    training,
                    status,
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
                    model_name(key),
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
                    model_name(key),
                    pct(row.get("accuracy")),
                    pct((robustness.get("30") or {}).get("accuracy")),
                    pct((robustness.get("20") or {}).get("accuracy")),
                    pct((robustness.get("10") or {}).get("accuracy")),
                    # 5 dB é o único SNR NÃO VISTO no augmentation de treino, e
                    # portanto a única coluna que mede generalização fora da
                    # distribuição. Ficou de fora até 2026-08-14: o artigo
                    # chamava esse nível de "resultado mais informativo do
                    # recorte" e não o tabulava em lugar nenhum.
                    pct((robustness.get("5") or {}).get("accuracy")),
                    # Pior locutor no teste LIMPO. O teste é speaker-disjoint
                    # (11 locutores não vistos), e o agregado esconde dispersão
                    # grande: RawNet2 tem 95,88% de média e 74,2% no pior
                    # locutor; SVM tem 85,31% e 53,2%. Sem esta coluna a tabela
                    # sugere uniformidade que não existe.
                    pct(row.get("worst_speaker_accuracy")),
                ]
            )
            + r" \\"
        )
    return "\n".join(table_rows)


STABILITY_DIAGNOSIS = {
    "stable": "estável",
    "unstable_oscillation": r"\textbf{oscilação}",
    "collapsed": r"\textbf{colapso}",
    "unknown": "sem histórico",
}


def stability_diagnosis(row: dict) -> str:
    status = ((row.get("training_stability") or {}).get("status")) or "unknown"
    return STABILITY_DIAGNOSIS.get(status, status)


def build_stability_table(rows: list[dict]) -> str:
    """Tabela de estabilidade: pico do monitor x época efetivamente avaliada.

    DUAS ÉPOCAS podem divergir: a coluna "Ép. sel." registra `best_epoch`,
    isto é, a época do CHECKPOINT AVALIADO, e "Ép. pico" é o máximo da
    acurácia de validação. Para as entradas Keras, o checkpoint é escolhido por
    `val_loss` nos nove históricos neurais consolidados. As duas épocas
    aparecem junto da acurácia de validação na época selecionada.

    A coluna "Queda" (pico menos final) saiu: a própria legenda advertia que ela
    não demonstra estabilidade, e o diagnóstico automático do pipeline —
    baseado em desvio da cauda e maior queda entre épocas consecutivas — é o
    indicador que o texto de fato usa.
    """
    table_rows = []
    for row in rows:
        key = row["key"]
        if key in {"SVM", "RandomForest"}:
            table_rows.append(
                "        "
                + " & ".join(
                    [
                        model_name(key),
                        "--",
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
        stability = row.get("training_stability") or {}
        peak_epoch = stability.get("best_epoch_by_monitor")
        selected_epoch = stability.get("best_epoch", row.get("best_epoch"))
        monitor_at_selected = stability.get("monitor_at_selected_epoch")
        table_rows.append(
            "        "
            + " & ".join(
                [
                    model_name(key),
                    pct(row.get("best_val")),
                    str(peak_epoch) if peak_epoch is not None else "--",
                    str(selected_epoch) if selected_epoch is not None else "--",
                    pct(monitor_at_selected),
                    pct(row.get("final_val")),
                    stability_diagnosis(row),
                ]
            )
            + r" \\"
        )
    return "\n".join(table_rows)


def build_decision_errors_table(rows: list[dict]) -> str:
    """Resume FPR/FNR do limiar fixo sem depender da figura de matrizes.

    O teste é balanceado. Logo, as duas taxas são recuperáveis da acurácia e do
    F1 já consolidados: o número total de erros é fixado pela acurácia e o de
    verdadeiros positivos pelo F1. Isso preserva a reprodutibilidade mesmo
    quando resultados brutos de execuções substituídas não estão mais no disco.
    """
    table_rows = []
    for row in rows:
        n_total = int((row.get("dataset") or {})["n_test"])
        if n_total % 2:
            raise ValueError("A tabela FPR/FNR exige teste balanceado")
        n_pos = n_total // 2
        n_neg = n_pos
        total_errors = round(n_total * (1 - float(row["accuracy"])))
        f1 = float(row["f1"])
        if f1 >= 1:
            true_positive = n_pos
        else:
            true_positive = round(f1 * total_errors / (2 * (1 - f1)))
        false_negative = n_pos - true_positive
        false_positive = total_errors - false_negative
        table_rows.append(
            "        "
            + " & ".join(
                [
                    model_name(row["key"]),
                    pct(false_positive / n_neg),
                    pct(false_negative / n_pos),
                ]
            )
            + r" \\"
        )
    return "\n".join(table_rows)


def build_blind_holdout_table(payload: dict | None) -> str:
    """Formata a avaliação fresca dos clássicos sem reordenar a bateria."""
    if not payload:
        return ""
    results = payload.get("resultados") or {}
    table_rows = []
    for source_name, key in (("SVM", "SVM"), ("Random Forest", "RandomForest")):
        result = results.get(source_name) or {}
        if not result:
            continue
        table_rows.append(
            "        "
            + " & ".join(
                [
                    model_name(key),
                    integer(result.get("n")),
                    pct(result.get("accuracy")),
                    num(result.get("auc_roc"), 4),
                    pct(result.get("eer")),
                    pct(result.get("worst_speaker_accuracy")),
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
    decision_errors_table: str,
    blind_holdout_table: str,
    figures_dir: str,
) -> str:
    fd = figures_dir.replace("\\", "/")
    return rf"""% ====================================================================
% Tabelas e figuras de benchmark -- GERADO automaticamente.
% Recorte oficial: Random Forest, SVM, CCT, AST, Res2Net, Conformer,
% RawNet2, AASIST, RawGAT-ST, WavLM Original e HuBERT Original.
% Nao editar a mao; regenerar com:
%   python scripts/reporting/consolidate_results.py <runs...> --prefer-last --copy-to data/results/paper/figures
%   python scripts/reporting/update_tcc_latex.py
% ====================================================================

\begin{{table}}[htbp]
\centering
\caption[Resultados consolidados no conjunto de teste limpo.]{{Resultados
consolidados no conjunto de teste limpo. Acurácia e EER trazem, na segunda
linha, IC 95\% por \textit{{bootstrap}} de locutores (unidade de reamostragem:
\num{{11}} locutores). A coluna
\textbf{{Treino}} traz a \textbf{{melhor época}} sobre o orçamento
(\textit{{melhor}}/\textit{{total}}): os pesos avaliados são os do
\textit{{checkpoint}} dessa época, restaurado ao final, e não os da última.}}
\label{{tab:resultados_consolidados}}
\resizebox{{\textwidth}}{{!}}{{%
\begin{{tabular}}{{lccccccc}}
\hline
Modelo & Acur. [IC 95\%] & EER [IC 95\%] & $t$-DCF$^\ast$ & AUC & F1 & Treino & Status \\
\hline
{results_table}
\hline
\end{{tabular}}
}}
\end{{table}}

\noindent\footnotesize{{$t$-DCF$^\ast$: proxy interno de custo normalizado,
sem acoplamento a um sistema ASV oficial.
\textbf{{Critério de seleção da época}}: as nove entradas neurais selecionam o
\textit{{checkpoint}} pela \textbf{{menor perda de validação}}
(\texttt{{val\_loss}}), conforme registrado nos históricos consolidados. A
época reportada é a do critério efetivamente usado. \textbf{{SVM}} e
\textbf{{Random Forest}} não têm época:
são ajustados por busca em grade com validação cruzada agrupada
(\texttt{{CV+fit}}). O sobrescrito \textsuperscript{{E}} identifica os dois
resultados exploratórios, pois o \textit{{frontend}} tabular foi revisto após
uma observação do teste.}}\normalsize

\begin{{table}}[htbp]
\centering
\caption{{Eficiência dos modelos consolidados: número de parâmetros, tamanho
do artefato serializado em disco e tempo de passagem direta do modelo, sem
\textit{{frontend}}. Os tempos pertencem a três \textit{{runtimes}} e só são
comparáveis dentro do mesmo \textit{{runtime}}.}}
\label{{tab:eficiencia_modelos}}
\resizebox{{\textwidth}}{{!}}{{%
\begin{{tabular}}{{lccccc}}
\hline
Modelo & Parâmetros & Artef.\,(MB) & Fwd.\,(ms) & Acur. & EER \\
\hline
{efficiency_table}
\hline
\end{{tabular}}
}}
\end{{table}}

\begin{{table}}[htbp]
\centering
\caption[Robustez a ruído AWGN por modelo consolidado.]{{Acurácia no limiar
fixo $\theta=0{{,}}5$ sob ruído AWGN aplicado à forma de onda antes dos
\textit{{frontends}}, por modelo consolidado. Os níveis de 30, 20 e
\SI{{10}}{{\decibel}} são \textbf{{casados}} com o \textit{{augmentation}} de
treino; \SI{{5}}{{\decibel}} é \textbf{{não visto}}. A última coluna traz a
acurácia no pior dos 11 locutores do teste (nenhum visto no treino), medida em
áudio limpo.}}
\label{{tab:robustez_awgn}}
\resizebox{{\textwidth}}{{!}}{{%
\begin{{tabular}}{{lcccccc}}
\hline
Modelo & Limpo & 30\,dB & 20\,dB & 10\,dB & 5\,dB$^\dagger$ & Pior locutor \\
\hline
{robustness_table}
\hline
\end{{tabular}}
}}

\noindent\footnotesize{{$^\dagger$ SNR ausente do \textit{{augmentation}} de
treino: é a única coluna que mede generalização a uma severidade de AWGN não
vista, ainda dentro da mesma família de degradação.}}\normalsize
\end{{table}}

\begin{{table}}[htbp]
\centering
\caption[Avaliação complementar fresca dos classificadores clássicos.]{{Avaliação
complementar fresca dos classificadores clássicos, sem retreino, fora da
subamostragem canônica de \num{{15000}} itens. Os \num{{4916}} exemplos incluem
\num{{22}} locutores e \num{{397}} textos disjuntos do ajuste. O ensaio reduz o
viés de seleção do \textit{{frontend}} v2, mas permanece no mesmo
\textit{{corpus}}/gerador e não é validação externa.}}
\label{{tab:holdout_classicos}}
\begin{{tabular}}{{lccccc}}
\hline
Modelo & $n$ & Acur. & AUC-ROC & EER & Pior locutor \\
\hline
{blind_holdout_table}
\hline
\end{{tabular}}
\end{{table}}

\begin{{table}}[ht]
\centering
\caption[Curvas de validação em execução única.]{{Curvas de validação em execução única. \textbf{{Val.\,pico}} é o
máximo da acurácia de validação e \textbf{{Ép.\,pico}} a época em que ocorre;
\textbf{{Ép.\,sel.}} é a época do \textit{{checkpoint}} efetivamente avaliado,
escolhido por perda de validação nas nove entradas neurais, e
\textbf{{Val.\,@sel.}} é a acurácia de validação nessa
época. O pico de acurácia e a época selecionada podem divergir. A última coluna traz
o diagnóstico automático do \textit{{pipeline}} (campo
\texttt{{training\_stability}}), baseado no desvio do monitor nas últimas
\num{{50}} épocas e na maior queda entre épocas consecutivas.}}
\label{{tab:estabilidade_treinamento}}
\resizebox{{\textwidth}}{{!}}{{%
\begin{{tabular}}{{lcccccc}}
\hline
Modelo & Val.\,pico & Ép.\,pico & Ép.\,sel. & Val.\,@sel. & Val.\,final & Diagnóstico \\
\hline
{stability_table}
\hline
\end{{tabular}}
}}
\end{{table}}

\begin{{table}}[htbp]
\centering
\caption{{Taxas de falso positivo e falso negativo no conjunto limpo, no
limiar fixo $\theta=0{{,}}5$.}}
\label{{tab:erros_decisao}}
\begin{{tabular}}{{lcc}}
\hline
Modelo & FPR & FNR \\
\hline
{decision_errors_table}
\hline
\end{{tabular}}
\end{{table}}
\FloatBarrier

\begin{{figure}}[htbp]\centering
\caption[Acurácia e AUC-ROC no conjunto de teste limpo.]{{Acurácia e AUC-ROC no
conjunto de teste limpo, com IC 95\% obtido por \textit{{bootstrap}} de
locutores. O eixo ampliado não parte de zero e serve para comparar a região de
alto desempenho; valores absolutos estão anotados.}}
\label{{fig:benchmark_accuracy_auc}}
\includegraphics[width=\textwidth]{{{fd}/benchmark_accuracy_auc.png}}
\end{{figure}}
\FloatBarrier

\begin{{figure}}[htbp]\centering
\caption{{Robustez a AWGN na forma de onda: acurácia no limiar fixo
$\theta=0{{,}}5$ em função da SNR. Os dois painéis separam as arquiteturas pela
acurácia limpa apenas para legibilidade; a separação não altera o ranking nem a
escala.}}
\label{{fig:benchmark_robustness}}
\includegraphics[width=\textwidth]{{{fd}/benchmark_robustness.png}}
\end{{figure}}
\FloatBarrier

\begin{{figure}}[htbp]\centering
\caption{{Robustez a AWGN por métricas livres de limiar. AUC-ROC maior e EER
menor indicam melhor separabilidade. A leitura conjunta com a
Figura~\ref{{fig:benchmark_robustness}} distingue perda de discriminação de
deriva do ponto de operação fixo.}}
\label{{fig:benchmark_robustness_threshold_free}}
\includegraphics[width=\textwidth]{{{fd}/benchmark_robustness_threshold_free.png}}
\end{{figure}}
\FloatBarrier

\begin{{figure}}[htbp]\centering
\caption{{Taxa de Erro Igual (EER) por configuração completa de modelo.}}
\label{{fig:benchmark_eer}}
\includegraphics[width=0.92\textwidth]{{{fd}/benchmark_eer.png}}
\end{{figure}}
\FloatBarrier

\begin{{figure}}[htbp]\centering
\caption[Curvas DET no conjunto de teste limpo.]{{Curvas DET no conjunto de
teste limpo, em eixos de probabilidade normal. FPR é a fração de áudios
\textit{{bonafide}} rejeitados e FNR a fração de ataques aceitos; a interseção
com a diagonal aproxima o EER. O recorte de \num{{0,1}}\% a \num{{40}}\% amplia
a região operacional e não exibe taxas fora desse intervalo.}}
\label{{fig:benchmark_det}}
\includegraphics[width=0.98\textwidth]{{{fd}/benchmark_det_curves.png}}
\end{{figure}}
\FloatBarrier

\begin{{figure}}[htbp]\centering
\caption{{$t$-DCF$^\ast$ por configuração completa de modelo.}}
\label{{fig:benchmark_tdcf}}
\includegraphics[width=0.92\textwidth]{{{fd}/benchmark_tdcf.png}}
\end{{figure}}
\FloatBarrier

\begin{{figure}}[htbp]\centering
\caption{{Tamanho dos artefatos serializados em disco, em escala logarítmica.}}
\label{{fig:benchmark_size}}
\includegraphics[width=0.95\textwidth]{{{fd}/benchmark_size.png}}
\end{{figure}}
\FloatBarrier

\begin{{figure}}[htbp]\centering
\caption[Compromisso entre acurácia, latência e tamanho do artefato.]{{Compromisso
descritivo entre acurácia limpa, latência de passagem direta e tamanho do
artefato. Cada painel tem escala própria e corresponde a um \textit{{runtime}};
latências são comparáveis apenas dentro do painel e não sustentam ranking global
de velocidade.}}
\label{{fig:benchmark_accuracy_latency_tradeoff}}
\includegraphics[width=\textwidth]{{{fd}/benchmark_accuracy_latency_tradeoff.png}}
\end{{figure}}
\FloatBarrier

\begin{{figure}}[htbp]\centering
\caption[Distribuições de escores de AASIST e RawGAT-ST.]{{Distribuições de
$p_{{fake}}$ por classe verdadeira em AASIST e RawGAT-ST, com contagens em
escala logarítmica. O percentual no título mede escores abaixo de \num{{0,05}}
ou acima de \num{{0,95}}. A figura descreve concentração e sobreposição; não
demonstra, isoladamente, calibração nem atribui causalidade à topologia.}}
\label{{fig:score_distributions_gat}}
\includegraphics[width=\textwidth]{{{fd}/score_distributions_gat.png}}
\end{{figure}}
\FloatBarrier

\begin{{figure}}[htbp]\centering
\caption[Importância por permutação do Random Forest.]{{Quinze maiores
importâncias por permutação do \textit{{Random Forest}} no teste canônico,
com média e desvio-padrão em \num{{30}} repetições. A análise é exploratória:
o \textit{{frontend}} tabular v2 foi definido após uma rodada de teste, e
descritores correlacionados podem repartir importância entre si. As barras não
são efeitos causais.}}
\label{{fig:rf_permutation_importance}}
\includegraphics[width=\textwidth]{{{fd}/rf_feature_importance.png}}
\end{{figure}}
\FloatBarrier

\begin{{figure}}[htbp]\centering
\caption[Trajetórias de validação e seleção de \textit{{checkpoint}}.]{{Trajetórias
de acurácia de validação das nove configurações neurais em execução única. A
linha tracejada marca o \textit{{checkpoint}} selecionado pela menor
\texttt{{val\_loss}}; divergências entre o pico de acurácia e a época escolhida
são esperadas. A figura mostra oscilação intraexecução, mas não estima variância
entre sementes.}}
\label{{fig:training_stability}}
\includegraphics[width=0.95\textwidth]{{{fd}/training_stability.png}}
\end{{figure}}
\FloatBarrier
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
    parser.add_argument("--blind-holdout", default=str(BLIND_HOLDOUT),
                        help="avaliação fresca dos classificadores clássicos")
    parser.add_argument("--figures-dir", default=FIGURES_DIR,
                        help="diretório das figuras visto pelo main.tex")
    parser.add_argument("--allow-incomplete", action="store_true",
                        help="permite gerar fragmento com menos de 11 modelos")
    parser.add_argument("--allow-legacy", action="store_true",
                        help="ignora validações do protocolo acadêmico")
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
            "Rode antes scripts/reporting/consolidate_results.py."
        )
    rows = model_rows(json.loads(summary.read_text(encoding="utf-8")))
    if len(rows) != len(MODEL_ORDER) and not args.allow_incomplete:
        sys.exit(
            f"ERRO: summary contém {len(rows)}/{len(MODEL_ORDER)} modelos oficiais. "
            "Use --allow-incomplete somente para depuração."
        )
    issues = academic_protocol_issues(rows)
    if issues and not args.allow_legacy:
        sys.exit("ERRO: protocolo acadêmico inválido:\n- " + "\n- ".join(issues))
    holdout_path = Path(args.blind_holdout)
    blind_holdout = (
        json.loads(holdout_path.read_text(encoding="utf-8"))
        if holdout_path.exists()
        else None
    )
    fragment = tables_fragment(
        results_table=build_results_table(rows),
        efficiency_table=build_efficiency_table(rows),
        robustness_table=build_robustness_table(rows),
        stability_table=build_stability_table(rows),
        decision_errors_table=build_decision_errors_table(rows),
        blind_holdout_table=build_blind_holdout_table(blind_holdout),
        figures_dir=args.figures_dir,
    )

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    # newline="\n": sem isso, no Windows o Python traduz \n para \r\n e o
    # fragmento inteiro aparece como reescrito no diff a cada regeração,
    # escondendo a mudança real de números entre duas execuções.
    output.write_text(fragment, encoding="utf-8", newline="\n")
    print(f"Fragmento escrito: {output.resolve()}")


if __name__ == "__main__":
    main()
