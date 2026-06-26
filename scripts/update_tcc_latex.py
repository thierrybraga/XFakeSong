"""Generate an updated TCC LaTeX file with consolidated benchmark results."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


# Defaults reprodutíveis (antes: SOURCE apontava para um .txt morto em
# C:\Users\...\.codex\attachments). Agora o template-base é o próprio tcc.tex
# e os dados vêm do summary gerado por scripts/consolidate_results.py. Todos
# podem ser sobrescritos por CLI (--source/--summary/--output/--in-place).
SOURCE = Path("tcc_overleaf/tcc.tex")
SUMMARY = Path("results/tcc_consolidated/benchmark_summary.json")
OUTPUT = Path("tcc_overleaf/tcc_atualizado.tex")

MODEL_ORDER = [
    "WavLM Original",
    "HuBERT Original",
    "RawNet2",
    "Sonic Sleuth",
    "AASIST",
    "RawGAT-ST",
    "Conformer",
    "Hybrid CNN-Transformer",
    "SpectrogramTransformer",
    "EfficientNet-LSTM",
    "MultiscaleCNN",
    "Ensemble",
    "SVM",
    "RandomForest",
]

DISPLAY_NAME = {
    "WavLM Original": "WavLM Original",
    "HuBERT Original": "HuBERT Original",
    "RawNet2": "RawNet2",
    "Sonic Sleuth": "Sonic Sleuth",
    "AASIST": "AASIST",
    "RawGAT-ST": "RawGAT-ST",
    "Conformer": "Conformer",
    "Hybrid CNN-Transformer": "Hybrid CNN-Transformer",
    "SpectrogramTransformer": "Spectrogram Transformer",
    "EfficientNet-LSTM": "EfficientNet-LSTM",
    "MultiscaleCNN": "MultiscaleCNN",
    "Ensemble": "Ensemble",
    "SVM": "SVM",
    "RandomForest": "Random Forest",
}

BEST_EPOCH = {
    "WavLM Original": 46,
    "HuBERT Original": 92,
    "RawNet2": 58,
    "Sonic Sleuth": 23,
    "AASIST": 100,
    "RawGAT-ST": 55,
    "Conformer": 49,
    "Hybrid CNN-Transformer": 96,
    "SpectrogramTransformer": 11,
    "EfficientNet-LSTM": 74,
    "MultiscaleCNN": 69,
    "Ensemble": 62,
}


def pct(value: float | None) -> str:
    if value is None:
        return "--"
    return f"{value * 100:.2f}".replace(".", ",") + r"\%"


def num(value: float | None, digits: int = 3) -> str:
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


def model_rows(data: list[dict]) -> list[dict]:
    # O summary consolidado (consolidate_results.py) já traz TODAS as
    # arquiteturas — inclusive RandomForest — com métricas e robustez. O
    # antigo caso especial que lia um metrics.json fixo do RF foi removido.
    by_key = {row["key"]: dict(row) for row in data if "key" in row}
    missing = [key for key in MODEL_ORDER if key not in by_key]
    if missing:
        # Tolerante: avisa e segue com o que houver (ex.: consolidação parcial).
        print(f"AVISO: faltando no summary: {missing}", file=sys.stderr)
    ordered = [by_key[key] for key in MODEL_ORDER if key in by_key]
    ordered += [r for k, r in by_key.items() if k not in MODEL_ORDER]
    return ordered


def build_results_table(rows: list[dict]) -> str:
    table_rows = []
    for row in rows:
        key = row["key"]
        status = r"\textcolor{warningorange}{Tuning}" if key == "SpectrogramTransformer" else r"\textcolor{successgreen}{OK}"
        training = "CV+fit" if key in {"SVM", "RandomForest"} else "100"
        table_rows.append(
            "        "
            + " & ".join(
                [
                    DISPLAY_NAME[key],
                    pct(row["accuracy"]),
                    pct(row["eer"]),
                    num(row["auc"]),
                    pct(row["f1"]),
                    ms(row["latency"]),
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
                    pct(row["accuracy"]),
                    pct((robustness.get("30") or {}).get("accuracy")),
                    pct((robustness.get("20") or {}).get("accuracy")),
                    pct((robustness.get("10") or {}).get("accuracy")),
                ]
            )
            + r" \\"
        )
    return "\n".join(table_rows)


def build_stability_table(rows: list[dict]) -> str:
    table_rows = []
    for row in rows:
        key = row["key"]
        if key in {"SVM", "RandomForest"}:
            continue
        best_epoch = row.get("best_epoch") or BEST_EPOCH.get(key, "--")
        drop = None
        if row.get("best_val") is not None and row.get("final_val") is not None:
            drop = max(0.0, row["best_val"] - row["final_val"])
        note = "checkpoint obrigatório" if key == "SpectrogramTransformer" else "estável"
        table_rows.append(
            "        "
            + " & ".join(
                [
                    DISPLAY_NAME[key],
                    pct(row.get("best_val")),
                    str(best_epoch),
                    pct(row.get("final_val")),
                    pct(drop),
                    note,
                ]
            )
            + r" \\"
        )
    return "\n".join(table_rows)


def _tables_fragment(results_table, efficiency_table, robustness_table,
                     stability_table, figures_dir: str) -> str:
    """Fragmento .tex autocontido (tabelas data-driven + figuras).

    Projetado para `\\input{}` no tcc.tex — NÃO toca na prosa da tese. Os
    valores vêm do benchmark_summary.json (consolidate_results.py), logo
    refletem o treinamento mais recente.
    """
    fd = figures_dir.replace("\\", "/")
    return rf"""% ====================================================================
% Tabelas e figuras de benchmark — GERADO automaticamente.
% Origem: results/.../results.json -> consolidate_results.py -> summary
% NÃO editar à mão; regenerar com:
%   python scripts/consolidate_results.py <runs...>
%   python scripts/update_tcc_latex.py
% ====================================================================

\begin{{table}}[ht]
\centering
\caption{{Desempenho no conjunto limpo.}}
\begin{{tabular}}{{lccccccc}}
\hline
Modelo & Acur. & EER & AUC & F1 & Lat.\,(ms) & Treino & Status \\
\hline
{results_table}
\hline
\end{{tabular}}
\end{{table}}

\begin{{table}}[ht]
\centering
\caption{{Eficiência (parâmetros, tamanho, latência).}}
\begin{{tabular}}{{lccccc}}
\hline
Modelo & Parâmetros & Tam.\,(MB) & Lat.\,(ms) & Acur. & EER \\
\hline
{efficiency_table}
\hline
\end{{tabular}}
\end{{table}}

\begin{{table}}[ht]
\centering
\caption{{Robustez a ruído (acurácia por SNR, AWGN).}}
\begin{{tabular}}{{lcccc}}
\hline
Modelo & Limpo & 30\,dB & 20\,dB & 10\,dB \\
\hline
{robustness_table}
\hline
\end{{tabular}}
\end{{table}}

\begin{{table}}[ht]
\centering
\caption{{Estabilidade de treinamento (validação).}}
\begin{{tabular}}{{lccccc}}
\hline
Modelo & Val.\,pico & Época & Val.\,final & Queda & Nota \\
\hline
{stability_table}
\hline
\end{{tabular}}
\end{{table}}

\begin{{figure}}[ht]\centering
\includegraphics[width=\textwidth]{{{fd}/benchmark_accuracy_auc.png}}
\caption{{Acurácia e AUC-ROC por arquitetura.}}
\end{{figure}}

\begin{{figure}}[ht]\centering
\includegraphics[width=\textwidth]{{{fd}/benchmark_robustness.png}}
\caption{{Robustez a ruído (acurácia vs.\ SNR).}}
\end{{figure}}

\begin{{figure}}[ht]\centering
\includegraphics[width=0.92\textwidth]{{{fd}/benchmark_eer.png}}
\caption{{EER por arquitetura.}}
\end{{figure}}

\begin{{figure}}[ht]\centering
\includegraphics[width=0.95\textwidth]{{{fd}/benchmark_latency.png}}
\includegraphics[width=0.95\textwidth]{{{fd}/benchmark_size.png}}
\caption{{Latência e tamanho dos modelos.}}
\end{{figure}}

\begin{{figure}}[ht]\centering
\includegraphics[width=0.95\textwidth]{{{fd}/training_stability.png}}
\caption{{Acurácia de validação ao longo das épocas.}}
\end{{figure}}
"""


def replace_between(text: str, start: str, end: str, replacement: str) -> str:
    start_index = text.index(start)
    end_index = text.index(end, start_index)
    return text[:start_index] + replacement.rstrip() + "\n\n" + text[end_index:]


def replace_abstract(text: str) -> str:
    start = r"\begin{center}\textbf{RESUMO}\end{center}"
    end = "\n\\newpage\n\\pagestyle{fancy}"
    replacement = r"""
\begin{center}\textbf{RESUMO}\end{center}
\vspace{0.5cm}

\begin{singlespace}
\noindent
Este trabalho apresenta o desenvolvimento e a avaliação de um \textit{pipeline}
modular e de código aberto para detecção de áudio sintético. A metodologia
integra pré-processamento de áudio, extração de características acústicas,
treinamento supervisionado, inferência e geração automática de relatórios de
\textit{benchmark}. A versão experimental consolidada avalia quatorze
arquiteturas: WavLM, HuBERT, RawNet2, Sonic Sleuth, AASIST, RawGAT-ST,
Conformer, Hybrid CNN-Transformer, Spectrogram Transformer,
EfficientNet-LSTM, MultiscaleCNN, Ensemble, SVM e Random Forest. Os
experimentos foram conduzidos sobre um \textit{dataset} balanceado com
15.000 amostras de áudio, dividido em 70/15/15 para treino, validação e teste,
com janelas padronizadas em 16\,kHz, mono e 5\,s. O treinamento neural utilizou
GPU NVIDIA RTX 3060 via WSL2/CUDA, enquanto SVM e Random Forest foram
otimizados por validação cruzada em CPU.

Os resultados mostram que Conformer e Sonic Sleuth atingiram 100,00\% de
acurácia e EER de 0,00\%, seguidos por Hybrid CNN-Transformer (99,96\%),
MultiscaleCNN (99,73\%), SVM (99,02\%), Random Forest (98,18\%), RawNet2
(96,36\%), Ensemble (95,82\%), RawGAT-ST (95,29\%), AASIST (93,64\%),
HuBERT original (92,71\%), EfficientNet-LSTM (91,16\%) e WavLM original
(86,36\%). O Spectrogram Transformer obteve 71,51\% no conjunto de teste,
apesar de ter alcançado 98,00\% de validação na época 11, caracterizando
instabilidade de treinamento e necessidade de novo ciclo de ajuste. Os
gráficos consolidados de acurácia, EER, AUC-ROC, latência, tamanho dos modelos,
robustez a ruído AWGN e estabilidade do treinamento foram gerados
automaticamente para apoiar a análise do TCC.

\vspace{0.4cm}
\noindent\textbf{Palavras-chave:} \textit{deepfake} de áudio.
\textit{benchmark}. Arquiteturas neurais. WavLM. HuBERT. Conformer.
MultiscaleCNN. Robustez a ruído.
\end{singlespace}

\vspace{1cm}
\begin{center}\textbf{ABSTRACT}\end{center}
\vspace{0.5cm}

\begin{singlespace}
\noindent
This work presents the development and evaluation of a modular open-source
pipeline for synthetic audio detection. The methodology integrates audio
preprocessing, acoustic feature extraction, supervised training, inference and
automatic benchmark report generation. The consolidated experimental version
evaluates fourteen architectures: WavLM, HuBERT, RawNet2, Sonic Sleuth,
AASIST, RawGAT-ST, Conformer, Hybrid CNN-Transformer, Spectrogram Transformer,
EfficientNet-LSTM, MultiscaleCNN, Ensemble, SVM and Random Forest. Experiments
were conducted on a balanced 15,000-sample audio dataset with a 70/15/15
train/validation/test split and 16\,kHz mono 5\,s windows. Neural models were
trained on an NVIDIA RTX 3060 GPU through WSL2/CUDA, while SVM and Random
Forest were optimized with CPU cross-validation.

Results show that Conformer and Sonic Sleuth achieved 100.00\% accuracy and
0.00\% EER, followed by Hybrid CNN-Transformer (99.96\%), MultiscaleCNN
(99.73\%), SVM (99.02\%), Random Forest (98.18\%), RawNet2 (96.36\%),
Ensemble (95.82\%), RawGAT-ST (95.29\%), AASIST (93.64\%), original HuBERT
(92.71\%), EfficientNet-LSTM (91.16\%) and original WavLM (86.36\%). The
Spectrogram Transformer reached 71.51\% on the test set despite reaching
98.00\% validation accuracy at epoch 11, indicating training instability and
the need for further tuning. Consolidated plots for accuracy, EER, AUC-ROC,
latency, model size, AWGN robustness and training stability were automatically
generated to support the TCC analysis.

\vspace{0.4cm}
\noindent\textbf{Keywords:} Audio deepfake. Benchmark. Neural architectures.
WavLM. HuBERT. Conformer. MultiscaleCNN. Noise robustness.
\end{singlespace}
"""
    return replace_between(text, start, end, replacement)


def dataset_block() -> str:
    return r"""
\subsection{Construção do Dataset BRSpeech-DF}
\label{sec:brspeech_df}

O \textit{dataset} experimental utilizado no \textit{benchmark} consolidado
foi estruturado como uma base balanceada de 15.000 amostras de áudio, sendo
7.500 amostras \textit{bonafide} e 7.500 amostras \textit{spoof}. A escolha
desse volume corrige a limitação da rodada exploratória inicial de 600
amostras e permite treinar arquiteturas de maior capacidade, inclusive modelos
\textit{raw-audio} e \textit{self-supervised learning} (SSL).

\begin{itemize}
    \item \textbf{Padronização}: todos os arquivos foram convertidos para
    16\,kHz, mono, janelas de 5\,s e representação PCM normalizada. Para os
    modelos de áudio bruto, a entrada final possui dimensão
    $80.000\times1$; para modelos espectrais, o mesmo áudio é convertido para
    representações Mel, LFCC, CQT ou MFCC conforme a arquitetura.
    \item \textbf{Balanceamento}: as classes real e falsa foram mantidas em
    proporção 1:1 em todas as partições, evitando viés de decisão por
    frequência de classe.
    \item \textbf{Divisão estratificada}: 70\% para treino
    (10.500 amostras), 15\% para validação (2.250 amostras) e 15\% para teste
    (2.250 amostras).
    \item \textbf{Processamento}: o \textit{pipeline} aplica validação de
    áudio, reamostragem, normalização, extração de características e geração
    de arquivos intermediários em formato NumPy para reprodutibilidade.
    \item \textbf{Robustez}: além do teste limpo, foram geradas versões com
    ruído AWGN em SNR 30, 20 e 10\,dB para medir degradação em cenários de
    transmissão e captura degradadas.
\end{itemize}

Os artefatos de entrada do \textit{benchmark} foram consolidados em
\path{app/datasets/benchmark_audio_raw_balanced_15k.npz}. Os resultados,
modelos treinados, métricas, matrizes de confusão e figuras são salvos em
pastas individuais por arquitetura dentro de \path{results/}, permitindo
auditoria e reexecução parcial por modelo.
"""


def architecture_block() -> str:
    return r"""
\section{Arquiteturas de \textit{Deep Learning}}
\label{sec:arquiteturas}

O \textit{pipeline} consolidado contempla quatorze arquiteturas, cobrindo desde
classificadores clássicos sobre características tabulares até modelos neurais
\textit{end-to-end} baseados em forma de onda, grafos, convoluções,
atenção e \textit{self-supervised learning}. Essa ampliação substitui a rodada
exploratória inicial com cinco famílias e permite comparar custo computacional,
estabilidade de treinamento, robustez a ruído e desempenho final.

\subsection{Modelos SSL e \textit{Raw-Audio}}

O WavLM original \cite{chen2022wavlm} e o HuBERT original
\cite{hsu2021hubert} foram avaliados com \textit{backbones} reais, não apenas
com implementações simplificadas. Ambos recebem áudio bruto a 16\,kHz e usam
representações aprendidas em larga escala, com cabeça supervisionada para a
tarefa binária real/\textit{fake}. No \textit{benchmark}, os \textit{backbones}
foram executados em PyTorch/CUDA e persistidos localmente para inferência.

O RawNet2 \cite{tak2021rawnet2} processa diretamente a forma de onda por meio
de filtros SincNet, blocos residuais com \textit{feature map scaling} e GRU,
preservando informação temporal fina sem necessidade de espectrograma. AASIST
\cite{jung2022aasist} e RawGAT-ST exploram atenção em grafos
espectro-temporais para modelar relações locais e globais entre regiões do
sinal. Esses modelos, antes inviáveis na rodada CPU de pequena escala, foram
treinados com 100 épocas no ambiente GPU.

\subsection{Modelos Espectrais Neurais}

Sonic Sleuth combina características acústicas clássicas (LFCC, MFCC e CQT)
com uma rede neural compacta. Conformer \cite{gulati2020conformer} une
convoluções locais e autoatenção para capturar padrões espectro-temporais em
diferentes escalas. O Hybrid CNN-Transformer \cite{kurnaz2024hybrid} utiliza
tokenização convolucional antes dos blocos Transformer, reduzindo o custo em
comparação com atenção aplicada diretamente a mapas espectrais densos.

O Spectrogram Transformer adapta a lógica de Vision Transformer para
espectrogramas \cite{gong2021ast}; no experimento consolidado, apresentou alta
validação intermediária, mas queda severa no final do treinamento, exigindo
uso rigoroso de \textit{checkpoint} e novo ciclo de regularização. O
EfficientNet-LSTM usa transferência de aprendizado convolucional seguida por
Bi-LSTM para resumir dependências temporais. A MultiscaleCNN incorpora blocos
multi-escala inspirados em Res2Net \cite{gao2021res2net}, enquanto o Ensemble
funde quatro ramos de características complementares.

\subsection{Modelos Clássicos}

SVM e Random Forest foram mantidos como baselines fortes e interpretáveis. O
SVM utiliza padronização estatística e kernel RBF; a Random Forest explora
paralelismo em CPU com múltiplas árvores e \texttt{n\_jobs=-1}. Ambos foram
submetidos a busca de hiperparâmetros antes do ajuste final, permitindo
comparação justa com as arquiteturas neurais.
"""


def results_discussion_block(
    results_table: str,
    efficiency_table: str,
    robustness_table: str,
    stability_table: str,
) -> str:
    block = r"""
\section{Experimentos e Resultados}
\label{sec:resultados}

\subsection{Ambiente Experimental e Hardware}

O \textit{benchmark} consolidado foi executado em ambiente Windows 11 com WSL2,
Docker e acesso a GPU NVIDIA via CUDA. A GPU principal foi uma NVIDIA GeForce
RTX 3060, com \textit{memory growth} habilitado, \texttt{cuda\_malloc\_async}
e precisão mista (\texttt{mixed\_float16}) para modelos compatíveis. O
treinamento dos modelos Keras/TensorFlow utilizou aceleração por GPU quando
disponível; os modelos WavLM e HuBERT originais utilizaram PyTorch/CUDA por
serem \textit{backbones} nativos do ecossistema Transformers. SVM e Random
Forest foram executados em CPU com paralelismo e validação cruzada.

\begin{table}[H]
\centering
\caption{Resumo do ambiente de execução do \textit{benchmark}.}
\label{tab:hardware_benchmark}
\begin{tabular}{ll}
\toprule
\textbf{Componente} & \textbf{Configuração} \\
\midrule
Sistema operacional & Windows 11 + WSL2 \\
GPU & NVIDIA GeForce RTX 3060 \\
Aceleração & CUDA, precisão mista e crescimento dinâmico de VRAM \\
Modelos Keras & TensorFlow/Keras com GPU quando disponível \\
Modelos SSL originais & PyTorch/CUDA + \textit{backbone} WavLM/HuBERT real \\
Modelos clássicos & CPU, validação cruzada e paralelismo \\
Dataset & 15.000 amostras balanceadas (7.500 reais + 7.500 falsas) \\
Split & 70/15/15: 10.500 treino, 2.250 validação, 2.250 teste \\
Áudio & 16\,kHz, mono, 5\,s, $80.000\times1$ no caminho \textit{raw-audio} \\
\bottomrule
\end{tabular}
\end{table}

\subsection{Protocolo de Treinamento}

Os modelos neurais foram treinados com o preset de \textit{benchmark} de
100 épocas, salvamento de melhor \textit{checkpoint}, registro de histórico,
matriz de confusão, curvas de acurácia/loss, ROC, Precision-Recall, métricas
por classe, latência e tamanho do modelo. Os classificadores clássicos foram
ajustados com busca de hiperparâmetros antes do treinamento final. Todos os
resultados foram armazenados em pastas próprias por arquitetura, e as figuras
consolidadas foram exportadas para
\path{results/tcc_consolidated/figures/}.

\begin{table}[H]
\centering
\caption{Resultados consolidados no conjunto de teste limpo.}
\label{tab:resultados_consolidados}
\resizebox{\textwidth}{!}{%
\begin{tabular}{lccccccc}
\toprule
\textbf{Modelo} & \textbf{Acurácia} & \textbf{EER} & \textbf{AUC} &
\textbf{F1} & \textbf{Lat. ms} & \textbf{Treino} & \textbf{Status} \\
\midrule
__RESULTS_TABLE__
\bottomrule
\end{tabular}}
\end{table}

\begin{figure}[H]
    \centering
    \includegraphics[width=\textwidth]{results/tcc_consolidated/figures/benchmark_accuracy_auc.png}
    \caption{Acurácia e AUC-ROC consolidadas por arquitetura.}
    \label{fig:benchmark_accuracy_auc}
\end{figure}

\begin{figure}[H]
    \centering
    \includegraphics[width=0.92\textwidth]{results/tcc_consolidated/figures/benchmark_eer.png}
    \caption{Equal Error Rate (EER) por arquitetura. Valores menores indicam melhor separação entre classes.}
    \label{fig:benchmark_eer}
\end{figure}

\subsection{Análise dos Resultados}

Os melhores desempenhos foram obtidos por Conformer e Sonic Sleuth, ambos com
100,00\% de acurácia e EER de 0,00\% no conjunto de teste limpo. Em seguida,
Hybrid CNN-Transformer, MultiscaleCNN e SVM permaneceram acima de 99\% de
acurácia, indicando que tanto arquiteturas espectrais profundas quanto
baselines clássicos bem ajustados extraem sinais discriminativos fortes no
\textit{dataset} definido.

RawNet2, Ensemble, RawGAT-ST, AASIST, HuBERT original e EfficientNet-LSTM
também convergiram adequadamente, com acurácia entre 91,16\% e 96,36\%. Esse
resultado invalida a conclusão preliminar da rodada CPU de 600 amostras, na
qual os modelos de maior complexidade não haviam convergido. Com dataset mais
robusto, GPU e 100 épocas, os modelos \textit{raw-audio} e SSL tornaram-se
funcionais no pipeline principal.

O caso discrepante foi o Spectrogram Transformer: embora tenha alcançado
98,00\% de acurácia de validação na época 11, o resultado final caiu para
71,51\% de acurácia no teste. Isso sugere instabilidade de otimização,
sensibilidade a hiperparâmetros ou necessidade de restaurar obrigatoriamente o
melhor \textit{checkpoint} para inferência. Portanto, o modelo é funcional,
mas requer novo ciclo de ajuste antes de ser tratado como resultado final
competitivo.

\subsection{Eficiência Computacional}

\begin{table}[H]
\centering
\caption{Parâmetros, tamanho, latência e desempenho por arquitetura.}
\label{tab:eficiencia_modelos}
\resizebox{\textwidth}{!}{%
\begin{tabular}{lccccc}
\toprule
\textbf{Modelo} & \textbf{Parâmetros} & \textbf{Tamanho MB} &
\textbf{Lat. ms} & \textbf{Acurácia} & \textbf{EER} \\
\midrule
__EFFICIENCY_TABLE__
\bottomrule
\end{tabular}}
\end{table}

\begin{figure}[H]
    \centering
    \includegraphics[width=0.95\textwidth]{results/tcc_consolidated/figures/benchmark_latency.png}
    \caption{Latência média de inferência por arquitetura.}
    \label{fig:benchmark_latency}
\end{figure}

\begin{figure}[H]
    \centering
    \includegraphics[width=0.95\textwidth]{results/tcc_consolidated/figures/benchmark_size.png}
    \caption{Tamanho dos artefatos treinados por arquitetura.}
    \label{fig:benchmark_size}
\end{figure}

O WavLM e o HuBERT originais apresentam os maiores artefatos, próximos de
361\,MB, por incorporarem \textit{backbones} SSL de aproximadamente 94,5
milhões de parâmetros. Apesar disso, o WavLM apresentou a menor latência média
medida no relatório consolidado (10,18\,ms), enquanto EfficientNet-LSTM e
MultiscaleCNN foram os mais custosos em latência entre os modelos Keras
avaliados. Para demonstrações em tempo real, Sonic Sleuth, Hybrid
CNN-Transformer, SVM e Ensemble oferecem melhor equilíbrio entre desempenho,
tamanho e tempo de resposta.

\subsection{Robustez a Ruído}

\begin{table}[H]
\centering
\caption{Acurácia sob ruído AWGN em diferentes SNRs.}
\label{tab:robustez_awgn}
\resizebox{\textwidth}{!}{%
\begin{tabular}{lcccc}
\toprule
\textbf{Modelo} & \textbf{Limpo} & \textbf{30 dB} & \textbf{20 dB} & \textbf{10 dB} \\
\midrule
__ROBUSTNESS_TABLE__
\bottomrule
\end{tabular}}
\end{table}

\begin{figure}[H]
    \centering
    \includegraphics[width=\textwidth]{results/tcc_consolidated/figures/benchmark_robustness.png}
    \caption{Degradação de acurácia sob ruído AWGN.}
    \label{fig:benchmark_robustness}
\end{figure}

A robustez sob ruído é heterogênea. Modelos que dependem de detalhes
espectrais finos tendem a sofrer queda em SNRs mais baixos, enquanto
arquiteturas com fusão de características ou treinamento mais estável mantêm
separação melhor em 30\,dB. Em 10\,dB, a degradação passa a ser relevante para
a maior parte das arquiteturas, reforçando a necessidade de incluir ruído,
compressão e reverberação no \textit{data augmentation} quando o objetivo for
implantação em campo.

\subsection{Estabilidade do Treinamento}

\begin{table}[H]
\centering
\caption{Estabilidade dos modelos neurais durante as 100 épocas.}
\label{tab:estabilidade_treinamento}
\resizebox{\textwidth}{!}{%
\begin{tabular}{lccccc}
\toprule
\textbf{Modelo} & \textbf{Melhor val.} & \textbf{Época} &
\textbf{Val. final} & \textbf{Queda} & \textbf{Observação} \\
\midrule
__STABILITY_TABLE__
\bottomrule
\end{tabular}}
\end{table}

\begin{figure}[H]
    \centering
    \includegraphics[width=0.95\textwidth]{results/tcc_consolidated/figures/training_stability.png}
    \caption{Melhor validação, validação final e queda de desempenho por arquitetura neural.}
    \label{fig:training_stability}
\end{figure}

A análise de estabilidade mostra que Conformer, Sonic Sleuth, MultiscaleCNN,
Hybrid CNN-Transformer, AASIST, RawGAT-ST, RawNet2, HuBERT e
EfficientNet-LSTM mantiveram diferença controlada entre a melhor validação e o
fim do treinamento. WavLM e Ensemble apresentaram queda moderada, mas ainda
com desempenho final utilizável. O Spectrogram Transformer apresentou queda
acentuada, sendo o principal candidato a novo ciclo de regularização, redução
de taxa de aprendizado, aumento de \textit{dropout}, \textit{weight decay} e
uso estrito do melhor \textit{checkpoint}.

\section{Discussão}
\label{sec:discussao}

O \textit{benchmark} consolidado altera substancialmente a interpretação dos
experimentos. Na fase exploratória, a combinação de CPU, apenas 600 amostras e
20 épocas favorecia arquiteturas menores e levava modelos \textit{raw-audio} a
não convergir. A rodada atual, com 15.000 amostras, GPU, 100 épocas e
hiperparâmetros revisados, mostra que a limitação estava no protocolo
experimental inicial, não necessariamente nas arquiteturas.

Do ponto de vista prático, os resultados sugerem três perfis de uso. Para
máxima acurácia no \textit{dataset} atual, Conformer, Sonic Sleuth, Hybrid
CNN-Transformer e MultiscaleCNN são as opções mais fortes. Para inferência
leve e integração em demonstrações, SVM e Sonic Sleuth são particularmente
atraentes por combinarem alto desempenho com baixa complexidade. Para estudo
de representações modernas e comparação com a literatura, WavLM, HuBERT,
RawNet2, AASIST e RawGAT-ST são essenciais, embora imponham maior custo de
treinamento e dependência de GPU.

Também se observa que alto desempenho no conjunto limpo não elimina a
necessidade de validação externa. O dataset utilizado é balanceado e
controlado, o que favorece separabilidade; portanto, os resultados devem ser
interpretados como avaliação do pipeline e das arquiteturas sob um protocolo
padronizado, não como prova definitiva de generalização para todos os sistemas
TTS/VC, codecs, microfones e idiomas. A etapa seguinte deve incluir
ASVspoof, WaveFake, In-the-Wild e cenários com ruído e compressão desde o
treinamento.

\subsection{Limitações}
\label{sec:limitacoes}

\begin{itemize}
    \item \textbf{Generalização externa}: os resultados são fortes no dataset
    consolidado, mas ainda precisam ser confirmados em bases públicas e
    multi-gerador.
    \item \textbf{Robustez}: a queda sob AWGN em SNRs baixos indica que o
    treinamento deve incorporar ruído, codec e reverberação.
    \item \textbf{Spectrogram Transformer}: o modelo treinou, mas apresentou
    instabilidade severa entre melhor validação e avaliação final.
    \item \textbf{Custo computacional}: WavLM e HuBERT originais exigem maior
    armazenamento e dependem de GPU para treinamento viável.
    \item \textbf{Interpretabilidade}: SHAP e análises locais por segmento
    ainda devem ser integradas ao relatório final automatizado.
\end{itemize}

\subsection{Trabalhos Futuros}
\label{sec:trabalhos_futuros}

\begin{itemize}
    \item Reexecutar o Spectrogram Transformer com \textit{checkpoint}
    obrigatório, \textit{learning rate} menor, \textit{weight decay},
    \textit{dropout} maior e \textit{early stopping}.
    \item Incluir ASVspoof 2019/2021/5, WaveFake e In-the-Wild no preset
    completo de validação externa.
    \item Adicionar \textit{data augmentation} com AWGN, codecs, reverberação
    e variação de ganho no treinamento de todos os modelos.
    \item Consolidar relatório automático em HTML/PDF com métricas, matrizes
    de confusão, curvas ROC/PR, curvas de treinamento e exemplos de inferência.
    \item Integrar SHAP, Grad-CAM espectral ou análise por oclusão temporal
    para explicar decisões dos modelos neurais.
\end{itemize}
"""
    return (
        block.replace("__RESULTS_TABLE__", results_table)
        .replace("__EFFICIENCY_TABLE__", efficiency_table)
        .replace("__ROBUSTNESS_TABLE__", robustness_table)
        .replace("__STABILITY_TABLE__", stability_table)
    )


def conclusion_block() -> str:
    return r"""
\section{Conclusões}
\label{sec:conclusao}

Este trabalho desenvolveu e avaliou um \textit{pipeline} modular de código
aberto para detecção de áudio sintético, cobrindo processamento de dados,
extração de características, treinamento, inferência, \textit{benchmark} e
geração de relatórios. A versão consolidada amplia o escopo para quatorze
arquiteturas e utiliza um dataset balanceado de 15.000 amostras, com treino em
GPU para modelos neurais e otimização por validação cruzada para modelos
clássicos.

Os resultados demonstram que o pipeline está funcional e é capaz de treinar,
avaliar e comparar arquiteturas heterogêneas. Conformer e Sonic Sleuth
atingiram 100,00\% de acurácia e EER de 0,00\%; Hybrid CNN-Transformer,
MultiscaleCNN, SVM e Random Forest também apresentaram desempenho elevado,
acima de 98\% de acurácia. Modelos anteriormente considerados inviáveis na
rodada exploratória, como RawNet2, AASIST e EfficientNet-LSTM, convergiram
adequadamente quando treinados com dataset maior, GPU e 100 épocas.

A principal exceção foi o Spectrogram Transformer, que apresentou forte
degradação entre o melhor ponto de validação e o resultado final. Esse achado
não invalida a arquitetura, mas evidencia a necessidade de restaurar o melhor
\textit{checkpoint} e revisar hiperparâmetros antes de utilizá-la como
resultado competitivo. A análise de robustez sob AWGN também mostra que
desempenho em áudio limpo não é suficiente para implantação em campo.

As principais contribuições consolidadas são:

\begin{enumerate}
    \item Implementação de um pipeline reprodutível para áudio
    \textit{bonafide}/\textit{spoof}, com treino, inferência e benchmark.
    \item Avaliação padronizada de quatorze arquiteturas, incluindo SSL,
    \textit{raw-audio}, grafos, CNNs, Transformers, Ensemble e modelos
    clássicos.
    \item Geração automática de métricas, matrizes de confusão, curvas de
    treinamento, ROC, Precision-Recall, robustez a ruído e relatórios por
    arquitetura.
    \item Consolidação de um conjunto de resultados numéricos e gráficos para
    apoiar a apresentação e discussão do TCC.
    \item Identificação objetiva dos modelos prontos para demonstração e dos
    modelos que exigem novo ciclo de ajuste.
\end{enumerate}
"""


def pipeline_config_block() -> str:
    return r"""
\section{Configurações Detalhadas do Pipeline}

\subsection{Hiperparâmetros Consolidados}

O preset final do \textit{benchmark} utiliza 100 épocas para modelos neurais,
batch adaptado ao consumo de VRAM, precisão mista quando segura e salvamento
do melhor \textit{checkpoint}. Modelos SSL originais utilizam seus
\textit{backbones} reais via PyTorch/CUDA. Os modelos clássicos utilizam busca
de hiperparâmetros com validação cruzada antes do ajuste final.

\begin{table}[H]
\centering
\caption{Resumo dos principais hiperparâmetros por família de modelo.}
\label{tab:hiperparametros_consolidados}
\resizebox{\textwidth}{!}{%
\begin{tabular}{llll}
\toprule
\textbf{Família} & \textbf{Entrada} & \textbf{Treinamento} & \textbf{Observação} \\
\midrule
WavLM/HuBERT & Áudio bruto 16\,kHz & 100 épocas + cabeça supervisionada & Backbone original salvo localmente \\
RawNet2 & PCM $80.000\times1$ & 100 épocas, GPU, GRU + SincNet & Convergente no preset GPU \\
AASIST/RawGAT-ST & Espectro-temporal & 100 épocas, atenção em grafos & Feedback e checkpoints por época \\
Conformer/Hybrid & Espectrograma & 100 épocas, Adam/AdamW & Melhor estabilidade entre Transformers \\
Spectrogram Transformer & Espectrograma & 100 épocas & Requer checkpoint e retuning \\
EfficientNet-LSTM & Espectrograma 2D & 100 épocas, transferência parcial & Boa convergência, maior latência \\
MultiscaleCNN/Ensemble & Features espectrais & 100 épocas & Forte desempenho no dataset limpo \\
SVM & Features tabulares & GridSearchCV + SVC RBF & Modelo leve e alta acurácia \\
Random Forest & Features tabulares & GridSearchCV + \texttt{n\_jobs=-1} & Paralelismo CPU \\
\bottomrule
\end{tabular}}
\end{table}

\subsection{Organização dos Artefatos}

Cada arquitetura possui diretório próprio em \path{results/}, contendo
modelo treinado, configuração, histórico de treinamento, métricas em JSON/CSV,
matriz de confusão, curvas ROC/PR, gráficos de acurácia e perda, resultados de
robustez e relatório Markdown. Os gráficos consolidados utilizados no TCC
ficam em \path{results/tcc_consolidated/figures/}.
"""


def appendix_results_block(
    results_table: str,
    efficiency_table: str,
    robustness_table: str,
    stability_table: str,
) -> str:
    block = r"""
\section{Resultados Completos dos Experimentos}

\subsection{Resumo Consolidado}

Esta seção registra os valores consolidados extraídos automaticamente dos
diretórios de \textit{benchmark}. Os mesmos dados alimentam as figuras da
Seção~\ref{sec:resultados}.

\begin{table}[H]
\centering
\caption{Tabela completa de desempenho por arquitetura.}
\resizebox{\textwidth}{!}{%
\begin{tabular}{lccccccc}
\toprule
\textbf{Modelo} & \textbf{Acurácia} & \textbf{EER} & \textbf{AUC} &
\textbf{F1} & \textbf{Lat. ms} & \textbf{Treino} & \textbf{Status} \\
\midrule
__RESULTS_TABLE__
\bottomrule
\end{tabular}}
\end{table}

\begin{table}[H]
\centering
\caption{Tabela completa de eficiência computacional.}
\resizebox{\textwidth}{!}{%
\begin{tabular}{lccccc}
\toprule
\textbf{Modelo} & \textbf{Parâmetros} & \textbf{Tamanho MB} &
\textbf{Lat. ms} & \textbf{Acurácia} & \textbf{EER} \\
\midrule
__EFFICIENCY_TABLE__
\bottomrule
\end{tabular}}
\end{table}

\begin{table}[H]
\centering
\caption{Tabela completa de robustez sob ruído AWGN.}
\resizebox{\textwidth}{!}{%
\begin{tabular}{lcccc}
\toprule
\textbf{Modelo} & \textbf{Limpo} & \textbf{30 dB} & \textbf{20 dB} & \textbf{10 dB} \\
\midrule
__ROBUSTNESS_TABLE__
\bottomrule
\end{tabular}}
\end{table}

\begin{table}[H]
\centering
\caption{Tabela completa de estabilidade do treinamento neural.}
\resizebox{\textwidth}{!}{%
\begin{tabular}{lccccc}
\toprule
\textbf{Modelo} & \textbf{Melhor val.} & \textbf{Época} &
\textbf{Val. final} & \textbf{Queda} & \textbf{Observação} \\
\midrule
__STABILITY_TABLE__
\bottomrule
\end{tabular}}
\end{table}

\subsection{Figuras Geradas}

\begin{figure}[H]
    \centering
    \includegraphics[width=\textwidth]{results/tcc_consolidated/figures/benchmark_accuracy_auc.png}
    \caption{Acurácia e AUC-ROC consolidadas.}
\end{figure}

\begin{figure}[H]
    \centering
    \includegraphics[width=0.92\textwidth]{results/tcc_consolidated/figures/benchmark_eer.png}
    \caption{EER consolidado.}
\end{figure}

\begin{figure}[H]
    \centering
    \includegraphics[width=0.95\textwidth]{results/tcc_consolidated/figures/benchmark_latency.png}
    \caption{Latência consolidada.}
\end{figure}

\begin{figure}[H]
    \centering
    \includegraphics[width=0.95\textwidth]{results/tcc_consolidated/figures/benchmark_size.png}
    \caption{Tamanho dos modelos consolidados.}
\end{figure}

\begin{figure}[H]
    \centering
    \includegraphics[width=\textwidth]{results/tcc_consolidated/figures/benchmark_robustness.png}
    \caption{Robustez sob ruído AWGN.}
\end{figure}

\begin{figure}[H]
    \centering
    \includegraphics[width=0.95\textwidth]{results/tcc_consolidated/figures/training_stability.png}
    \caption{Estabilidade de treinamento.}
\end{figure}
"""
    return (
        block.replace("__RESULTS_TABLE__", results_table)
        .replace("__EFFICIENCY_TABLE__", efficiency_table)
        .replace("__ROBUSTNESS_TABLE__", robustness_table)
        .replace("__STABILITY_TABLE__", stability_table)
    )


def main() -> None:
    p = argparse.ArgumentParser(
        description="Atualiza o .tex do TCC a partir do summary consolidado"
    )
    p.add_argument("--source", default=str(SOURCE),
                   help="tex base (default: tcc_overleaf/tcc.tex)")
    p.add_argument("--summary", default=str(SUMMARY),
                   help="benchmark_summary.json (consolidate_results.py)")
    p.add_argument("--output", default=str(OUTPUT),
                   help="saída (default: tcc_overleaf/tcc_atualizado.tex)")
    p.add_argument("--in-place", action="store_true",
                   help="(só com --full-rewrite) sobrescreve o --source com backup .bak")
    p.add_argument("--full-rewrite", action="store_true",
                   help="reescreve a TESE inteira injetando prosa+tabelas. FRÁGIL: "
                        "exige que os marcadores de seção batam EXATAMENTE com o "
                        "--source. Por padrão, gera só o fragmento de tabelas.")
    p.add_argument("--figures-dir",
                   default="results/tcc_consolidated/figures",
                   help="caminho das figuras referenciado no fragmento de tabelas")
    args = p.parse_args()

    summary = Path(args.summary)
    if not summary.exists():
        sys.exit(f"ERRO: summary não encontrado: {summary}. "
                 "Rode antes: python scripts/consolidate_results.py <runs...>")
    rows_all = json.loads(summary.read_text(encoding="utf-8"))

    # ---- Modo padrão: fragmento de TABELAS (data-driven, seguro) ----
    if not args.full_rewrite:
        rows = model_rows(rows_all)
        fragment = _tables_fragment(
            results_table=build_results_table(rows),
            efficiency_table=build_efficiency_table(rows),
            robustness_table=build_robustness_table(rows),
            stability_table=build_stability_table(rows),
            figures_dir=args.figures_dir,
        )
        out = Path(args.output)
        if out.name == "tcc_atualizado.tex":  # default → nome de fragmento
            out = out.with_name("tabelas_benchmark.tex")
        out.write_text(fragment, encoding="utf-8")
        print(f"Fragmento de tabelas escrito: {out.resolve()}")
        print("Use no tcc.tex com:  \\input{tabelas_benchmark.tex}")
        return

    # ---- Modo legado: reescrita completa da tese (frágil) ----
    source = Path(args.source)
    output = Path(args.source) if args.in_place else Path(args.output)
    if not source.exists():
        sys.exit(f"ERRO: tex base não encontrado: {source}")
    if args.in_place:
        backup = source.with_suffix(source.suffix + ".bak")
        backup.write_text(source.read_text(encoding="utf-8"), encoding="utf-8")
        print(f"Backup: {backup}")

    text = source.read_text(encoding="utf-8")
    text = text.replace(
        "\\usepackage{microtype}   % melhora justificação e hifenização",
        "\\usepackage{microtype}   % melhora justificação e hifenização\n"
        "\\microtypesetup{expansion=false}",
    )
    text = text.replace(
        "\\pagestyle{fancy}\n\\fancyhf{}",
        "\\pagestyle{fancy}\n\\setlength{\\headheight}{14pt}\n\\fancyhf{}",
    )
    text = text.replace(
        "Prof.\\ [Nome do Membro 1]\\\\\n[Instituição]",
        "Prof.\\ \\textit{[Nome do Membro 1]}\\\\\n\\textit{[Instituição]}",
    )
    text = text.replace(
        "Prof.\\ [Nome do Membro 2]\\\\\n[Instituição]",
        "Prof.\\ \\textit{[Nome do Membro 2]}\\\\\n\\textit{[Instituição]}",
    )
    rows = model_rows(json.loads(summary.read_text(encoding="utf-8")))

    results_table = build_results_table(rows)
    efficiency_table = build_efficiency_table(rows)
    robustness_table = build_robustness_table(rows)
    stability_table = build_stability_table(rows)

    text = replace_abstract(text)
    text = text.replace(
        "    \\item Implementar e comparar cinco arquiteturas neurais especializadas\n"
        "    sobre um \\textit{dataset} brasileiro balanceado.",
        "    \\item Implementar e comparar quatorze arquiteturas especializadas\n"
        "    sobre um \\textit{dataset} balanceado de 15.000 amostras.",
    )
    text = replace_between(
        text,
        r"\subsection{Construção do Dataset BRSpeech-DF}",
        r"\section{Arquiteturas de \textit{Deep Learning}}",
        dataset_block(),
    )
    text = replace_between(
        text,
        r"\section{Arquiteturas de \textit{Deep Learning}}",
        r"\section{Componentes de Software Livre}",
        architecture_block(),
    )
    text = replace_between(
        text,
        r"\section{Experimentos e Resultados}",
        r"\section{Conclusões}",
        results_discussion_block(
            results_table, efficiency_table, robustness_table, stability_table
        ),
    )
    text = replace_between(
        text,
        r"\section{Conclusões}",
        "% =====================================================================\n%  REFERÊNCIAS",
        conclusion_block(),
    )
    text = replace_between(
        text,
        r"\section{Configurações Detalhadas do Pipeline}",
        r"\section{Implementação de Referência",
        pipeline_config_block(),
    )
    text = replace_between(
        text,
        r"\section{Resultados Completos dos Experimentos}",
        r"\end{document}",
        appendix_results_block(
            results_table, efficiency_table, robustness_table, stability_table
        ),
    )

    output.write_text(text, encoding="utf-8")
    print(f"Wrote {output.resolve()}")


if __name__ == "__main__":
    main()
