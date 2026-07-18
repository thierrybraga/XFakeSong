# Estudo Experimental de Detecção de Áudio Sintético

Esta página organiza, em formato de documentação navegável, a fundamentação
técnica e a análise experimental do XFakeSong. A fonte LaTeX em
`tcc_overleaf/main.tex` continua sendo um artefato acadêmico separado; esta
versão existe para consulta técnica no GitHub Pages.

## Resumo técnico

O XFakeSong é um pipeline modular e de código aberto para detecção de áudio
sintético. A metodologia integra pré-processamento, extração de características
acústicas, treinamento supervisionado, inferência e geração automática de
relatórios de benchmark.

O harness de benchmark suporta **14 arquiteturas**; o recorte oficial
consolidado no artigo (`tcc_overleaf/main.tex`) usa **11** delas, sobre o
tier `medium` canônico de 15k, com **15.000 amostras alvo** de áudio,
divididas em 70/15/15 para treino, validação e teste. As janelas foram
padronizadas em **16 kHz**, mono e **5 s**. A base ativa consolidada contém
15.000 WAVs PCM lineares, 16 bits, mono, 16 kHz, somando **2.045,61 min** de
áudio validado; o `.npz` canônico tem **2.769,01 MiB** e representa
**1.250,00 min** após o corte/padding de 5 s por amostra. Modelos neurais
foram treinados em GPU NVIDIA RTX 3060 via WSL2/CUDA; SVM e Random Forest
foram otimizados por validação cruzada em CPU.

Principais resultados no conjunto de teste limpo (recorte oficial dos 11
modelos, atualizado em 2026-07-02 — fonte de verdade em
`tcc_overleaf/tabelas_benchmark.tex`, não editar esta tabela à mão):

| Modelo | Accuracy | EER | AUC-ROC | Acc.\ @10dB |
|---|---:|---:|---:|---:|
| Res2Net | 99,69% | 0,44% | 1,000 | 96,18% |
| Conformer | 99,69% | 0,27% | 1,000 | 98,44% |
| AST | 98,71% | 1,33% | 0,995 | 97,38% |
| Random Forest | 98,18% | 1,69% | 0,998 | 68,04% |
| RawNet2 | 97,38% | 2,89% | 0,998 | 90,80% |
| SVM | 96,00% | 4,31% | 0,991 | 66,44% |
| CCT | 96,04% | 3,91% | 0,991 | 81,20% |
| AASIST | 92,49% | 7,42% | 0,926 | 88,93% |
| HuBERT Original | 88,76% | 11,29% | 0,963 | 80,98% |
| RawGAT-ST | 86,98% | 12,80% | 0,951 | 82,93% |
| WavLM Original | 84,67% | 15,24% | 0,930 | 75,91% |

Sonic Sleuth, Ensemble e EfficientNet-LSTM são suportados pelo harness mas
não integram o recorte oficial (ver `docs/15_BENCHMARK.md` e
`docs/RETREINO_AJUSTES.md`).

## Objetivos

### Objetivo geral

Desenvolver um pipeline modular, reprodutível e extensível para detecção de
áudio sintético, capaz de comparar arquiteturas clássicas e neurais sob o mesmo
protocolo experimental.

### Objetivos específicos

- Implementar e comparar quatorze arquiteturas especializadas.
- Padronizar pré-processamento, extração de características, treinamento e
  inferência.
- Avaliar robustez sob ruído AWGN em múltiplos níveis de SNR.
- Gerar relatórios, gráficos, matrizes de confusão, métricas e artefatos por
  arquitetura.
- Consolidar um conjunto de resultados numéricos para o TCC e para a
  demonstração da interface Gradio/API.

## Fundamentação: síntese e detecção

### Clonagem de voz

A clonagem de voz sintetiza uma nova fala preservando características do
falante. No artigo, esse processo é descrito por:

$$
\mathbf{y}_{clone} = f_{\theta}(\mathbf{x}_{text}, \mathbf{s}_{ref})
$$

em que \(\mathbf{y}_{clone}\) é o áudio sintético gerado,
\(\mathbf{x}_{text}\) é o texto de entrada, \(\mathbf{s}_{ref}\) representa a
referência vocal e \(f_{\theta}\) é o modelo neural treinado.

### Conversão de voz

A conversão de voz transforma a identidade vocal de um áudio de entrada sem
necessariamente alterar o conteúdo linguístico:

$$
\mathbf{y}_{vc} = g_{\phi}(\mathbf{x}_{source}, \mathbf{s}_{target})
$$

em que \(\mathbf{x}_{source}\) é a fala original e
\(\mathbf{s}_{target}\) representa a identidade vocal alvo.

### Síntese de fala

Sistemas TTS modernos mapeiam texto para representações acústicas e, depois,
para forma de onda:

$$
\mathbf{s}_{mel} = h_{\psi}(\mathbf{t}_{text}), \qquad
\mathbf{y}_{tts} = v_{\omega}(\mathbf{s}_{mel})
$$

\(\mathbf{t}_{text}\) é o texto, \(\mathbf{s}_{mel}\) é o espectrograma mel,
\(h_{\psi}\) é o modelo acústico e \(v_{\omega}\) é o vocoder.

### Manipulação em tempo real

Em cenários de baixa latência, a transformação pode ser vista como:

$$
\mathbf{y}_{rt}[n] = r_{\eta}(\mathbf{x}[n-L:n])
$$

em que \(L\) é a janela de contexto e \(r_{\eta}\) é o modelo de conversão em
tempo real.

## Gerações de métodos de detecção

| Geração | Representação | Classificador | Pontos fortes | Limitações |
|---|---|---|---|---|
| 1ª, até 2015 | MFCC, LFCC, CQCC | GMM-UBM, SVM | Baixo custo | Baixa generalização |
| 2ª, 2019 | Espectrograma mel | CNN, LSTM, LCNN | Artefatos de vocoder | Sensível a geradores não vistos |
| 3ª, 2021 | Forma de onda bruta, grafos | RawNet2, AASIST | EER baixo em bases controladas | GPU e dados suficientes |
| 4ª, 2024+ | Representação SSL | WavLM, HuBERT + MLP | Melhor generalização externa | Alto custo computacional |

## Fluxograma do pipeline

```mermaid
flowchart LR
    subgraph P["Predição"]
        A["Áudio de entrada"] --> B["VAD + AGC"]
        B --> C["Extração de características"]
        C --> D["Normalização"]
        D --> E["Inferência"]
        E --> F["Score 0-1"]
        F --> G{"score > 0,5?"}
        G -->|"Sim"| H["REAL"]
        G -->|"Não"| I["FAKE"]
    end

    subgraph T["Treinamento"]
        J["Dataset real/fake"] --> K["Pré-processamento"]
        K --> L["Extração de características"]
        L --> M["Arquitetura neural/clássica"]
        M --> N["Treinamento"]
        N --> O{"Convergiu?"}
        O -->|"Sim"| Q["Modelo salvo"]
        O -->|"Não"| P2["Ajustar hiperparâmetros"]
        P2 --> N
    end
```

## Pré-processamento

### Normalização AGC

O controle automático de ganho ajusta o nível médio de loudness:

$$
x_{norm}[n] = x[n] \cdot 10^{\frac{L_{target} - L_{measured}}{20}}
$$

No experimento, \(L_{target} = -23\,\mathrm{LUFS}\).

### Detecção de atividade de voz

A energia por quadro é:

$$
E_m = \sum_{n=mH}^{mH+N-1} x^2[n] w[n-mH]
$$

O quadro é classificado como voz quando:

$$
V_m =
\begin{cases}
1, & 10\log_{10}(E_m) > \theta_{VAD}\\
0, & \text{caso contrário}
\end{cases}
$$

em que \(H\) é o passo de avanço, \(N\) é o tamanho da janela,
\(w[n]\) é a janela de análise e \(\theta_{VAD}\) é o limiar.

## Equações de extração de características

### Centroide espectral

$$
C_t = \frac{\sum_{k=0}^{K-1} f_k |X_t[k]|}{\sum_{k=0}^{K-1} |X_t[k]|}
$$

### Largura de banda espectral

$$
B_t = \left(
\frac{\sum_k (f_k - C_t)^2 |X_t[k]|}{\sum_k |X_t[k]|}
\right)^{1/2}
$$

### Roll-off espectral

$$
\sum_{k=0}^{k_r} |X_t[k]| = \alpha \sum_{k=0}^{K-1} |X_t[k]|
$$

com \(\alpha = 0{,}85\).

### Zero Crossing Rate

$$
ZCR_t = \frac{1}{2N}\sum_{n=1}^{N-1}
\left|\operatorname{sgn}(x_t[n]) - \operatorname{sgn}(x_t[n-1])\right|
$$

### Flatness espectral

$$
SF_t =
\frac{\left(\prod_{k=0}^{K-1}|X_t[k]|\right)^{1/K}}
{\frac{1}{K}\sum_{k=0}^{K-1}|X_t[k]|}
$$

### Contraste espectral

$$
SC_b = 10\log_{10}\left(\frac{\mu_{peaks,b}}{\mu_{valleys,b}}\right)
$$

### MFCC

$$
c_n = \sum_{m=0}^{M-1} \log(S_m)
\cos\left[\frac{\pi n}{M}\left(m+\frac{1}{2}\right)\right]
$$

### Espectrograma mel

$$
M[m,t] = \sum_{k=0}^{K-1} |X_t[k]|^2 H_m[k]
$$

### Características prosódicas

Frequência fundamental média:

$$
F0_{mean} = \frac{1}{T}\sum_{t=1}^{T} F0_t
$$

Jitter:

$$
Jitter = \frac{1}{N-1}\sum_{i=1}^{N-1}
\frac{|T_i - T_{i+1}|}{\bar{T}}
$$

Shimmer:

$$
Shimmer = \frac{1}{N-1}\sum_{i=1}^{N-1}
\frac{|A_i - A_{i+1}|}{\bar{A}}
$$

### Constant-Q Transform

$$
Q = \frac{f_k}{\Delta f_k}
$$

### Delta e delta-delta

$$
d_t =
\frac{\sum_{n=1}^{N} n(c_{t+n} - c_{t-n})}
{2\sum_{n=1}^{N} n^2}
$$

### Normalização estatística

Min-Max:

$$
X_{norm} = \frac{X - X_{min}}{X_{max} - X_{min}}
$$

Z-score:

$$
X_{norm} = \frac{X - \mu}{\sigma}
$$

## Ranking de características

| Grupo | Dimensão | Custo | Validação na literatura |
|---|---:|---|---|
| LFCC/CQCC | 20-84 | Médio | Muito usado em ASVspoof e front-ends clássicos |
| Mel espectrograma | 80-128 | Médio | Forte com CNN/Transformer |
| CQT | 84 | Alto | Boa resolução logarítmica |
| MFCC | 13-40 | Baixo | Baseline clássico |
| Prosódicas | 6 | Médio | Complementares, fracas isoladamente |

## Dataset consolidado

O benchmark utiliza `data/datasets/benchmark_audio_raw_balanced_15k.npz`.

| Atributo | Valor |
|---|---:|
| Total de amostras | 15.000 |
| Bonafide/real | 7.500 |
| Spoof/fake | 7.500 |
| Treino | 10.500 |
| Validação | 2.250 |
| Teste | 2.250 |
| Taxa de amostragem | 16 kHz |
| Duração | 5 s |
| Formato bruto | PCM mono normalizado, \(80.000 \times 1\) |
| Semente | 42 |
| Ruído | AWGN em 30, 20 e 10 dB |

Limitações atuais do manifesto: metadados completos de falante, gênero/idioma,
licença, duração original e duplicatas perceptuais ainda devem ser consolidados
para validação externa mais rigorosa.

## Arquiteturas avaliadas

| Família | Modelos | Entrada |
|---|---|---|
| SSL e áudio bruto | WavLM, HuBERT, RawNet2 | PCM 16 kHz |
| Grafos | AASIST, RawGAT-ST | Espectro-temporal / waveform |
| Espectrograma + atenção | Conformer, Hybrid CNN-Transformer, Spectrogram Transformer | Log-mel/espectrograma |
| CNN e fusão | Sonic Sleuth, EfficientNet-LSTM, MultiscaleCNN, Ensemble | LFCC/MFCC/CQT/log-mel |
| Clássicos | SVM, Random Forest | Características tabulares |

### Observações por modelo

- **WavLM e HuBERT**: usam backbones SSL reais quando disponíveis; preservam
  artefatos completos em `app/models/benchmark_final/<modelo>/`.
- **RawNet2**: processa forma de onda com filtros SincNet, blocos residuais e
  GRU.
- **AASIST e RawGAT-ST**: modelam dependências espectro-temporais com atenção em
  grafos.
- **Conformer**: combina convolução local e atenção global; foi o melhor modelo
  para demonstração principal.
- **Sonic Sleuth**: modelo leve baseado em características acústicas clássicas,
  com desempenho máximo no conjunto limpo.
- **Spectrogram Transformer**: treinou, mas apresentou instabilidade entre a
  melhor validação e o artefato final.
- **SVM e Random Forest**: baselines clássicos com GridSearchCV e paralelismo em
  CPU.

## Treinamento

### Função de perda

Para classificação binária:

$$
\mathcal{L}_{BCE} =
-\frac{1}{N}\sum_{i=1}^{N}
\left[y_i\log(\hat{y}_i) + (1-y_i)\log(1-\hat{y}_i)\right]
$$

### Regularização

O treinamento usa combinações de dropout, penalização L2, salvamento do melhor
checkpoint, redução de taxa de aprendizado e parada antecipada quando aplicável.

### Otimizador

Adam combina médias móveis de gradientes e gradientes ao quadrado:

$$
m_t = \beta_1 m_{t-1} + (1-\beta_1)g_t
$$

$$
v_t = \beta_2 v_{t-1} + (1-\beta_2)g_t^2
$$

$$
\theta_t = \theta_{t-1} -
\alpha \frac{\hat{m}_t}{\sqrt{\hat{v}_t}+\epsilon}
$$

## Ambiente experimental

| Item | Valor |
|---|---|
| Sistema | Windows 11 + WSL2 |
| GPU | NVIDIA GeForce RTX 3060 |
| CUDA | Usado no WSL2/Linux para modelos neurais |
| CPU | Usada para SVM/Random Forest e validação cruzada |
| Frameworks | TensorFlow/Keras, PyTorch, scikit-learn |
| Épocas neurais | até 120 (parada antecipada quando aplicável) |
| Métricas | Accuracy, F1, AUC-ROC, EER, latência, robustez |

## Robustez a ruído

O benchmark aplica AWGN no espaço de entrada do modelo, mantendo o mesmo
protocolo para arquiteturas de áudio bruto, espectrograma e features
tabulares. Recorte oficial dos 11 modelos, atualizado em 2026-07-02 (fonte de
verdade: `tcc_overleaf/tabelas_benchmark.tex`, `Tabela~\ref{tab:robustez_awgn}`):

| Modelo | Limpo | 30 dB | 20 dB | 10 dB |
|---|---:|---:|---:|---:|
| Conformer | 99,69% | 99,73% | 99,69% | 98,44% |
| AST | 98,71% | 98,76% | 98,67% | 97,38% |
| Res2Net | 99,69% | 99,64% | 99,60% | 96,18% |
| RawNet2 | 97,38% | 96,84% | 94,40% | 90,80% |
| AASIST | 92,49% | 92,76% | 92,67% | 88,93% |
| RawGAT-ST | 86,98% | 86,89% | 86,62% | 82,93% |
| CCT | 96,04% | 95,64% | 93,60% | 81,20% |
| HuBERT Original | 88,76% | 87,07% | 85,47% | 80,98% |
| WavLM Original | 84,67% | 81,33% | 79,47% | 75,91% |
| Random Forest | 98,18% | 93,51% | 84,53% | 68,04% |
| SVM | 96,00% | 95,07% | 87,42% | 66,44% |

SVM e Random Forest são hoje os modelos menos robustos a 10 dB do recorte,
apesar de figurarem entre os melhores no conjunto limpo — a robustez sob
ruído depende mais de o treinamento incluir exemplos ruidosos compatíveis
com o teste do que da família arquitetural em si (ver
`docs/RETREINO_AJUSTES.md`).

## Estabilidade de treinamento

Recorte oficial, atualizado em 2026-07-02 (fonte de verdade:
`Tabela~\ref{tab:estabilidade_treinamento}` em `tabelas_benchmark.tex`):

| Modelo | Melhor validação | Época | Validação final | Queda | Status |
|---|---:|---:|---:|---:|---|
| Conformer | 100,00% | 17 | 99,64% | 0,36% | Estável |
| Res2Net | 99,91% | 40 | 99,73% | 0,18% | Estável |
| AST | 99,16% | 29 | 99,07% | 0,09% | Estável |
| RawNet2 | 98,18% | 62 | 97,56% | 0,62% | Estável |
| AASIST | 92,76% | 104 | 92,40% | 0,36% | Estável |
| RawGAT-ST | 89,16% | 21 | 87,29% | 1,87% | Convergência precoce |
| HuBERT Original | 84,92% | 17 | 84,24% | 0,68% | Estável |
| CCT | 95,91% | 30 | 92,13% | 3,78% | Flutuação moderada |
| WavLM Original | 79,72% | 10 | 78,99% | 0,73% | Estável |

Random Forest e SVM não têm trajetória por época (ajuste via
`GridSearchCV` + validação cruzada, não aplicável).

## Discussão

Os resultados sugerem três perfis de uso:

- **Máxima acurácia e robustez no conjunto atual**: Conformer, Res2Net e AST.
- **Inferência leve e demonstração**: SVM e RandomForest (atenção: robustez a
  ruído baixa, ver tabela acima).
- **Pesquisa e comparação com literatura moderna**: WavLM Original, HuBERT
  Original, RawNet2, AASIST e RawGAT-ST.

Alto desempenho no conjunto limpo não elimina a necessidade de validação
externa. A base é balanceada e controlada, o que favorece separabilidade. Os
resultados devem ser lidos como avaliação padronizada do pipeline e das
arquiteturas, não como prova definitiva de generalização para todos os vocoders,
idiomas, microfones e codecs.

## Ameaças à validade e limitações

- **Validade interna**: risco de vazamento se amostras do mesmo falante, canal,
  música ou gerador aparecerem em partições distintas.
- **Validade externa**: resultados precisam ser confirmados em ASVspoof,
  WaveFake, In-the-Wild e bases multi-gerador.
- **Metadados da base**: falantes, licenças, idiomas, duração original e fontes
  devem ser enriquecidos no manifesto.
- **Validade de construção**: acurácia, F1, AUC e EER não substituem análise de
  limiar operacional.
- **Robustez**: quedas sob AWGN indicam necessidade de treino com ruído,
  codecs e reverberação.
- **Custo computacional**: WavLM e HuBERT originais exigem armazenamento e GPU
  para treinamento viável.
- **Interpretabilidade**: SHAP, Grad-CAM espectral e oclusão temporal ainda
  devem ser integrados ao relatório automatizado.

## Considerações éticas e LGPD

Detectores de voz sintética devem ser usados como ferramentas de apoio, não como
mecanismos definitivos de acusação ou autenticação. Falsos positivos podem
prejudicar usuários legítimos; falsos negativos podem permitir aceitação de
áudios sintéticos como reais. Decisões sensíveis exigem revisão humana,
calibração de limiar, governança, controle de acesso e descarte seguro de
arquivos de entrada.

## Trabalhos futuros

- Reexecutar Spectrogram Transformer com checkpoint obrigatório, menor taxa de
  aprendizado, maior dropout, weight decay e early stopping.
- Incluir ASVspoof 2019/2021/5, WaveFake e In-the-Wild no preset de validação
  externa.
- Implementar validação cross-dataset e ablações por família de características.
- Registrar intervalos de confiança, testes pareados e curvas de calibração por
  arquitetura.
- Adicionar aumento de dados com AWGN, codecs, reverberação e variação de ganho.
- Integrar SHAP, Grad-CAM espectral ou oclusão temporal.

## Artefatos e rastreabilidade

| Artefato | Caminho | Finalidade |
|---|---|---|
| Fonte do artigo | `tcc_overleaf/main.tex` | Fonte única para Overleaf |
| Figuras finais | `tcc_overleaf/figures/*.png` | Gráficos usados no artigo |
| Dataset consolidado | `data/datasets/benchmark_audio_raw_balanced_15k.npz` | Entrada única do benchmark |
| Modelos padrão | `app/models/bench_*` | Inferência na Gradio/API |
| Modelos completos | `app/models/benchmark_final/<modelo>/` | Artefatos finais por arquitetura |
| Métricas | `results/<run>/architectures/<modelo>/metrics.json` | Auditoria por modelo |
| Relatório | `results/<run>/tcc_report.md` | Registro textual do benchmark |

## Comandos de reprodução

```bash
python main.py --bootstrap-dirs
python main.py --gradio
```

```bash
python scripts/benchmark/run_tcc_pipeline.py \
  --download \
  --target-per-class 7500 \
  --full-benchmark \
  --epochs 100 \
  --device-profile gpu \
  --npz data/datasets/benchmark_audio_raw_balanced_15k.npz
```

```bash
python scripts/benchmark/run_benchmark.py \
  --full \
  --epochs 100 \
  --device-profile gpu \
  --dataset data/datasets/benchmark_audio_raw_balanced_15k.npz
```

```bash
python scripts/benchmark/run_benchmark.py \
  --model Conformer \
  --epochs 100 \
  --device-profile gpu \
  --dataset data/datasets/benchmark_audio_raw_balanced_15k.npz
```

## Matrizes de confusão completas

As matrizes completas ficam em:

```text
tcc_overleaf/figures/confusion_matrices/
```

Também são geradas por arquitetura dentro de:

```text
results/<run>/architectures/<modelo>/
```
