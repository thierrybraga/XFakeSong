# 15 — Sistema de Benchmark e Teste (dados do TCC)

O pacote `benchmarks/` gera, de forma **reprodutível** e usando
o **pipeline real** (`TrainingService → ModelLoader → Predictor →
MetricsCalculator`) e a **API** (FastAPI `TestClient`), os dados empíricos do
trabalho: desempenho por arquitetura, eficiência computacional, robustez a ruído
e teste de sistema da API.

> Importante: este harness usa o pipeline **já corrigido** (treino→salvar→
> carregar→prever funcional). Os números aqui substituem com fidelidade os
> medidos manualmente — incluindo os modelos `raw-audio` e os baselines
> clássicos (SVM/RF) que faltavam.

## Como rodar

```bash
# 1) Verificação do harness (sintético, 1 época) — segundos:
python scripts/run_benchmark.py --quick

# 2) Pipeline completo do TCC: download, processamento, split, treino,
#    inferência, gráficos PNG e relatórios Markdown:
python scripts/run_tcc_pipeline.py \
    --tcc-full-dataset \
    --out results/tcc_full_20k \
    --npz app/datasets/benchmark_audio_raw_20k.npz

# 3) Execução do TCC direto no benchmark, usando dataset real .npz já exportado:
python scripts/run_benchmark.py --full --dataset app/datasets/brspeech_df.npz

# 4) Sob medida:
python scripts/run_benchmark.py \
    --archs MultiscaleCNN Ensemble EfficientNet-LSTM AASIST RawNet2 SVM RandomForest \
    --dataset data.npz --epochs 20 --snr 30 20 10 --api --out results/bench_tcc
```

O preset `--tcc-full-dataset` configura o alvo oficial do experimento:
`10.000` amostras reais + `10.000` amostras fake, com composição balanceada
entre BRSpeech-DF, CommonVoice/FLEURS PT-BR e Fake Voices XTTS. Ele também
ativa o benchmark completo e o probe da API. Use `--skip-download` quando os
WAVs e splits já estiverem prontos localmente.

O `.npz` deve conter `X_train`/`y_train` (e opcionalmente `X_val`/`X_test`); o
harness **reconcatena e re-divide 70/15/15 estratificado** com semente fixa,
garantindo um conjunto de teste *held-out* controlado. Sem `--dataset`, usa um
dataset sintético separável (apenas para validar o harness).

## O que é medido

| Dimensão | Métricas |
|---|---|
| Desempenho (teste limpo) | acurácia, precisão, recall, F1, **EER**, **AUC-ROC**, **min-tDCF** |
| Eficiência | nº de parâmetros, tamanho em disco (MB), **latência** (ms/amostra, mediana) |
| Robustez | acurácia/EER/AUC sob **AWGN** em cada SNR (`--snr`) |
| Convergência | flag por arquitetura (AUC ≥ limiar) + curva de validação |
| API (`--api`) | status + latência por endpoint (lê a superfície OpenAPI real) |

O ruído AWGN é aplicado no **espaço de entrada** do modelo (forma de onda para
raw-audio; espectrograma para os demais) — escolha deliberada para um teste
uniforme e reprodutível em todas as arquiteturas, documentada no relatório.

## Saídas → mapeamento para as tabelas/figuras do TCC

Tudo é gravado em `--out` (default `results/benchmark/`):

| Arquivo | Uso no TCC |
|---|---|
| `tables/tab_resultados.tex` | **Tabela "Desempenho das arquiteturas"** (acur/EER/AUC/min-tDCF/lat/conv) |
| `tables/tab_eficiencia.tex` | **Tabela "Eficiência computacional"** (params/MB/latência) |
| `tables/tab_robustez.tex` | **Tabela "Robustez sob ruído AWGN"** (acur/EER por SNR) |
| `dataset.md` / `dataset_manifest.json` | Composição, origem, split, processamento e hiperparâmetros globais do dataset |
| `tcc_report.md` | Relatório Markdown com dataset, hiperparâmetros, métricas, inferências e imagens PNG |
| `figures/roc.png` | Curvas ROC (visualiza a AUC) |
| `figures/confusion_matrices.png` | Matrizes de confusão agregadas |
| `figures/score_distributions.png` | Distribuição dos scores por classe |
| `figures/robustez.png` | Acurácia × SNR (degradação sob ruído) |
| `figures/eficiencia.png` | Latência × acurácia (verde=convergiu) |
| `figures/convergencia.png` | Curvas de acurácia de validação por época |
| `architectures/<modelo>/*.png` | Matrizes de confusão, ROC, scores e convergência por arquitetura |
| `results.csv` / `results.json` | Dados brutos (reprodutibilidade / anexos) |
| `summary.md` | Resumo legível (ambiente, dataset, tabela-resumo, API) |

As tabelas `.tex` usam `\singlespacing`, decimais com vírgula e as cores
`successgreen`/`dangerred` — **basta `\input{}`** no documento (o preâmbulo do
TCC já define esses pacotes/cores).

## Reprodutibilidade

`results.json` registra o **ambiente** (SO, Python, TensorFlow, GPU/CPU,
dispositivo), a **configuração** completa (sementes, épocas, SNRs) e o
**balanceamento** do conjunto de teste — anexe-o para garantir reprodutibilidade.
