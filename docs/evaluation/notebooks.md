# 16 — Guia de Notebooks

> **Estado de alinhamento:** notebooks de pipeline por tier e janela de 5 s
> são históricos/genéricos e não reproduzem o artigo. Para a bateria canônica,
> siga `benchmarks/run_models_sequential.py` e o
> [inventário do estudo](canonical-study-and-legacy-inventory.md). Use os
> notebooks como material didático ou de análise auxiliar; confira a célula e
> o dataset configurado antes de executar.

Os notebooks cobrem estudo, treinamento, inferência, análise de features e
fluxos históricos de benchmark. A organização é útil para consulta e estudo por
arquitetura; as células de execução precisam ser revisadas contra o protocolo
canônico antes de serem usadas para gerar resultados do TCC.

!!! tip "Rodar no Google Colab (com GPU)"
    Todos os notebooks são **auto-suficientes no Colab**: a primeira célula
    detecta o ambiente, clona o repositório e instala as dependências. Abra por
    `colab.research.google.com/github/thierrybraga/SpeechChecker/blob/main/docs/notebooks/<caminho>`,
    selecione um runtime com **GPU** e rode a 1ª célula. Os notebooks de modelo
    **treinam por padrão** (defina `XFAKE_RUN_EVAL=0` para pular). Passo a passo
    no [Guia Google Colab](../getting-started/google-colab.md).

## Catálogo, requisitos e resultados esperados

Use os cadernos como material didático. Resultados de dados sintéticos e
pipelines históricos **não são resultados confirmatórios do artigo**.

| Caderno | Estado e entrada | Requisitos principais | Resultado esperado |
|---|---|---|---|
| `00_index.ipynb` | Índice; não consome dataset | Jupyter/IPython | Links e orientação para os demais cadernos |
| `features/01_feature_extraction_study.ipynb` | Didático; sinal sintético de 1 s | TensorFlow, NumPy e bibliotecas de áudio; GPU não necessária | Visualizações e vetores LFCC, log-mel, raw-audio e features clássicas |
| `pipeline/01_benchmark_tcc_full_pipeline.ipynb` | Histórico/genérico; sintético para smoke ou dataset por tier no fluxo completo | Stack de benchmark; GPU, armazenamento e downloads para execução completa | Relatório, métricas, tabelas e figuras do runner; não reproduz o artigo |
| `pipeline/02_training_model.ipynb` | Tutorial ativo; NPZ sintético por padrão; dataset real é opcional | TensorFlow/Keras; GPU opcional | Modelo de demonstração e configuração de inferência em diretório temporário |
| `pipeline/03_inference.ipynb` | Tutorial ativo; treina modelo de demonstração com NPZ sintético | TensorFlow/Keras; GPU opcional | Predição via `Predictor` e `DetectionService` sobre amostra de áudio |
| `pipeline/04_all_architectures_full_benchmark.ipynb` | Histórico/genérico; dataset por tier, não o NPZ canônico | GPU, storage persistente, espaço em disco e acesso aos datasets | Métricas, predições, gráficos e modelos; execução pesada desativada por padrão |
| `models/*.ipynb` (14 cadernos) | Didáticos; dados sintéticos para inspeção/smoke | Dependências da família do modelo; GPU recomendada para redes neurais; WavLM/HuBERT podem baixar backbones | Construção/avaliação de demonstração e métricas sintéticas, sem validade científica |

Os cadernos de modelos treinam a demonstração por padrão. Defina
`XFAKE_RUN_EVAL=0` para abrir e inspecionar sem iniciar esse fluxo. Os cadernos
01 e 04 de pipeline ficam identificados como **históricos** porque usam o
runner genérico e tiers; a reprodução do artigo usa somente o runner e os
artefatos definidos no [inventário canônico](canonical-study-and-legacy-inventory.md).

Os arquivos `.ipynb` permanecem disponíveis para download. O `README.md` da
pasta de notebooks orienta quem navega pelo repositório; esta página é o guia
completo publicado no site.

## Estrutura

```text
notebooks/
├── 00_index.ipynb
├── README.md
├── features/
│   └── 01_feature_extraction_study.ipynb
├── pipeline/
│   ├── 01_benchmark_tcc_full_pipeline.ipynb
│   ├── 02_training_model.ipynb
│   ├── 03_inference.ipynb
│   └── 04_all_architectures_full_benchmark.ipynb
├── models/
│   ├── 01_wavlm.ipynb
│   ├── 02_hubert.ipynb
│   ├── 03_rawnet2.ipynb
│   ├── 04_sonic_sleuth.ipynb
│   ├── 05_aasist.ipynb
│   ├── 06_rawgat_st.ipynb
│   ├── 07_conformer.ipynb
│   ├── 08_hybrid_cnn_transformer.ipynb
│   ├── 09_spectrogram_transformer.ipynb
│   ├── 10_efficientnet_lstm.ipynb
│   ├── 11_multiscale_cnn.ipynb
│   ├── 12_ensemble.ipynb
│   ├── 13_svm.ipynb
│   └── 14_random_forest.ipynb
```

## Ordem Recomendada

1. `notebooks/00_index.ipynb`: mapa executável dos notebooks.
2. `notebooks/features/01_feature_extraction_study.ipynb`: estudo visual e
   numérico das features.
3. `notebooks/pipeline/01_benchmark_tcc_full_pipeline.ipynb`: pipeline histórico
   genérico por tiers; não é a execução oficial do artigo.
4. `notebooks/pipeline/02_training_model.ipynb`: treino prático de um modelo.
5. `notebooks/pipeline/03_inference.ipynb`: leitura de predições e inferência.
6. `notebooks/pipeline/04_all_architectures_full_benchmark.ipynb`: execução
   operacional de todas as arquiteturas, com storage, download robusto, treino,
   auditoria de gráficos/relatórios e modelos salvos.
7. `notebooks/models/*.ipynb`: estudo individual por arquitetura.

## Notebook histórico de Benchmark

`pipeline/01_benchmark_tcc_full_pipeline.ipynb` documenta o fluxo genérico
legado parametrizado por **tier** (ver [docs/data/public-datasets.md](../data/public-datasets.md)):

```bash
python benchmarks/run_tcc_pipeline.py \
  --download \
  --tier medium \
  --full-benchmark \
  --epochs 100 \
  --device-profile gpu \
  --out data/results/tcc_medium_15k \
  --npz data/datasets/benchmark_dataset.npz
```

Esse roteiro não deve ser citado como reprodução do benchmark do artigo. O
notebook mantém a execução completa desativada por padrão; para a receita
canônica, use a configuração e o comando descritos no inventário do estudo.

## Notebook Completo de Todas as Arquiteturas

`pipeline/04_all_architectures_full_benchmark.ipynb` é um caderno operacional
histórico. Ele:

- configura `XFAKE_STORAGE_DIR`, `XFAKE_DATASETS_DIR`, `XFAKE_MODELS_DIR` e
  `XFAKE_LOGS_DIR` para storage persistente;
- executa um smoke test antes do treino pesado;
- usa um **tier** de dataset, diferente do NPZ e dos splits do artigo;
- chama `benchmarks/run_tcc_pipeline.py --download --tier <tier> --archs`
  com as 14 arquiteturas;
- audita `dataset.md`, `dataset_manifest.json`, `results.json`, `results.csv`,
  `predictions_clean.csv`, figuras PNG agregadas e artefatos por arquitetura;
- verifica o campo `model_artifact` e a consolidação em `data/models/`, garantindo
  que os modelos treinados foram salvos para uso posterior no Gradio/API.

A flag `RUN_FULL_BENCHMARK = False` impede execução acidental. Ative somente
após confirmar GPU, espaço em disco/storage e credenciais necessárias para os
datasets.

## Notebooks por Modelo

Cada notebook em `notebooks/models/` contém:

- descrição da família da arquitetura;
- entrada esperada;
- motivo técnico para estudar o modelo;
- foco de análise no TCC;
- célula de preparação de dados sintéticos;
- célula de inspeção do modelo ou smoke benchmark;
- referências para arquitetura, treino e benchmark.

Os notebooks neurais usam o factory de arquiteturas quando possível. Os
baselines clássicos (`SVM` e `RandomForest`) usam o harness de benchmark rápido.

## Notebook de Features

`features/01_feature_extraction_study.ipynb` demonstra o **front-end real** de
inferência (`audio_preprocessing.prepare_audio_for_model`, baseado em
`tf.signal`, in-graph): gera um sinal sintético de 1 s e extrai os três
front-ends que o sistema usa de verdade, com visualização:

- **LFCC** (default desde a melhoria P0 — supera o mel em anti-spoofing);
- **log-mel**;
- **raw-audio** (forma de onda PCM 1D);
- features clássicas de estudo: **MFCC**, centroide espectral, bandwidth,
  **ZCR** e RMS.

É o mesmo front-end reproduzido na detecção, garantindo paridade
treino↔inferência via `input_contract`. O catálogo completo de features
(MFCC, CQT, espectrais, prosódicas, formantes, qualidade vocal) está em
`docs/architecture/audio-features.md`.

## Manutenção e geração

Os notebooks ativos (`00_index`, `features/`, `models/`, `pipeline/`) são
**gerados programaticamente** por:

```bash
python tools/ops/build_notebooks.py
```

O gerador valida com `compile()` **todas** as células de código antes de gravar
— se ele roda sem erro, todo o código dos notebooks é sintaticamente válido. As
células usam a API real do projeto (`BenchmarkData.prepare_for_architecture`,
`create_model_by_name`, `BenchmarkConfig`/`run_benchmark`, `TrainingService`,
`ModelLoader`+`Predictor`) e o `source` é gravado como lista de linhas em UTF-8
(sem indentação espúria que viraria bloco de código no Jupyter).

O teste de notebooks também verifica:

- 14 notebooks de modelos com contrato de entrada resolvido;
- 4 notebooks de pipeline, incluindo o caderno completo de todas as
  arquiteturas;
- notebook de benchmark com execução completa guardada por
  `RUN_FULL_PIPELINE = False`;
- presença dos artefatos esperados (`dataset.md`, `tcc_report.md`, PNGs);
- notebook de features cobrindo front-end real e features clássicas;
- execução real opcional de uma amostra leve quando `nbformat`/`nbclient` estão
  instalados, cobrindo **ambos os caminhos**: espectrograma (`11_multiscale_cnn`)
  e **raw-audio** (`03_rawnet2`, SincConv), além de `00_index`, features, SVM,
  RandomForest e os notebooks de `pipeline/`.

Além disso, `tests/integration/test_architectures_build.py` constrói as **10
arquiteturas neurais não-SSL** pelo MESMO caminho dos notebooks de modelo
(`BenchmarkData.synthetic → prepare_for_architecture → create_model_by_name`) e
faz um forward pass — rápido (sem treino) e dentro do gate bloqueante do CI
(`not smoke`). WavLM/HuBERT (download SSL) e os clássicos SVM/RandomForest ficam
fora desse teste e seguem cobertos pelo smoke e pelos notebooks, respectivamente.

O gerador é **determinístico e TF-free**: o `input_type` de cada modelo vem do
catálogo `MODELS` (cross-check opcional com o registry quando o factory está
importável). Use-o quando a lista de modelos, os comandos de benchmark ou a
estrutura didática mudarem. O teste `tests/unit/test_notebooks_compile.py`
garante que todas as células de código continuam compilando.
