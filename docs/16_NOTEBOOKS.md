# 16 — Guia de Notebooks

Os notebooks foram reorganizados para separar estudo, execução do benchmark,
treinamento, inferência e análise de features. A estrutura atual evita um
notebook monolítico e facilita revisar cada arquitetura isoladamente.

## Estrutura

```text
notebooks/
├── 00_index.ipynb
├── features/
│   └── 01_feature_extraction_study.ipynb
├── pipeline/
│   ├── 01_benchmark_tcc_full_pipeline.ipynb
│   ├── 02_training_model.ipynb
│   └── 03_inference.ipynb
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
└── legacy/
```

## Ordem Recomendada

1. `notebooks/00_index.ipynb`: mapa executável dos notebooks.
2. `notebooks/features/01_feature_extraction_study.ipynb`: estudo visual e
   numérico das features.
3. `notebooks/pipeline/01_benchmark_tcc_full_pipeline.ipynb`: pipeline oficial
   de download, processamento, treino, inferência e relatório.
4. `notebooks/pipeline/02_training_model.ipynb`: treino prático de um modelo.
5. `notebooks/pipeline/03_inference.ipynb`: leitura de predições e inferência.
6. `notebooks/models/*.ipynb`: estudo individual por arquitetura.

## Notebook de Benchmark

`pipeline/01_benchmark_tcc_full_pipeline.ipynb` documenta o comando oficial:

```bash
python scripts/run_tcc_pipeline.py \
  --tcc-full-dataset \
  --out results/tcc_full_20k \
  --npz app/datasets/benchmark_audio_raw_20k.npz
```

Esse preset usa o alvo experimental de `10.000` amostras reais + `10.000`
amostras fake, ativa o benchmark completo e inclui o probe da API. O notebook
mantém a execução completa desativada por padrão para evitar download e treino
longos sem revisão do ambiente.

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
`docs/04_FEATURES.md`.

## Manutenção e geração

Os notebooks ativos (`00_index`, `features/`, `models/`, `pipeline/`) são
**gerados programaticamente** por:

```bash
python scripts/build_notebooks.py
```

O gerador valida com `compile()` **todas** as células de código antes de gravar
— se ele roda sem erro, todo o código dos notebooks é sintaticamente válido. As
células usam a API real do projeto (`BenchmarkData.prepare_for_architecture`,
`create_model_by_name`, `BenchmarkConfig`/`run_benchmark`, `TrainingService`,
`ModelLoader`+`Predictor`) e o `source` é gravado como lista de linhas em UTF-8
(sem indentação espúria que viraria bloco de código no Jupyter).

O teste de notebooks também verifica:

- 14 notebooks de modelos com contrato de entrada resolvido;
- notebook de benchmark com execução completa guardada por
  `RUN_FULL_PIPELINE = False`;
- presença dos artefatos esperados (`dataset.md`, `tcc_report.md`, PNGs);
- notebook de features cobrindo front-end real e features clássicas;
- notebooks legados com aviso de arquivamento e células de código compiláveis.
- execução real opcional de uma amostra leve (`00_index`, features, SVM e
  RandomForest) quando `nbformat` e `nbclient` estiverem instalados.

Use o gerador quando a lista de modelos, os comandos de benchmark ou a estrutura
didática mudarem. Os notebooks antigos ficam preservados em `notebooks/legacy/`
(não são regenerados). O teste `tests/unit/test_notebooks_compile.py` garante
que todas as células de código continuam compilando.
