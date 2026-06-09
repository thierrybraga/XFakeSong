---
title: XFakeSong
emoji: 🛡️
colorFrom: blue
colorTo: slate
sdk: gradio
sdk_version: 4.31.0
app_file: app.py
pinned: false
license: mit
---

# XFakeSong

XFakeSong é uma plataforma open source para detecção de deepfakes de áudio.
O projeto combina extração de features acústicas, modelos neurais modernos,
baselines clássicos e um pipeline de benchmark reprodutível para gerar os
resultados numéricos e gráficos do TCC.

![Python Version](https://img.shields.io/badge/python-3.11%2B-blue?style=for-the-badge&logo=python)
![License](https://img.shields.io/badge/license-MIT-green?style=for-the-badge)
![Benchmark](https://img.shields.io/badge/benchmark-TCC_20k-informational?style=for-the-badge)
[![CI](https://github.com/thierrybraga/XFakeSong/actions/workflows/ci.yml/badge.svg)](https://github.com/thierrybraga/XFakeSong/actions/workflows/ci.yml)

## Foco do Projeto

O objetivo central é comparar arquiteturas de detecção de áudio deepfake em um
fluxo local, auditável e repetível:

1. baixar e organizar datasets reais e sintéticos;
2. normalizar áudio bruto e criar splits estratificados;
3. treinar cada arquitetura com hiperparâmetros rastreáveis;
4. executar inferência no conjunto de teste;
5. gerar métricas, matrizes de confusão, curvas ROC, robustez, latência e
   relatórios Markdown com imagens PNG.

O pipeline oficial usa o preset `--tcc-full-dataset`, com alvo de `10.000`
amostras reais + `10.000` amostras fake.

## Início Rápido

```bash
git clone https://github.com/thierrybraga/XFakeSong.git
cd XFakeSong
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python main.py --bootstrap-dirs
python main.py --gradio
```

A interface Gradio fica disponível em `http://localhost:7860`.

No Windows, o menu interativo também pode ser iniciado com:

```bash
start.bat
```

## Benchmark do TCC

Validação rápida do harness, sem download de dataset:

```bash
python scripts/run_tcc_pipeline.py --smoke --epochs 1 --batch-size 4
```

Execução completa planejada para o TCC:

```bash
python scripts/run_tcc_pipeline.py ^
  --tcc-full-dataset ^
  --out results/tcc_full_20k ^
  --npz app/datasets/benchmark_audio_raw_20k.npz
```

Saídas principais:

| Artefato | Conteúdo |
| --- | --- |
| `dataset.md` | composição, split, processamento e hiperparâmetros globais |
| `dataset_manifest.json` | manifesto estruturado do dataset |
| `results.json` / `results.csv` | métricas completas por arquitetura |
| `tcc_report.md` | relatório final com métricas, inferências e imagens PNG |
| `figures/*.png` | gráficos agregados |
| `architectures/<modelo>/*.png` | matriz de confusão, ROC, scores e convergência por modelo |

Para detalhes do desenho experimental, consulte
[docs/15_BENCHMARK.md](docs/15_BENCHMARK.md).

## Modelos Avaliados

O benchmark cobre 14 arquiteturas/baselines:

| Família | Modelos |
| --- | --- |
| Raw audio | WavLM, HuBERT, RawNet2 |
| Espectrograma e Transformers | Sonic Sleuth, AASIST, RawGAT-ST, Conformer, Hybrid CNN-Transformer, SpectrogramTransformer |
| CNN e fusão | EfficientNet-LSTM, MultiscaleCNN, Ensemble |
| Clássicos | SVM, Random Forest |

## Notebooks

Os notebooks foram reorganizados para estudo e reprodução:

```text
notebooks/
├── 00_index.ipynb
├── features/
│   └── 01_feature_extraction_study.ipynb
├── pipeline/
│   ├── 01_benchmark_tcc_full_pipeline.ipynb
│   ├── 02_training_model.ipynb
│   └── 03_inference.ipynb
└── models/
    ├── 01_wavlm.ipynb
    ├── ...
    └── 14_random_forest.ipynb
```

Cada notebook em `notebooks/models/` documenta uma arquitetura, seu contrato de
entrada, objetivo de estudo e célula prática de inspeção. O diretório
`notebooks/pipeline/` contém o notebook do benchmark completo, um notebook de
treino e um notebook de inferência. O diretório `notebooks/features/` concentra
a extração e estudo de features acústicas.

## Documentação

A documentação técnica está em `docs/` e é publicada via MkDocs:
[thierrybraga.github.io/XFakeSong](https://thierrybraga.github.io/XFakeSong/).

| Objetivo | Documento |
| --- | --- |
| Visão geral | [Introdução](docs/01_INTRODUCAO.md) |
| Instalação local, Docker e HF Spaces | [Instalação e Configuração](docs/02_INSTALACAO_CONFIGURACAO.md) |
| Arquitetura do sistema | [Arquitetura](docs/03_ARQUITETURA.md) |
| Extração de features | [Features de Áudio](docs/04_FEATURES.md) |
| Modelos | [Arquiteturas](docs/08_ARQUITETURAS.md) |
| Treinamento | [Treinamento](docs/10_TREINAMENTO.md) |
| Inferência | [Inferência](docs/09_INFERENCIA.md) |
| Datasets | [Datasets Públicos](docs/12_DATASETS.md) |
| Benchmark e resultados do TCC | [Benchmark e TCC](docs/15_BENCHMARK.md) |
| Notebooks | [Guia de Notebooks](docs/16_NOTEBOOKS.md) |

## Comandos Essenciais

```bash
python main.py --bootstrap-dirs
python main.py --gradio
./scripts/run_tests.sh fast
./scripts/run_tests.sh cov
docker compose up --build -d
docker compose logs -f
docker compose down
```

## Contribuição e Segurança

Leia [CONTRIBUTING.md](CONTRIBUTING.md) antes de abrir pull requests e
[SECURITY.md](SECURITY.md) para reportar vulnerabilidades. O projeto segue a
licença MIT; consulte [LICENSE](LICENSE).
