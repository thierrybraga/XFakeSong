---
title: XFakeSong
emoji: 🛡️
colorFrom: blue
colorTo: slate
sdk: docker
app_port: 7860
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
![Benchmark](https://img.shields.io/badge/benchmark-TCC_15k-informational?style=for-the-badge)
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

O benchmark consolidado do TCC usa o dataset
`app/datasets/benchmark_audio_raw_balanced_20k.npz`, com `7.500` amostras reais
+ `7.500` amostras fake.
Excedentes baixados durante a curadoria são arquivados em
`app/datasets/overflow/`, preservando os WAVs brutos para novas rotas.

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

A interface Gradio fica disponível em `http://localhost:7860/gradio/`.
O uso das abas, análises e notificações está documentado em
[docs/23_INTERFACE_GRADIO.md](docs/23_INTERFACE_GRADIO.md).

## Runtime Local

Os caminhos padrão são mantidos separados para evitar artefatos soltos na raiz:

| Uso | Caminho padrão |
| --- | --- |
| Banco SQLite local | `data/app.db` |
| Uploads da API/Gradio | `data/uploads/` |
| Resultados regeneráveis | `results/` |
| Raiz de modelos usada pela inferência | `app/models/` |
| Modelos finais consolidados | `app/models/benchmark_final/` |

`DEEPFAKE_MODELS_DIR` deve continuar apontando para `app/models`; os modelos
treinados prontos para demonstração ficam em `app/models/benchmark_final/` e
também são materializados como `app/models/bench_*` quando necessário pelo
loader.

No Windows, o menu interativo também pode ser iniciado com:

```bash
start.bat
```

## Ambientes de Treinamento

O projeto possui uma estrutura consolidada por família computacional em
`environments/`, com requirements, Dockerfiles e READMEs dedicados.

| Família | Modelos | Entrada |
| --- | --- | --- |
| `classical-ml` | SVM, RandomForest | `python scripts/train_classical.py` |
| `tensorflow-keras` | Sonic Sleuth, EfficientNet-LSTM, MultiscaleCNN, SpectrogramTransformer | `python scripts/train_tensorflow.py` |
| `pytorch-audio` | RawNet2, AASIST, RawGAT-ST, Conformer, Hybrid CNN-Transformer | `python scripts/train_pytorch.py` |
| `ssl-transformers` | WavLM, HuBERT | `python scripts/train_ssl.py` |
| `inference-api` | Gradio/FastAPI com modelos treinados | `python main.py --gradio` |

Todos os wrappers usam `scripts/run_models_sequential.py`, preservando pasta
própria por modelo, logs, retomada, `results.json`, figuras e artefatos.

```bash
python scripts/train_classical.py --plan-only
python scripts/train_tensorflow.py --models MultiscaleCNN --epochs 100 --device-profile gpu
python scripts/train_pytorch.py --models Conformer RawNet2 --epochs 100 --device-profile gpu
python scripts/train_ssl.py --models WavLM HuBERT --epochs 100 --device-profile gpu
```

Execução via Docker por perfil:

```bash
# CPU/onboard, sem CUDA
docker compose -f docker/compose/train.cpu.yml run --rm classical-ml
docker compose -f docker/compose/train.cpu.yml run --rm tensorflow-keras

# NVIDIA CUDA via Linux/WSL2/Docker Desktop GPU
docker compose -f docker/compose/train.nvidia.yml run --rm tensorflow-keras
docker compose -f docker/compose/train.nvidia.yml run --rm pytorch-audio
docker compose -f docker/compose/train.nvidia.yml run --rm ssl-transformers

# Inferência
docker compose -f docker/compose/inference.cpu.yml up --build inference-api
docker compose -f docker/compose/inference.nvidia.yml up --build inference-api
```

Também há um helper único:

```bash
python scripts/docker_build.py train-nvidia config
python scripts/docker_build.py inference-cpu up
python scripts/docker_build.py benchmark-nvidia run
```

Via `make`, os targets de Docker usam os mesmos perfis segmentados:

```bash
make build              # docker/compose/inference.cpu.yml
make up                 # inferência CPU/onboard
make up GPU=1           # inferência NVIDIA
make train-cpu          # treino clássico/CPU
make train-nvidia       # treino neural TensorFlow/Keras com NVIDIA
make benchmark-nvidia   # benchmark completo NVIDIA/WSL2
make docker-config      # valida todos os compose segmentados
```

O plano técnico e os critérios de aceite estão em
[docs/26_PLANO_AMBIENTES_TREINAMENTO.md](docs/26_PLANO_AMBIENTES_TREINAMENTO.md).

## Benchmark do TCC

Validação rápida do harness, sem download de dataset:

```bash
python scripts/run_tcc_pipeline.py --smoke --epochs 1 --batch-size 4
```

Execução completa planejada para o TCC:

```bash
python scripts/run_tcc_pipeline.py ^
  --download ^
  --target-per-class 7500 ^
  --full-benchmark ^
  --epochs 100 ^
  --device-profile gpu ^
  --out results/tcc_full_15k ^
  --npz app/datasets/benchmark_audio_raw_balanced_20k.npz
```

Saídas principais:

| Artefato | Conteúdo |
| --- | --- |
| `benchmark_plan.md` / `benchmark_plan.json` | preset, dataset e hiperparâmetros efetivos antes do treino |
| `dataset.md` | composição, split, processamento e hiperparâmetros globais |
| `dataset_manifest.json` | manifesto estruturado do dataset |
| `results.json` / `results.csv` | métricas completas por arquitetura |
| `tcc_report.md` | relatório final com métricas, inferências e imagens PNG |
| `figures/*.png` | gráficos agregados |
| `architectures/<modelo>/*.png` | matriz de confusão, ROC, scores e convergência por modelo |
| `app/models/bench_*` | modelos/configs salvos por padrão para uso direto na Gradio/API |
| `app/models/benchmark_final/` | cópia completa dos modelos finais por arquitetura |
| `tcc_overleaf/main.tex` | fonte acadêmica para Overleaf |
| `tcc_overleaf.zip` | pacote limpo com `.tex` e figuras, sem PDF/auxiliares |

Use `--models-dir outro/diretorio` apenas quando quiser isolar os modelos de uma
execução específica. Caminhos relativos de `--out`, `--models-dir` e `--dataset`
são ancorados na raiz do projeto.

Para revisar o plano sem iniciar treinamento:

```bash
python scripts/run_benchmark.py --full ^
  --dataset app/datasets/benchmark_audio_raw_balanced_20k.npz ^
  --epochs 100 ^
  --out results/tcc_full_15k ^
  --plan-only
```

Benchmark de um modelo individual:

```bash
python scripts/run_benchmark.py --model AASIST ^
  --dataset app/datasets/benchmark_audio_raw_balanced_20k.npz ^
  --epochs 100 ^
  --out results/bench_aasist
```

Para detalhes do desenho experimental, consulte
[docs/15_BENCHMARK.md](docs/15_BENCHMARK.md).

## Publicar Modelos no Hugging Face

Depois de consolidar os artefatos em `app/models/`, envie os modelos finais
para um repositório do tipo **Model** no Hugging Face Hub:

```bash
python scripts/upload_models_to_hf.py \
  --repo-id SEU_USUARIO/xfakesong-models \
  --dry-run

python scripts/upload_models_to_hf.py \
  --repo-id SEU_USUARIO/xfakesong-models \
  --private
```

Use `HF_TOKEN` ou `HUGGINGFACE_HUB_TOKEN` como variável de ambiente. O script
envia `app/models/bench_*`, `app/models/benchmark_final/` e o manifesto dos
modelos; opções extras permitem anexar `tcc_overleaf/` e resultados
consolidados. O passo a passo completo está em
[docs/11_DEPLOY_HUGGINGFACE.md](docs/11_DEPLOY_HUGGINGFACE.md).

Para baixar modelos pré-treinados do benchmark em outra máquina ou no Space:

```bash
MODEL_REPO_ID=SEU_USUARIO/xfakesong-models \
python scripts/sync_hf_models.py --models-dir app/models --force
```

Use `HF_TOKEN` se o repositório de modelos for privado.

### Deploy como Hugging Face Space

O repositório já está preparado para **Docker Space** (`sdk: docker`,
`app_port: 7860`). Para demonstração com os modelos já treinados, configure no
Space:

| Tipo | Nome | Valor recomendado |
| --- | --- | --- |
| Variable | `MODEL_REPO_ID` | `SEU_USUARIO/xfakesong-models` |
| Variable | `ENABLE_TRAINING` | `false` |
| Variable | `XFAKE_SYNC_MODELS_ON_BOOT` | `true` |
| Variable | `DEEPFAKE_MODELS_DIR` | `app/models` |
| Secret | `HF_TOKEN` | token com leitura do model repo, se privado |

No boot, `scripts/sync_hf_models.py` sincroniza os artefatos do Model Hub para
`app/models`. Se `MODEL_REPO_ID` não estiver definido, a aplicação usa os
modelos já empacotados/localmente disponíveis. O frontend lista os modelos sem
carregar todos os pesos no startup; cada modelo é carregado sob demanda ao ser
selecionado para inferência.

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
| Conceitos da área (anti-spoofing) | [Conceitos e Fundamentos](docs/00_CONCEITOS.md) |
| Visão geral | [Introdução](docs/01_INTRODUCAO.md) |
| Instalação local, Docker e HF Spaces | [Instalação e Configuração](docs/02_INSTALACAO_CONFIGURACAO.md) |
| Arquitetura do sistema | [Arquitetura](docs/03_ARQUITETURA.md) |
| Extração de features | [Features de Áudio](docs/04_FEATURES.md) |
| Modelos | [Arquiteturas](docs/08_ARQUITETURAS.md) |
| Treinamento | [Treinamento](docs/10_TREINAMENTO.md) |
| Inferência | [Inferência](docs/09_INFERENCIA.md) |
| Datasets | [Datasets Públicos](docs/12_DATASETS.md) |
| Benchmark e resultados | [Benchmark e Resultados](docs/15_BENCHMARK.md) |
| Estudo experimental no GitHub Pages | [Estudo Experimental](docs/20_ESTUDO_EXPERIMENTAL.md) |
| Notebooks | [Guia de Notebooks](docs/16_NOTEBOOKS.md) |
| Interface Gradio e abas | [Interface Gradio](docs/23_INTERFACE_GRADIO.md) |
| GitHub Pages e Hugging Face | [Publicação GitHub/HF](docs/24_PUBLICACAO_GITHUB_HF.md) |
| Dúvidas frequentes | [Perguntas Frequentes (FAQ)](docs/19_FAQ.md) |

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
