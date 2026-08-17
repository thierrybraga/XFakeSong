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
![Benchmark](https://img.shields.io/badge/benchmark-40980_amostras-informational?style=for-the-badge)
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

O benchmark consolidado do TCC usa o dataset canônico
`data/datasets/benchmark_dataset.npz` — CETUC pareado com clones XTTS-v2, com
disjunção dupla de locutor e frase entre treino, validação e teste: `40.980`
amostras (`20.490` reais + `20.490` falsas), áudio bruto `(48000, 1)` mono a
16 kHz (3 s). O protocolo completo está em
[docs/data/dataset-protocol.md](docs/data/dataset-protocol.md).

| Métrica | Valor |
| --- | ---: |
| Amostras no NPZ | 40.980 (20.490 reais + 20.490 falsas) |
| Entrada | áudio bruto `(48000, 1)` — 3 s, mono, 16 kHz |
| Tamanho do NPZ canônico | 7,99 GB |
| Formato dos WAVs | PCM linear, 16 bits, mono, 16 kHz |
| Locutores | 56, todos nas duas classes |
| Splits | 33.226 treino / 3.976 validação / 3.778 teste |

A partição é disjunta por locutor **e** por frase (semente 42), e o conjunto de
teste é selado antes de qualquer treino por
`benchmark_dataset.npz.test-lock.json`:

| Split | Amostras | Locutores | Frases | Horas |
| --- | ---: | ---: | ---: | ---: |
| treino | 33.226 | 34 (23F/11M) | 602 | 45,47 |
| validação | 3.976 | 11 (7F/4M) | 201 | 5,49 |
| teste | 3.778 | 11 (7F/4M) | 197 | 5,17 |

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
[docs/interfaces/gradio.md](docs/interfaces/gradio.md).

## Runtime Local

Os caminhos padrão são mantidos separados para evitar artefatos soltos na raiz:

| Uso | Caminho padrão |
| --- | --- |
| Banco SQLite local | `data/app.db` |
| Uploads da API/Gradio | `data/uploads/` |
| Resultados regeneráveis | `data/results/` |
| Raiz de modelos usada pela inferência | `data/models/` |
| Modelos finais consolidados | `data/models/benchmark_final/` |

`DEEPFAKE_MODELS_DIR` deve continuar apontando para `data/models`; os modelos
treinados prontos para demonstração ficam em `data/models/benchmark_final/` e
também são materializados como `data/models/bench_*` quando necessário pelo
loader.

No Windows, o menu interativo também pode ser iniciado com:

```bash
start.bat
```

## Ambientes de Treinamento

O projeto possui uma estrutura consolidada por família computacional em
`docker/environments/`, com requirements, Dockerfiles e READMEs dedicados.

No Windows nativo, TensorFlow roda em CPU. Treino/benchmark com GPU NVIDIA deve
ser executado via WSL2/Docker Desktop GPU usando os perfis `*-nvidia`.

| Família metodológica | Modelos oficiais |
| --- | --- |
| `classical-tabular` | RandomForest, SVM |
| `spectral-convolutional` | MultiscaleCNN |
| `spectral-attention` | Conformer, Hybrid CNN-Transformer, SpectrogramTransformer |
| `waveform-end-to-end` | RawNet2, AASIST, RawGAT-ST |
| `ssl-pretrained` | WavLM Original, HuBERT Original |
| `extended` | Sonic Sleuth, EfficientNet-LSTM, Ensemble, WavLM (porte Keras), HuBERT (porte Keras) — fora do artigo |
Todos os wrappers usam `scripts/benchmark/run_models_sequential.py`, preservando pasta
própria por modelo, logs, retomada, `results.json`, figuras e artefatos.

```bash
python scripts/training/train_by_family.py --family classical-tabular --plan-only
python scripts/training/train_by_family.py --family spectral-convolutional --models MultiscaleCNN --epochs 100 --device-profile gpu
python scripts/training/train_by_family.py --family waveform-end-to-end --models RawNet2 AASIST --epochs 100 --device-profile gpu
python scripts/training/train_by_family.py --family ssl-pretrained --models "WavLM Original" "HuBERT Original" --epochs 100 --device-profile gpu
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
python scripts/ops/docker_build.py train-nvidia config
python scripts/ops/docker_build.py inference-cpu up
python scripts/ops/docker_build.py benchmark-nvidia run
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
[docs/archive/training-environments-plan.md](docs/archive/training-environments-plan.md).

## Benchmark do TCC

Validação rápida do harness, sem download de dataset:

```bash
python scripts/benchmark/run_tcc_pipeline.py --smoke --epochs 1 --batch-size 4
```

Execução completa planejada para o TCC:

```bash
python scripts/benchmark/run_tcc_pipeline.py ^
  --download ^
  --tier medium ^
  --full-benchmark ^
  --epochs 100 ^
  --device-profile gpu ^
  --out data/results/benchmark_confirmatory ^
  --npz data/datasets/benchmark_dataset.npz
```

No Windows com GPU, use o perfil Docker/WSL2:

```powershell
$env:DOCKER_TRAIN_CPU_LIMIT='8'
docker compose -f docker\compose\benchmark.nvidia.yml --env-file .env run --rm benchmark `
  python scripts/benchmark/run_tcc_pipeline.py `
    --download `
    --tier medium `
    --full-benchmark `
    --epochs 100 `
    --batch-size 32 `
    --device-profile gpu `
    --npz data/datasets/benchmark_dataset.npz `
    --out data/results/benchmark_confirmatory
```

Saídas principais:

| Artefato | Conteúdo |
| --- | --- |
| `benchmark_plan.md` / `benchmark_plan.json` | preset, dataset e hiperparâmetros efetivos antes do treino |
| `dataset.md` | composição, split, processamento e hiperparâmetros globais |
| `dataset_manifest.json` | manifesto estruturado do dataset |
| `data/datasets/speaker_manifest.json` | IDs reais de falante quando a fonte fornece o metadado |
| `data/datasets/speaker_table.csv` | tabela por arquivo com classe, fonte, split, falante, duração e tamanho |
| `results.json` / `results.csv` | métricas completas por arquitetura |
| `tcc_report.md` | relatório final com métricas, inferências e imagens PNG |
| `data/results/<run>/figures/*.png` | gráficos agregados |
| `architectures/<modelo>/*.png` | matriz de confusão, ROC, scores e convergência por modelo |
| `data/models/bench_*` | modelos/configs salvos por padrão para uso direto na Gradio/API |
| `data/models/benchmark_final/` | cópia completa dos modelos finais por arquitetura |
| `data/results/paper/main.tex` | fonte acadêmica do artigo (LaTeX) |
| `data/results/paper.zip` | pacote limpo com `.tex` e figuras, sem PDF/auxiliares |

Use `--models-dir outro/diretorio` apenas quando quiser isolar os modelos de uma
execução específica. Caminhos relativos de `--out`, `--models-dir` e `--dataset`
são ancorados na raiz do projeto.

> `data/models/` é global e chaveado só pela arquitetura: qualquer execução
> (benchmark, smoke, retreino) grava em `data/models/bench_<arch>.*`. Por isso o
> runner também guarda uma cópia do artefato **dentro do run**
> (`architectures/<modelo>/models/`) — sem ela, um smoke posterior deixa as
> métricas do run sem o modelo que as produziu. O `model_artifact_fingerprint`
> no `results.json` detecta a troca.

Para revisar o plano sem iniciar treinamento:

```bash
python scripts/benchmark/run_benchmark.py --full ^
  --dataset data/datasets/benchmark_dataset.npz ^
  --epochs 100 ^
  --out data/results/benchmark_confirmatory ^
  --plan-only
```

Benchmark de um modelo individual:

```bash
python scripts/benchmark/run_benchmark.py --model AASIST ^
  --dataset data/datasets/benchmark_dataset.npz ^
  --epochs 100 ^
  --out data/results/bench_aasist
```

Para detalhes do desenho experimental, consulte
[docs/evaluation/benchmark.md](docs/evaluation/benchmark.md).

Para auditar falantes depois de montar o dataset:

```bash
python scripts/dataset/rebuild_speaker_manifest.py --dataset-dir data/datasets
python scripts/dataset/export_speaker_table.py --dataset-dir data/datasets --scope all
python scripts/dataset/audit_speaker_manifest.py --dataset-dir data/datasets --scope splits
```

## Publicar Modelos no Hugging Face

Depois de consolidar os artefatos em `data/models/`, envie os modelos finais
para um repositório do tipo **Model** no Hugging Face Hub:

```bash
python scripts/ops/upload_models_to_hf.py \
  --repo-id SEU_USUARIO/xfakesong-models \
  --dry-run

python scripts/ops/upload_models_to_hf.py \
  --repo-id SEU_USUARIO/xfakesong-models \
  --private
```

Use `HF_TOKEN` ou `HUGGINGFACE_HUB_TOKEN` como variável de ambiente. O script
envia `data/models/bench_*`, `data/models/benchmark_final/` e o manifesto dos
modelos; opções extras permitem anexar `data/results/paper/` e resultados
consolidados. O passo a passo completo está em
[docs/deployment/hugging-face-spaces.md](docs/deployment/hugging-face-spaces.md).

Para baixar modelos pré-treinados do benchmark em outra máquina ou no Space:

```bash
MODEL_REPO_ID=SEU_USUARIO/xfakesong-models \
python scripts/ops/sync_hf_models.py --models-dir data/models --force
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
| Variable | `DEEPFAKE_MODELS_DIR` | `data/models` |
| Secret | `HF_TOKEN` | token com leitura do model repo, se privado |

No boot, `scripts/ops/sync_hf_models.py` sincroniza os artefatos do Model Hub para
`data/models`. Se `MODEL_REPO_ID` não estiver definido, a aplicação usa os
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

Elas rendem **11 entradas** no escopo oficial e 5 no estendido
(`benchmarks/config.py`): WavLM e HuBERT entram no oficial como `Original` —
backbone congelado com cabeça treinada, que é a configuração usada pelos
sistemas de topo do ASVspoof 5 — e no estendido em porte Keras.

Resultados do run vigente (`data/results/clean_benchmark_15k/`) em
[docs/evaluation/benchmark.md](docs/evaluation/benchmark.md#run-vigente--clean_benchmark_15k).

## Notebooks

Os notebooks foram reorganizados para estudo e reprodução:

```text
docs/notebooks/
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

Cada notebook em `docs/notebooks/models/` documenta uma arquitetura, seu contrato de
entrada, objetivo de estudo e célula prática de inspeção. O diretório
`docs/notebooks/pipeline/` contém o notebook do benchmark completo, um notebook de
treino e um notebook de inferência. O diretório `docs/notebooks/features/` concentra
a extração e estudo de features acústicas.

## Documentação

A documentação técnica está em `docs/` e é publicada via MkDocs:
[thierrybraga.github.io/XFakeSong](https://thierrybraga.github.io/XFakeSong/).

| Objetivo | Documento |
| --- | --- |
| Conceitos da área (anti-spoofing) | [Conceitos e Fundamentos](docs/getting-started/concepts.md) |
| Visão geral | [Introdução](docs/getting-started/introduction.md) |
| Instalação local, Docker e HF Spaces | [Instalação e Configuração](docs/getting-started/installation.md) |
| Arquitetura do sistema | [Arquitetura](docs/architecture/overview.md) |
| Extração de features | [Features de Áudio](docs/architecture/audio-features.md) |
| Modelos | [Arquiteturas](docs/models/architectures.md) |
| Treinamento | [Treinamento](docs/models/training.md) |
| Inferência | [Inferência](docs/models/inference.md) |
| Datasets | [Datasets Públicos](docs/data/public-datasets.md) |
| Benchmark e resultados | [Benchmark e Resultados](docs/evaluation/benchmark.md) |
| Dataset usado no benchmark | [Dataset do Benchmark Utilizado](docs/data/benchmark-dataset.md) |
| Pipeline e auditoria de dataset | [Dataset Pipeline](docs/data/pipeline-and-audit.md) |
| Estudo experimental no GitHub Pages | [Estudo Experimental](docs/evaluation/experimental-study.md) |
| Notebooks | [Guia de Notebooks](docs/evaluation/notebooks.md) |
| Frontend Gradio e abas | [Frontend Gradio](docs/interfaces/gradio.md) |
| GitHub Pages e Hugging Face | [Publicação GitHub/HF](docs/deployment/github-pages-and-hugging-face.md) |
| Dúvidas frequentes | [Perguntas Frequentes (FAQ)](docs/reference/faq.md) |

Índice canônico e completo: [docs/index.md](docs/index.md).

## Comandos Essenciais

```bash
python main.py --bootstrap-dirs
python main.py --gradio
./scripts/ops/run_tests.sh fast
./scripts/ops/run_tests.sh cov
docker compose up --build -d
docker compose logs -f
docker compose down
```

## Contribuição e Segurança

Leia [CONTRIBUTING.md](CONTRIBUTING.md) antes de abrir pull requests e
[SECURITY.md](SECURITY.md) para reportar vulnerabilidades. O projeto segue a
licença MIT; consulte [LICENSE](LICENSE).
