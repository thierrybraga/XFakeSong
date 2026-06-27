# 26 - Plano de Refatoracao de Ambientes de Treinamento

Este documento define o plano de acao para separar ambientes de treinamento,
inferencia, benchmark e publicacao do XFakeSong por familia computacional. O
objetivo e reduzir conflitos de dependencias, melhorar reprodutibilidade
cientifica e manter a aplicacao Gradio/FastAPI estavel para demonstracao e
inferencia com modelos ja treinados.

!!! note "Estado de implementacao"
    A estrutura base ja foi materializada em `environments/`,
    `configs/training/`, `configs/dataset.yaml`, `configs/inference.yaml`,
    `docker/compose/*.yml`, `Makefile` e `scripts/train_*.py`. O Dockerfile,
    os requirements raiz e os arquivos `docker-compose*.yml` da raiz permanecem
    como compatibilidade legada enquanto os ambientes por familia passam por
    validacao incremental.

## Estrategia central

Nao sera criado um ambiente por modelo individual. A divisao recomendada e por
ecossistema computacional:

| Familia | Modelos e uso | Ambiente recomendado |
| --- | --- | --- |
| Machine Learning classico | SVM, Random Forest, PCA, MFCC, prosodia, wavelets | `classical-ml` |
| TensorFlow/Keras | CNN, LSTM, EfficientNet-LSTM, MultiscaleCNN, Sonic Sleuth, SpectrogramTransformer | `tensorflow-keras` |
| PyTorch audio | RawNet2, AASIST, RawGAT-ST, Conformer, Hybrid CNN-Transformer | `pytorch-audio` |
| SSL/Transformers | WavLM, HuBERT, wav2vec2, Whisper embeddings | `ssl-transformers` |
| Inferencia/API | FastAPI, Gradio, carregamento de modelos treinados | `inference-api` |

Essa separacao preserva a coesao tecnica de cada stack e evita imagens Docker
excessivamente grandes para tarefas que nao precisam de todas as dependencias.

!!! note "Compatibilidade com a implementacao atual"
    No codigo atual, RawNet2, AASIST, RawGAT-ST, Conformer,
    Hybrid CNN-Transformer, WavLM e HuBERT ainda sao integrados ao benchmark por
    TensorFlow/Keras. Por isso `pytorch-audio` e `ssl-transformers` incluem
    TensorFlow durante a migracao. A separacao sera mais enxuta quando houver
    portas PyTorch nativas para essas arquiteturas.

## Diagnostico tecnico

O projeto XFakeSong possui escopo amplo:

- treinamento de modelos classicos e neurais;
- inferencia local via Gradio e API;
- benchmark consolidado para avaliacao academica;
- geracao de metricas, graficos e relatorios;
- documentacao tecnica e material para TCC;
- suporte a Docker, Hugging Face Spaces e WSL2 com GPU.

O problema principal do ambiente monolitico e carregar dependencias conflitantes
ou pesadas no mesmo runtime:

- TensorFlow/Keras;
- PyTorch/Torchaudio;
- Transformers/Hugging Face;
- Gradio/FastAPI;
- librosa, soundfile, scikit-learn e ferramentas cientificas;
- dependencias de CPU e GPU no mesmo ambiente;
- scripts de benchmark, relatorio e exportacao.

Isso aumenta tempo de build, tamanho da imagem, consumo de RAM/VRAM, chance de
incompatibilidade entre versoes e risco de quebrar a inferencia ao alterar uma
dependencia usada apenas no treinamento.

## Estado atual do repositorio

Hoje o projeto ainda parte de um runtime centralizado:

| Area | Estado atual |
| --- | --- |
| Dependencias | `requirements.txt`, `requirements-base.txt`, `requirements-cpu.txt`, `requirements-dev.txt` |
| Docker | Principal: `docker/compose/*.yml`; legado: `Dockerfile`, `docker-compose.yml`, `docker-compose.gpu.yml`, `docker-compose.benchmark.yml`, `docker-compose.train.yml` |
| Benchmark | `scripts/run_tcc_pipeline.py`, `scripts/run_benchmark.py`, `benchmarks/` |
| Dataset principal | `app/datasets/benchmark_audio_raw_balanced_20k.npz` |
| Modelos default | `app/models/bench_*` e `app/models/benchmark_final/` |
| Resultados | `results/`, `results/tcc_consolidated/`, `reports/` |
| Documentacao | `docs/` via MkDocs Material |

A pasta `environments/` ja existe como camada de consolidacao. Cada ambiente
possui requirements, Dockerfile(s) e README. A validacao completa deve ocorrer
por smoke test e benchmark por familia antes de remover os arquivos legados.

## Arquitetura-alvo

```text
XFakeSong/
├── app/
│   ├── domain/
│   ├── application/
│   ├── core/
│   ├── interfaces/
│   └── routers/
│
├── environments/
│   ├── classical-ml/
│   │   ├── Dockerfile.cpu
│   │   ├── requirements.txt
│   │   └── README.md
│   ├── tensorflow-keras/
│   │   ├── Dockerfile.cpu
│   │   ├── Dockerfile.nvidia
│   │   ├── requirements.txt
│   │   └── README.md
│   ├── pytorch-audio/
│   │   ├── Dockerfile.cpu
│   │   ├── Dockerfile.nvidia
│   │   ├── requirements.txt
│   │   └── README.md
│   ├── ssl-transformers/
│   │   ├── Dockerfile.cpu
│   │   ├── Dockerfile.nvidia
│   │   ├── requirements.txt
│   │   └── README.md
│   └── inference-api/
│       ├── Dockerfile.cpu
│       ├── Dockerfile.nvidia
│       ├── requirements.txt
│       └── README.md
│
├── scripts/
│   ├── train_classical.py
│   ├── train_tensorflow.py
│   ├── train_pytorch.py
│   ├── train_ssl.py
│   ├── benchmark_all.py
│   ├── validate_artifacts.py
│   └── export_model_card.py
│
├── configs/
│   ├── training/
│   │   ├── classical.yaml
│   │   ├── tensorflow.yaml
│   │   ├── pytorch.yaml
│   │   └── ssl.yaml
│   ├── inference.yaml
│   └── dataset.yaml
│
├── datasets/
│   ├── raw/
│   ├── processed/
│   ├── manifests/
│   └── splits/
│
├── models/
│   ├── classical/
│   ├── tensorflow/
│   ├── pytorch/
│   ├── ssl/
│   └── registry.json
│
├── reports/
│   ├── metrics/
│   ├── figures/
│   ├── confusion_matrices/
│   ├── roc_curves/
│   └── markdown/
│
├── cache/
│   ├── huggingface/
│   ├── torch/
│   └── tensorflow/
│
├── docker/
│   ├── README.md
│   ├── build.env.example
│   └── compose/
│       ├── inference.cpu.yml
│       ├── inference.nvidia.yml
│       ├── train.cpu.yml
│       ├── train.nvidia.yml
│       └── benchmark.nvidia.yml
│
├── docker-compose.yml              # legado/compatibilidade
├── docker-compose.train.yml        # legado/compatibilidade
├── docker-compose.gpu.yml          # legado/compatibilidade
├── Makefile                        # atalhos para docker/compose/*.yml
├── README.md
└── docs/
```

## Contratos entre ambientes

Todos os ambientes devem compartilhar os mesmos contratos de entrada e saida.
Isso evita que cada stack gere modelos, metricas e graficos em formatos
incompativeis.

| Contrato | Obrigatorio |
| --- | --- |
| Dataset | manifesto com origem, classe, duracao, taxa de amostragem, split e hash |
| Treinamento | `training_config.json`, `history.csv`, `metrics.json`, `results.json` |
| Modelo | arquivo exportado, metadados de arquitetura, versao da biblioteca e `input_contract` |
| Graficos | loss/accuracy, matriz de confusao, ROC, distribuicao de scores quando aplicavel |
| Inferencia | funcao de carregamento compativel com Gradio/API |
| Relatorio | Markdown por arquitetura com dataset, hiperparametros, metricas e observacoes |

## Perfis Docker consolidados

| Perfil | Compose | Dockerfiles | Hardware |
| --- | --- | --- | --- |
| Inferencia CPU/onboard | `docker/compose/inference.cpu.yml` | `environments/inference-api/Dockerfile.cpu` | CPU, Intel/AMD integrado, sem CUDA |
| Inferencia NVIDIA | `docker/compose/inference.nvidia.yml` | `environments/inference-api/Dockerfile.nvidia` | NVIDIA CUDA via Linux/WSL2 |
| Treino CPU/onboard | `docker/compose/train.cpu.yml` | `Dockerfile.cpu` por familia | CPU, smoke tests, SVM/RF |
| Treino NVIDIA | `docker/compose/train.nvidia.yml` | `Dockerfile.nvidia` por familia | NVIDIA CUDA via Linux/WSL2 |
| Benchmark NVIDIA | `docker/compose/benchmark.nvidia.yml` | `Dockerfile` raiz com `TF_VARIANT=gpu` | benchmark sequencial completo |

Use `scripts/docker_build.py` como entrada padronizada para validar ou executar
os perfis:

```bash
python scripts/docker_build.py inference-cpu config
python scripts/docker_build.py train-cpu config
python scripts/docker_build.py train-nvidia config
python scripts/docker_build.py benchmark-nvidia run
```

## Fases de migracao

### P0 - Inventario e congelamento do baseline

- Registrar versoes atuais de Python, TensorFlow, PyTorch, CUDA, cuDNN,
  scikit-learn, librosa, Gradio e FastAPI.
- Gerar `pip freeze` do ambiente usado no benchmark atual.
- Validar que os modelos default em `app/models/` carregam pela interface
  Gradio.
- Executar smoke tests de inferencia e benchmark sintetico antes de qualquer
  separacao de dependencias.

**Criterio de aceite:** o pipeline atual continua executando sem regressao.

### P1 - Separacao de requirements por familia

- Criar `environments/<familia>/requirements.txt`. **Implementado como base.**
- Mover dependencias pesadas para o ambiente da familia correspondente.
- Manter `requirements.txt` como compatibilidade temporaria.
- Documentar matriz de versoes por ambiente.

**Criterio de aceite:** cada arquivo de requirements instala em ambiente limpo e
os testes da familia passam.

### P2 - Dockerfiles por familia

- Criar Dockerfiles CPU/GPU para TensorFlow, PyTorch e SSL. **Implementado como base.**
- Criar Dockerfile leve para `inference-api`. **Implementado como base.**
- Criar composes segmentados por perfil de hardware. **Implementado em
  `docker/compose/`.**
- Padronizar volumes para `datasets/`, `app/models/`, `results/` e `cache/`.
- Configurar cache Hugging Face, Torch e TensorFlow fora da imagem.

**Criterio de aceite:** cada imagem executa um smoke test de treino ou
inferencia sem baixar dependencias em tempo de execucao.

### P3 - Entrypoints de treinamento

- Criar scripts por familia: `train_classical.py`, `train_tensorflow.py`,
  `train_pytorch.py` e `train_ssl.py`. **Implementado como wrappers de
  `scripts/run_models_sequential.py`.**
- Criar utilitarios de consolidacao: `benchmark_all.py`,
  `validate_artifacts.py` e `export_model_card.py`. **Implementado como camada
  leve sobre o pipeline atual.**
- Permitir treino por arquitetura individual e por preset.
- Padronizar logs de progresso, checkpoints, early stopping e exportacao de
  artefatos.
- Garantir que o benchmark possa chamar qualquer familia por CLI.

**Criterio de aceite:** qualquer modelo pode ser treinado com dataset real ou
synthetic smoke usando o mesmo contrato de saida.

### P4 - Registro de modelos e artefatos

- Consolidar `models/registry.json` ou equivalente em `app/models/`.
- Registrar caminho do modelo, arquitetura, hash do dataset, metricas,
  hiperparametros, data de treino e ambiente usado.
- Validar automaticamente se um modelo pode ser usado no frontend.
- Sincronizar modelos default com Hugging Face Hub quando configurado.

**Criterio de aceite:** a interface Gradio lista apenas modelos carregaveis e
informa metricas/versao do artefato.

### P5 - Benchmark reprodutivel

- Executar benchmark por familia e consolidar os resultados em `results/`.
- Gerar figuras padronizadas por arquitetura.
- Gerar relatorio Markdown e insumos para TCC/Overleaf.
- Incluir teste CI de smoke para todas as arquiteturas com dataset sintetico
  pequeno.

**Criterio de aceite:** o benchmark completo pode ser reproduzido a partir de
dataset, configs e containers versionados.

### P6 - Limpeza do ambiente monolitico

- Remover dependencias de treinamento do ambiente de inferencia.
- Marcar `requirements.txt` monolitico como legado ou substitui-lo por extras.
- Remover scripts duplicados e caminhos antigos apos migracao validada.

**Criterio de aceite:** a imagem de inferencia e menor, mais rapida de construir
e independente das dependencias de treinamento.

## Comandos-alvo

```bash
# Inferencia local
docker compose up --build app

# Inferencia usando o ambiente dedicado
docker compose -f docker/compose/inference.cpu.yml up --build inference-api
docker compose -f docker/compose/inference.nvidia.yml up --build inference-api

# Treino classico
docker compose -f docker/compose/train.cpu.yml run --rm classical-ml \
  python scripts/train_classical.py --models SVM --config configs/training/classical.yaml

# Treino TensorFlow/Keras com GPU
docker compose -f docker/compose/train.nvidia.yml run --rm tensorflow-keras \
  python scripts/train_tensorflow.py --models MultiscaleCNN --epochs 100

# Treino PyTorch audio com GPU
docker compose -f docker/compose/train.nvidia.yml run --rm pytorch-audio \
  python scripts/train_pytorch.py --models RawNet2 --epochs 100

# Treino SSL/Transformers com GPU
docker compose -f docker/compose/train.nvidia.yml run --rm ssl-transformers \
  python scripts/train_ssl.py --models WavLM --epochs 100

# Benchmark NVIDIA completo
docker compose -f docker/compose/benchmark.nvidia.yml run --rm benchmark

# Validacao de artefatos
python scripts/validate_artifacts.py --models-dir app/models --results-dir results
```

## Regras de compatibilidade

- O frontend Gradio nao deve depender de bibliotecas de treinamento quando usar
  modelos ja exportados.
- Todo modelo treinado deve carregar pelo mesmo servico de inferencia usado pela
  aplicacao.
- O benchmark deve salvar graficos e metricas em pasta propria por arquitetura.
- Modelos que dependem de backbone externo devem registrar se usaram backbone
  real ou fallback.
- Containers de treino podem ser pesados; containers de inferencia devem ser
  enxutos.
- Cache, datasets e resultados nao devem ser embutidos na imagem Docker.

## Riscos principais

| Risco | Mitigacao |
| --- | --- |
| Divergencia treino-inferencia | Validar `input_contract` e smoke de inferencia apos cada treino |
| Quebra de modelos ja treinados | Manter carregadores legados durante a transicao |
| Diferencas CPU/GPU | Registrar ambiente, seeds, versoes e device em `training_config.json` |
| Imagens Docker grandes | Separar inferencia de treinamento e usar cache por volume |
| Duplicacao de scripts | Criar entrypoints por familia, mas compartilhar servicos de dataset, metricas e relatorio |

## Checklist de aceite final

- [x] `environments/` criado com cinco familias documentadas.
- [x] Cada familia possui requirements e Dockerfile CPU/NVIDIA nomeado por perfil.
- [x] Perfis Compose segmentados em `docker/compose/`.
- [ ] Cada familia possui Dockerfile validado por build completo.
- [ ] Benchmark sintetico passa para todas as 14 arquiteturas.
- [ ] Benchmark real gera `results.json`, graficos e relatorio por arquitetura.
- [ ] Modelos treinados carregam na Gradio e na API.
- [ ] Hugging Face Hub/Spaces documentado para download e uso de modelos.
- [ ] Documentacao de instalacao, treinamento, benchmark e deploy atualizada.
- [ ] Ambiente de inferencia nao instala dependencias desnecessarias de treino.
