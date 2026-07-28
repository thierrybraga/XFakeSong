# XFakeSong — Guia para Claude Code

XFakeSong e uma plataforma open source de deteccao de audio deepfake (analise
espectral classica + Transformers SSL como WavLM/HuBERT), com foco em XAI e
processamento 100% local. Interface Gradio, Python 3.11+, licenca MIT.

Este arquivo e o guia operacional para agentes de codigo. A fonte canonica de
arquitetura e convencoes e o [AGENTS.md](AGENTS.md); os detalhes tecnicos estao
em `docs/`. Em caso de divergencia, AGENTS.md + docs/ prevalecem.

| Tema | Fonte canonica |
| --- | --- |
| Arquitetura e limites de camadas | [AGENTS.md](AGENTS.md), [docs/architecture/overview.md](docs/architecture/overview.md) |
| Desenvolvimento e padroes | [docs/development/developer-guide.md](docs/development/developer-guide.md) |
| Testes | [docs/development/quality-and-testing.md](docs/development/quality-and-testing.md) |
| Arquiteturas neurais | [docs/models/architectures.md](docs/models/architectures.md) |
| Treinamento e hiperparametros | [docs/models/training.md](docs/models/training.md) |
| Benchmark e metricas | [docs/evaluation/benchmark.md](docs/evaluation/benchmark.md) |
| Datasets | [docs/data/public-datasets.md](docs/data/public-datasets.md) |
| Retreino apos diagnostico | [docs/evaluation/retraining-adjustments.md](docs/evaluation/retraining-adjustments.md) |

---

## Atalhos operacionais

```bash
# App / dev
python main.py --gradio                 # interface Gradio (porta 7860)
python main.py --bootstrap-dirs         # cria estrutura de diretorios padrao
make dev                                # Gradio local (sem Docker)
make install-dev                        # cria .venv + instala deps + dev tools

# Testes / qualidade
pytest tests/                           # suite completa
make test                               # suite rapida (tudo exceto smoke) — run da CI
make test-unit | test-api | test-integration | test-smoke
make lint                               # ruff
make format                             # black + isort
flake8 app/ tests/ ; bandit -r app/

# Docker
make build | up | down | logs | shell   # ciclo de vida dos containers
docker compose up --build -d            # alternativa direta
```

---

## Estrutura de pastas

```
XFakeSong/
├── main.py                 # entrypoint (--gradio | --bootstrap-dirs | --deploy | --port)
├── app.py                  # entrada para Hugging Face Spaces (importa app/interfaces/gradio/app.py)
├── Makefile                # build/up/down, train-*, benchmark-*, test-*, lint/format
├── Dockerfile              # imagem multi-stage (CPU/GPU)
├── docker-compose*.yml     # compose raiz + .train / .gpu / .benchmark
├── docker/compose/         # composes segmentados (inference/train/benchmark x cpu/nvidia)
├── configs/                # YAMLs de configuracao
│   ├── dataset.yaml, inference.yaml
│   └── training/           # presets de treino por familia + retune_ajustado.yaml
├── benchmarks/             # motor do benchmark (runner, evaluate, planning, report...)
├── scripts/                # CLIs por categoria: dataset/, training/, benchmark/, reporting/, ops/ (ver scripts/README.md)
├── app/                    # codigo-fonte (Clean Architecture)
│   ├── domain/             # regras de negocio puras (sem frameworks de UI)
│   │   ├── features/       # extractors/, adapters/, registry de features
│   │   ├── models/         # architectures/, training/ (inclui secure_training_pipeline.py), inference/
│   │   ├── services/       # DetectionService, TrainingService, UploadService, ...
│   │   ├── dataset_metadata/  # dataset_catalog.py, speaker_manifest.py (catalogo/manifesto do dataset)
│   │   └── xai/            # gradcam.py, shap_explainer.py, tabular.py (explicabilidade dos modelos)
│   ├── core/               # infra transversal: config/, db/, auth/, contracts/ (interfaces SOLID)
│   ├── interfaces/         # as 3 interfaces de entrada do projeto:
│   │   ├── gradio/         # app.py (montagem, ex-gradio_app.py raiz), schema_patch.py, 5 abas role-based (tabs/), utils/
│   │   ├── cli/            # menu interativo (context.py, menus/)
│   │   └── web/            # FastAPI: main_fastapi.py, routers/, schemas/, static/, templates/
│   ├── models/             # artefatos treinados (.keras/.pkl) — benchmark_final/
│   └── utils/              # audio_utils/file_utils/helpers/silero_vad/system_utils (centrais)
│                           #   + colab.py (helper isolado so para execucao via Google Colab)
├── data/datasets/          # RAIZ CANONICA dos dados (real/, fake/, raw/, .npz) — nao versionado.
│                           #   (app/datasets/ foi descontinuado em 2026-07-14 — causava fragmentacao)
├── docs/                   # documentacao MkDocs (00_..30_, RETREINO_AJUSTES.md) — indice completo em docs/index.md
├── notebooks/, data/results/paper/   # material academico (TCC)
└── tests/                  # unit/, integration/, api/, functional/ espelhando app/
```

**Regra de ouro**: `app/domain/` nunca importa de `app/interfaces/` (gradio, cli
ou web) nem de frameworks de UI. Bibliotecas externas entram via wrapper em
`app/core/` ou `app/domain/features/adapters/`.

---

## Funcoes e modulos-chave

| Necessidade | Onde |
| --- | --- |
| Instanciar modelo por nome | `app/domain/models/architectures/factory.py` -> `create_model` |
| Registro/hiperparametros default das 14 arquiteturas | `app/domain/models/architectures/registry.py` (`ArchitectureRegistry`, `default_params`) |
| Implementacao de cada arquitetura | `architectures/<nome>.py` (`create_model(...)` compila o modelo) |
| Camadas customizadas (SincConv, GAT, AMSoftmax...) | `architectures/layers.py` |
| Orquestracao de treino | `app/domain/services/training_service.py`, `app/domain/models/training/secure_training_pipeline.py` |
| Deteccao/inferencia | `app/domain/services/detection_service.py` |
| Extracao de features | `app/domain/features/` (implementar `IFeatureExtractor`, registrar no registry) |
| Config global de treino (LR, early stop, augmentation, calibracao) | `app/core/config/settings.py` (`TrainingConfig`) |
| Motor de benchmark | `benchmarks/` (`runner.py`, `evaluate.py`, `planning.py`, `report.py`) |

---

## Build

Imagens Docker multi-stage (CPU e NVIDIA), orquestradas por composes segmentados
em `docker/compose/`. Pelo Makefile:

```bash
make build            # build com cache (SERVICE=inference-api por padrao)
make build-nocache    # build limpo + pull das imagens base
make up / make down   # sobe/derruba; app em http://localhost:7860
make docker-config    # valida a config dos perfis (inference/train/benchmark)
```

Dev local sem Docker: `make venv && make install-dev` (cria `.venv`,
instala `requirements.txt` + `requirements-dev.txt`), depois `make dev`.

---

## Treinamento

Presets por familia em `configs/training/`:
`tensorflow.yaml` (Sonic Sleuth, EfficientNet-LSTM, MultiscaleCNN,
SpectrogramTransformer), `pytorch.yaml` (RawNet2, AASIST, RawGAT-ST, Conformer,
Hybrid CNN-Transformer), `ssl.yaml` (WavLM, HuBERT) e `classical.yaml`
(SVM, RandomForest). Campos: `dataset`, `epochs`, `batch_size`, `device_profile`,
`snr`, `optimize_hyperparameters`.

Hiperparametros por modelo vivem em **tres lugares** (cuidado com drift; chaves
como `dropout_rate`/`l2_reg_strength` se sobrepoem):
1. `registry.py` (`default_params`: dropout, l2, patience, gradient_clip,
   augmentation_strength) — consumido pelo `training_service` (app/Gradio/
   `train_advanced`);
2. o `create_model(...)` de cada `architectures/<nome>.py` (LR, weight_decay,
   clipnorm, loss);
3. `benchmarks/planning.py` (`NEURAL_BENCHMARK_HPARAMS`: lr, batch, optimizer,
   scheduler, warmup, label_smoothing) — usado **pelo benchmark** quando
   `optimize_hyperparameters=True` (default). Ao ajustar um modelo, revise as 3
   fontes para nao divergir.

A config global (augmentation com `snr_range_db`, class weighting, calibracao de
temperatura, SWA, mixup) esta em `app/core/config/settings.py`.

> **WavLM/HuBERT no caminho Keras (importante para o TCC):** desde 2026-07-27 o
> caminho TF usa os **backbones SSL reais**. O `transformers` nao entrega WavLM
> em TF (e seus modelos TF nem importam com Keras 3), entao
> `architectures/ssl_backbone.py` le o **state_dict PyTorch** do checkpoint e
> reimplementa o forward em Keras — inclusive o vies posicional relativo com
> gating do WavLM. O backbone fica inteiramente congelado; treinam so a soma
> ponderada dos hidden-states (receita SUPERB) e a cabeca.
>
> O fallback CNN-1D do zero **ainda existe** para quando o checkpoint nao esta
> acessivel. Nesse caso o benchmark grava `provenance.variant =
> "*_fallback_cnn1d_scratch_nao_e_o_ssl_real"` e `ssl_backbone.pretrained =
> false` — nenhum artefato alega SSL real onde nao houve. `XFAKE_STRICT_SSL=1`
> aborta em vez de degradar. As entradas de manifesto sao "WavLM"/"HuBERT"
> (escopo **extended**); "WavLM Original"/"HuBERT Original" sao as do escopo
> oficial, treinadas pelo runner PyTorch dedicado.

Execucao:

```bash
make train-nvidia     # perfil TensorFlow/Keras em GPU (Docker)
make train-cpu        # perfil classical/CPU (Docker)

# Sequencial, um modelo por vez (timeout, --resume, log por modelo):
# Dataset canônico: benchmark_dataset.npz (Protocolo de Dataset — CETUC pareado com
# clones XTTS-v2; ver docs/data/dataset-protocol.md).
python scripts/benchmark/run_models_sequential.py \
  --dataset data/datasets/benchmark_dataset.npz \
  --models AASIST Ensemble --epochs 100 --snr 30 20 10 5 \
  --device-profile gpu --out data/results/<run> --resume

# Um modelo isolado:
python scripts/benchmark/run_benchmark.py --model AASIST --dataset <npz> --out data/results/bench_aasist
```

Pre-requisitos: dataset `.npz` em `data/datasets/`, TensorFlow/PyTorch e GPU
(ver [docs/models/training.md](docs/models/training.md)).

### Retreino dos modelos ajustados
Apos diagnostico de um benchmark, os ajustes ficam aplicados no codigo
(registry + arquiteturas) e o retreino dos modelos afetados roda por:
`bash scripts/training/retrain_ajustado.sh` (ou `scripts\training\retrain_ajustado.bat` no
Windows), config em `configs/training/retune_ajustado.yaml`. Mapa
diagnostico->ajuste e checklist de verificacao em
[docs/evaluation/retraining-adjustments.md](docs/evaluation/retraining-adjustments.md).

---

## Benchmark

Motor em `benchmarks/`; orquestracao em `scripts/`. As 14 arquiteturas sao
cobertas em DOIS escopos (`benchmarks/config.py`):

- **oficial** (`--experiment-scope official`, default): 2 classicas (SVM,
  RandomForest) + 7 neurais Keras (RawNet2, AASIST, RawGAT-ST, Conformer,
  Hybrid CNN-Transformer, SpectrogramTransformer, MultiscaleCNN) com
  hiperparametros de `planning.py::NEURAL_BENCHMARK_HPARAMS`, mais
  **WavLM Original** e **HuBERT Original**, que rodam por um runner PyTorch
  separado (`scripts/benchmark/run_wavlm_original_benchmark.py`) — sao os SSL
  REAIS, nao o fallback TF;
- **estendido** (`--experiment-scope extended`): Sonic Sleuth,
  EfficientNet-LSTM e Ensemble. Esse escopo forca `optimize_hyperparameters=
  False`, entao nao passa por `NEURAL_BENCHMARK_HPARAMS` (pedir essas tres no
  escopo oficial e erro de configuracao, nao limitacao).

Avalia em condicoes limpas e sob ruido
(SNR 30/20/10 dB casados com o augmentation de treino, mais 5 dB
NAO VISTO — a coluna que mede generalizacao a ruido), gerando metricas (accuracy,
precision, recall, f1, AUC-ROC, EER, min t-DCF), eficiencia (params, MB,
latencia) e artefatos (figuras, tabelas LaTeX, summary.md, tcc_report.md).

```bash
make benchmark-nvidia                       # benchmark completo (Docker/WSL2 GPU)
python scripts/benchmark/run_benchmark.py --full --dataset <npz>
python scripts/benchmark/run_clean_benchmark_pipeline.py   # run limpo, sem misturar artefatos
python scripts/benchmark/run_benchmark.py --plan-only      # valida e grava benchmark_plan.* sem treinar

# Pos-processamento:
python scripts/reporting/consolidate_results.py --results data/results/<run>
python scripts/reporting/validate_artifacts.py  --results data/results/<run>
python scripts/reporting/sync_completed_benchmark_artifacts.py --results data/results/<run>
```

Os artefatos promovidos ficam em `data/models/benchmark_final/<arch>/` com
`data/results/` (metrics.json, results.csv/json, predictions_clean.csv,
robustness.csv, figuras, tabelas .tex) e `index.json` consolidado.
Promova um modelo apenas se ele melhorar (ou empatar) o baseline, com atencao
especial a robustez a 10 dB.

Scripts uteis: `build_dataset.py`, `preprocess_dataset.py`,
`audit_dataset_leakage.py`, `robustness_test.py`, `benchmark_latency.py`,
`export_model_card.py`, `update_tcc_latex.py`.

---

## Edicao e convencoes

- Formatacao: `black` (max 88 colunas) + `isort`; lint via `ruff`/`flake8`;
  seguranca via `bandit`. Rode `make format` antes de commitar.
- Type hints obrigatorios em funcoes publicas; logging via
  `logging.getLogger(__name__)` — nunca `print()`.
- Novas regras de negocio em `app/domain/` sem dependencia de UI/frameworks.
- Nova feature: implementar `IFeatureExtractor` e registrar no
  `FeatureExtractorRegistry`.
- Nova arquitetura: criar `architectures/<nome>.py` com `create_model(...)` e
  registrar no `ArchitectureRegistry` (registry.py) com `default_params` e
  `input_requirements`.
- Ajuste de hiperparametros: preferir `registry.default_params` (dropout, l2,
  patience, gradient_clip, augmentation_strength); LR/optimizer/loss no
  `create_model` da arquitetura; flags globais em `settings.py`.
- Testes espelham `app/` em `tests/unit|integration|api/`; rode `make test`
  (a CI usa essa suite rapida) antes de abrir PR.
- `.gitignore` ignora `data/results/`, `data/models/*.keras|*.pkl`,
  `logs/` e caches — artefatos de treino sao regeneraveis, nao versione.
- Copie `.env.example` para `.env` antes de executar.
