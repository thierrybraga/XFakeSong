# XFakeSong — Guia para Claude Code

XFakeSong e uma plataforma open source de deteccao de audio deepfake (analise
espectral classica + Transformers SSL como WavLM/HuBERT), com foco em XAI e
processamento 100% local. Interface Gradio, Python 3.11+, licenca MIT.

Este arquivo e o guia operacional para agentes de codigo. A fonte canonica de
arquitetura e convencoes e o [AGENTS.md](AGENTS.md); os detalhes tecnicos estao
em `docs/`. Em caso de divergencia, AGENTS.md + docs/ prevalecem.

| Tema | Fonte canonica |
| --- | --- |
| Arquitetura e limites de camadas | [AGENTS.md](AGENTS.md), [docs/03_ARQUITETURA.md](docs/03_ARQUITETURA.md) |
| Desenvolvimento e padroes | [docs/05_GUIA_DEV.md](docs/05_GUIA_DEV.md) |
| Testes | [docs/06_TESTES.md](docs/06_TESTES.md) |
| Arquiteturas neurais | [docs/08_ARQUITETURAS.md](docs/08_ARQUITETURAS.md) |
| Treinamento e hiperparametros | [docs/10_TREINAMENTO.md](docs/10_TREINAMENTO.md) |
| Benchmark e metricas | [docs/15_BENCHMARK.md](docs/15_BENCHMARK.md) |
| Datasets | [docs/12_DATASETS.md](docs/12_DATASETS.md) |
| Retreino apos diagnostico | [docs/RETREINO_AJUSTES.md](docs/RETREINO_AJUSTES.md) |

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
├── app.py                  # entrada para Hugging Face Spaces
├── gradio_app.py           # montagem da interface Gradio
├── Makefile                # build/up/down, train-*, benchmark-*, test-*, lint/format
├── Dockerfile              # imagem multi-stage (CPU/GPU)
├── docker-compose*.yml     # compose raiz + .train / .gpu / .benchmark
├── docker/compose/         # composes segmentados (inference/train/benchmark x cpu/nvidia)
├── configs/                # YAMLs de configuracao
│   ├── dataset.yaml, inference.yaml
│   └── training/           # presets de treino por familia + retune_ajustado.yaml
├── benchmarks/             # motor do benchmark (runner, evaluate, planning, report...)
├── scripts/                # CLIs de build/dataset/treino/benchmark/sync (ver abaixo)
├── app/                    # codigo-fonte (Clean Architecture)
│   ├── domain/             # regras de negocio puras (sem frameworks de UI)
│   │   ├── features/       # extractors/, adapters/, registry de features
│   │   ├── models/         # architectures/, training/, inference/
│   │   └── services/       # DetectionService, TrainingService, UploadService, ...
│   ├── application/        # casos de uso + pipeline (stages/workflows/dto)
│   ├── core/               # infra transversal: config/, training/, utils/, interfaces/
│   ├── interfaces/         # entrada: gradio/ (8 abas) e cli/
│   ├── routers/            # endpoints FastAPI
│   ├── schemas/            # modelos Pydantic (request/response)
│   ├── models/             # artefatos treinados (.keras/.pkl) — benchmark_final/
│   └── datasets/           # dados (real/ e fake/) e .npz de benchmark (nao versionado)
├── docs/                   # documentacao MkDocs (01_..15_, RETREINO_AJUSTES.md)
├── notebooks/, figures/, tcc_overleaf/   # material academico (TCC)
└── tests/                  # unit/, integration/, api/ espelhando app/
```

**Regra de ouro**: `app/domain/` nunca importa de `app/interfaces/`, `app/routers/`
ou frameworks de UI. Bibliotecas externas entram via wrapper em `app/core/` ou
`app/domain/features/adapters/`.

---

## Funcoes e modulos-chave

| Necessidade | Onde |
| --- | --- |
| Instanciar modelo por nome | `app/domain/models/architectures/factory.py` -> `create_model` |
| Registro/hiperparametros default das 14 arquiteturas | `app/domain/models/architectures/registry.py` (`ArchitectureRegistry`, `default_params`) |
| Implementacao de cada arquitetura | `architectures/<nome>.py` (`create_model(...)` compila o modelo) |
| Camadas customizadas (SincConv, GAT, AMSoftmax...) | `architectures/layers.py` |
| Orquestracao de treino | `app/domain/services/training_service.py`, `app/core/training/` |
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

> **Caveat WavLM/HuBERT (importante para o TCC):** o caminho TF do benchmark usa
> **fallback CNN-1D treinado do zero**, nao os backbones SSL reais. WavLM e
> *sempre* fallback (PyTorch-only, sem conversao TF). HuBERT tenta o backbone
> real (`from_pt=True`) e cai no simplificado se indisponivel. Logo, resultados
> rotulados "WavLM/HuBERT" no benchmark TF nao refletem os modelos SSL originais
> — reporte isso ao comparar com a literatura. Os artefatos `*_original.pt`
> (PyTorch) sao os reais, usados so na inferencia/demonstracao do Gradio.

Execucao:

```bash
make train-nvidia     # perfil TensorFlow/Keras em GPU (Docker)
make train-cpu        # perfil classical/CPU (Docker)

# Sequencial, um modelo por vez (timeout, --resume, log por modelo):
python scripts/run_models_sequential.py \
  --dataset app/datasets/benchmark_audio_raw_balanced_15k.npz \
  --models AASIST Ensemble --epochs 100 --snr 30 20 10 \
  --device-profile gpu --out results/<run> --resume

# Um modelo isolado:
python scripts/run_benchmark.py --model AASIST --dataset <npz> --out results/bench_aasist
```

Pre-requisitos: dataset `.npz` em `app/datasets/`, TensorFlow/PyTorch e GPU
(ver [docs/10_TREINAMENTO.md](docs/10_TREINAMENTO.md)).

### Retreino dos modelos ajustados
Apos diagnostico de um benchmark, os ajustes ficam aplicados no codigo
(registry + arquiteturas) e o retreino dos modelos afetados roda por:
`bash scripts/retrain_ajustado.sh` (ou `scripts\retrain_ajustado.bat` no
Windows), config em `configs/training/retune_ajustado.yaml`. Mapa
diagnostico->ajuste e checklist de verificacao em
[docs/RETREINO_AJUSTES.md](docs/RETREINO_AJUSTES.md).

---

## Benchmark

Motor em `benchmarks/`; orquestracao em `scripts/`. Avalia as 14 arquiteturas
em condicoes limpas e sob ruido (SNR 30/20/10 dB), gerando metricas (accuracy,
precision, recall, f1, AUC-ROC, EER, min t-DCF), eficiencia (params, MB,
latencia) e artefatos (figuras, tabelas LaTeX, summary.md, tcc_report.md).

```bash
make benchmark-nvidia                       # benchmark completo (Docker/WSL2 GPU)
python scripts/run_benchmark.py --full --dataset <npz>
python scripts/run_clean_benchmark_pipeline.py   # run limpo, sem misturar artefatos
python scripts/run_benchmark.py --plan-only      # valida e grava benchmark_plan.* sem treinar

# Pos-processamento:
python scripts/consolidate_results.py --results results/<run>
python scripts/validate_artifacts.py  --results results/<run>
python scripts/sync_completed_benchmark_artifacts.py --results results/<run>
```

Os artefatos promovidos ficam em `app/models/benchmark_final/<arch>/` com
`results/` (metrics.json, results.csv/json, predictions_clean.csv,
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
- `.gitignore` ignora `results/`, `app/results/`, `app/models/*.keras|*.pkl`,
  `logs/` e caches — artefatos de treino sao regeneraveis, nao versione.
- Copie `.env.example` para `.env` antes de executar.
