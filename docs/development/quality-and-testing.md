# 06 — Qualidade e Testes

Esta é a fonte canônica da estratégia de testes do XFakeSong. A suíte é
organizada por categoria, usa marcadores automáticos por pasta e separa o que é
rápido do que é pesado com TensorFlow real.

## TL;DR

```bash
./scripts/ops/run_tests.sh fast          # suíte rápida: tudo exceto smoke
./scripts/ops/run_tests.sh unit          # só unitários
./scripts/ops/run_tests.sh api           # contrato HTTP
./scripts/ops/run_tests.sh integration   # integração entre serviços
./scripts/ops/run_tests.sh smoke         # opt-in: TensorFlow real, minutos
./scripts/ops/run_tests.sh cov           # rápida + cobertura app/benchmarks
```

Equivalentes principais:

```bash
make test
make test-unit
make test-api
make test-integration
make test-smoke
make test-cov
```

## Estrutura

```text
tests/
├── conftest.py          # fixtures globais + marcação automática por pasta
├── unit/                # 44 arquivos
├── api/                 # 5 arquivos
├── functional/          # 2 arquivos
├── integration/         # 6 arquivos
└── smoke/               # 5 arquivos, opt-in
```

Total atual: **71 arquivos de teste**.

| Categoria | Marcador | Arquivos | Objetivo | Run padrão |
|---|---:|---:|---|---|
| Unit | `unit` | 53 | Componentes isolados, utilitários, treinamento, benchmark, notebooks, segurança local | Sim |
| API | `api` | 5 | Contratos FastAPI com `TestClient` e serviços mockados | Sim |
| Functional | `functional` | 2 | Fluxos de usuário e rotas/frontend | Sim |
| Integration | `integration` | 6 | Cooperação real entre serviços, podendo treinar modelos pequenos | Sim |
| Smoke | `smoke` | 5 | Sanidade ponta a ponta com TensorFlow real, modelos e app | Não |

`pyproject.toml` define `addopts = "-ra -q --ignore=data --ignore=logs -m 'not smoke'"`.
Assim, `pytest tests/` não roda `tests/smoke/` por acidente.

## Contrato Frontend, API e Comunicação

A suíte cobre três fronteiras do sistema:

| Fronteira | Onde testar | O que deve falhar cedo |
|---|---|---|
| Frontend Gradio | `tests/functional/` e testes unitários de utilitários da UI | rotas quebradas, callbacks sem serviço mockável, gráficos/estados inválidos |
| API FastAPI | `tests/api/` | OpenAPI inválido, rotas ausentes, status HTTP incompatível, schemas Pydantic regressivos |
| Comunicação entre camadas | `tests/integration/` | import indevido de web layer pelo domínio, descoberta de modelos, fluxo DetectionService/TrainingService |

O domínio continua sendo a linha mais protegida: `tests/integration/test_domain_imports_without_web_layer.py`
garante que `app/domain/` carrega sem depender de FastAPI, Gradio ou adapters de
interface.

## Marcadores

Os marcadores são aplicados automaticamente por `tests/conftest.py`, usando a
pasta do arquivo:

```python
_TEST_CATEGORIES = ("unit", "api", "functional", "integration", "smoke")
```

Não marque testes manualmente se eles já vivem em uma dessas pastas. Para criar
uma nova categoria, atualize em conjunto:

1. subpasta em `tests/`;
2. `_TEST_CATEGORIES` em `tests/conftest.py`;
3. `markers` em `pyproject.toml`;
4. `scripts/ops/run_tests.sh`;
5. esta página.

## Mapa de Cobertura

### `tests/unit/`

Cobertura principal:

- utilitários core, arquivos, áudio, resample, schemas, exceções e middleware;
- modelos e treino: architectures, trainer, save/load, classical fit,
  RawBoost, mixed precision, device/GPU support;
- benchmark: dados, métricas, relatórios, API probe, gráficos e artefatos TCC;
- paridade treino<->inferencia do front-end do benchmark
  (`tests/unit/test_benchmark_frontend.py`);
- XAI: contrato tabular (63 descritores), Grad-CAM (2D/1D, busca
  automática de camada) e wrappers SHAP (`tests/unit/test_xai.py`);
- notebooks: estrutura, contratos de entrada, geração e compilação;
- segurança local: headers, file utils e validações auxiliares;
- melhorias P1/P2/P3: SpecAugment/SSL, RawGAT-ST/AASIST, min-tDCF/OC-Softmax.

Arquivos unitários atuais:

```text
test_architectures.py
test_audio_resample_safety.py
test_audio_utils.py
test_benchmark.py
test_build_dataset.py
test_classical_fit.py
test_colab_utils.py
test_core_utils.py
test_dataset_catalog.py
test_detection_utils.py
test_detection_model_loader_predictor.py
test_device_support.py
test_exceptions.py
test_file_utils.py
test_frontend_rawboost.py
test_gpu_diagnosis.py
test_helpers.py
test_i18n.py
test_middleware.py
test_notebooks_compile.py
test_p1_specaug_ssl.py
test_p2_rawgatst_sslaasist.py
test_p3_metrics_ocsoftmax.py
test_retraining_adjustments.py
test_schemas.py
test_security_headers.py
test_sinc_layers_mixed_precision.py
test_system_utils.py
test_tier1_perf.py
test_test_documentation.py
test_train_save_load_roundtrip.py
test_trainer.py
test_trainer_compile_respect.py
test_training_charts.py
test_tuning_charts.py
test_upload_service.py
test_version_check.py
```

### `tests/api/`

Usa `TestClient` e fixtures de mock em `tests/conftest.py`.

```text
test_datasets.py
test_detection.py
test_features.py
test_smoke.py
test_training.py
```

`tests/api/test_smoke.py` é smoke **de contrato HTTP leve** e recebe marcador
`api`. O marcador `smoke` fica reservado para `tests/smoke/`.

### `tests/functional/`

```text
test_detection_flow.py
test_frontend_routes.py
```

### `tests/integration/`

```text
test_architectures_build.py
test_detection_integration.py
test_domain_imports_without_web_layer.py
test_models_dir_unification.py
test_notebook_train_system_inference.py
test_training_integration.py
```

### `tests/smoke/`

```text
test_all_architectures.py
test_app_startup.py
test_inference_integrated.py
test_inference_pipeline.py
test_wizard_pipeline.py
```

Esses testes constroem modelos reais e podem levar minutos. Rode quando tocar em
arquiteturas, `TrainingService`, `Predictor`, app startup, wizard ou pipeline de
inferência.

## Fixtures Globais

`tests/conftest.py` fornece:

| Fixture | Uso |
|---|---|
| `client` | `TestClient` com dependências sobrescritas por mocks |
| `mock_detection_service` | serviço de detecção mockado |
| `mock_upload_service` | upload/dataset mockado com `ProcessingResult` realista |
| `mock_training_service` | treino mockado para rotas API |
| `api_key_headers` | header `X-API-Key` válido para endpoints protegidos |

Convenções:

- use `tmp_path`/`tmp_path_factory` para arquivos temporários;
- não grave em `data/models/`, `data/results/` ou datasets reais;
- mocke rede, pesos grandes e downloads;
- force `MPLBACKEND=Agg` para testes com gráficos;
- helpers que não são testes devem começar com `_`.

## Comandos Padronizados

### Runner

`scripts/ops/run_tests.sh` escolhe automaticamente `.venv/bin/python`,
`.venv/Scripts/python.exe` ou `python`.

```bash
./scripts/ops/run_tests.sh fast
./scripts/ops/run_tests.sh unit -v -k benchmark
./scripts/ops/run_tests.sh api
./scripts/ops/run_tests.sh functional
./scripts/ops/run_tests.sh integration
./scripts/ops/run_tests.sh smoke
./scripts/ops/run_tests.sh all
./scripts/ops/run_tests.sh cov
./scripts/ops/run_tests.sh list
```

### Pytest direto

```bash
pytest tests/
pytest tests/unit/test_benchmark.py -q
pytest -m "api or functional" tests/
pytest -m smoke tests/smoke/ -v
pytest -m "" tests/              # inclui smoke, anulando addopts
```

### Cobertura

```bash
./scripts/ops/run_tests.sh cov
pytest --cov=app --cov=benchmarks --cov-report=term-missing tests/
pytest --cov=app --cov=benchmarks --cov-report=html tests/
```

`htmlcov/` é artefato local e não deve ser versionado.

## CI

Workflows ativos:

| Workflow | Quando roda | Papel |
|---|---|---|
| `.github/workflows/ci.yml` | push, PR, manual | ruff advisório, testes+cobertura, segurança, docs, drift de notebooks, Docker CPU em PR |
| `.github/workflows/static.yml` | push na `main`, manual | build e deploy da documentação no GitHub Pages |
| `.github/workflows/notebooks-execute.yml` | manual | execução best-effort de notebooks self-contained |

Gates locais equivalentes:

```bash
./scripts/ops/run_tests.sh cov
mkdocs build --strict
bandit -r app benchmarks scripts -lll
python scripts/ops/build_notebooks.py
git diff --exit-code -- notebooks/
```

## Padrão Para Novos Testes

1. Escolha a categoria pela intenção do teste.
2. Crie `tests/<categoria>/test_<assunto>.py`.
3. Use nomes `test_*` só para casos coletáveis.
4. Use fixtures existentes antes de criar novas.
5. Prefira entradas mínimas e determinísticas.
6. Adicione teste de regressão junto com correção de bug.
7. Atualize esta página se criar nova categoria, runner ou workflow.

Para mudanças de frontend/API, prefira testar a borda pública do comportamento:
rotas com `TestClient`, callbacks com serviços mockados e integrações que
comprovem o caminho entre adapter e domínio sem exigir pesos reais.

## Checklist Antes de Commit

```bash
python -m py_compile tests/conftest.py
./scripts/ops/run_tests.sh unit
./scripts/ops/run_tests.sh fast
mkdocs build --strict
```

Quando tocar em notebooks:

```bash
python scripts/ops/build_notebooks.py
pytest tests/unit/test_notebooks_compile.py -q
```

Quando tocar em modelos, treino ou inferência:

```bash
./scripts/ops/run_tests.sh smoke
```

## Armadilhas Conhecidas

- `-m 'not smoke'` deseleciona depois da coleta; erro de import em qualquer
  arquivo ainda quebra a coleta.
- `MagicMock()` cru cria atributos truthy. Ao mockar `ModelInfo`, defina
  explicitamente `onnx_session=None`, `temperature=1.0`, `eer_threshold` e
  `input_contract` quando o predictor puder acessá-los.
- TensorFlow no Windows nativo não usa GPU com versões modernas; smoke pesado
  deve preferir WSL2/Linux quando GPU for necessária.
- Notebooks ativos são gerados por `scripts/ops/build_notebooks.py`; edite o
  gerador, não só o `.ipynb`.
