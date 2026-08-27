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
├── unit/                # 69 arquivos
├── api/                 # 5 arquivos
├── functional/          # 2 arquivos
├── integration/         # 7 arquivos
└── smoke/               # 5 arquivos, opt-in
```

Total atual: **94 arquivos de teste**.

| Categoria | Marcador | Arquivos | Objetivo | Run padrão |
|---|---:|---:|---|---|
| Unit | `unit` | 74 | Componentes isolados, utilitários, treinamento, benchmark, notebooks, segurança local | Sim |
| API | `api` | 5 | Contratos FastAPI com `TestClient` e serviços mockados | Sim |
| Functional | `functional` | 2 | Fluxos de usuário e rotas/frontend | Sim |
| Integration | `integration` | 7 | Cooperação real entre serviços, podendo treinar modelos pequenos | Sim |
| Smoke | `smoke` | 6 | Sanidade ponta a ponta com TensorFlow real, modelos e app | Não |

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

68 arquivos, organizados por **sujeito**. A lista abaixo é o mapa: cada linha
diz qual módulo está sob contrato, não em que semana o teste foi escrito.

**Arquiteturas e camadas**

| Arquivo | Sujeito |
|---|---|
| `test_architectures.py` | registry, factory, e a sincronia das TRÊS fontes de hiperparâmetro |
| `test_rawgat_aasist_ssl_backends.py` | RawGAT-ST/AASIST fiéis ao paper + back-ends SSL |
| `test_sinc_layers_mixed_precision.py` | `SincConvLayer` sob `mixed_float16` |
| `test_ssl_backbone.py` | backbone SSL portado do PyTorch para Keras |
| `test_ssl_head_contract.py` | contrato de embedding SSL + augmentation dinâmico |
| `test_torch_ssl_aasist.py` | back-end AASIST em PyTorch (ablação fora do escopo) |
| `test_metrics_ocsoftmax.py` | min t-DCF, OC-Softmax e calibração |

**Treino**

| Arquivo | Sujeito |
|---|---|
| `test_trainer.py` | contratos básicos do `ModelTrainer` |
| `test_trainer_compile_respect.py` | compile-respect, `from_logits`, augmentation |
| `test_training_guards.py` | `CollapseAbort` e `PersistentEpochHistory` |
| `test_checkpoint_monitor.py` | `checkpoint_monitor`, `ValidationEER` e o encanamento |
| `test_resumable_checkpoint.py` | melhor valor sobrevive a retomadas |
| `test_guarded_checkpoint_restore.py` | restauração validada contra os pesos em memória |
| `test_train_save_load_roundtrip.py` | treino → salvar → carregar → prever |
| `test_classical_fit.py` | SVM/RF construídos pela factory treinam de fato |
| `test_specaugment_ssl_finetune.py` | SpecAugment e descongelamento parcial dos SSL |
| `test_retraining_adjustments.py` | split por fonte + ruído SNR do retreino |
| `test_perf_optimizations.py` | otimizações neutras em acurácia |

**Benchmark**

| Arquivo | Sujeito |
|---|---|
| `test_benchmark.py` | split, AWGN, métricas, relatório + smoke SVM ponta a ponta |
| `test_benchmark_families.py` | escopos oficial x estendido |
| `test_benchmark_frontend.py` | paridade treino↔inferência (raw / log-Mel / tabular v1 e v2) |
| `test_benchmark_partition_integrity.py` | disjunção locutor x frase; cross-generator |
| `test_benchmark_protocol_fixes.py` | contratos de protocolo e relatório (multicrop, limiar, timeout) |
| `test_benchmark_protocol_guards.py` | test-lock, preflight do NPZ, congelamento do teste |
| `test_benchmark_provenance.py` | variante, git, versões e checkpoints declarados |
| `test_benchmark_reporting_fidelity.py` | o que o JSON declara bate com o que o run fez |
| `test_benchmark_seed_repetitions.py` | repetições por semente e publicação da incerteza |
| `test_classical_retune.py` | grid de fonte única, CV agrupada, calibração isotônica |
| `test_academic_rigor_metrics.py` | IC bootstrap, ECE e sanitização de scores |
| `test_run_summary_rebuild.py` | regeneração do `run_summary` a partir dos `results.json` |
| `test_stft_coverage.py` | nenhuma amostra invisível ao log-mel |
| `test_band_correction.py` | remoção da assinatura de reamostragem entre as classes |
| `test_lfcc_frontend_rawboost.py` | front-end LFCC e augmentation RawBoost |

**Dataset**

| Arquivo | Sujeito |
|---|---|
| `test_dataset_protocol.py` | Protocolo de Dataset (CETUC pareado com clones XTTS-v2) |
| `test_dataset_catalog.py` | catálogo de locutores, frases e origens |
| `test_dataset_pipeline_regressions.py` | balanceamento, janela e proveniência |
| `test_build_dataset.py` | corpus bruto → `.npz` |
| `test_split_speaker_disjoint.py` | o split da interface não repete amostra nem falante |

**Inferência e serviços**

| Arquivo | Sujeito |
|---|---|
| `test_detection_model_loader_predictor.py` | resolver artefato, carregar e pontuar |
| `test_detection_utils.py` | utilitários do serviço de detecção |
| `test_audio_resample_safety.py` | rede de segurança de reamostragem |
| `test_device_support.py` | seleção de dispositivo CPU/GPU |
| `test_gpu_diagnosis.py` | matriz de diagnóstico + probe read-only |
| `test_upload_service.py` | criação de dataset e recepção de arquivos |
| `test_experiment_store.py` | persistência científica consolidada |
| `test_results_paths.py` | raiz canônica de resultados |
| `test_clean_bootstrap.py` | primeira execução sem artefatos, idempotente |

**Interfaces**

| Arquivo | Sujeito |
|---|---|
| `test_gradio_tabs.py` | cada aba constrói e degrada sem derrubar o resto |
| `test_training_wizard_feedback.py` | custo à frente, parada e sobreajuste no assistente |
| `test_training_charts.py` | figura de curvas de treino |
| `test_tuning_charts.py` | gráficos de busca de hiperparâmetros |
| `test_interface_uses_pipeline_hparams.py` | a interface propõe o que o benchmark treina |
| `test_forensic_math.py` | correção das medidas da análise forense |
| `test_i18n.py` | i18n da UI |
| `test_schemas.py` | schemas Pydantic da API |
| `test_middleware.py` | cadeia de middleware HTTP |

**XAI**

| Arquivo | Sujeito |
|---|---|
| `test_xai.py` | Grad-CAM, SHAP e contrato tabular (63 no v1, 183 no v2) |

**Infraestrutura e segurança**

| Arquivo | Sujeito |
|---|---|
| `test_security_boundaries.py` | autenticação/autorização no ponto de entrada |
| `test_security_headers.py` | headers de segurança em todas as respostas |
| `test_exceptions.py` | hierarquia de exceções |
| `test_file_utils.py`, `test_audio_utils.py`, `test_system_utils.py`, `test_helpers.py`, `test_core_utils.py` | utilitários |
| `test_version_check.py` | guard de compatibilidade no startup |
| `test_colab_utils.py` | helper isolado do Google Colab |
| `test_notebooks_compile.py` | notebooks ativos seguem compilando |
| `test_test_documentation.py` | esta página não desatualiza (ver convenção abaixo) |

## Convenção de nomes e docstrings

Adotada em 2026-08-17 e verificada por
`tests/unit/test_test_documentation.py`.

**1. O nome do arquivo é o SUJEITO, nunca o episódio.** Um teste vive muito
mais tempo do que a correção que o motivou. Nomes como `test_p1_*`,
`test_tier1_*` ou "as correções de tal data" envelhecem em semanas e, pior,
escondem que dois arquivos cobrem a mesma coisa.

O caso concreto que fixou a regra: `CollapseAbort` era coberto por
`test_resume_guards_and_artifacts.py` (agrupado pela data 2026-08-06) e por
`test_collapse_never_learns.py`. Como nada no nome dizia que eram o mesmo
sujeito, um prazo novo acrescentado ao callback passou a contradizer uma
regressão do outro arquivo — e só a suíte inteira revelou o conflito. Hoje é
um arquivo só, `test_training_guards.py`.

**2. Todo arquivo abre com docstring de módulo** que diz o sujeito na primeira
linha e, no corpo, o que está sob contrato. Quando o teste nasceu de um
incidente, o incidente entra como MOTIVAÇÃO no corpo — não no nome.

**3. Nome de teste único no repositório.** `pytest` aceita homônimos em
arquivos diferentes, mas isso quebra a seleção por `-k` e esconde duplicação
real. Quando dois arquivos testam a mesma rota em camadas diferentes, o nome
diz qual é qual (`test_create_dataset_via_api` x `test_create_dataset_no_servico`).

**4. Um sujeito, um arquivo.** Se um arquivo precisa de seções separadas por
assunto, provavelmente são dois arquivos. A exceção é o par
publicador/consumidor que só faz sentido junto — como o `ValidationEER` e o
monitor de checkpoint, que compartilham o mesmo defeito de ordem.

**5. `pytest.importorskip` no módulo, não no helper.** Se o módulo sob teste
importa TensorFlow no topo, a dependência é do arquivo inteiro: repetir a
guarda em cada função mascara isso e deixa a coleção quebrar.

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
test_calibration_matches_saved_weights.py
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
| `_isolate_models_dir` | **autouse, escopo de sessão**: aponta `XFAKE_MODELS_DIR` para um diretório temporário |

`_isolate_models_dir` não é conveniência, é proteção. `benchmarks/runner.py::
_models_dir` cai em `cfg.models_dir`, cujo default é `data/models` — o
diretório de PRODUÇÃO —, e onze `BenchmarkConfig` da suíte não passam
`models_dir`. Foi assim que o `bench_svm.pkl` do `clean_benchmark_15k` (63
features, 3,6 MB) virou um artefato de smoke de 47 KB: as métricas do run
sobreviveram, o modelo não. Como `_models_dir` consulta as variáveis de
ambiente ANTES de `cfg`, apontar uma delas cobre todos os caminhos de uma vez.

Convenções:

- use `tmp_path`/`tmp_path_factory` para arquivos temporários;
- não grave em `data/models/`, `data/results/` ou datasets reais — a fixture
  acima cobre o caso do `models_dir`, mas um caminho escrito à mão escapa dela;
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
| `.github/workflows/ci.yml` | push, PR, manual | ruff advisório, testes+cobertura, segurança, docs, Docker CPU em PR — **não** roda notebooks |
| `.github/workflows/static.yml` | push na `main`, manual | build e deploy da documentação no GitHub Pages |
| `.github/workflows/notebooks-execute.yml` | manual (`workflow_dispatch`) | execução best-effort de notebooks self-contained |

O drift de notebooks (`build_notebooks.py` + `git diff --exit-code`, abaixo)
é um gate **local**, não roda em nenhum workflow do GitHub Actions — rode
manualmente antes de abrir PR se tocou em notebooks.

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
