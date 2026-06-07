# Caderno de Testes e Qualidade

Este caderno é a fonte canônica da estratégia de testes do **XFakeSong**.
Documenta a organização da suíte, o que cada arquivo cobre, como executar por
categoria, medir cobertura e manter a qualidade contínua.

> **TL;DR**
> ```bash
> ./scripts/run_tests.sh fast     # suíte rápida (tudo EXCETO smoke) — run padrão
> ./scripts/run_tests.sh unit     # só uma categoria
> ./scripts/run_tests.sh smoke    # smoke pesado (TF real, minutos)
> ./scripts/run_tests.sh cov       # rápida + cobertura
> # equivalente via Make: make test | make test-unit | make test-smoke | make test-cov
> ```

---

## 1. Filosofia e pirâmide de testes

A suíte segue uma pirâmide de 5 camadas, da base (rápida, isolada, executada
sempre) ao topo (lenta, integrada, opt-in):

```
        ▲  smoke         (5)  — TF real: cria 14 archs, treina 1 epoch, sobe o app
       ╱ ╲ integration   (2)  — cooperação entre serviços (pode treinar de verdade)
      ╱   ╲ functional   (2)  — fluxos de usuário e rotas do frontend
     ╱     ╲ api         (5)  — contrato HTTP (FastAPI TestClient + mocks)
    ╱_______╲ unit       (23) — componentes isolados, sem I/O pesado
```

**Total: 37 arquivos de teste.** A cada camada corresponde uma **subpasta** de
`tests/` e um **marcador pytest homônimo** (ver §3). O run padrão da CI executa
tudo **exceto** `smoke` (lento demais para cada commit).

| Camada        | Objetivo                                                              | Velocidade | No run padrão? |
| ------------- | --------------------------------------------------------------------- | ---------- | -------------- |
| `unit`        | Lógica e contratos internos, sem dependências externas reais          | rápida     | ✅ sim         |
| `api`         | Contratos HTTP estáveis (status, payloads, auth) com serviços mockados | rápida     | ✅ sim         |
| `functional`  | Fluxos de usuário e disponibilidade das rotas/UI                       | média      | ✅ sim         |
| `integration` | Cooperação real entre serviços/pipelines (pode treinar modelos)        | lenta      | ✅ sim*        |
| `smoke`       | Sanidade ponta a ponta com TensorFlow real (modelos, app, wizard)      | muito lenta | ❌ opt-in      |

\* `integration` entra no run padrão, mas alguns testes podem treinar modelos
pequenos — use `-k` para selecionar quando quiser rapidez.

---

## 2. Organização da suíte (mapa de arquivos)

```
tests/
├── conftest.py                 # fixtures globais + auto-marcação por pasta (§3)
├── unit/                       # 23 arquivos
├── api/                        # 5 arquivos
├── functional/                 # 2 arquivos
├── integration/                # 2 arquivos
└── smoke/                      # 5 arquivos + conftest.py (registra marker smoke)
```

### 2.1 `tests/unit/` — 23 arquivos

| Arquivo                                | Cobre                                                                       |
| -------------------------------------- | --------------------------------------------------------------------------- |
| `test_core_utils.py`                   | Utilidades base do core (`ProcessingResult`/`ProcessingStatus`, validações) |
| `test_schemas.py`                      | Schemas Pydantic da API (validação de payloads e modelos de resposta)       |
| `test_middleware.py`                   | Middlewares FastAPI (API Key, TrustedHost, tratamento de erro)              |
| `test_version_check.py`                | `app/core/version_check.py` — guarda da incompatibilidade gradio×starlette  |
| `test_upload_service.py`               | `AudioUploadService` (criar/deletar dataset, contrato `ProcessingResult`)   |
| `test_trainer.py`                      | `ModelTrainer` (callbacks, class weighting, remap de labels, save)          |
| `test_architectures.py`                | Construção + forward pass das arquiteturas neurais do registry              |
| `test_classical_fit.py`               | Pipeline clássico SVM / RandomForest (fit + calibração)                     |
| `test_train_save_load_roundtrip.py`    | Round-trip `train → save → load → predict` (contrato de inferência)         |
| `test_tier1_perf.py`                   | Otimizações Tier-1 (save sem optimizer, cache de features, warm-up, ONNX)   |
| `test_sinc_layers_mixed_precision.py`  | `SincNetLayer` estável em `mixed_float16` (sem NaN/overflow)                |
| `test_audio_resample_safety.py`        | Resample/normalização de áudio sem NaN/Inf nem clipping                     |
| `test_device_support.py`               | Seleção de dispositivo (CPU/GPU) no predictor e utils                       |
| `test_gpu_diagnosis.py`                | `app/core/gpu.py` — `setup_gpu`/`describe`/diagnóstico (idempotente)        |
| `test_frontend_rawboost.py`            | RawBoost ligado ao frontend de treino raw-audio                             |
| `test_p1_specaug_ssl.py`               | P1 — SpecAugment + fine-tuning parcial de SSL (WavLM/HuBERT)                 |
| `test_p2_rawgatst_sslaasist.py`        | P2 — RawGAT-ST fiel ao paper + variantes SSL→AASIST                         |
| `test_p3_metrics_ocsoftmax.py`         | P3 — métricas min-tDCF (ASVspoof) e camada/loss OC-Softmax                  |
| `test_benchmark.py`                    | Pacote `benchmarks/` (config, dados sintéticos, evaluate, efficiency)       |
| `test_training_charts.py`              | Geração dos gráficos de treino (matplotlib, backend Agg)                    |
| `test_tuning_charts.py`                | Gráficos do Optuna (convergência + importância de hiperparâmetros)          |
| `test_colab_utils.py`                  | Utilitários do guia Colab (`docs/13`)                                       |
| `test_notebooks_compile.py`            | Notebooks ativos compilam + estrutura (index + features + 14 modelos + 3 pipeline) |

### 2.2 `tests/api/` — 5 arquivos

Todos usam o `TestClient` do FastAPI com os serviços **mockados** via
`dependency_overrides` (ver fixtures em §4). São rápidos — não tocam TensorFlow.

| Arquivo              | Cobre                                                                 |
| -------------------- | --------------------------------------------------------------------- |
| `test_detection.py`  | Rotas `/detection` (single, multi-model fusion, uncertainty/MC dropout) |
| `test_training.py`   | Rotas `/training` (treinar, cross-validate, status)                   |
| `test_datasets.py`   | Rotas `/datasets` (criar, listar, deletar)                            |
| `test_features.py`   | Rotas `/features` (extração e metadados)                              |
| `test_smoke.py`      | **Smoke de contrato** (rápido): 33 rotas + OpenAPI + schemas presentes |

> ⚠️ `tests/api/test_smoke.py` é um *smoke de contrato HTTP* (leve) e recebe o
> marcador **`api`** (não `smoke`). O marcador `smoke` é exclusivo da pasta
> `tests/smoke/` (TF real). Não confundir.

### 2.3 `tests/functional/` — 2 arquivos

| Arquivo                    | Cobre                                                              |
| -------------------------- | ----------------------------------------------------------------- |
| `test_detection_flow.py`   | Fluxo `DetectionService.detect_single` com modelo falso (mock)    |
| `test_frontend_routes.py`  | Disponibilidade das rotas Gradio montadas sobre o FastAPI         |

### 2.4 `tests/integration/` — 2 arquivos

| Arquivo                          | Cobre                                                          |
| -------------------------------- | -------------------------------------------------------------- |
| `test_detection_integration.py`  | Detecção end-to-end com cooperação entre serviços reais        |
| `test_training_integration.py`   | Treino end-to-end (pode treinar um modelo pequeno — mais lento) |

### 2.5 `tests/smoke/` — 5 arquivos (opt-in, TF real)

Constroem modelos TensorFlow reais e executam pipelines completos. Levam
**minutos** e ficam **fora** do run padrão (ver §3 e §5).

| Arquivo                       | Cobre                                                                   |
| ----------------------------- | ----------------------------------------------------------------------- |
| `test_all_architectures.py`   | Cria + forward pass de **todas as 14 arquiteturas**                     |
| `test_wizard_pipeline.py`     | Pipeline do training wizard: treina **1 epoch** por arquitetura          |
| `test_inference_pipeline.py`  | Pipeline de inferência (extract → prepare → predict)                    |
| `test_inference_integrated.py`| Inferência integrada ponta a ponta (serviço + modelo)                   |
| `test_app_startup.py`         | App sobe (FastAPI + Gradio montados, `setup_gpu`) sem exceções          |

> Cada arquivo smoke pode ser rodado **como script** (`python tests/smoke/<arquivo>.py`)
> com `--models` para filtrar arquiteturas, além de via pytest. Funções
> auxiliares internas usam prefixo `_` (ex.: `_check_architecture`,
> `_train_one_epoch`) **propositalmente**, para não serem coletadas como testes —
> o teste pytest real é sempre `test_smoke_*`.

---

## 3. Marcadores (markers) e categorias

Os marcadores espelham as subpastas e são **aplicados automaticamente** — você
**não** precisa anotar os 37 arquivos. O hook em `tests/conftest.py` marca cada
teste pela pasta em que vive:

```python
# tests/conftest.py
_TEST_CATEGORIES = ("unit", "api", "functional", "integration", "smoke")

def pytest_collection_modifyitems(config, items):
    """Aplica o marcador da categoria (pasta) a cada item coletado."""
    for item in items:
        parts = set(pathlib.Path(str(item.fspath)).parts)
        for cat in _TEST_CATEGORIES:
            if cat in parts:
                item.add_marker(getattr(pytest.mark, cat))
                break
```

Os 5 marcadores estão registrados em `pyproject.toml` (`[tool.pytest.ini_options]
→ markers`), então `pytest --markers` os lista e não há *warning* de marker
desconhecido.

Selecionar por categoria com `-m`:

```bash
pytest -m unit                  # só unit
pytest -m "api or functional"   # combinar
pytest -m "not smoke"           # tudo menos smoke (é o default, ver abaixo)
```

### Exclusão do smoke no run padrão

`pyproject.toml` injeta `-m 'not smoke'` em `addopts`, então `pytest tests/`
**nunca** roda smoke por acidente:

```toml
[tool.pytest.ini_options]
addopts = "-ra -q --ignore=data --ignore=logs -m 'not smoke'"
testpaths = ["tests"]
```

Para rodar smoke explicitamente, sobrescreva o filtro:

```bash
pytest -m smoke tests/smoke/         # só smoke
pytest -m "" tests/                  # TUDO (anula o filtro do addopts)
```

---

## 4. Fixtures globais (`tests/conftest.py`)

Além do hook de auto-marcação, o conftest provê as fixtures usadas pela camada
`api`:

| Fixture                  | O que entrega                                                              |
| ------------------------ | -------------------------------------------------------------------------- |
| `client`                 | `TestClient` com `get_detection/upload/training_service` sobrescritos por mocks |
| `mock_detection_service` | `DetectionService` mockado (modelos/arquiteturas disponíveis falsos)       |
| `mock_upload_service`    | `AudioUploadService` mockado (retorna `ProcessingResult[DatasetMetadata]`) |
| `mock_training_service`  | `TrainingService` mockado (treino retorna métricas falsas)                 |
| `api_key_headers`        | Header `X-API-Key` válido (a env `XFAKESONG_API_KEY` é setada no import)   |

`tests/smoke/conftest.py` apenas reforça o registro do marcador `smoke` e
documenta como rodar a camada.

**Convenções de isolamento**: use `tmp_path`/`tmp_path_factory` para não tocar o
ambiente real; mocke o que for caro (pesos de modelo, I/O de disco); endpoints
sensíveis exigem `api_key_headers`.

---

## 5. Como executar

### 5.1 Runner padronizado — `scripts/run_tests.sh`

O script escolhe automaticamente o Python do venv (`.venv/bin/python` ou
`.venv/Scripts/python.exe` no Windows) e expõe subcomandos por categoria. Args
extras são repassados ao pytest.

```bash
./scripts/run_tests.sh fast          # tudo EXCETO smoke (run padrão da CI)
./scripts/run_tests.sh unit          # -m unit  tests/unit
./scripts/run_tests.sh api           # -m api   tests/api
./scripts/run_tests.sh functional    # -m functional tests/functional
./scripts/run_tests.sh integration   # -m integration tests/integration
./scripts/run_tests.sh smoke         # -m smoke tests/smoke  (TF real, minutos)
./scripts/run_tests.sh all           # -m ""    tests/        (inclui smoke)
./scripts/run_tests.sh cov           # rápida + cobertura (app + benchmarks)
./scripts/run_tests.sh list          # --collect-only (não executa)

# args extras: verbose, filtro por nome, parar no 1º erro…
./scripts/run_tests.sh unit -v -k roundtrip
./scripts/run_tests.sh fast -x
```

### 5.2 Via Makefile

Os targets delegam ao runner (mesma semântica):

```bash
make test              # = run_tests.sh fast
make test-unit         # make test-api / test-functional / test-integration
make test-smoke        # smoke pesado
make test-all          # inclui smoke
make test-cov          # cobertura
```

### 5.3 pytest direto

```bash
pytest tests/                                   # run padrão (sem smoke)
pytest tests/unit/test_trainer.py -v            # um arquivo
pytest tests/ -k "detection and not integration"  # por nome
pytest -m smoke tests/smoke/ -v                 # só smoke
```

---

## 6. Cobertura de código

```bash
./scripts/run_tests.sh cov
# equivale a:
pytest --cov=app --cov=benchmarks --cov-report=term-missing tests/
```

- A cobertura abrange `app/` **e** `benchmarks/`.
- Para HTML navegável: acrescente `--cov-report=html` (gera `htmlcov/`, já no
  `.gitignore`).
- Smoke não entra na medição padrão de cobertura (lento e opt-in).

---

## 7. Integração Contínua (CI)

- O único workflow atual (`.github/workflows/static.yml`) publica a documentação
  (MkDocs/Pages); **não** executa pytest.
- **Recomendação** para um job de testes: rodar a suíte rápida em cada push/PR e
  deixar o smoke como job manual/agendado (lento). Esboço:

  ```yaml
  # .github/workflows/tests.yml (sugestão)
  jobs:
    fast:
      runs-on: ubuntu-latest
      steps:
        - uses: actions/checkout@v4
        - uses: actions/setup-python@v5
          with: { python-version: "3.13" }
        - run: pip install -r requirements.txt -r requirements-dev.txt
        - run: ./scripts/run_tests.sh fast
  ```

- A suíte rápida deve permanecer **verde** antes de qualquer merge no PR.

---

## 8. Adicionando novos testes

1. **Escolha a camada** e crie o arquivo na pasta correspondente
   (`tests/<categoria>/test_*.py`). O marcador é aplicado sozinho pela pasta.
2. **Nomeie** funções de teste como `test_*`. Funções auxiliares que **não** são
   testes (especialmente as que recebem parâmetros que não são fixtures) **devem**
   começar com `_` para não serem coletadas pelo pytest.
3. **Mocke o caro**: pesos de modelo, rede e disco. Em `api`, reaproveite as
   fixtures de `conftest.py`.
4. **Smoke** (TF real) vai em `tests/smoke/` e deve expor um `test_smoke_*` que
   chama um `main()` reaproveitável também como script (`if __name__ == "__main__"`).
5. **Antes do commit**: `./scripts/run_tests.sh fast` verde; rode `smoke` quando
   tocar em arquiteturas, treino ou inferência.

---

## 9. Notas e armadilhas conhecidas

- **`MagicMock` de `ModelInfo`**: um `MagicMock()` cru devolve sub-mocks *truthy*
  para atributos novos (`onnx_session`, `temperature`, `eer_threshold`,
  `input_contract`). Ao falsificar um `ModelInfo`, **defina esses atributos com os
  defaults reais** (`onnx_session=None`, `temperature=1.0`, …) — senão o predictor
  tenta usar uma sessão ONNX falsa e estoura `IndexError`/`TypeError`. Veja
  `tests/functional/test_detection_flow.py`.
- **Coleção antes da deselção**: o filtro `-m 'not smoke'` só **deseleciona** após
  a coleção. Um erro de *import*/assinatura em qualquer arquivo (mesmo smoke)
  quebra a coleção e aparece como `ERROR`. Por isso helpers `test_*` com
  parâmetros não-fixture foram renomeados para `_*`.
- **Backend matplotlib**: os testes de gráfico forçam o backend `Agg` (headless).
- **Artefatos**: `app/models/` (pesos) e `results/` (saídas geradas) estão no
  `.gitignore` — testes não devem versioná-los.
