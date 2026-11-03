# Plano de ação — auditoria de engenharia e refatoração

Auditoria de 2026-10-18 sobre a árvore de trabalho (inclui a reorganização
ainda não commitada `scripts/` → `tools/` + `benchmarks/` e `docker/` →
`infrastructure/docker/`). Este documento foi escrito **antes** das
modificações; a seção [Registro de execução](#registro-de-execucao) diz o que
foi aplicado e o que ficou pendente.

Ele complementa, sem substituir:

- [Auditoria e organização do projeto](../architecture/project-structure.md)
  (decisões sobre pastas de primeiro nível);
- [Plano de auditoria para publicação e TCC](public-release-and-tcc-audit-plan.md)
  (dados, licenças, reprodutibilidade científica);
- [Gates de release](release-gates.md) (critérios automáticos de publicação).

---

## 1. Prompt estruturado da tarefa

| Campo | Definição |
| --- | --- |
| **Objetivo** | Deixar o SpeechChecker organizado, sem redundância nem código órfão, aderente a SOLID e a uma separação MVC/Clean Architecture verificável, e operacional para produção. |
| **Escopo** | `app/` (core, domain, interfaces, utils), `benchmarks/`, `tools/`, `configs/`, `tests/`, manifests de dependência, Docker/CI e a documentação que descreve esses itens. |
| **Fora do escopo** | Mudar hiperparâmetros, protocolo do benchmark, front-ends de features ou qualquer coisa que altere os números do TCC; apagar dados, modelos, resultados ou o banco local; reescrever o histórico do Git. |
| **Restrições** | 1) Paridade treino→inferência e contratos de artefato (`feature_frontend`, `input_contract`) intocados. 2) Toda remoção precisa de prova de ausência de chamadores (grafo de imports + busca por nome em `.py`, `.ipynb`, `.sh`, `.bat`, `.yml`, `Makefile`). 3) Nenhuma mudança de comportamento observável da API sem registro. 4) Nada é commitado sem pedido explícito. |
| **Critérios de aceite** | Nenhum caminho quebrado em código ativo; `ruff` sem achados (gate de release); zero violações de camada no grafo de imports para os itens corrigidos; documentação coerente com a árvore; suíte `make test` verde no ambiente suportado (Python 3.11/3.12). |
| **Validação disponível nesta máquina** | Só Python 3.14 (sem TensorFlow para essa versão) e Docker parado: valida-se por AST/compilação, grafo de imports, `ruff`, `bandit` e testes que não dependem de TF. A suíte completa é gate obrigatório no ambiente suportado — ver [§7](#7-validacao-e-gates). |

---

## 2. Mapa do sistema

### 2.1 Pontos de entrada

| Entrada | Arquivo | O que sobe |
| --- | --- | --- |
| App unificada (Gradio + API) | `main.py --gradio` | Uvicorn com `app.interfaces.web.main_fastapi:app`; Gradio montado em `/gradio`. |
| CLI interativa | `main.py` (sem flags) | `app.interfaces.cli.menus.main_menu.MainMenu`. |
| Bootstrap | `main.py --bootstrap-dirs` | `app.core.bootstrap.bootstrap_application`. |
| Deploy HF | `main.py --deploy` / `app.py` | `app.deploy_hf` / `app.interfaces.gradio.app.demo` (Spaces). |
| API pura | `XFAKE_API_ONLY=1` | FastAPI sem Gradio; `/` redireciona para `/api/docs`. |
| Benchmark | `benchmarks/run_*.py` | Harness experimental (`benchmarks.runner`). |
| Operação | `tools/{dataset,training,reporting,ops}/*.py` | CLIs finas sobre `app/` e `benchmarks/`. |

### 2.2 Camadas e mapeamento MVC

```text
            View                     Controller                        Model
  ┌────────────────────────┐  ┌──────────────────────────┐  ┌───────────────────────────┐
  │ gradio/tabs (layout)   │  │ gradio/tabs (handlers)   │  │ domain/services           │
  │ gradio/utils (plots,   │─▶│ web/routers (FastAPI)    │─▶│ domain/models (ML)        │
  │   i18n, notificações)  │  │ cli/menus                │  │ domain/features, xai      │
  │ web/schemas (DTOs)     │  │ app/dependencies.py (DI) │  │ core/db (persistência)    │
  └────────────────────────┘  └──────────────────────────┘  └───────────────────────────┘
                          core/ (config, contracts, auth, segurança) — transversal
```

| Camada | Pacote | Responsabilidade | Pode depender de |
| --- | --- | --- | --- |
| Domínio (Model) | `app/domain/` | Regras de negócio, ML, features, XAI | `app/core/`, `app/utils/` |
| Persistência | `app/core/db/` | Sessão SQLAlchemy, store de experimentos | `app/core/` |
| Contratos | `app/core/contracts/` | Interfaces (DIP) | stdlib |
| Controllers | `app/interfaces/{web/routers,cli,gradio handlers}` | Traduzir entrada → serviço → saída | domínio, core |
| View | `app/interfaces/gradio`, `web/schemas` | Layout, gráficos, DTOs | controllers |
| Composição | `app/dependencies.py` | Singletons dos serviços (DI) | domínio, core |
| Harness | `benchmarks/`, `tools/` | Experimentos e operação | `app/` (nunca o inverso) |

### 2.3 API REST (`/api/v1`)

| Método | Rota | Auth | Rate limit | Serviço |
| --- | --- | --- | --- | --- |
| POST | `/detection/analyze` | público | 10/min | `DetectionService.detect_from_file` |
| POST | `/detection/multi-model` | público | 5/min | `DetectionService.detect_multi_model` |
| POST | `/detection/uncertainty` | público | 5/min | `Predictor.predict_with_uncertainty` |
| GET | `/detection/models`, `/detection/architectures` | público | 30/min | `DetectionService` |
| POST | `/features/extract` · GET `/features/types` | público | 10/30/min | `AudioFeatureExtractionService` |
| GET/POST/DELETE | `/datasets/…` (4 rotas) | API key | 5–20/min | `AudioUploadService` |
| GET/DELETE | `/history/…` (3 rotas) | API key | 10–30/min | ORM `AnalysisResult` (direto no router) |
| GET/POST/PUT/DELETE | `/profiles/…` (9 rotas) | API key | 3–30/min | `VoiceProfileService` |
| POST | `/training/start` | API key | **sem limite** | `TrainingService` (job em `BackgroundTasks`) |
| GET | `/training/status/{id}`, `/training/architectures` | público | **sem limite** | cache TTL + ORM `TrainingJob` |
| POST/GET | `/training/cross-validate…` | API key/público | 3/30/min | `TrainingService.cross_validate_model` |
| POST | `/training/export-onnx` | API key | 5/min | `onnx_export` |
| GET | `/system/{status,health,version,info,bootstrap}` | público | 10–60/min | core (`feedback`, `performance`, `db`) |
| GET/POST | `/system/feedback{,/read,/clear}` | público/API key | 10–60/min | `app.core.feedback` |

### 2.4 Interface Gradio

| Seção | Abas (`app/interfaces/gradio/tabs/`) | Serviços usados |
| --- | --- | --- |
| Painel | `dashboard.py` | `DetectionService`, ORM `AnalysisResult`, `core.gpu` |
| Detectar | `detection.py`, `voice_profiles.py` | `DetectionService`, `VoiceProfileService` |
| Investigar | `forensic_analysis.py` | `forensic_visualization` |
| Treinar | `training_wizard.py`, `optimization.py` | pipeline próprio (ver P1-4), `hyperparameter_tuning`, `benchmarks.planning` |
| Gerenciar | `dataset_management.py`, `features.py`, `history.py` | `tools.dataset.*` (ver P1-3), `AudioFeatureExtractionService` |

### 2.5 Persistência

SQLite (`data/app.db`, WAL), tabelas criadas por `app/core/db/setup.py::init_db`:
`analysis_results`, `architecture_configs`, `training_jobs`, `voice_profiles`,
`experiment_runs`, `model_runs`, `metric_records`, `configuration_entries`,
`system_snapshots`, `artifact_records` e `users` (sem uso — ver P2-6).

---

## 3. Achados

Severidade: **P0** quebra execução; **P1** viola arquitetura/gate de release ou
causa comportamento incorreto; **P2** redundância, código órfão ou dívida.

### P0 — caminhos quebrados pela migração de pastas

Um find/replace transformou `scripts/benchmark/…` em `data/benchmarks/…` e
`scripts/…` em `data/scripts/…`. Nenhum desses caminhos existe.

| Arquivo | Efeito |
| --- | --- |
| `tools/training/train_by_family.py:118` | `make train-*` e os serviços de treino chamam um script inexistente. |
| `benchmarks/run_models_sequential.py:190-192` | `_code_identity()` faz SHA-256 de arquivos inexistentes → o fingerprint de todo run falha. |
| `benchmarks/run_tcc_pipeline.py:33,765` | `SCRIPTS_DIR` e o caminho do `run_benchmark.py` errados → pipeline do TCC quebra. |
| `tools/ops/build_notebooks.py:699,837,1140,1173` | Notebooks gerados chamam um script inexistente. |
| 7 arquivos de teste (`test_benchmark_reporting_fidelity.py` ×12, `test_torch_ssl_aasist.py` ×2, `test_classical_retune.py`, `test_ssl_registra_protocolo.py`, `test_melhor_epoca_na_tabela.py`, `test_run_summary_rebuild.py`) | Testes leem/executam arquivos inexistentes → falham com `FileNotFoundError`. |

### P1 — arquitetura, gate de release e comportamento

| ID | Achado | Evidência | Ação |
| --- | --- | --- | --- |
| P1-1 | **Gate de release bloqueado**: `ruff` encontra 171 achados (o gate exige zero). | 77 I001, 35 E402, 15 B023, 12 W293, 9 B017, 9 C408… | Corrigir; E402 em scripts com bootstrap de `sys.path` vira `per-file-ignores`. |
| P1-2 | **`__init__.py` com re-export ansioso**: `import app.domain.qualquer` carrega librosa + 12 extratores; `import app.core.qualquer` carrega FastAPI, numpy e todos os contratos. Nenhum consumidor usa esses nomes pelo pacote. | `app/core/__init__.py` (60 nomes), `app/domain/__init__.py`, `app/domain/services/__init__.py`, `app/domain/features/__init__.py`, `app/utils/__init__.py` | Reduzir a docstring (+ `__version__`). Imports sempre pelo submódulo. |
| P1-3 | **Interface importa CLIs operacionais**: a aba de datasets importa `tools.dataset.download_datasets` / `preprocess_dataset` (scripts de 2.000/1.200 linhas). | `gradio/tabs/dataset_management.py:1025,1305` | Fase 3: mover a lógica para `app/domain/services/dataset/` e deixar `tools/` como CLI fina. |
| P1-4 | **Lógica de negócio na View**: `training_wizard._run_training` tem ~1.125 linhas (carga de dados, front-end, treino, avaliação, persistência) e `_run_classical_training` ~360, duplicando o `TrainingService`. | `gradio/tabs/training_wizard.py:952-2475` | Fase 4: extrair para um serviço de domínio que emite eventos de progresso; a aba só renderiza. |
| P1-5 | **Ciclo `core.db` ↔ `domain`**: entidades ORM vivem em `app/domain/models/` e `core/db/setup.py` importa o domínio para criar tabelas e semear arquiteturas. | `core/db/setup.py`, `core/db/experiment_store.py` | Fase 3: mover entidades para `app/core/db/models/`; semeadura passa ao ponto de composição. |
| P1-6 | **Controllers com acesso a dados**: `history.py` e `system.py::get_info` montam consultas SQLAlchemy; `training.py` mantém estado de jobs (cache TTL, `_update_job`) no router. | `web/routers/{history,system,training}.py` | Fase 3: repositórios em `core/db` e `TrainingJobService` no domínio. |
| P1-7 | **Apresentação no domínio**: `forensic_visualization.py` (1.238 linhas de matplotlib) está em `domain/services`. | `domain/services/forensic_visualization.py` | Fase 4: mover plotagem para `gradio/presenters/`; cálculos ficam no domínio. |
| P1-8 | **`.env` só é carregado como efeito colateral** do import de `app.core.security`; CLI, scripts e o caminho de logs do `main.py` ignoram o `.env`. | `core/security.py:23` | Carregamento explícito em `app/core/config/env.py`, chamado nos pontos de entrada. |
| P1-9 | **Uploads gravados relativos ao CWD**: `get_upload_service` usa `"data/uploads"` relativo; o health check confere `Path("data/models")` relativo. | `app/dependencies.py`, `web/routers/system.py:116` | Resolver por `OperationalPaths`. |
| P1-10 | **`train_profile_model` com ~850 linhas** num único método. | `domain/services/voice_profile_service.py:476-1326` | Fase 4: decompor (dados, front-end, treino, avaliação, persistência). |
| P1-11 | `POST /training/start`, `GET /training/status/{id}` e `GET /training/architectures` sem rate limit, enquanto todo o resto tem. | `web/routers/training.py` | Aplicar `@limiter.limit`. |

### P2 — redundância, órfãos e dívida

| ID | Achado | Ação |
| --- | --- | --- |
| P2-1 | Helpers de ambiente duplicados: `_env_flag`/`_env_bool`/`_env_int`/`_env_path` em `dependencies.py`, `main_fastapi.py`, `auth_handler.py`, `middleware.py`, `performance.py`. | Um módulo `app/core/config/env.py`. |
| P2-2 | Shim `HfFolder` copiado em `gradio/app.py` e `gradio/tabs/__init__.py`, além do canônico `core/hf_compat.py`. | Usar só `ensure_hf_folder_shim()`. |
| P2-3 | `sys.path.insert` dentro do pacote (`gradio/app.py`, `web/main_fastapi.py`, `gradio/tabs/dataset_management.py`). Pacote não deve mexer em `sys.path`. | Remover; entrypoints da raiz já resolvem. |
| P2-4 | Shims de compatibilidade sem consumidor ativo: `app/core/interfaces/` (só `benchmarks/runner.py:1221`), `domain/features/interfaces.py` e `types.py`. | Migrar chamadores para `app.core.contracts.*` e remover. |
| P2-5 | Contratos nunca implementados: `IAudioValidator`, `IAudioRepository`, `IFeatureRepository`, `IModelRepository`, `IDatasetRepository`, `INotificationService`, `IMonitoringService`. | Remover (YAGNI); reintroduzir junto com a primeira implementação. |
| P2-6 | Entidade `User` (com hash PBKDF2) sem nenhum uso: autenticação é por API key + Basic Auth do Gradio. | Remover o modelo; a tabela existente fica intocada nos bancos já criados. |
| P2-7 | Código morto (nome ocorre só na própria definição): `LossFactory` e `LearningRateScheduler` inteiros; 19 métodos `_compute_*` do extrator temporal; `optimize_*`/`evaluate_*` de SVM/RandomForest; `MetricsCalculator` (5 métodos); `TrainingAnalyticsVisualizer` (5); `ModelTrainer.predict_with_scaler/save_training_artifacts/predict_with_tta`; `create_secure_pipeline`; `set_memory_growth`; wrappers de `upload_service`; utilitários de `helpers`, `colab`, `extractor_registry`, `segmented_*`, `csv_feature_exporter`, `dataset_catalog`, `speaker_manifest`, `session.get_readonly_db`, DTOs `DetectionRequest`/`FeatureExtractionRequest`/`ErrorResponse`, helpers não usados da aba de perfis e de `CommonErrors`. | Remover. Classes legadas de `layers.py` ficam (desserialização). |
| P2-8 | CLI com caminhos fixos (ignora `MODELS_DIR`, `XFAKE_STORAGE_DIR`), lista de arquiteturas digitada à mão (sem SVM/RandomForest) e dois menus que só imprimem "em desenvolvimento". | `OperationalPaths` + `architecture_registry`; menus de Resultados e Configurações funcionais. |
| P2-9 | `configs/inference.yaml` não é lido por nenhum código: editá-lo não muda nada. | Remover e corrigir as referências na documentação. |
| P2-10 | Dependências sem import: `redis`, `a2wsgi`, `seaborn`, `jsonschema`. | Remover de `requirements-base.txt`, do ambiente `classical-ml`, do `constraints.txt` e do `verify_environments.py`. |
| P2-11 | Docstrings/READMEs citam `templates/`, `static/` e Jinja2 da interface web removida; comentários de login desativado em `gradio/app.py`. | Atualizar/remover. |
| P2-12 | `GET /system/bootstrap` devolve `{"status": "ok"}` fixo — resquício da página de loading removida; duplica `/system/health`. | Manter o contrato (testes de smoke dependem dele) e marcar `deprecated` no OpenAPI. |
| P2-13 | `app/file_storage.py` solto na raiz do pacote (só existe ali para não puxar numpy via `app/utils/__init__`). | Com P1-2 resolvido, mover para `app/utils/file_storage.py`. |
| P2-14 | Funções privadas mortas no harness (`benchmarks/data.py::_audio_flat`, `runner.py::_stratified_test_labels`, `_audit_predefined_provenance`, `config.py::unseen_speaker_tcc`). | Fase 3, junto com a suíte do benchmark rodando (há testes que leem o fonte do `runner.py`). |
| P2-15 | Módulos gigantes: `layers.py` (2.910), `runner.py` (2.891), `training_wizard.py` (2.887), `trainer.py` (2.186), `download_datasets.py` (1.995). | Fase 4, um por vez, com a suíte verde antes e depois. |

---

## 4. Estrutura-alvo

Mudanças de nome/local (as demais pastas ficam como estão — ver
[Auditoria e organização](../architecture/project-structure.md)):

| Hoje | Alvo | Fase | Motivo |
| --- | --- | --- | --- |
| `app/file_storage.py` | `app/utils/file_storage.py` | 2 | Utilitário de arquivo junto dos demais. |
| — | `app/core/config/env.py` | 2 | Fonte única de leitura de ambiente e do `.env`. |
| `app/core/interfaces/` | *(removido)* | 2 | Alias legado; `app/core/contracts/` é o canônico. |
| `app/domain/features/{interfaces,types}.py` | *(removido)* | 2 | Re-export do contrato canônico. |
| `configs/inference.yaml` | *(removido)* | 2 | Nunca lido pelo código. |
| `app/domain/models/{base_model,analysis,architecture_config,experiment,training_job,voice_profile}.py` | `app/core/db/models/` | 3 | Persistência fora do domínio; quebra o ciclo `core.db ↔ domain`. |
| — | `app/core/db/repositories/` | 3 | Consultas saem dos routers (`history`, `system`, `training`). |
| — | `app/domain/services/training_jobs.py` | 3 | Estado de jobs sai do router de treino. |
| `tools/dataset/{download_datasets,preprocess_dataset}.py` (lógica) | `app/domain/services/dataset/` | 3 | A UI deixa de importar CLIs; `tools/` vira casca fina. |
| `training_wizard._run_training` / `_run_classical_training` | `app/domain/services/interactive_training.py` | 4 | Controller/View separados do pipeline de treino. |
| `app/domain/services/forensic_visualization.py` | `app/interfaces/gradio/presenters/forensic_plots.py` | 4 | matplotlib é View. |

---

## 5. Fases de execução

### Fase 1 — correções bloqueantes (P0) · *esta sessão*

1. Corrigir os 10 arquivos com `data/benchmarks` / `data/scripts`.
2. Varredura final: nenhum caminho `data/(benchmarks|scripts|tools)` em código ativo.

**Aceite:** `grep` vazio; os scripts referenciados existem.

### Fase 2 — higiene, DRY e órfãos (P1-1, P1-2, P1-8, P1-9, P1-11, P2-1…P2-13) · *esta sessão*

1. `__init__.py` enxutos (P1-2).
2. `app/core/config/env.py` + carregamento explícito do `.env` (P1-8, P2-1).
3. `dependencies.py` e health check por `OperationalPaths`; provider do
   `VoiceProfileService` em `dependencies.py` (P1-9).
4. Shim `HfFolder` único, sem `sys.path` no pacote, sem comentários mortos (P2-2, P2-3, P2-11).
5. Remover shims, contratos sem implementação, `User` e código morto (P2-4…P2-7).
6. CLI: caminhos e arquiteturas da fonte única; menus funcionais (P2-8).
7. `configs/inference.yaml` e dependências sem uso (P2-9, P2-10).
8. Rate limit nas rotas de treino; `/system/bootstrap` como `deprecated` (P1-11, P2-12).
9. `file_storage` para `app/utils/` (P2-13).
10. `ruff` em zero (P1-1); documentação alinhada.

**Aceite:** grafo de imports sem referência a módulo removido; `ruff check` limpo;
`bandit` sem HIGH; testes independentes de TF verdes.

### Fase 3 — camadas (P1-3, P1-5, P1-6, P2-14) · *requer suíte completa*

1. Entidades ORM → `app/core/db/models/`; atualizar ~15 imports e testes.
2. Repositórios para `AnalysisResult`, `TrainingJob` e estatísticas de persistência.
3. `TrainingJobService` (domínio) assumindo o estado de jobs do router.
4. Lógica de download/pré-processamento → `app/domain/services/dataset/`.
5. Código morto do harness com a suíte do benchmark rodando.

**Aceite:** `test_domain_imports_without_web_layer.py` estendido para proibir
`core → domain` e `app → tools`; `make test` verde.

### Fase 4 — decomposição de módulos gigantes (P1-4, P1-7, P1-10, P2-15) · *requer suíte completa*

Um módulo por PR, nesta ordem: `training_wizard.py` → `voice_profile_service.py`
→ `forensic_visualization.py` → `trainer.py` → `runner.py` → `layers.py`.
Cada PR mantém a API pública do módulo e roda `make test` + smoke do fluxo
afetado (`tests/smoke/test_wizard_pipeline.py`, `test_inference_pipeline.py`).

### Fase 5 — gates de produção

`.github/workflows/release-gate.yml` completo (instalação 3.11/3.12, ruff, testes,
bandit, pip-audit, MkDocs, build Docker) + execução manual dos itens de
[release-gates.md](release-gates.md#revisao-manual-obrigatoria).

---

## 6. Riscos e decisões

| Risco | Mitigação |
| --- | --- |
| Remover algo usado por nome dinâmico (`getattr`, string de módulo). | Busca de tokens em todo o repositório (inclui notebooks, shell, YAML) e checagem de `importlib`/`getattr(..., f"...")`; os extratores carregados por string (`extractor_loader.py`) e as arquiteturas resolvidas pelo `registry` não foram tocados. |
| Testes que leem o fonte de `runner.py` como texto. | Harness só muda na Fase 3, com a suíte rodando. |
| Arquivos com alterações locais não commitadas do usuário. | Edições mínimas e localizadas; nada foi revertido nem commitado. |
| Sem TensorFlow nesta máquina. | Fases 3–4 ficam condicionadas à suíte completa em Python 3.11/3.12 ou Docker. |

---

## 7. Validação e gates

| Verificação | Comando | Onde |
| --- | --- | --- |
| Sintaxe | `python -m compileall -q app benchmarks tools tests main.py app.py` | qualquer Python |
| Lint | `ruff check app/ benchmarks/ tools/ tests/ main.py app.py` | qualquer Python |
| SAST | `bandit -q -r app/ benchmarks/ tools/ -lll` | qualquer Python |
| Suíte rápida | `make test` | Python 3.11/3.12 com `requirements-dev.txt` |
| Smoke | `pytest -m smoke tests/smoke/` | idem, com TensorFlow |
| Docs | `mkdocs build --strict --clean` + `python tools/ops/check_docs_site.py infrastructure/site` | `requirements-docs.txt` |
| Containers | `make docker-config` e `make build` | Docker |

---

## Registro de execução

Sessão de 2026-10-18. Nada foi commitado; as mudanças estão no working tree,
misturadas à reorganização local que já existia.

### Aplicado

| Item | O que mudou |
| --- | --- |
| P0 (caminhos) | `data/benchmarks`/`data/scripts` → `benchmarks`/`tools` em 10 arquivos; e mais 4 encontrados durante a execução: `tools/dataset/build_dataset.py` (`SCRIPTS_DIR` → `TOOLS_DIR`), `tools/reporting/build_paper_from_benchmark.py`, `tools/training/train_advanced.py` e o teste que lia `docker/compose/` (agora `infrastructure/docker/compose/`). |
| P0 (import) | `tools/reporting/build_paper_from_benchmark.py` importava `atomic_write_text`/`link_or_copy_file` de `app.utils.file_utils`, onde não existem — `ImportError` ao iniciar. |
| P1-1 | `ruff` de 171 achados para **zero**. Ordenação de imports, closures com variável de laço ligadas por default (`B023`), `pytest.raises` específicos (`ValidationError`/`ValueError`), `noqa` justificados (`gr.Progress()` é injetado pelo Gradio) e `E402` ignorado só em `tools/**` (cabeçalho de `sys.path` documentado). |
| P1-2 | `__init__.py` sem re-export em `app/core`, `app/core/config`, `app/core/contracts`, `app/domain`, `app/domain/models`, `app/domain/services`, `app/domain/features`, `app/utils`. |
| P1-5 | Entidades ORM movidas para `app/core/db/models/` (`base_model.py` virou `base.py`); `setup.py` não importa mais entidades do domínio. A semeadura (`seed_architectures`, último `core → domain`) saiu na Fase 3 — ver [abaixo](#fase-3-camadas-p1-3-p1-5-p1-6). |
| P1-8 | `app/core/config/env.py` com `load_environment`, `env_flag`, `env_int`, `env_positive_int`, `first_env`; `.env` carregado explicitamente em `main.py`, `app.py` e `main_fastapi.py`, **antes** de `configure_runtime_environment()` (antes, flags do `.env` como `XFAKE_ENABLE_XLA` eram ignoradas no `main.py`). |
| P1-9 | `dependencies.py` resolve tudo por `OperationalPaths` (agora honra `UPLOAD_DIR`/`DEEPFAKE_UPLOADS_DIR`/`XFAKE_UPLOADS_DIR`); health check usa o diretório real de modelos; provider `get_voice_profile_service` substitui três singletons (classe, router e aba Gradio). |
| P1-11 | Rate limit em `POST /training/start` (5/min), `GET /training/status/{id}` (60/min) e `GET /training/architectures` (30/min). |
| P2-1…P2-3 | Helpers de ambiente únicos; shim `HfFolder` único; `sys.path` removido de dentro do pacote; aba de datasets usa `OperationalPaths`. |
| P2-4, P2-5 | Removidos `app/core/interfaces/`, `app/domain/features/{interfaces,types}.py`, 16 contratos sem implementação e os métodos de interface que só tinham stub/implementação falsa (`TrainingService.evaluate_model/fine_tune_model/get_training_progress`, `AudioUploadService.upload_dataset/get_upload_progress`, `AudioFeatureExtractionService.extract_from_dataset` — este devolvia sucesso sem processar nada). |
| P2-6 | Entidade `User` removida (tabela existente fica no banco; ver `app/core/db/README.md`). |
| P2-7 | ~70 funções/classes/métodos sem chamador removidos (três varreduras, incluindo cascata). Removidos também `app/domain/features/segmented_feature_loader.py` e `app/utils/system_utils.py` — este apagava `data/results/*.json` e datasets e não tinha chamador. |
| P2-8 | CLI: diretórios e serviços de `OperationalPaths`/`app.dependencies`; arquiteturas do registry; treino pelo mesmo contrato `.npz` do `POST /training/start` (antes tentava treinar redes de áudio com features tabulares); inferência lista os modelos do `DetectionService`; análise em lote, Resultados e Configurações funcionais. `DetectionService.reload_models()` substitui o acesso a `model_loader` pela CLI e pelo wizard. |
| P2-9, P2-10 | `configs/inference.yaml` removido; `redis`, `a2wsgi`, `seaborn`, `jsonschema` removidos de `requirements-base.txt`, `classical-ml/requirements.txt`, `constraints.txt` e `verify_environments.py`. |
| P2-11, P2-12 | Docstrings/READMEs sem Jinja2/templates; `/system/bootstrap` marcado `deprecated`. |
| P2-13 | `app/file_storage.py` → `app/utils/file_storage.py` (19 importadores). |
| DRY de arquivos | `ensure_directory`, `get_file_hash` e `safe_filename` existiam em `helpers.py` **e** `file_utils.py` com contratos diferentes; a versão canônica fica em `file_utils.py`. O sanitizador estrito de upload saiu de `core/security.py` (que puxa FastAPI) para `file_utils.sanitize_filename` e substitui o terceiro sanitizador do `VoiceProfileService`. |
| Backups | `copy_file_atomic` em `file_storage`; os `.bak` de `backfill_artifact_metadata`, `rebuild_inference_contracts`, `rebuild_run_summary` e `recover_truncated_history` voltam a ser cópias independentes (como hardlink, uma escrita in-place no original alterava o "backup"). |

### Validação nesta máquina (Python 3.14, sem TensorFlow)

- `compileall`, verificação estática de todos os `from <interno> import <nome>`
  (incluindo notebooks) e `ruff check`: limpos. `bandit -lll`: zero achados.
- `pytest tests/unit tests/integration tests/functional --noconftest` (o
  `conftest.py` importa TensorFlow): **542 aprovados**, 37 pulados, antes e
  depois da mudança das entidades ORM. Das 102 falhas/erros, 99 são de
  ambiente (TensorFlow/Gradio/Torch/Optuna ausentes, ou fixtures do
  `conftest.py` desligado) e três são divergências pré-existentes:

| Teste | Divergência | Decisão necessária |
| --- | --- | --- |
| `test_melhor_epoca_na_tabela::test_o_criterio_de_selecao_e_declarado_no_rodape` e `test_benchmark_reporting_fidelity::test_fragmento_declara_sete_keras_e_duas_ssl` | O gerador LaTeX declarava "nove entradas neurais por `val_loss`" (texto já presente no `HEAD`); o teste e o `CLAUDE.md` dizem `val_eer` nas entradas Keras. | **Resolvido**: o run canônico usou `val_eer` no caminho Keras. Rodapé, legenda da tabela de estabilidade e comentários de `update_tcc_latex.py`, legenda da figura e aviso de `consolidate_results.py` corrigidos; os dois testes passam. No artigo (`data/results/paper/main.tex`), os parágrafos do regime de treino (§ Protocolo de Treinamento) e do preset final (seção de hiperparâmetros) também foram corrigidos; não resta menção a `val_loss` no texto. Recompilar o PDF. |
| `test_benchmark_reporting_fidelity::test_artefato_e_copiado_para_dentro_do_run` | `_preserve_run_artifact` passou a usar hardlink (o runner desassocia antes de retreinar); o teste exige cópia independente. | Manter o hardlink (e adaptar o teste ao protocolo) ou voltar à cópia. |

### Validação da suíte completa (Python 3.11, Docker)

Rodada em 2026-10-18 num contêiner descartável `python:3.11-slim` com
`requirements-dev.txt` + `requirements-docs.txt` (TensorFlow 2.21.0, Keras
3.15.1, torch 2.5.1+cpu, scikit-learn 1.8.0, Gradio 4.44.1, NumPy 2.2.6) e as
variáveis do `ci.yml` (`XFAKE_API_ONLY=1`, `MPLBACKEND=Agg`). A entrada é um
instantâneo do working tree com o que o CI veria se ele fosse commitado
(arquivos rastreados + não rastreados não ignorados): os artefatos ignorados
de `data/` ficam de fora e nenhum teste escreve neles.

| Gate | Instantâneo | Resultado |
| --- | --- | --- |
| `make test` (`tools/ops/run_tests.sh fast`) | 13:16 | **937 aprovados**, 1 falha, 10 pulados, 10 desmarcados (smoke) — 7 min 58 s |
| `pytest -m smoke tests/smoke/` | 13:24 | **10 aprovados** (14 arquiteturas, wizard, inferência integrada, startup do Gradio, paridade do backbone SSL) — 5 min 09 s |
| Os 10 pulados acima, com `requirements-notebooks.txt` + `requirements-xai.txt` (+ `ipykernel`) | 13:24 | **45 aprovados** em `test_notebooks_compile.py` e `test_xai.py`, com os 8 notebooks executados de ponta a ponta |
| `compileall`, `ruff check`, `bandit -lll` | 13:31 | limpos; zero achados |
| `mkdocs build --strict --clean` + `check_docs_site.py infrastructure/site` | — | aprovado após corrigir um link (abaixo) |
| `make docker-config` (5 perfis) | — | os 5 compose são válidos com `.env.example` + segredos fictícios. Com o `.env` local, os dois perfis de inferência param por falta de `ALLOWED_HOSTS`, `ALLOWED_ORIGINS`, `SPEECHCHECKER_API_KEY`, `XFAKE_GRADIO_USERNAME` e `XFAKE_GRADIO_PASSWORD` — fail-closed proposital; o `.env` local é anterior a essas variáveis. |

- A única falha é a do hardlink em `_preserve_run_artifact` (tabela acima). O
  `HEAD` usava `shutil.copy2`; a troca por `link_or_copy_file` é anterior a
  esta refatoração. **Nenhuma falha vem das Fases 1–2.**
- Os 10 pulados da suíte rápida (8 por `nbformat`, 2 por `shap`) são perfis
  opcionais fora de `requirements-dev.txt`, como no CI; a terceira linha da
  tabela os cobre.
- `docs/architecture/project-structure.md` apontava para
  `../../tools/ops/migrate_voice_profiles_storage.py`, fora de `docs/`, e
  derrubava o `--strict`. O link virou caminho em código, como no resto da
  documentação. O arquivo é da reorganização anterior, não da refatoração.
- Os instantâneos incluem a Fase 3 em andamento (outra sessão editando o mesmo
  checkout). Uma segunda rodada da suíte rápida no instantâneo das 13:31 deu
  958 aprovados e 5 falhas: a do hardlink e quatro em arquivos que a Fase 3
  editava naquele momento. Na árvore das 13:41, três delas já passavam; resta
  `test_clean_bootstrap::test_main_bootstrap_twice_from_another_cwd` (ver
  pendências).

### Fase 3 — camadas (P1-3, P1-5, P1-6)

Sessão lateral de 2026-10-18, no mesmo working tree. Nada foi commitado.

| Item | O que mudou |
| --- | --- |
| P1-6 (repositórios) | `app/core/db/repositories/` com `AnalysisResultRepository`, `TrainingJobRepository`, `ArchitectureConfigRepository` e `PersistenceStatsRepository` — uma sessão por método, como o `ExperimentStore`; as entidades saem destacadas, com as colunas já carregadas. Instâncias únicas em `app.dependencies`. `web/routers/history.py`, `system.py::get_info`, as abas Histórico e Painel e `gradio/utils/hyperparameters.py` usam os repositórios: não resta `sqlalchemy`, `SessionLocal` nem `.query(` em `app/interfaces/`. `session.get_db`/`get_background_db` ficaram sem chamador e saíram. |
| P1-6 (jobs de treino) | `app/domain/services/training_jobs.py::TrainingJobService` assume o `TTLCache`, `_update_job`, `run_training_task`, `_run_cv_task` e a reconstrução do agregado do K-fold; provider `get_training_job_service`. O router valida e delega. O `TrainingService` continua injetado por requisição (`Depends(get_training_service)`) e passado à tarefa — é a instância que os testes da API sobrescrevem. |
| P1-5 (semeadura) | `seed_architectures` saiu de `core/db/setup.py`: `init_db` só cria/migra o schema. `app.dependencies.seed_architecture_configs` lê o registry e grava por `ArchitectureConfigRepository.seed_defaults` (as mesmas linhas, só com a tabela vazia); `app.dependencies.bootstrap_application` é o `bootstrap_application` do core com `seed_database=`. `main.py` (CLI e `--bootstrap-dirs`) e o lifespan da API usam o da composição. Para isso `app/dependencies.py` importa serviços e repositórios **dentro** dos providers: importá-lo não cria o `engine` (que lê `DATABASE_URL` no import) nem carrega o TensorFlow. Com os imports no topo, o `--bootstrap-dirs` gravava o schema no banco padrão — a falha que a validação acima encontrou. |
| P1-3 | A lógica de `tools/dataset/download_datasets.py` e `preprocess_dataset.py` foi para `app/domain/services/dataset/{download,preprocessing}.py`. Os 56 corpos de função são idênticos byte a byte ao original (conferido por AST); mudam só os loggers (`__name__`), a raiz (`PROJECT_ROOT` de `app.core.bootstrap`, mesmo caminho) e o nome público `warn_if_datasets_incompatible`. As CLIs ficam com argparse, `logging.basicConfig` e o mesmo `--help`. A aba de datasets, `build_paired_pt_corpus.py`, `download_cetuc_paired.py` e os testes importam o domínio. |
| Fronteiras | `tests/unit/test_layer_boundaries.py` lê o grafo de imports do fonte (AST; conta imports tardios, relativos e `import_module` com literal): `app.core` não importa `app.domain`, `app.interfaces` nem `app.dependencies`; `app.domain` não importa `app.interfaces` nem `app.dependencies`; nada em `app/` importa `tools`. As três regras passam. |
| Testes novos | `test_db_repositories.py`, `test_training_job_service.py`, `test_layer_boundaries.py` e, em `test_clean_bootstrap.py`, a semeadura pela composição e a garantia (em subprocesso) de que importar `app.dependencies` não cria o `engine`. `quality-and-testing.md` passa a 96 arquivos. |

Mudanças observáveis (nenhuma no caminho de sucesso da API):

- `ExperimentStore.ensure_schema()` (benchmark, `TrainingService`) continua
  criando o schema, mas não semeia mais `architecture_configs`; só o bootstrap
  das interfaces semeia. Num banco onde nenhuma interface subiu (o Spaces via
  `app.py` nunca chama o bootstrap), a tabela fica vazia:
  `registry.get_active_config` e a aba de hiperparâmetros caem no
  `default_params` do registry — o mesmo conteúdo que seria semeado —, mas
  `load_hyperparameters_json` (metadado `recommended_hyperparameters` da
  detecção) cai em `get_recommended_hyperparameters`.
- `load_history` (aba Histórico) ganhou `@ui_safe`: falha de banco vira
  "Falha ao carregar o histórico: …" em vez do erro genérico do Gradio. A
  auditoria `test_todo_handler_de_evento_trata_excecao` só passava antes por
  causa do `try/finally` que fechava a sessão.
- Importar `app.dependencies` não carrega mais os serviços: cada um é
  importado na primeira chamada do seu provider.
- `build_paired_pt_corpus.py` passa a usar o próprio `datefmt` (`%H:%M:%S`) no
  console; antes o import de `download_datasets` executava um `basicConfig`
  que anulava o dele.
- A mensagem de `ImportError` da aba de download cita
  `app/domain/services/dataset/download.py`.

| Gate | Ambiente | Resultado |
| --- | --- | --- |
| `make test` antes da Fase 3 | Python 3.12 local; instantâneo da árvore com `data/` ligado e `DATABASE_URL` num SQLite de teste | 932 aprovados, 5 falhas, 11 pulados |
| `make test` com a Fase 3 | idem | 960 aprovados, 3 falhas, 11 pulados. JUnit comparado: nenhum teste perdido nem regredido; 26 novos; as duas falhas de `val_eer` passaram com a correção da tabela acima |
| `make test` com a Fase 3 | Python 3.11, `speechchecker-test:py311`, instantâneo CI | **963 aprovados**, 1 falha (hardlink), 10 pulados |
| `pytest -m smoke tests/smoke/` | idem | **10 aprovados** (4 min 17 s) |
| `ruff check`, `compileall`, `bandit -lll` | Python 3.12 | limpos; zero achados |

Das três falhas locais, só a do hardlink é do código. As outras duas vêm do
ambiente local e passam no contêiner: `test_gradio_detection_tabs_share_api_singleton`
(a junção de `data/` resolve para fora do instantâneo) e
`test_main_bootstrap_twice_from_another_cwd` (o `.env` local define
`DATABASE_URL`, que o `main.py` relê depois de o teste apagá-la). As duas
também falhavam localmente antes da Fase 3.

### Verificação do app depois da Fase 3 (2026-10-18/31)

Correções encontradas ao exercitar o app de ponta a ponta (treino pelo
assistente, inferência dos 9 modelos promovidos, API):

| Achado | Correção |
| --- | --- |
| `@ui_safe` embrulhava handlers geradores numa função comum; o Gradio recebia o objeto gerador e recusava a saída. Validar, treinar e salvar do assistente quebravam (o assistente não passava da etapa 1). Existia no `HEAD`. | O decorador devolve um invólucro gerador para funções geradoras (`gradio/utils/components.py`), com teste. |
| `tests/conftest.py` fixava a chave de teste só pelo alias legado e depois de importar o app; com o `.env` do `.env.example` (`ALLOWED_HOSTS` ativo), ~30 testes de API falhavam fora do CI com 400. | O conftest fixa chave, hosts e origens antes de importar o app. |
| Sidecars de SVM e Random Forest sem `band_correction`: o SVM acertava 4% do áudio real. | `rebuild_inference_contracts.py --stamp-band-correction` (só esse campo, só com sha256 conferido). Ver `docs/models/inference.md`. |
| A inferência filtrava e nivelava o arquivo inteiro e recortava a janela depois; o treino faz o inverso. | `FeaturePreparer` recorta a janela canônica antes. SVM 80,5% → 91,0% e RawGAT-ST 90,0% → 93,5% em validação. |

### Fase 4 — `training_wizard.py` (P1-4)

Primeiro módulo da Fase 4. O pipeline de treino do assistente saiu da aba para
`app/domain/services/interactive_training.py`:

- `InteractiveTrainingService.train()` gera eventos tipados (`DatasetReady`,
  `FitStarted`, `FitHeartbeat`, `EpochCompleted`, `FitCompleted`, os
  equivalentes do caminho clássico e `TrainingFailed`); progresso fino por
  callback e parada por um `threading.Event` que a aba controla.
  `save_trained_model()` grava `.keras`/`.pkl` + `_config.json`.
- O pipeline Keras foi transplantado por intervalo de linhas — mesmo front-end
  LFCC, mesmo float32 + `clipvalue`, mesmo teste de fumaça, mesmo `fit` em
  thread. Só os `yield` de tuplas da UI viraram eventos.
- A aba (`training_wizard.py`, 2.891 → 1.638 linhas) só traduz eventos em
  painel, log, gráficos e notificações; `_run_training` manteve nome e
  assinatura, então o handler da etapa 4 não mudou.
- O plano de hiperparâmetros (`benchmarks.planning`) entra pela composição
  (`get_interactive_training_service`): o domínio não importa o harness. O
  extrator do caminho clássico continua sendo o do `DetectionService`.
- O diretório de gravação passou a ser o de `OperationalPaths` (o mesmo que o
  `ModelLoader` lê; caminhos relativos resolvidos pela raiz, não pelo CWD).
- `tests/integration/test_interactive_training.py` exercita o serviço de ponta
  a ponta (Sonic Sleuth 2 épocas e SVM): eventos, gravação, recusa de nome
  repetido, recarga e detecção pelo `DetectionService`, e o pedido de parada
  pendente que não pode matar o treino seguinte (conferido por mutação).
- Conferido na interface: assistente completo com AASIST, 2 épocas — painel ao
  vivo, linhas por época e os 8 gráficos de avaliação, como antes.

| Gate | Ambiente | Resultado |
| --- | --- | --- |
| `make test` | Python 3.11.9, `.venv` do projeto, árvore real | **970 aprovados**, 2 falhas (as duas pendências abaixo), 10 pulados |
| `pytest -m smoke tests/smoke/` | idem | **10 aprovados** |

### Fase 4 — `voice_profile_service.py` (P1-10)

O `train_profile_model` já delegava para métodos auxiliares (~190 linhas; as
"~850" do achado contavam o bloco inteiro até a detecção). O problema que
restava era o módulo: 1.445 linhas misturando CRUD e amostras com extração de
features, fakes sintéticos, seis construtores Keras e dois laços de treino — e
**sem teste nenhum**.

- Primeiro, `tests/integration/test_voice_profile_service.py` fixou o
  comportamento pelo caminho público (perfil → amostras → validação → treino
  SVM e Keras → verificação → remoção, dataset insuficiente e o construtor de
  cada família), rodando contra o código ANTIGO. Os mesmos 9 testes passam
  depois; só o import do construtor mudou.
- As etapas foram para `app/domain/services/voice_profile/` (como `detection/`
  para o `detection_service.py`): `features.py` (vetor de 144 features e
  fakes sintéticos), `models.py` (um modelo por família + MLP de reserva) e
  `training.py` (`prepare_training_split`, treino Keras/clássico, EER). Os
  corpos foram transplantados por intervalo de linhas; mudaram só `self.` →
  funções do módulo e `profile_id` → `model_dir`.
- O serviço (1.445 → 758 linhas) mantém a API pública e as constantes do
  módulo; `train_profile_model` virou a sequência de etapas (amostras reais →
  fakes → split → treino → persistência no perfil). As cinco cópias do
  conjunto de extensões de áudio viraram `AUDIO_EXTENSIONS`.
- O docstring dizia "146 dimensões"; o vetor tem 144 (o
  `DEFAULT_FEATURE_CONFIG` já dizia 144, e o teste agora o fixa).
- Conferido na interface: aba Perfis de Voz lista o perfil existente. Criar ou
  treinar um perfil pela UI gravaria no banco e em `data/voice_profiles/` do
  usuário; o fluxo completo está coberto pelo teste de integração.

| Gate | Ambiente | Resultado |
| --- | --- | --- |
| `make test` | Python 3.11.9, `.venv` do projeto | **979 aprovados** (+9), as mesmas 2 falhas, 10 pulados |
| `pytest -m smoke tests/smoke/` | idem | **10 aprovados** |

### Fase 4 — `forensic_visualization.py` (P1-7)

- Primeiro, `tests/integration/test_forensic_analysis.py` fixou a aba
  Investigar pelos handlers (13 figuras + resumo, confiança por segmento vinda
  do detector, radar de oito eixos, lote e CSV), com detector falso e
  determinístico, rodando contra o código ANTIGO.
- Cálculos → `app/domain/services/forensic_measures.py`: formantes por LPC,
  HNR, jitter/shimmer, F0/vibrato, LFCC, DET/EER, varredura de limiar, os oito
  eixos do radar (que viviam na ABA) e o resumo do lote. Transformadas padrão
  do librosa exibidas como imagem (Mel, STFT, CQT, cromograma, fase) ficam na
  apresentação.
- matplotlib → `app/interfaces/gradio/presenters/forensic_plots.py` (mesmas
  classes e métodos). O módulo antigo saiu; aba Investigar e assistente de
  treino importam do presenter.
- A/B de 36 saídas (cada linha, imagem, faixa, texto e legenda de cada
  figura, mais lote, CSV e relatório): idênticas. Uma mutação proposital no
  eixo RAP foi detectada pelo mesmo A/B.
- `test_layer_boundaries.py` ganhou a regra "domínio sem gradio, fastapi,
  starlette, matplotlib ou plotly"; o `forensic_visualization.py` era o único
  infrator. `tests/unit/test_forensic_measures.py` verifica as medidas contra
  sinais de valor conhecido.
- Conferido no app em execução, pela API do próprio Gradio
  (`/run_forensic_analysis`, sessão autenticada): 13 figuras com o modelo real
  carregado e nenhuma falha no log.

### Fase 4 — `trainer.py` (P2-15)

`trainer.py` tinha 2.053 linhas: cinco callbacks Keras, três funções de EER,
o salvamento de inferência e a calibração pós-treino em volta do
`ModelTrainer`.

- `callbacks.py`: `ResumableModelCheckpoint`, `PersistentEpochHistory`,
  `CollapseAbort`, `ValidationEER`, `EpochProgressLogger`.
- `metrics.py` (que já era puro numpy/sklearn): `scores_from_predictions`,
  `eer_from_scores`, `validation_eer` — o EER de validação agora importa sem
  TensorFlow.
- `calibration.py`: conjunto sob ruído, temperatura, limiar de OOD, limiar de
  EER e métricas finais; `artifacts.py`: `save_inference_keras` e
  `build_input_contract`.
- `trainer.py` (2.053 → 1.006 linhas) mantém o `ModelTrainer` e os métodos
  privados que o `TrainingService` e os testes chamam
  (`_auto_calibrate_temperature`, `_compute_ood_threshold`,
  `_compute_eer_threshold`, `_calculate_final_metrics`,
  `_build_input_contract`), agora delegando. `_export_onnx_artifacts` saiu:
  nunca teve chamador, e `docs/models/training.md` afirmava que o treino
  exportava ONNX.
- A/B de um treino determinístico (op determinism, semente fixa; forma de
  onda, saída de 2 colunas, `checkpoint_monitor=val_eer`, class weighting,
  backup, CSV): histórico, métricas finais, temperatura, OOD, limiar e valor
  de EER, conjunto de calibração, contrato, arquivos gravados, predição do
  `.keras` recarregado e as 24 mensagens de log — idênticos. Duas execuções do
  código antigo já eram idênticas entre si; uma mutação na semente do ruído
  de calibração foi detectada (11 diferenças).
- Mudança observável única: o NOME do logger das mensagens movidas
  (`...training.callbacks`/`calibration`/`artifacts` em vez de
  `...training.trainer`); o texto é o mesmo e `training.progress` não mudou.
- `tests/unit/test_training_calibration.py` cobre as funções novas isoladas
  (SNR realizado do AWGN, recusa de ruído fora da forma de onda, campos do
  contrato).

| Gate (P1-7 + `trainer.py`) | Ambiente | Resultado |
| --- | --- | --- |
| `make test` | Python 3.11.9, `.venv` do projeto | **1.002 aprovados** (+23), as mesmas 2 falhas, 10 pulados |
| `pytest -m smoke tests/smoke/` | idem | **10 aprovados** |
| `ruff check app/ benchmarks/ tools/ tests/` | idem | limpo |

### Fase 4 — `runner.py` (P2-15)

`benchmarks/runner.py` tinha 2.890 linhas e é o motor do estudo, então o critério
foi mais estrito que nos módulos anteriores: nenhum corpo de função mudou.

- Cortes: `common.py` (defaults de ambiente, nomes, caminhos, artefato do run),
  `provenance.py`, `split_audit.py`, `protocol.py` (partições preparadas, política
  de banda, carimbos do contrato), `fit_neural.py`, `fit_classical.py` e
  `seed_aggregation.py`. O `runner.py` (2.890 → 773 linhas) fica com
  `_benchmark_one`, `plan_benchmark` e `run_benchmark`; `run_benchmark` e
  `plan_benchmark` seguem como a API pública do pacote.
- Corpos transplantados por intervalo de linhas e conferidos pela AST: as 38
  funções restantes têm árvore idêntica à original. As 2 que saíram são as mortas
  do P2-14 (`_stratified_test_labels`, `_audit_predefined_provenance`). Os nomes
  com `_` foram mantidos: são privados ao pacote e citados em documentos e testes.
- **Acoplamento encontrado: o `--resume`.** `run_models_sequential._code_identity`
  decide pular um modelo por um hash de `config.py`, `planning.py` e `runner.py`.
  Mover o treino para outro arquivo sem atualizar a lista faria uma mudança de
  protocolo em `fit_neural.py` reaproveitar em silêncio o artefato treinado com o
  código antigo. Os sete módulos entraram na lista, e
  `test_fingerprint_do_resume_cobre_o_motor_que_era_o_runner` fixa isso. Os
  fingerprints gravados em `data/results/benchmark/` já não conferiam antes desta
  divisão (schema antigo e os quatro arquivos com hash diferente), e não havia run
  em andamento: nada retomável foi invalidado.
- Os defaults de ambiente (`XFAKE_STRICT_SSL`, `TF_USE_LEGACY_KERAS`…) foram para
  `common.py`, importado por `provenance.py`: o runner SSL, que os herdava ao
  importar o `runner`, continua com os mesmos valores. Importar o `runner` segue
  sem carregar TensorFlow.
- Testes: o `monkeypatch` de `prepare_input_for_architecture` foi para `protocol`,
  e os três testes que leem o texto do arquivo apontam para onde o trecho está
  agora. Os `inspect.getsource(runner._run_neural)` e o `monkeypatch` de
  `runner._run_neural`/`_run_classical` não precisaram mudar.
- A/B de um benchmark REAL em CPU (`run_benchmark` sobre um NPZ de forma de onda
  com identificadores de locutor e frase: SVM, RandomForest, AASIST, Conformer e
  MultiscaleCNN, correção de banda, `val_eer`, cópia AWGN, SNR casado e não visto,
  mais um run de 2 sementes e um só de plano). Comparados `results.json`,
  `metrics.json`, CSVs de predição, tabelas, sidecars e plano: **idênticos**. Duas
  execuções do código antigo já eram iguais entre si, a menos de tempo, latência,
  ambiente, hash dos binários e da 16ª casa do ECE do RandomForest (paralelismo),
  que a comparação ignora. Uma mutação de controle (semente do ruído de treino e
  `ddof` do desvio entre sementes) produziu 1.460 diferenças.
- `test_main_bootstrap_twice_from_another_cwd` abria o `data/app.db` real a cada
  `make test`: removia `DATABASE_URL` do ambiente e o `main.py` a relia do `.env`.
  Agora o teste fixa um banco próprio, passa, e a suíte inteira deixa o arquivo
  intocado (data de modificação conferida antes e depois).

| Gate (`runner.py`) | Ambiente | Resultado |
| --- | --- | --- |
| `make test` | Python 3.11.9, `.venv` do projeto | **1.004 aprovados** (+2), 1 falha (a do hardlink), 10 pulados |
| `pytest -m smoke tests/smoke/` | idem | **10 aprovados** |
| `ruff check app/ benchmarks/ tools/ tests/` | idem | limpo |

### Verificação de modelos e inferência (2026-11-01)

Só leitura em `data/`; a partição de TESTE não foi lida. Em CPU, Python 3.11.9.

- **Carregamento:** os 9 `data/models/bench_*` carregam, sem aviso de versão, com
  `input_shape` igual ao do contrato. O hash de 8 bate com o `metrics.json` do run
  em `data/results/benchmark/`; o do RawNet2 bate com `_superseded/rawnet2_d9f23c0d`.
- **Paridade treino → app:** 60 janelas de validação do NPZ pelos dois caminhos
  (`prepare_input_for_architecture` + correção de banda + temperatura + multicrop,
  contra `detect_single` e `_detect_with_config`). Diferença máxima de `p_fake`:
  0,00008. Concordância de decisão: 100% nos 9.
- **200 WAVs de validação** (`detect_from_file`, zero erros): AASIST 99,0%, Conformer
  99,0%, Hybrid 100%, MultiscaleCNN 98,5%, RandomForest 89,0%, RawGAT-ST 93,5%,
  RawNet2 94,5%, SpectrogramTransformer 99,5%, SVM 91,0%. Idênticos aos da
  verificação anterior nos cinco medidos lá: as divisões de módulo não mudaram a
  inferência. São números de validação, não os do artigo.
- **Entradas:** WAV 16 kHz, 44,1 kHz estéreo, 8 kHz, PCM16, FLAC, OGG e MP3
  funcionam; 30 s usam a janela central; abaixo de 0,3 s, vazio, corrompido, não
  áudio e inexistente são recusados sem exceção. M4A/AAC passam na validação de
  extensão, mas não foram testados (sem ffmpeg nesta máquina).


### Correções da verificação (2026-11-01)

Os dois primeiros achados da verificação acima, com aprovação do usuário para a
mudança de contrato. Testes escritos antes, falhando pelo motivo certo.

- **Aba Investigar:** `run_forensic_analysis` plotava `result.data.confidence`, a
  confiança do VEREDITO, como "Confianca (Fake)", com limiar fixo de 0,5. A linha,
  o mapa de anomalias e o histograma do lote agora usam `probabilities["fake"]` e o
  limiar do modelo (`classification_threshold`). A tabela e o CSV do lote seguem
  com a confiança do veredito, rotulada assim. Segmento que falha sai do gráfico:
  antes entrava como 0,5, um número não medido. Métodos renomeados para
  `plot_fake_score_timeline`/`plot_fake_score_distribution`.
- **Sidecar dos clássicos:** `_classical_input_contract` copia
  `protocol["band_correction"]`, como `_stamp_benchmark_frontend` faz nas neurais,
  e só quando há front-end declarado. Os artefatos atuais não mudam: já tinham o
  carimbo. Como `fit_classical.py` entra no fingerprint, o `--resume` retreina.

Prova com dados reais, só validação:

| Verificação | Antes | Depois |
| --- | --- | --- |
| AASIST, 20 clipes reais: segmentos acima do limiar | 100% (linha média 0,94) | 3 de 183 (P(fake) média 0,10) |
| AASIST, 20 clipes fake: segmentos acima do limiar | não distinguia das reais | 164 de 187 (P(fake) média 0,94) |
| SVM de um run novo, app x benchmark, dif. máx. de `p_fake` | 0,844 | 0,000000 |
| RandomForest de um run novo, idem | 0,240 | 0,000000 |

| Gate (correções) | Ambiente | Resultado |
| --- | --- | --- |
| `make test` | Python 3.11.9, `.venv` do projeto | **1.009 aprovados** (+5), 1 falha (a do hardlink), 10 pulados; `data/app.db` intocado |
| `pytest -m smoke tests/smoke/` | idem | **10 aprovados** |
| `ruff check app/ benchmarks/ tools/ tests/` | idem | limpo |

### Fase 4 — `layers.py` (P2-15)

`layers.py` tinha 2.910 linhas e 60 definições compartilhadas. O risco aqui era outro:
os `.keras` gravados citam `app.domain.models.architectures.layers` no campo `module`,
e o carregamento depende do registro no Keras.

- O Keras 3 resolve classe própria SÓ pelo registro (`_retrieve_class_or_fn`:
  `registered_name` e `custom_objects`; `module` só vale para pacotes `keras*`). Mover
  as classes é seguro desde que `import layers` continue registrando. Por isso, ao
  contrário dos módulos anteriores, este fica como **porta de entrada**: reexporta
  tudo e mantém o bloco de `get_custom_objects().update`, verbatim. Os importadores não
  mudam. Um `.keras` novo grava o `module` temático; o antigo segue abrindo.
- Definições por tema, sem dependência entre módulos: `plumbing_layers.py`,
  `spectral_layers.py`, `sinc_layers.py`, `block_layers.py`, `head_layers.py` e
  `graph_layers.py` (legadas). `layers.py`: 2.910 → 203 linhas; o maior módulo tem 864.
- Conservação pela AST: as 60 definições têm árvore idêntica à original. Comentários
  soltos foram levados junto, exceto uma nota que ficaria falsa ("ResidualBlock1D is
  defined below") e dois títulos de seção; o `layers` deixou de reexportar
  `SafeInstanceNormalization`, que ninguém importava dali.
- Achado ao escrever o teste: `import layers` sozinho nunca registrou
  `AudioNormalizationLayer` sem prefixo. Quem o faz é `rawnet2.py`, e o carregador sempre
  o importa. Mantido como está e fixado em `tests/unit/test_layers_deserialization.py`,
  que abre configs gravadas num processo limpo.
- A/B: 25 `.keras` do projeto (geração atual, promovida anterior e arquivadas, que
  exercitam as classes legadas) pelo `ModelLoader`, com saída exata para entrada fixa;
  12 arquiteturas construídas com semente fixa, com config (sem `module`), parâmetros e
  saída iguais, salvas e recarregadas. O registro completo também foi comparado. Duas
  execuções do código antigo eram idênticas entre si; depois da divisão, a única
  diferença é a esperada (`SafeInstanceNormalization` fora de `layers`).
- Mutação de controle: sem o registro de `GraphPoolLayer`, 4 artefatos legados deixam de
  abrir, e multiplicar `MagnitudeLayer` por 1,001 muda a saída do AASIST e do RawGAT-ST
  (35 diferenças no A/B). A execução foi interrompida por um encerramento de sessão
  depois de restaurar a mutação; os arquivos foram conferidos contra os backups e a
  suíte, refeita do zero.

| Gate (`layers.py`) | Ambiente | Resultado |
| --- | --- | --- |
| `make test` | Python 3.11.9, `.venv` do projeto | **1.013 aprovados** (+4), 1 falha (a do hardlink), 10 pulados; `data/app.db` intocado |
| `pytest -m smoke tests/smoke/` | idem | **10 aprovados** |
| `ruff check app/ benchmarks/ tools/ tests/` | idem | limpo |

### Fase 5 — gates de produção (2026-11-02)

- **Pilha web:** o `pip-audit` encontrava 197 avisos em 8 pacotes, quase todos em
  Gradio 4, Starlette, datasets e dependências. Com aprovação do usuário, a pilha subiu
  para Gradio 6.17.3, FastAPI 0.162, Starlette 1.10, Pillow 12.3 e datasets 5, sem tocar
  em `torch`, `transformers`, TensorFlow, Keras, NumPy e scikit-learn. Sobram 84 IDs em
  3 pacotes. A cadeia de tetos está em
  [release-gates.md](release-gates.md#pilha-web-e-seus-tetos).
- **Mudanças que a migração exigiu:** `theme`, `css` e `head` saíram de `gr.Blocks`,
  onde o Gradio 6 os ignora com um aviso e a interface sobe sem tema (agora em
  `APPEARANCE`, passado a `launch()` e `mount_gradio_app()`); `trust_remote_code` saiu
  do download de datasets; o teste de rotas lê `app.openapi()`, porque o FastAPI novo
  guarda os routers incluídos aninhados; `warn_if_datasets_incompatible` passou a
  avisar só na 4.x (disparava em qualquer major >=4 e mandava rebaixar), e as dicas de
  instalação, a documentação e o notebook de treino pararam de pedir `datasets<4.0`.
- **Gate:** `pip-audit` puro reprovaria toda release por causa dos pinos deliberados.
  `tools/ops/check_pip_audit.py` só tolera os IDs de
  `configs/security/pip-audit-exceptions.json` e falha fechado se o `pip-audit` não
  concluir. A CI agora instala com `constraints.txt`: sem isso auditava outras versões,
  e não as que as imagens publicam.
- **Justificativas conferidas no código, aviso por aviso.** Duas hipóteses iniciais não
  se sustentaram: `weights_only=True` não protege no torch 2.5.1 (CVE-2025-32434,
  CVE-2026-24747), e "nenhuma rota escolhe o repositório" é falso para a API de treino.
  `parameters` chega a `create_model`, e `model_name` ou `wavlm_model` vão ao
  `from_pretrained`. A rota exige a chave da API, e o Gradio não alcança esse caminho.
  Ficou declarado como risco residual, não corrigido: bloquear muda o contrato da API.
- **Revisão manual:** feita e registrada em [release-review.md](release-review.md). A
  publicação acadêmica segue **bloqueada** por licença do corpus e proveniência das
  métricas, e as duas decisões são humanas.

Verificação da migração no app real, com banco de rascunho:

| Fluxo | Resultado |
| --- | --- |
| Tema e CSS | Restaurados, sem aviso de `Blocks` |
| Navegador: upload, forma de onda, análise forense | 13 figuras renderizadas; todas as requisições para `localhost`, nenhum erro HTTP |
| Detecção, modo simples e avançado | Modelo escolhido é respeitado, com a variante vinda de `/update_variants` |
| Lote e exportação CSV | Funcionam |
| Assistente, na ordem do botão | Dataset de 12 WAVs: 9 s, streaming, 8 figuras |
| Paridade com o caminho de treino, 60 janelas de validação | 9 modelos: decisões iguais, diferença máxima 8e-05 |
| 200 WAVs de validação | Iguais à verificação anterior: AASIST 99,0; Conformer 99,0; Hybrid 100; MultiscaleCNN 98,5; RF 89,0; RawGAT-ST 93,5; RawNet2 94,5; AST 99,5; SVM 91,0 |
| Download pelo datasets 5, sem `torchcodec` | Funciona em streaming com `hf-internal-testing/librispeech_asr_dummy`. Repositórios Common Voice da Mozilla não testados: podem cair no fallback |

Duas falhas aparentes eram do meu teste, não do código. O assistente não terminava
porque o script pulava `attach_path_to_scan`: sem o caminho, ele cai em `data/datasets` e
treinava 19.706 lotes do corpus real. Já `/update_variants` devolve uma tupla, que o
script passava inteira ao Dropdown.

AASIST e SpectrogramTransformer repetem a mesma confiança em real e fake (0,97891 e
0,997973). Não é defeito da migração: é o teto da calibração para a saída saturada,
`sigmoid(logit(1-1e-7)/T)`, com T de 4,2 e 2,6.

| Gate (Fase 5) | Ambiente | Resultado |
| --- | --- | --- |
| `make test` | Python 3.11.9, `.venv` do projeto | **1.049 aprovados**, 0 falhas, 10 pulados; `data/app.db` e `data/models` intocados |
| Os 10 pulados (`nbformat`, `nbclient`, `shap`), que a CI instala | idem, com esses pacotes numa pasta de rascunho | 8 notebooks executados e 2 testes de SHAP aprovados |
| `pytest -m smoke tests/smoke/` | idem | **10 aprovados** |
| `ruff check` e `bandit -lll` | idem | limpo; nenhum achado HIGH |
| `mkdocs build --strict --clean` e `check_docs_site.py` | ambiente de docs | passam |
| `pip check` | `.venv` do projeto | sem conflitos |
| `docker_build.py <perfil> config`, 5 perfis | env fictício do workflow | válidos |
| `check_pip_audit.py` | `.venv` (160 pacotes); todos os perfis, Windows (288) e Linux (303) | aprova; 84 IDs tolerados, nenhum aviso novo |
| Mutação do gate | ID a menos, versão diferente, sem a exceção, data vencida, aviso novo | todas reprovam; justificativa vazia sai com código 2 |
| Resolução com `constraints.txt` | Linux, Python 3.11 e 3.12, `requirements.txt` e `requirements-cpu.txt`; todos os perfis em 3.11 | resolve (`uv`, com marcadores de Linux) |

**Não verificado aqui:** o workflow no GitHub e o build Docker, que rodam na primeira
execução, e a instalação real em Linux. O Python 3.12 não está instalado nesta máquina,
então em 3.12 as dependências só foram resolvidas, sem executar nada. A resolução usou o
`uv`, que pode escolher versões diferentes das do pip. Pode faltar disco no runner ao
instalar `torch` e CUDA junto com o build da imagem.

### Revisão da interface (2026-11-02)

Auditoria no navegador em 1.440, 768 e 375 px, seguida de ajustes. A organização e o que
mudou por tela estão em [Frontend Gradio](../interfaces/gradio.md) e no `CHANGELOG.md`.
Três achados valem o registro, porque nenhum teste os via:

- **O tema tinha sido escrito para o DOM do Gradio 4.** `.tab-nav`, `.gr-accordion` e
  `.svelte-*` já não existem no 6, e `margin-inline: auto` em `.contain` e `.wrap`
  desligava o esticamento do flex: a página tinha a largura do conteúdo de cada aba. O
  CSS caiu de 1.917 para cerca de 1.400 linhas, só com classes presentes hoje.
- **Opções que não faziam nada.** A escolha do modelo só valia com uma caixa escondida
  marcada, sem aviso. O campo de hiperparâmetros JSON nunca era lido. No assistente, SWA e
  mixup paravam no handler, e calibração de temperatura e limiar EER eram caixas soltas:
  `interactive_training.py` não faz nenhum dos dois. Saíram da tela, sem prometer em troca.
- **A análise por arquivo perdia a calibração.** `_detect_with_config` não repassava
  `classification_threshold` nem `is_ood`, e o JSON afirmava "limiar fixo 0,5" para o
  AASIST, cujo corte é 0,94. Com o limiar à vista, apareceu o caso em que a confiança sozinha
  engana: P(fake) de 0,80 abaixo de um corte de 0,94 dá "real" com confiança de 0,20. Por
  isso o cartão compara P(fake) com o limiar e avisa quando a margem é pequena.

Essa mudança de domínio é aditiva: `metadata` ganha chaves na detecção por arquivo.

### Pontuação centrada no limiar (2026-11-02)

O cartão dizia "provável voz real" com "80,2% de probabilidade de ser sintética" e
confiança de 19,8%. A decisão estava certa e o número enganava: cada modelo decide em
`p_fake >= limiar`, e o limiar vai de 0,05 (SVM) a 0,99 (Hybrid), então o mesmo valor tinha
significados opostos. Com aprovação do usuário, `confidence` passou a ser a confiança no
veredito, e o novo `fake_score` (`detection/operating_point.py`) leva o limiar a 0,5.

A fórmula veio de medição, com 200 áudios de validação por modelo (1.800 inferências; a
partição de teste não foi lida). Nos 37 casos em que P(fake) e o veredito discordam:

| Escala | Confiança média declarada | Acerto real |
| --- | ---: | ---: |
| Linear em dois segmentos (adotada) | 54,6% | 59,5% |
| Translação em logit | 66,4% | 59,5% |

A translação em logit, a primeira hipótese, declararia cerca de 95% para P(fake) de 0,50
com corte em 0,95. Nas faixas gerais da linear, o acerto real fica acima do declarado (0,5
a 0,6: 83,7%; 0,6 a 0,7: 73,9%; 0,7 a 0,8: 94,0%; 0,8 a 0,9: 94,9%), salvo a mais alta
(97,9% contra 98,7%). P(fake) bruta continua mais bem calibrada como probabilidade (ECE
0,039 contra 0,078), então ela segue em `probabilities`, e a pontuação não é chamada de
probabilidade na tela.

Vale em todo caminho do `Predictor` e na janela segmentada. O `DetectionService.detect_batch`,
que gravava a confiança como probabilidade de fake, passou a usar `p_fake`.

### Pendente

- **Decisão: API de treino escolhe o repositório do modelo.** Validar `parameters`
  contra uma lista de checkpoints permitidos fecharia o caminho e encolheria o risco
  residual das exceções.
- **Checkpoints sem `revision`:** `DEFAULT_CHECKPOINTS` e `DEFAULT_AST_CHECKPOINT` usam o
  ramo principal do Hugging Face. Fixar o commit protege e melhora a reprodutibilidade.
- **Analytics do Gradio fora do Docker:** `GRADIO_ANALYTICS_ENABLED` só é desligado no
  Dockerfile e nos composes. Rodando localmente, o servidor consulta
  `https://api.gradio.app/pkg-version`, o que contradiz "processamento 100% local".
- **Código de compatibilidade do Gradio 4:** o shim de `HfFolder` não faz nada (entre as
  dependências, só `transformers/commands/user.py` cita a classe). Remover
  `hf_compat.py`. Para `schema_patch.py`, verificar antes se o Gradio 6 ainda precisa.
- **Figuras no tema claro:** as figuras têm fundo escuro fixo no servidor. Funcionam,
  mas destoam da tela clara. Gerar nos dois temas exige saber o tema no servidor.
- **Microfone na aba Detectar não testado no navegador:** não há microfone neste
  ambiente. O caminho de streaming manteve o código, com a saída trocada pelo cartão.
- **Tela de login:** é do Gradio e não recebe CSS, título nem `head`; os painéis laterais
  ficam como estão.
- **Menu principal no celular:** o Gradio move o que não cabe para o menu "…". As abas
  continuam acessíveis, mas só as primeiras ficam à vista.
- **Notebooks defasados do gerador:** `tools/ops/build_notebooks.py` já não reproduz
  `docs/notebooks/`: gera outra numeração em `models/` (nove arquivos novos) e reescreve
  cinco notebooks. Por isso o `datasets` foi ajustado à mão no notebook versionado e no
  gerador. Falta regenerar e revisar. O comando `git diff --exit-code -- notebooks/` em
  `quality-and-testing.md` também aponta para a pasta antiga.
- **Reconciliar o artigo com os resultados** e gerar o PDF de novo
  ([release-review.md](release-review.md)).
- Os limites `gradio<6.18` e `pillow<13` saem junto com o pino do `transformers`.
- O `ci.yml` não instala com `constraints.txt`, ao contrário do gate de release, e não
  dispara em push para `refactor/**`.
- Fase 3, restante: `benchmarks.planning` consumido pela UI (inverter a
  dependência `app → benchmarks`, que o teste de fronteiras ainda não
  proíbe); código morto do harness que resta do P2-14
  (`data.py::_audio_flat`, `config.py::unseen_speaker_tcc`).
- `app.py` (Spaces) não chama o bootstrap: sem ele não há schema nem
  semeadura até o primeiro treino. Chamar
  `app.dependencies.bootstrap_application()` ali alinharia o Spaces às
  demais interfaces.
- O domínio ainda consulta o ORM direto (`registry.get_active_config`,
  `hyperparameter_defaults`, gravação de `AnalysisResult` em
  `detection_service`, `voice_profile_service`). As regras permitem
  (`domain → core`); passar pelos repositórios tiraria a duplicação das
  consultas de `architecture_configs`.
- O mapa de `docs/development/quality-and-testing.md` lista 73 dos 78 arquivos
  de `tests/unit/` (faltam 5 anteriores à Fase 3).
- Fase 4 concluída na ordem do plano (`training_wizard.py`, `voice_profile_service.py`,
  `forensic_visualization.py`, `trainer.py`, `runner.py`, `layers.py`). Do P2-15 resta
  `tools/dataset/download_datasets.py`, cuja lógica foi para
  `app/domain/services/dataset/download.py` na Fase 3 sem ser repartida.
- **WavLM/HuBERT Original não chegam à inferência:** o carregador procura
  `data/models/benchmark_final/`, e a pasta é `data/models/benchmark/`. Com o
  backbone disponível os `.pt` da geração atual funcionam (WavLM 96,7%, HuBERT 91,7%
  em 60 janelas; a correção de banda não mudou nenhuma decisão), mas estão só em
  `data/results/benchmark/`, sem `input_contract` (limiar fixo 0,5). NÃO basta
  renomear: `data/models/benchmark/` guarda a geração ANTERIOR (`clean_benchmark_15k`,
  sem correção de banda, outro teste), e o carregador dá prioridade à pasta
  promovida, então os 9 modelos corretos seriam trocados pelos antigos. É preciso
  promover a geração atual. Nesta máquina Windows os snapshots do cache do HF são
  links que o `from_pretrained` não resolve.
- **Pasta do RawNet2 no run corrente:** o retreino de 25/08 parou na época 5, e
  `data/results/benchmark/rawnet2/` ficou sem `results.json`. O `run_summary.json` e o
  modelo em uso têm o resultado válido, do run arquivado em
  `_superseded/rawnet2_d9f23c0d` (21/08, 100 épocas, `val_eer`, banda, mesmo teste:
  acurácia 91,17%, EER 8,10%). Falta organizar a pasta, e o `results.json` está no
  arquivo.
- Análise em lote: arquivo que falha entra como `REAL` na tabela e na pizza
  (`is_fake=False`, confiança 0), e o `test_lote_resume_e_exporta_csv` fixa isso
  como está. Uma ferramenta forense deveria marcá-lo como erro.
- **Sidecars recalibrados, `results.json` não:** em 2026-11-02 os sete neurais de
  `data/models` passaram a `temperature_on_logits`, com autorização, pela
  `tools/reporting/recalibrate_temperature.py --apply` (os anteriores estão em
  `*.json.pre-temperature.bak`; `data/models` não é versionado). Nenhum dos 1.456 vereditos de
  validação mudou em nenhum modelo, conferido pelo serviço real. A acurácia em 0,5 e o ECE dos
  `results.json` e do artigo descrevem a calibração anterior; EER, AUC e t-DCF continuam valendo.
- **RawGAT-ST com o limiar dentro de um platô:** as pontuações de áudio sintético param numa
  faixa estreita logo acima do limiar, e por isso o cartão mostra perto de 50% de confiança em
  todo veredito "sintética" (erro de confiança de 25%, com ou sem a recalibração). Vem do
  modelo, não do corte: pede revisão junto da correção de fidelidade da arquitetura.
- **Fusão AASIST + Hybrid CNN-Transformer:** é o conjunto mais barato e o de maior ganho sob
  ruído na validação, em locutor não visto e com a temperatura nova (EER 2,68% a 10 dB e
  7,90% a 5 dB, contra 5,22% e 11,13% do melhor modelo sozinho). Foi escolhido depois de ver
  esses números, e a validação já elegeu a época de cada modelo: falta a medida única no teste.
- **Histórico antigo:** análises gravadas antes da pontuação mantêm a confiança antiga.
- **Veredito sobre silêncio:** silêncio digital recebe decisão (AASIST: fake a 97,9%;
  SVM: fake; Conformer: real). Só há guarda de duração mínima; uma ferramenta forense
  deveria recusar entrada sem sinal.
- Arquivo corrompido devolve `Erro ao carregar arquivo: ` com a mensagem vazia.
- `_code_identity` não cobre `data.py`, `evaluate.py`, `efficiency.py`,
  `stability.py` nem `run_wavlm_original_benchmark.py`: mudar o preparo, as
  métricas ou o runner SSL não invalida o `--resume`. A cobertura foi mantida
  como estava; ampliá-la é decisão, porque cada arquivo a mais força retreino.
- **HNR da aba Investigar é enviesado** (achado da P1-7, não corrigido
  porque muda o que a aba exibe): `forensic_measures.hnr_contour` usa a
  autocorrelação enviesada, sem dividir pela da janela. Num quadro de 25 ms
  a 16 kHz o pico possível no atraso mínimo é (400-32)/400, então o HNR
  nunca passa de ~10,6 dB; um tom PURO de 200 Hz marca 6,0 dB, abaixo da
  linha "voz ruidosa" (7 dB), e a linha "voz limpa" (20 dB) é
  inalcançável. Correção usual: normalizar pela autocorrelação da janela
  (Boersma, 1993).
- `TrainingConfig.export_onnx`/`export_onnx_int8` não são lidas por nada: o
  método do trainer que as usaria nunca teve chamador (removido na divisão).
  Ligar a exportação ao fim do treino ou remover as flags.
- `_scan_dataset` (validação da etapa 1) ainda vive na aba; é varredura de
  disco com mensagens de UI — candidato a ir para o serviço de treino
  interativo junto com o próximo ajuste do assistente.
- `benchmarks/robustness_test.py` ainda usa o caminho legado
  `data/results/models` (script fora do protocolo canônico).
- `benchmarks/data.py::_architecture_input_contract` cai em silêncio no
  contrato `spectrogram` quando o import da factory falha (ex.: sem TF); um
  erro explícito evitaria formatos de entrada errados.
