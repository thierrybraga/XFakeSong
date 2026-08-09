# Guia do Desenvolvedor

Para quem mantém ou estende o **XFakeSong**. Veja também
[`AGENTS.md`](https://github.com/thierrybraga/XFakeSong/blob/main/AGENTS.md) e
[`CONTRIBUTING.md`](https://github.com/thierrybraga/XFakeSong/blob/main/CONTRIBUTING.md).

## Arquitetura (onde colocar cada coisa)

Clean Architecture — as dependências apontam para dentro (domínio não conhece
framework):

| Camada | Pasta | Conteúdo |
| --- | --- | --- |
| Domínio | `app/domain/` | Modelos/arquiteturas, serviços (detecção, treino, features), regras de negócio |
| Core | `app/core/` | Config, logging, segurança, middleware, exceções, GPU, utilitários, `contracts/` (interfaces SOLID) |
| Interfaces | `app/interfaces/` | UI Gradio (`gradio/tabs/`, `gradio/utils/`), CLI e Web (FastAPI) |
| API HTTP | `app/interfaces/web/routers/` + `app/interfaces/web/schemas/` | Rotas FastAPI e modelos Pydantic |

Regra prática: bibliotecas externas (librosa, TF, sklearn) entram via
adaptadores; o domínio permanece testável sem elas.

## Logging

Configurado em `app/core/feedback.py::configure_logging` (chamado no startup de
`app/interfaces/web/main_fastapi.py`). O arquivo padrão é **`system.log`** na raiz; os
diretórios e nível vêm de `app/core/config/settings.py` (`LoggingConfig`,
`logs_dir = ./data/logs`, resolvido contra a raiz do repositório — **não**
`app/data/logs`).

```python
import logging
logger = logging.getLogger(__name__)

logger.info("Iniciando processamento...")
logger.exception("Falha ao carregar arquivo")  # inclui o traceback
```

Variáveis de ambiente úteis (ver `settings.py` e `app/core/middleware.py`):

| Variável | Efeito |
| --- | --- |
| `DEEPFAKE_ENV` / `DEEPFAKE_DEBUG` | ambiente e modo debug |
| `DEEPFAKE_DEVICE` | dispositivo preferido (CPU/GPU) |
| `XFAKE_LOG_EVERY_REQUEST` | loga toda request (default só erros/lentas) |
| `XFAKE_MAX_UPLOAD_MB` | limite de upload (default 100) |
| `XFAKESONG_API_KEY` | chave da API |
| `ALLOWED_ORIGINS` / `ALLOWED_HOSTS` | CORS e TrustedHost |

## Artefatos gerados

- **Modelos treinados default**: `data/models/` (`bench_*.keras`,
  `bench_*.pkl` + `bench_*_config.json` com o `input_contract`). É o diretório
  carregado pela Gradio/API.
- **Modelos finais completos**: `data/models/benchmark_final/<arquitetura>/`.
  Preserva backbones SSL, README dos modelos originais e artefatos auxiliares.
- **Manifesto de modelos**: `data/models/registry.json`.
- **Resultados/benchmark**: `data/results/` (figuras, JSON/CSV, relatórios,
  métricas por arquitetura e cópia da execução original). Ignorado (gitignore).
  **Convenção de nome para runs novos** (aplica-se daqui pra frente — não é
  uma reorganização retroativa dos runs existentes):
  `data/results/<descrição_curta>_<AAAAMMDD>/`, com `run_summary.json`/`.md` na
  raiz do run e uma subpasta por arquitetura (`<run>/<arquitetura>/results.json`,
  `predictions_clean.csv`, `figures/`, etc. — o formato que
  `scripts/benchmark/run_models_sequential.py` já produz). A promoção para
  `data/models/benchmark_final/` acontece **só** via
  `scripts/reporting/sync_completed_benchmark_artifacts.py --summary
  data/results/<run>/run_summary.json` (o script exige `--summary` explícito, sem
  adivinhar qual é o "run atual").
- **Material acadêmico**: `data/results/paper/` concentra a fonte LaTeX
  (`main.tex`, fonte única), as tabelas geradas do benchmark
  (`tabelas_benchmark.tex`, via `scripts/reporting/update_tcc_latex.py`) e as
  figuras (`figures/`, via `scripts/reporting/consolidate_results.py`).
- **Notebooks**: `notebooks/` — gerados por `scripts/ops/build_notebooks.py`
  (fonte de verdade; não edite o `.ipynb` à mão).

## Como adicionar uma arquitetura

1. Implemente o modelo em `app/domain/models/architectures/<nome>.py`
   (função `create_model(input_shape, num_classes, **kwargs)`).
2. Registre no `factory`/`registry` (`app/domain/models/architectures/`) com o
   `input_requirements` correto (`input_type`: `raw_audio` ou `spectrogram`).
3. Garanta que o **wizard** e o **benchmark** reconhecem o nome (o smoke
   `tests/smoke/test_all_architectures.py` valida a criação de todas).
4. Adicione um notebook em `MODELS` de `scripts/ops/build_notebooks.py` e regenere.

## Como adicionar um extrator de features

1. Crie o extrator em `app/domain/features/extractors/<família>/` e o adapter
   em `app/domain/features/adapters/`.
2. Use uma chave do enum `FeatureType` (`app/core/contracts/audio.py`).
3. Registre no `FeatureExtractorRegistry`. Detalhes em
   [Features de Áudio](../architecture/audio-features.md).

## Dependências

Adicione com **versão mínima** (e upper bound quando houver major arriscado) ao
`requirements.txt` — não use `pip freeze`. Deps de desenvolvimento vão em
`requirements-dev.txt`. O Dependabot mantém tudo atualizado semanalmente.

## Estilo e commits

- Formatação: `black` + `isort`; lint: `ruff` (config em `pyproject.toml`).
- Alvo Python 3.11 (ambiente de referência). Prefira funções puras e testáveis; logue erros com
  `logger.exception`.
- Commits pequenos e escopados; PRs passam pelos gates de CI (abaixo).

## Testes e gates de CI

Espelhe a estrutura de `app/` em `tests/` (ver [Qualidade e Testes](quality-and-testing.md)):

```bash
./scripts/ops/run_tests.sh fast        # suíte rápida (sem smoke)
./scripts/ops/run_tests.sh cov         # + cobertura
mkdocs build --strict              # docs
bandit -r app benchmarks scripts -lll   # SAST (bloqueia HIGH)
python scripts/ops/build_notebooks.py  # regenera notebooks
```

A CI (`.github/workflows/ci.yml`) roda testes+cobertura, docs, segurança
(bandit/pip-audit) e build Docker em PRs. Ver
[CI/CD e Segurança](ci-cd-and-security.md).
