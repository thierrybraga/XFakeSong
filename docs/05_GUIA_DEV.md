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
| Casos de uso | `app/application/` | Orquestração de fluxos (pipelines) |
| Core | `app/core/` | Config, logging, segurança, middleware, exceções, GPU, utilitários |
| Interfaces | `app/interfaces/` | UI Gradio (`gradio/tabs/`, `gradio/utils/`) e CLI |
| API HTTP | `app/routers/` + `app/schemas/` | Rotas FastAPI e modelos Pydantic |

Regra prática: bibliotecas externas (librosa, TF, sklearn) entram via
adaptadores; o domínio permanece testável sem elas.

## Logging

Configurado em `app/core/feedback.py::configure_logging` (chamado no startup de
`app/main_fastapi.py`). O arquivo padrão é **`system.log`** na raiz; os
diretórios e nível vêm de `app/core/config/settings.py` (`LoggingConfig`,
`logs_dir = ./app/logs`).

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

- **Modelos treinados**: `app/models/` (`.keras`/`.pkl` + `_config.json` com o
  `input_contract`). Ignorados pelo git.
- **Resultados/benchmark**: `results/` (figuras, JSON/CSV, relatórios). Ignorado.
- **Notebooks**: `notebooks/` — gerados por `scripts/build_notebooks.py`
  (fonte de verdade; não edite o `.ipynb` à mão).

## Como adicionar uma arquitetura

1. Implemente o modelo em `app/domain/models/architectures/<nome>.py`
   (função `create_model(input_shape, num_classes, **kwargs)`).
2. Registre no `factory`/`registry` (`app/domain/models/architectures/`) com o
   `input_requirements` correto (`input_type`: `raw_audio` ou `spectrogram`).
3. Garanta que o **wizard** e o **benchmark** reconhecem o nome (o smoke
   `tests/smoke/test_all_architectures.py` valida a criação de todas).
4. Adicione um notebook em `MODELS` de `scripts/build_notebooks.py` e regenere.

## Como adicionar um extrator de features

1. Crie o extrator em `app/domain/features/extractors/<família>/` e o adapter
   em `app/domain/features/adapters/`.
2. Use uma chave do enum `FeatureType` (`app/core/interfaces/audio.py`).
3. Registre no `FeatureExtractorRegistry`. Detalhes em
   [Features de Áudio](04_FEATURES.md).

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

Espelhe a estrutura de `app/` em `tests/` (ver [Testes e Qualidade](06_TESTES.md)):

```bash
./scripts/run_tests.sh fast        # suíte rápida (sem smoke)
./scripts/run_tests.sh cov         # + cobertura
mkdocs build --strict              # docs
bandit -r app benchmarks scripts -lll   # SAST (bloqueia HIGH)
python scripts/build_notebooks.py  # regenera notebooks
```

A CI (`.github/workflows/ci.yml`) roda testes+cobertura, docs, segurança
(bandit/pip-audit) e build Docker em PRs. Ver
[CI/CD e Segurança](17_CICD_SEGURANCA.md).
