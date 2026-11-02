# CI/CD e Segurança

Este documento descreve o pipeline de integração contínua e o modelo de
segurança do SpeechChecker. As políticas de reporte e governança estão em
[`SECURITY.md`](https://github.com/thierrybraga/XFakeSong/blob/main/SECURITY.md)
e [`CONTRIBUTING.md`](https://github.com/thierrybraga/XFakeSong/blob/main/CONTRIBUTING.md).

## Pipeline de CI (`.github/workflows/ci.yml`)

Roda em cada `push` (main + `feature/**`) e `pull_request` para a `main`.
Princípio do menor privilégio: `permissions: contents: read`.

| Job | Bloqueia? | O que faz |
| --- | --- | --- |
| **Testes + cobertura** | ✅ sim | `run_tests.sh cov` (suíte rápida, exclui smoke) em Python 3.11 |
| **Docs** | ✅ sim | Build limpo `mkdocs build --strict --clean` + verificação de páginas, links e fragmentos |
| **Segurança** | ✅ parcial | `bandit -lll` bloqueia achados HIGH; `pip-audit -r requirements.txt` audita o runtime, mas é advisório no CI normal |
| **Docker build** | ✅ (só em PR) | `docker build --build-arg TF_VARIANT=cpu` + `docker compose config` dos perfis segmentados |
| **Lint (ruff)** | ⚠️ advisório | publica achados como anotações (`--exit-zero`) sem bloquear |

O workflow de release `.github/workflows/release-gate.yml` é estrito e roda em
tags `v*` ou manualmente: bloqueia a promoção se falharem compatibilidade de
dependências (Python 3.11/3.12), Ruff, testes, Bandit HIGH, pip-audit, MkDocs,
build Docker CPU ou validação dos perfis Compose. A política e seus limites
estão em [Dependências e Gates de Release](release-gates.md).

Workflows auxiliares:

- **`static.yml`** — publica a documentação (MkDocs) no GitHub Pages a cada push
  na `main`.
- **`dependabot.yml`** — PRs semanais de atualização de dependências `pip` e
  `github-actions`.

Para reproduzir todos os gates localmente (instalação maior, com todos os extras):

```bash
python -m pip install \
  -r requirements-notebooks.txt \
  -r requirements-docs.txt \
  -r requirements-xai.txt \
  -r requirements-experiments.txt
bash tools/ops/run_tests.sh cov            # testes + cobertura
ruff check app tests benchmarks tools  # lint estrito
bandit -r app benchmarks tools -lll    # SAST (HIGH)
pip-audit                                    # todos os grupos instalados
mkdocs build --strict --clean                # docs
python tools/ops/check_docs_site.py infrastructure/site   # páginas, links e fragmentos
docker build --build-arg TF_VARIANT=cpu .    # imagem CPU
python tools/ops/docker_build.py inference-cpu config
python tools/ops/docker_build.py inference-nvidia config
python tools/ops/docker_build.py train-cpu config
python tools/ops/docker_build.py train-nvidia config
python tools/ops/docker_build.py benchmark-nvidia config
```

Para auditar ou reproduzir os grupos de dependências, veja
[requirements e perfis](release-gates.md#perfis-de-dependencias).

## Modelo de segurança da aplicação

| Camada | Mecanismo | Onde |
| --- | --- | --- |
| **Headers HTTP** | `X-Content-Type-Options`, `X-Frame-Options`, `Referrer-Policy`, `Permissions-Policy`, `Cross-Origin-Opener-Policy` em toda resposta; HSTS/CSP opt-in via env | `app/core/middleware.py` |
| **Extração de arquivos** | ZIP/TAR validados contra path traversal (Zip Slip / CVE-2007-4559); `filter='data'` no tar | `app/utils/file_utils.py` (`safe_extract_zip/tar`) |
| **Rate limiting** | `slowapi` por IP, decorators nas rotas sensíveis | `app/core/security.py` + `app/interfaces/web/routers/*` |
| **Host / CORS** | `TrustedHostMiddleware` (anti DNS-rebinding), CORS configurável por env | `app/core/security.py` |
| **Upload** | limite por `Content-Length` (413), sanitização de filename | `app/core/middleware.py`, `app/core/security.py` |
| **Erros** | RFC 7807 (Problem Details); handler genérico nunca vaza stack interno | `app/core/exceptions.py` |

### Variáveis de ambiente de segurança

| Variável | Recomendação em produção |
| --- | --- |
| `ALLOWED_ORIGINS` | lista explícita (não `*`) |
| `ALLOWED_HOSTS` | hosts públicos do serviço |
| `SPEECHCHECKER_API_KEY` | chave forte e secreta |
| `XFAKE_ENABLE_HSTS` | `1` quando atrás de HTTPS |
| `XFAKE_CSP` | política CSP adequada ao deploy |
| `XFAKE_MAX_UPLOAD_MB` | limite de upload (default 100) |

## Reportando vulnerabilidades

Use o **GitHub Private Vulnerability Reporting** (aba *Security* →
"Report a vulnerability"). **Não** abra issue pública. Detalhes em
[`SECURITY.md`](https://github.com/thierrybraga/XFakeSong/blob/main/SECURITY.md).
