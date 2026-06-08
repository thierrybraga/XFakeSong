# CI/CD e Segurança

Este documento descreve o pipeline de integração contínua e o modelo de
segurança do XFakeSong. As políticas de reporte e governança estão em
[`SECURITY.md`](https://github.com/thierrybraga/XFakeSong/blob/main/SECURITY.md)
e [`CONTRIBUTING.md`](https://github.com/thierrybraga/XFakeSong/blob/main/CONTRIBUTING.md).

## Pipeline de CI (`.github/workflows/ci.yml`)

Roda em cada `push` (main + `feature/**`) e `pull_request` para a `main`.
Princípio do menor privilégio: `permissions: contents: read`.

| Job | Bloqueia? | O que faz |
| --- | --- | --- |
| **Testes + cobertura** | ✅ sim | `run_tests.sh cov` (suíte rápida, exclui smoke) em Python 3.13 |
| **Docs** | ✅ sim | `mkdocs build --strict` — pega links/refs quebrados |
| **Segurança** | ✅ HIGH | `bandit -lll` (SAST, falha em severidade alta); `pip-audit` advisório |
| **Docker build** | ✅ (só em PR) | `docker build --build-arg TF_VARIANT=cpu` valida o Dockerfile |
| **Lint (ruff)** | ⚠️ advisório | publica achados como anotações (`--exit-zero`) sem bloquear |

Workflows auxiliares:

- **`static.yml`** — publica a documentação (MkDocs) no GitHub Pages a cada push
  na `main`.
- **`dependabot.yml`** — PRs semanais de atualização de dependências `pip` e
  `github-actions`.

Para reproduzir os gates localmente:

```bash
./scripts/run_tests.sh cov                 # testes + cobertura
mkdocs build --strict                      # docs
bandit -r app benchmarks scripts -lll      # SAST (HIGH)
docker build --build-arg TF_VARIANT=cpu .  # valida o Dockerfile
```

## Modelo de segurança da aplicação

| Camada | Mecanismo | Onde |
| --- | --- | --- |
| **Headers HTTP** | `X-Content-Type-Options`, `X-Frame-Options`, `Referrer-Policy`, `Permissions-Policy`, `Cross-Origin-Opener-Policy` em toda resposta; HSTS/CSP opt-in via env | `app/core/middleware.py` |
| **Extração de arquivos** | ZIP/TAR validados contra path traversal (Zip Slip / CVE-2007-4559); `filter='data'` no tar | `app/core/utils/file_utils.py` (`safe_extract_zip/tar`) |
| **Rate limiting** | `slowapi` por IP, decorators nas rotas sensíveis | `app/core/security.py` + `app/routers/*` |
| **Host / CORS** | `TrustedHostMiddleware` (anti DNS-rebinding), CORS configurável por env | `app/core/security.py` |
| **Upload** | limite por `Content-Length` (413), sanitização de filename | `app/core/middleware.py`, `app/core/security.py` |
| **Erros** | RFC 7807 (Problem Details); handler genérico nunca vaza stack interno | `app/core/exceptions.py` |
| **Auth** | mensagem genérica anti-enumeração na recuperação de senha | `app/domain/services/auth_service.py` |

### Variáveis de ambiente de segurança

| Variável | Recomendação em produção |
| --- | --- |
| `ALLOWED_ORIGINS` | lista explícita (não `*`) |
| `ALLOWED_HOSTS` | hosts públicos do serviço |
| `XFAKESONG_API_KEY` | chave forte e secreta |
| `XFAKE_ENABLE_HSTS` | `1` quando atrás de HTTPS |
| `XFAKE_CSP` | política CSP adequada ao deploy |
| `XFAKE_MAX_UPLOAD_MB` | limite de upload (default 100) |

## Reportando vulnerabilidades

Use o **GitHub Private Vulnerability Reporting** (aba *Security* →
"Report a vulnerability"). **Não** abra issue pública. Detalhes em
[`SECURITY.md`](https://github.com/thierrybraga/XFakeSong/blob/main/SECURITY.md).
