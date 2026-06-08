# Política de Segurança

## Versões suportadas

Apenas a versão mais recente do XFakeSong recebe correções de segurança.

| Versão | Suportada          |
| ------ | ------------------ |
| 1.1.x  | :white_check_mark: |
| < 1.1  | :x:                |

## Reportando uma vulnerabilidade

**Não abra uma issue pública.** Use o canal privado do GitHub:

1. Acesse a aba **Security** do repositório
   (`https://github.com/thierrybraga/XFakeSong/security`).
2. Clique em **"Report a vulnerability"** (GitHub Private Vulnerability Reporting).
3. Descreva o problema com passos de reprodução, impacto e versão afetada.

Faremos a triagem o quanto antes e coordenaremos a divulgação responsável
(correção antes da publicação). Pedimos um prazo razoável antes de qualquer
divulgação pública.

## Práticas de segurança do projeto

- **SAST**: `bandit` roda na CI e **bloqueia** achados de severidade HIGH
  (`.github/workflows/ci.yml`).
- **Dependências**: monitoradas pelo Dependabot (`.github/dependabot.yml`) e
  auditadas por `pip-audit` na CI.
- **Extração de arquivos**: ZIP/TAR são extraídos com validação contra path
  traversal (Zip Slip / CVE-2007-4559) — ver `safe_extract_zip`/`safe_extract_tar`
  em `app/core/utils/file_utils.py`.
- **Headers HTTP**: respostas incluem `X-Content-Type-Options`,
  `X-Frame-Options`, `Referrer-Policy`, `Permissions-Policy` e
  `Cross-Origin-Opener-Policy`; HSTS e CSP são opt-in via env.
- **Hardening da API**: rate limiting (`slowapi`), `TrustedHostMiddleware`,
  CORS configurável, limite de upload e sanitização de nomes de arquivo.
- **Container**: imagem multi-stage slim, executando como usuário não-root.

## Configuração recomendada em produção

| Variável            | Recomendação                              |
| ------------------- | ----------------------------------------- |
| `ALLOWED_ORIGINS`   | lista explícita de origens (não use `*`)  |
| `ALLOWED_HOSTS`     | hosts públicos do serviço                 |
| `XFAKESONG_API_KEY` | chave forte e secreta                     |
| `XFAKE_ENABLE_HSTS` | `1` quando atrás de HTTPS/TLS             |
| `XFAKE_CSP`         | política CSP adequada ao seu deploy       |
