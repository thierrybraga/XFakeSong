# auth

Autenticacao simples por API key.

## Responsabilidade

Fornece dependencias FastAPI para validar `X-API-Key` em rotas protegidas.

## Quando usar

Use apenas nos adaptadores HTTP. O dominio deve receber identidade/permissao ja
resolvida por quem chama.

## Arquivos

- `auth_handler.py`: dependencia `get_api_key` e nome do header.
