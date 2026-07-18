# core

Infraestrutura transversal do XFakeSong.

## Responsabilidade

Centraliza configuracao, contratos abstratos, seguranca HTTP, middleware,
conexao de banco, feedback/logging, GPU/runtime e excecoes comuns.

## Quando usar

Use `core` para capacidades genericas que nao pertencem ao dominio de deteccao
de deepfake: configuracao, autenticacao, contratos, middleware, DB e suporte de
runtime.

## Dependencias

`core` pode depender de bibliotecas de infraestrutura, mas deve evitar depender
de regras de negocio. A excecao atual e `core/db/setup.py`, que importa modelos
SQLAlchemy do dominio para registrar metadados; isso deve ser isolado em uma
camada de infraestrutura numa refatoracao futura.

## Arquivos importantes

- `config/settings.py`: configuracoes por ambiente.
- `contracts/`: interfaces SOLID canonicas.
- `db/session.py`: engine e sessoes SQLAlchemy.
- `security.py`: CORS, TrustedHost e rate limiting.
- `middleware.py`: headers, request id e tratamento HTTP transversal.
- `gpu.py` e `performance.py`: setup de TensorFlow/GPU e runtime.
- `interfaces/`: aliases temporarios para `contracts/`.
