# db

Infraestrutura de banco de dados.

## Responsabilidade

Configura engine, sessoes e bootstrap do banco usado pela API, UI e servicos.

## Quando usar

Use para criar sessoes SQLAlchemy, verificar saude do banco e inicializar tabelas.

## Arquivos

- `session.py`: engine, `SessionLocal`, `Base`, helpers de sessao e healthcheck.
- `setup.py`: criacao de tabelas e seed inicial. Carrega os modelos de dominio
  de forma tardia durante `init_db()` para evitar acoplamento em import-time.
