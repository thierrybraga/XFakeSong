# Persistência consolidada em SQLite

O arquivo canônico é `data/app.db`, configurável por `DATABASE_URL`. JSON, CSV,
figuras e sidecars de modelos permanecem como exportações e artefatos portáveis;
não são a única fonte de verdade para cálculos ou resultados.

## Modelo de dados

As entidades ORM ficam em `models/` (`base.py`, `analysis.py`,
`architecture_config.py`, `experiment.py`, `training_job.py`, `voice_profile.py`).
Importar `app.core.db.models` registra todas no `Base.metadata`.

As consultas das tabelas operacionais ficam em `repositories/`
(`analysis_results.py`, `training_jobs.py`, `architecture_configs.py`,
`persistence.py`); routers e abas os obtêm por `app.dependencies` e não montam
SQL. `init_db` (`setup.py`) só cria/migra o schema: a semeadura de
`architecture_configs` com o registry de arquiteturas é feita pela composição
(`app.dependencies.bootstrap_application`), para que `app.core` não importe o
domínio. `ExperimentStore.ensure_schema()` (benchmark, treino) cria o schema
sem semear.

- `experiment_runs`: configuração, dataset, ambiente e payload completo da execução.
- `model_runs`: família, parâmetros, hiperparâmetros e contrato de entrada por modelo.
- `metric_records`: métricas normalizadas por condição (`clean`, ruído, codec e eficiência).
- `configuration_entries`: constantes, variáveis de sistema permitidas, manifests e arquivos de configuração, com escopo e hash.
- `system_snapshots`: ambiente reprodutível, hardware e versões; segredos são redigidos.
- `artifact_records`: ligação entre execuções e arquivos gerados.

As tabelas operacionais anteriores (`training_jobs`, `analysis_results`, perfis de
voz e configurações de arquitetura) continuam no mesmo banco para
compatibilidade. Novos consumidores devem usar `ExperimentStore` para resultados
e configuração científica.

## Esquema estranho ao projeto (auditado em 2026-07-31)

O arquivo carregava **18 tabelas de outro domínio** — `workouts`, `exercises`,
`mentors`, `workout_sessions`, `workout_exercises`, `workout_session_sets`,
`events`, `event_registrations`, `event_prize_tiers`, `runs`, `run_points`,
`achievements`, `user_achievements`, `follows`, `store_items`,
`store_redemptions`, `wallet_transactions`, `fraud_flags`. Todas vazias e
**sem uma única referência no código**: o `Base.metadata` do projeto declara 11
tabelas, então elas não vieram do `create_all` — vieram no próprio arquivo,
reaproveitado de outro projeto.

Onze delas declaram FK para `users`, o que faz qualquer limpeza futura de
usuários esbarrar em dependências que não existem no código. `integrity_check`
e `foreign_key_check` passam limpos; o problema é de esquema morto, não de
corrupção.

Desde 2026-10-18 a própria `users` saiu do ORM: a autenticação é por API key
e Basic Auth do Gradio, e a entidade `User` nunca teve chamador. Bancos já
criados mantêm a tabela; o auditor passa a listá-la entre as órfãs.

Auditar e remover:

```bash
python tools/ops/audit_database_schema.py            # só relata
python tools/ops/audit_database_schema.py --apply    # derruba e compacta
```

A lista de tabelas legítimas é derivada do `Base.metadata` (nunca escrita à
mão, senão uma tabela nova viraria "órfã") e nenhuma órfã **com linhas** é
derrubada sem `--force-drop-non-empty`.

## Migração e manutenção

```bash
python tools/ops/consolidate_sqlite.py
```

O comando cria um backup antes da primeira consolidação, aplica o schema, importa
configurações e resultados legados encontrados e registra um snapshot do sistema.
Use `--no-backup` apenas em reexecuções automatizadas. A migração é idempotente.

Verificação rápida:

```bash
python -m pytest tests/unit/test_experiment_store.py
```

Nunca persista todo o ambiente indiscriminadamente. Apenas prefixos permitidos
são coletados, e chaves com nomes de senha, token, segredo ou credencial recebem
`<redacted>` antes da gravação.
