# Persistência consolidada em SQLite

O arquivo canônico é `data/app.db`, configurável por `DATABASE_URL`. JSON, CSV,
figuras e sidecars de modelos permanecem como exportações e artefatos portáveis;
não são a única fonte de verdade para cálculos ou resultados.

## Modelo de dados

- `experiment_runs`: configuração, dataset, ambiente e payload completo da execução.
- `model_runs`: família, parâmetros, hiperparâmetros e contrato de entrada por modelo.
- `metric_records`: métricas normalizadas por condição (`clean`, ruído, codec e eficiência).
- `configuration_entries`: constantes, variáveis de sistema permitidas, manifests e arquivos de configuração, com escopo e hash.
- `system_snapshots`: ambiente reprodutível, hardware e versões; segredos são redigidos.
- `artifact_records`: ligação entre execuções e arquivos gerados.

As tabelas operacionais anteriores (`training_jobs`, `analysis_results`, usuários,
perfis de voz e configurações de arquitetura) continuam no mesmo banco para
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

Auditar e remover:

```bash
python scripts/ops/audit_database_schema.py            # só relata
python scripts/ops/audit_database_schema.py --apply    # derruba e compacta
```

A lista de tabelas legítimas é derivada do `Base.metadata` (nunca escrita à
mão, senão uma tabela nova viraria "órfã") e nenhuma órfã **com linhas** é
derrubada sem `--force-drop-non-empty`.

## Migração e manutenção

```bash
python scripts/ops/consolidate_sqlite.py
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
