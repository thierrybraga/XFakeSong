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
