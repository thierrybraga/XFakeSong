# Testes SpeechChecker

Esta pasta contém a suíte pytest do projeto. A documentação completa está em
[`docs/development/quality-and-testing.md`](../docs/development/quality-and-testing.md).

## Categorias

| Pasta | Marker | Uso |
|---|---|---|
| `unit/` | `unit` | componentes isolados e regressões rápidas |
| `api/` | `api` | contratos FastAPI com serviços mockados |
| `functional/` | `functional` | fluxos de usuário e rotas/frontend |
| `integration/` | `integration` | cooperação real entre serviços |
| `smoke/` | `smoke` | TensorFlow real, modelos e app, opt-in |

Os markers são aplicados automaticamente por `tests/conftest.py`.

## Comandos

```bash
./tools/ops/run_tests.sh fast
./tools/ops/run_tests.sh unit
./tools/ops/run_tests.sh api
./tools/ops/run_tests.sh integration
./tools/ops/run_tests.sh smoke
./tools/ops/run_tests.sh cov
```

`pytest tests/` exclui `smoke` por padrão via `pyproject.toml`.

## Convenções

- Crie arquivos como `tests/<categoria>/test_<assunto>.py`.
- Use `tmp_path` para arquivos temporários.
- Mocke rede, downloads, pesos grandes e I/O caro.
- Não grave artefatos versionáveis em `data/models/`, `data/results/` ou datasets reais.
- Funções auxiliares não coletáveis devem começar com `_`.
- Notebooks ativos são gerados por `tools/ops/build_notebooks.py`; teste com
  `pytest tests/unit/test_notebooks_compile.py -q`.
