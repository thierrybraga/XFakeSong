# Notebooks XFakeSong

Estrutura reorganizada para estudo e reprodução dos experimentos do TCC.

## Estrutura

| Pasta | Conteúdo |
|---|---|
| `00_index.ipynb` | Índice executável dos notebooks |
| `pipeline/` | Benchmark completo, treino e inferência |
| `features/` | Extração e estudo de features |
| `models/` | Um notebook por modelo/arquitetura |

## Ordem recomendada

1. `00_index.ipynb`
2. `features/01_feature_extraction_study.ipynb`
3. `pipeline/01_benchmark_tcc_full_pipeline.ipynb`
4. `pipeline/02_training_model.ipynb`
5. `pipeline/03_inference.ipynb`
6. `models/<modelo>.ipynb`

O experimento oficial do TCC usa `scripts/run_tcc_pipeline.py --tcc-full-dataset`,
com alvo de 10.000 amostras reais + 10.000 amostras fake.

## Geração e validação

Os notebooks ativos (`00_index`, `features/`, `models/`, `pipeline/`) são
**gerados** por `python scripts/build_notebooks.py` — todas as células de código
são validadas com `compile()` e usam a API real do projeto. O teste
`tests/unit/test_notebooks_compile.py` garante que continuam funcionais. O
gerador é determinístico e não depende de TensorFlow no build.
