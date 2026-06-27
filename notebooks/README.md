# Notebooks XFakeSong

Estrutura reorganizada para estudo e reprodução dos experimentos do TCC.

## Estrutura

| Pasta | Conteúdo |
|---|---|
| `00_index.ipynb` | Índice executável dos notebooks |
| `pipeline/` | Benchmark completo, todas as arquiteturas, treino e inferência |
| `features/` | Extração e estudo de features |
| `models/` | Um notebook por modelo/arquitetura |

## Ordem recomendada

1. `00_index.ipynb`
2. `features/01_feature_extraction_study.ipynb`
3. `pipeline/01_benchmark_tcc_full_pipeline.ipynb`
4. `pipeline/02_training_model.ipynb`
5. `pipeline/03_inference.ipynb`
6. `pipeline/04_all_architectures_full_benchmark.ipynb`
7. `models/<modelo>.ipynb`

O experimento oficial do TCC usa `scripts/run_tcc_pipeline.py --download
--target-per-class 7500 --full-benchmark`, com alvo de 7.500 amostras reais +
7.500 amostras fake (`app/datasets/benchmark_audio_raw_balanced_20k.npz`).
Para treinar e auditar todas as arquiteturas em um único caderno, use
`pipeline/04_all_architectures_full_benchmark.ipynb`.

## Relação com documentação e interface

| Tema | Documento |
|---|---|
| Benchmark, dataset, artefatos e gráficos | `docs/15_BENCHMARK.md` |
| Uso das abas Gradio e fluxos de análise | `docs/23_INTERFACE_GRADIO.md` |
| Deploy em Hugging Face Spaces | `docs/11_DEPLOY_HUGGINGFACE.md` |
| GitHub Pages e publicação coordenada | `docs/24_PUBLICACAO_GITHUB_HF.md` |

## Geração e validação

Os notebooks ativos (`00_index`, `features/`, `models/`, `pipeline/`) são
**gerados** por `python scripts/build_notebooks.py` — todas as células de código
são validadas com `compile()` e usam a API real do projeto. O teste
`tests/unit/test_notebooks_compile.py` garante que continuam funcionais. O
gerador é determinístico e não depende de TensorFlow no build.
