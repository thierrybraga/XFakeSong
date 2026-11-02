# Notebooks

Esta pasta contém os notebooks executáveis do SpeechChecker. O catálogo
canônico de finalidade, status, dados de entrada, requisitos e resultados
esperados está no [Guia de Notebooks](../evaluation/notebooks.md).

## Organização

- `00_index.ipynb`: índice navegável.
- `features/`: demonstrações de extração de features.
- `pipeline/`: treinamento, inferência e pipelines de benchmark.
- `models/`: demonstrações por arquitetura.

Os notebooks são gerados por `python tools/ops/build_notebooks.py`. Edite o
gerador, não os arquivos `.ipynb` diretamente. Antes de executar, confira o
status do caderno no guia: alguns são históricos e nenhum resultado sintético
deve ser citado como resultado do artigo.
