# Docker e ambientes

Os artefatos Docker foram organizados por perfil de execução, com a estrutura
principal concentrada em [docker/compose](compose).

## Perfis principais

| Perfil | Compose file | Objetivo |
| --- | --- | --- |
| Inferência CPU | [docker/compose/inference.cpu.yml](compose/inference.cpu.yml) | Gradio/FastAPI com modelos treinados, sem CUDA |
| Inferência NVIDIA | [docker/compose/inference.nvidia.yml](compose/inference.nvidia.yml) | Gradio/FastAPI com suporte CUDA/TensorFlow |
| Treino CPU | [docker/compose/train.cpu.yml](compose/train.cpu.yml) | Treino clássico e smoke em CPU |
| Treino NVIDIA | [docker/compose/train.nvidia.yml](compose/train.nvidia.yml) | Treino neural/SSL com GPU |
| Benchmark NVIDIA | [docker/compose/benchmark.nvidia.yml](compose/benchmark.nvidia.yml) | Benchmark sequencial completo |

## Estrutura de pastas

- [compose/](compose): perfis de execução recomendados para novos builds.
- [build.env.example](build.env.example): variáveis comuns para build.
- [../Dockerfile](../Dockerfile): imagem principal do runtime.
- [../docker-entrypoint.sh](../docker-entrypoint.sh): bootstrap do container.
- [environments/](environments): definições específicas por família de ambiente.

## Caminhos padrão

| Host | Container | Uso |
| --- | --- | --- |
| [data](../data) | /app/data | banco SQLite, uploads e dados persistidos |
| [results](../results) | /app/results | saídas de benchmark e artefatos regeneráveis |
| [data/datasets](../data/datasets) | /app/data/datasets | datasets de treino e benchmark |
| [app/models](../app/models) | /app/app/models | modelos de inferência |

Os arquivos raiz com prefixo docker-compose são compatibilidade legada. Para
novos fluxos, prefira os arquivos em [docker/compose](compose) e as definições
em [docker/environments](environments).
