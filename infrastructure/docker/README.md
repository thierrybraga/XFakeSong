# Docker e ambientes

Os artefatos Docker foram organizados por perfil de execução, com a estrutura
principal concentrada em [infrastructure/docker/compose](compose).

## Perfis principais

| Perfil | Compose file | Objetivo |
| --- | --- | --- |
| Inferência CPU | [infrastructure/docker/compose/inference.cpu.yml](compose/inference.cpu.yml) | Gradio/FastAPI com modelos treinados, sem CUDA |
| Inferência NVIDIA | [infrastructure/docker/compose/inference.nvidia.yml](compose/inference.nvidia.yml) | Gradio/FastAPI com suporte CUDA/TensorFlow |
| Treino CPU | [infrastructure/docker/compose/train.cpu.yml](compose/train.cpu.yml) | Treino clássico e smoke em CPU |
| Treino NVIDIA | [infrastructure/docker/compose/train.nvidia.yml](compose/train.nvidia.yml) | Treino neural/SSL com GPU |
| Benchmark NVIDIA | [infrastructure/docker/compose/benchmark.nvidia.yml](compose/benchmark.nvidia.yml) | Benchmark sequencial completo |

## Antes de subir

1. Copie `.env.example` para `.env` na raiz e defina `SPEECHCHECKER_API_KEY`,
   `XFAKE_GRADIO_USERNAME` e `XFAKE_GRADIO_PASSWORD`. Esses valores valem para todos os
   perfis e continuam os mesmos depois de reconstruir as imagens.
2. Use `make`, `start.bat`, `start.sh` ou `tools/ops/docker_build.py`, que já passam
   o arquivo ao Compose. Em um comando manual, inclua `--env-file .env`: sem a flag, o
   Compose procura o `.env` na pasta do arquivo compose.

Os perfis de inferência rodam sempre com `DEEPFAKE_ENV=production` e publicam a porta
só em `127.0.0.1`. Para abrir na rede, defina `GRADIO_BIND_HOST` e use um proxy HTTPS.

## Estrutura de pastas

- [compose/](compose): perfis de execução, único caminho de build do projeto.
- [build.env.example](build.env.example): variáveis comuns para build.
- [environments/](environments): Dockerfiles por família de ambiente;
  [environments/inference-api/docker-entrypoint.sh](environments/inference-api/docker-entrypoint.sh)
  é o bootstrap do container (dirs, writability, sync de modelos do HF Hub).

## Caminhos padrão

| Host | Container | Uso |
| --- | --- | --- |
| [data](../../data) | /app/data | banco SQLite, uploads e dados persistidos |
| [data/results](../../data/results) | /app/data/results | saídas de benchmark e artefatos regeneráveis |
| [data/datasets](../../data/datasets) | /app/data/datasets | datasets de treino e benchmark |
| [data/models](../../data/models) | /app/data/models | modelos de inferência |

Todo fluxo (dev, treino, benchmark, deploy) usa os arquivos em
[infrastructure/docker/compose](compose) e as definições em [infrastructure/docker/environments](environments).
