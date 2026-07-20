# Docker, ambientes e execução

Este guia consolida a visão operacional do projeto para Docker, ambientes de treino/inferência e execução local.

## Pontos de entrada

- `main.py`: ponto de entrada principal do CLI e da interface Gradio.
- `app/interfaces/gradio/app.py`: montagem da interface unificada.
- `app.py`: entrada para Hugging Face Spaces.
- `docker/environments/inference-api/docker-entrypoint.sh`: bootstrap do container.

## Estrutura recomendada

- [`docker/compose`](https://github.com/thierrybraga/XFakeSong/tree/main/docker/compose): perfis de execução.
- [`docker/environments`](https://github.com/thierrybraga/XFakeSong/tree/main/docker/environments): Dockerfiles por família de ambiente.
- [`results`](https://github.com/thierrybraga/XFakeSong/tree/main/results): artefatos regeneráveis do benchmark.
- [`data/models`](https://github.com/thierrybraga/XFakeSong/tree/main/data/models): modelos inferidos e exportados.

## Fluxos principais

- Inferência local: `python main.py --gradio`
- Inferência Docker: `docker compose -f docker/compose/inference.cpu.yml up --build inference-api`
- Treino Docker: `docker compose -f docker/compose/train.nvidia.yml run --rm tensorflow-keras`
- Benchmark Docker: `docker compose -f docker/compose/benchmark.nvidia.yml run --rm benchmark`
