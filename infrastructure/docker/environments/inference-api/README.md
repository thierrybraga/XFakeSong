# inference-api

Lightweight runtime for Gradio and FastAPI inference. It consumes trained models
from `data/models` and should not run benchmark or full training jobs.

Recommended command:

```bash
python main.py --gradio --gradio-port 7860
```

Docker:

```bash
docker compose --env-file .env -f infrastructure/docker/compose/inference.cpu.yml up --build inference-api
docker compose --env-file .env -f infrastructure/docker/compose/inference.nvidia.yml up --build inference-api
```

Inside the container the app always listens on 7860. On the host, the CPU profile
publishes 7861 and the NVIDIA profile publishes 7860. Port map:
`docs/development/docker-environments.md`.
