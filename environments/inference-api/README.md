# inference-api

Lightweight runtime for Gradio and FastAPI inference. It consumes trained models
from `app/models` and should not run benchmark or full training jobs.

Recommended command:

```bash
python main.py --gradio --gradio-port 7860
```

Docker:

```bash
docker compose -f docker/compose/inference.cpu.yml up --build inference-api
docker compose -f docker/compose/inference.nvidia.yml up --build inference-api
```
