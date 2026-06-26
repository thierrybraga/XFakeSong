# XFakeSong Docker Profiles

Docker assets are organized by execution profile:

| Profile | Compose file | Purpose |
| --- | --- | --- |
| Inference CPU/onboard | `docker/compose/inference.cpu.yml` | Gradio/FastAPI with trained models, no CUDA |
| Inference NVIDIA | `docker/compose/inference.nvidia.yml` | Gradio/FastAPI with CUDA-capable TensorFlow |
| Training CPU/onboard | `docker/compose/train.cpu.yml` | Classical ML and CPU smoke training |
| Training NVIDIA/WSL2 | `docker/compose/train.nvidia.yml` | Neural/SSL training with NVIDIA GPU |
| Benchmark NVIDIA/WSL2 | `docker/compose/benchmark.nvidia.yml` | Full sequential benchmark |

CPU/onboard means the container does not request GPU devices. This is the
portable profile for Windows native Docker, Intel/AMD integrated graphics and
machines without NVIDIA CUDA.

NVIDIA profiles require WSL2/Docker Desktop GPU support on Windows, or NVIDIA
Container Toolkit on Linux. Validate with:

```bash
docker run --rm --gpus all nvidia/cuda:12.4.1-base-ubuntu22.04 nvidia-smi
```

Use `scripts/docker_build.py` for a stable CLI over these files.

```bash
python scripts/docker_build.py inference-cpu config
python scripts/docker_build.py inference-nvidia up
python scripts/docker_build.py benchmark-nvidia run
```

Runtime paths are standardized across profiles:

| Host path | Container path | Use |
| --- | --- | --- |
| `data/app.db` | `/app/data/app.db` | SQLite runtime database |
| `data/uploads/` | `/app/data/uploads/` | Gradio/API uploads |
| `results/` | `/app/results/` | Regenerable benchmark outputs |
| `app/datasets/` | `/app/app/datasets/` | Benchmark/training datasets |
| `app/models/` | `/app/app/models/` | Inference model root |
| `app/models/benchmark_final/` | `/app/app/models/benchmark_final/` | Consolidated trained models |

Root-level `docker-compose*.yml` files are legacy compatibility aliases. Prefer
`docker/compose/*.yml` for new builds and CI validation.
