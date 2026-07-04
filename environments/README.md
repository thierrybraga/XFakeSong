# XFakeSong - Training and inference environments

This directory separates runtime definitions by computational family. The
current root `Dockerfile` and `requirements*.txt` remain compatible during the
migration; the files here are the target structure for reproducible training,
benchmarking, and lightweight inference.

| Environment | Purpose | Main entrypoint |
| --- | --- | --- |
| `classical-ml` | SVM, Random Forest, tabular audio features | `scripts/training/train_by_family.py --family classical-ml` |
| `tensorflow-keras` | TensorFlow/Keras neural models | `scripts/training/train_by_family.py --family tensorflow-keras` |
| `pytorch-audio` | Audio architectures grouped for future PyTorch ports; currently TensorFlow-compatible | `scripts/training/train_by_family.py --family pytorch-audio` |
| `ssl-transformers` | WavLM, HuBERT and SSL backbones; currently TensorFlow + Torch compatible | `scripts/training/train_by_family.py --family ssl-transformers` |
| `inference-api` | Gradio/FastAPI inference with trained artifacts | `python main.py --gradio` |

Dockerfile naming:

- `Dockerfile.cpu`: portable CPU/onboard profile, no CUDA device requested.
- `Dockerfile.nvidia`: NVIDIA CUDA profile for Linux/WSL2/Docker GPU.

Shared project volumes:

- `app/datasets`: canonical benchmark datasets.
- `app/models`: default trained artifacts consumed by Gradio/API.
- `results`: benchmark outputs, figures, reports and metrics.
- `cache`: external caches for Hugging Face, Torch and TensorFlow.

The orchestrator remains `scripts/benchmark/run_models_sequential.py`, which creates one
output directory per model and calls `scripts/benchmark/run_benchmark.py --model <name>`.

Prefer the segmented compose files under `docker/compose/` for new builds.
