# XFakeSong - Training and inference environments

This directory separates runtime definitions by computational family. The
current root `Dockerfile` and `requirements*.txt` remain compatible during the
migration; the files here are the target structure for reproducible training,
benchmarking, and lightweight inference.

| Environment | Purpose | Main entrypoint |
| --- | --- | --- |
| `classical-ml` | SVM, Random Forest, tabular audio features | `scripts/train_classical.py` |
| `tensorflow-keras` | TensorFlow/Keras neural models | `scripts/train_tensorflow.py` |
| `pytorch-audio` | Audio architectures grouped for future PyTorch ports; currently TensorFlow-compatible | `scripts/train_pytorch.py` |
| `ssl-transformers` | WavLM, HuBERT and SSL backbones; currently TensorFlow + Torch compatible | `scripts/train_ssl.py` |
| `inference-api` | Gradio/FastAPI inference with trained artifacts | `python main.py --gradio` |

Dockerfile naming:

- `Dockerfile.cpu`: portable CPU/onboard profile, no CUDA device requested.
- `Dockerfile.nvidia`: NVIDIA CUDA profile for Linux/WSL2/Docker GPU.

Shared project volumes:

- `app/datasets`: canonical benchmark datasets.
- `app/models`: default trained artifacts consumed by Gradio/API.
- `results`: benchmark outputs, figures, reports and metrics.
- `cache`: external caches for Hugging Face, Torch and TensorFlow.

The orchestrator remains `scripts/run_models_sequential.py`, which creates one
output directory per model and calls `scripts/run_benchmark.py --model <name>`.

Prefer the segmented compose files under `docker/compose/` for new builds.
