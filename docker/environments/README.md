# Ambientes do projeto

Esta pasta concentra os ambientes Docker específicos usados pelos perfis do
projeto, com foco em inferência, treino clássico e treino com TensorFlow,
PyTorch e transformers SSL.

| Ambiente | Objetivo | Entrada principal |
| --- | --- | --- |
| `classical-ml` | SVM, Random Forest e features tabulares | `scripts/training/train_by_family.py --family classical-ml` |
| `tensorflow-keras` | Modelos neurais TensorFlow/Keras | `scripts/training/train_by_family.py --family tensorflow-keras` |
| `pytorch-audio` | Arquiteturas de áudio em PyTorch | `scripts/training/train_by_family.py --family pytorch-audio` |
| `ssl-transformers` | WavLM, HuBERT e backbones SSL | `scripts/training/train_by_family.py --family ssl-transformers` |
| `inference-api` | Inferência Gradio/FastAPI com artefatos treinados | `python main.py --gradio` |

## Convenção de Dockerfiles

- `Dockerfile.cpu`: perfil portátil CPU/onboard, sem request de GPU.
- `Dockerfile.nvidia`: perfil NVIDIA para Linux/WSL2/Docker com GPU.

## Volumes compartilhados

- [data/datasets](../data/datasets): datasets canônicos de benchmark.
- [app/models](../app/models): artefatos treinados consumidos por Gradio/API.
- [results](../results): métricas, figuras e relatórios.
- [cache](../cache): caches externos de Hugging Face, Torch e TensorFlow.

Prefira os perfis segmentados em [docker/compose](../docker/compose) para novos builds e CI.
