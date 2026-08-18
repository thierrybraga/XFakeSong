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
| `inference-api` | Inferência Gradio/FastAPI **e o benchmark** | `python main.py --gradio` |

O benchmark completo roda na imagem `inference-api`
([docker/compose/benchmark.nvidia.yml](../compose/benchmark.nvidia.yml)) porque
é a única que reúne TensorFlow-GPU, PyTorch e `transformers` — necessários para
cobrir as 14 arquiteturas numa execução só.

## Convenção de Dockerfiles

- `Dockerfile.cpu`: perfil portátil CPU/onboard, sem request de GPU.
- `Dockerfile.nvidia`: perfil NVIDIA para Linux/WSL2/Docker com GPU.

## PyTorch: CUDA onde treina, CPU onde só lê pesos

Decisão de 2026-07-28. O `torch` aparece em quatro imagens, mas por dois motivos
diferentes:

- **`pytorch-audio`, `ssl-transformers`, `inference-api`** — o PyTorch **treina**
  (WavLM/HuBERT Original via `run_wavlm_original_benchmark.py`). Wheel padrão,
  que no Linux é a build CUDA.
- **`tensorflow-keras`** — o PyTorch apenas **lê `state_dict`**: o AST parte dos
  pesos AudioSet e o port Keras de WavLM/HuBERT carrega o checkpoint via
  `from_pretrained(...).state_dict()`, converte para numpy e descarta o modelo
  Torch. Aqui o wheel vem do índice **CPU**
  (`--index-url https://download.pytorch.org/whl/cpu`).

O motivo de separar: o wheel CUDA do torch 2.5.1 fixa
`nvidia-cudnn-cu12==9.1.0.70`, enquanto o TensorFlow exige 9.3+. Nas imagens em
que os dois convivem, os Dockerfiles corrigem isso com um
`pip install --upgrade --no-deps 'nvidia-cudnn-cu12>=9.3.0.75,<10'` posterior —
funciona, mas deixa a metadata do pip inconsistente. Na imagem Keras o conflito
simplesmente não existe, porque o torch CPU não traz wheel nvidia nenhum.

Sem `transformers` nessa imagem, o AST levantava `ASTPretrainedUnavailable`
(quebrando `make train-nvidia`, já que a família `spectral-attention` inclui o
SpectrogramTransformer) e WavLM/HuBERT degradavam para o CNN-1D do zero.

## Versões travadas

Todas as imagens definem `PIP_CONSTRAINT=/app/constraints.txt`, então **todo**
`pip install` do build respeita os pinos de [constraints.txt](../../constraints.txt).
Os requirements continuam declarando faixas; as constraints fixam a combinação
validada. `tensorflow`, `torch` e os `nvidia-*` ficam de fora de propósito —
variam por imagem (CPU × CUDA).

Os arquivos `requirements.txt` de cada ambiente listam apenas os **extras** além
de `requirements-base.txt`, que eles incluem com `-r ../../../requirements-base.txt`
(três níveis até a raiz — com dois, o `-r` aponta para `docker/` e o build falha).

## Volumes compartilhados

- [data/datasets](../data/datasets): datasets canônicos de benchmark.
- [data/models](../data/models): artefatos treinados consumidos por Gradio/API.
- [data/results](../../data/results): métricas, figuras e relatórios.
- [cache](../cache): caches externos de Hugging Face, Torch e TensorFlow.

Todo build (dev, treino, benchmark, CI) usa os perfis em [docker/compose](../docker/compose).
