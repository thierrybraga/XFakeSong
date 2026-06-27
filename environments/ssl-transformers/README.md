# ssl-transformers

Environment for self-supervised audio backbones:

- WavLM
- HuBERT
- wav2vec2-style experiments
- Whisper embeddings, when added

Recommended command:

```bash
python scripts/train_ssl.py \
  --dataset app/datasets/benchmark_audio_raw_balanced_20k.npz \
  --epochs 100 \
  --device-profile gpu \
  --out results/ssl_benchmark
```

Set `HF_HOME` and `TORCH_HOME` to persistent volumes to avoid downloading
backbones in every container run.

Current repository note: WavLM/HuBERT still use the TensorFlow/Keras benchmark
harness for training/evaluation, while original SSL artifacts require
Torch/Transformers. This environment includes both stacks intentionally.

GPU Docker:

```bash
docker compose -f docker/compose/train.nvidia.yml run --rm ssl-transformers
```

CPU/onboard Docker:

```bash
docker compose -f docker/compose/train.cpu.yml run --rm ssl-transformers
```
