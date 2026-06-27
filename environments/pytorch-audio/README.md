# pytorch-audio

Environment for PyTorch-oriented audio architectures:

- RawNet2
- AASIST
- RawGAT-ST
- Conformer
- Hybrid CNN-Transformer

Current repository note: these architectures are implemented through the
TensorFlow/Keras benchmark harness today. This environment therefore includes
TensorFlow plus Torch/Torchaudio so the existing registry and future native
PyTorch ports can coexist during migration.

Recommended command:

```bash
python scripts/train_pytorch.py \
  --dataset app/datasets/benchmark_audio_raw_balanced_20k.npz \
  --epochs 100 \
  --device-profile gpu \
  --out results/pytorch_benchmark
```

GPU Docker:

```bash
docker compose -f docker/compose/train.nvidia.yml run --rm pytorch-audio
```

CPU/onboard Docker:

```bash
docker compose -f docker/compose/train.cpu.yml run --rm pytorch-audio
```
