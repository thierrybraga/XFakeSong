# tensorflow-keras

Environment for TensorFlow/Keras training:

- Sonic Sleuth
- EfficientNet-LSTM
- MultiscaleCNN
- SpectrogramTransformer

Recommended command:

```bash
python scripts/training/train_by_family.py --family tensorflow-keras \
  --dataset data/datasets/benchmark_dataset.npz \
  --epochs 100 \
  --device-profile gpu \
  --out data/results/tensorflow_benchmark
```

GPU Docker:

```bash
docker compose -f docker/compose/train.nvidia.yml run --rm tensorflow-keras
```

CPU/onboard Docker:

```bash
docker compose -f docker/compose/train.cpu.yml run --rm tensorflow-keras
```
