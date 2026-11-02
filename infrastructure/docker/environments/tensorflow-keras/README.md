# tensorflow-keras

Environment for TensorFlow/Keras training:

- Sonic Sleuth
- EfficientNet-LSTM
- MultiscaleCNN
- SpectrogramTransformer

Recommended command:

```bash
python tools/training/train_by_family.py --family tensorflow-keras \
  --dataset data/datasets/benchmark_dataset_15k.npz \
  --epochs 100 \
  --device-profile gpu \
  --out data/results/tensorflow_benchmark
```

GPU Docker:

```bash
docker compose -f infrastructure/docker/compose/train.nvidia.yml run --rm tensorflow-keras
```

CPU/onboard Docker:

```bash
docker compose -f infrastructure/docker/compose/train.cpu.yml run --rm tensorflow-keras
```
