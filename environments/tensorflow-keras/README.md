# tensorflow-keras

Environment for TensorFlow/Keras training:

- Sonic Sleuth
- EfficientNet-LSTM
- MultiscaleCNN
- SpectrogramTransformer

Recommended command:

```bash
python scripts/train_tensorflow.py \
  --dataset app/datasets/benchmark_audio_raw_balanced_15k.npz \
  --epochs 100 \
  --device-profile gpu \
  --out results/tensorflow_benchmark
```

GPU Docker:

```bash
docker compose -f docker/compose/train.nvidia.yml run --rm tensorflow-keras
```

CPU/onboard Docker:

```bash
docker compose -f docker/compose/train.cpu.yml run --rm tensorflow-keras
```
