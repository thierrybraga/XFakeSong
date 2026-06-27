# classical-ml

Environment for CPU-oriented classical models:

- SVM
- RandomForest
- tabular features such as MFCC, LFCC, prosody, PCA and wavelets

Recommended command:

```bash
python scripts/train_classical.py \
  --dataset app/datasets/benchmark_audio_raw_balanced_15k.npz \
  --epochs 100 \
  --out results/classical_benchmark
```

Docker:

```bash
docker compose -f docker/compose/train.cpu.yml run --rm classical-ml
```
