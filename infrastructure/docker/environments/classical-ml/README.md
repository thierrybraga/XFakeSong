# classical-ml

Environment for CPU-oriented classical models:

- SVM
- RandomForest
- tabular features such as MFCC, LFCC, prosody, PCA and wavelets

Recommended command:

```bash
python tools/training/train_by_family.py --family classical-ml \
  --dataset data/datasets/benchmark_dataset_15k.npz \
  --epochs 100 \
  --out data/results/classical_benchmark
```

Docker:

```bash
docker compose --env-file .env -f infrastructure/docker/compose/train.cpu.yml run --rm classical-ml
```
