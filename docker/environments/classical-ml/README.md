# classical-ml

Environment for CPU-oriented classical models:

- SVM
- RandomForest
- tabular features such as MFCC, LFCC, prosody, PCA and wavelets

Recommended command:

```bash
python scripts/training/train_by_family.py --family classical-ml \
  --dataset data/datasets/benchmark_dataset.npz \
  --epochs 100 \
  --out data/results/classical_benchmark
```

Docker:

```bash
docker compose -f docker/compose/train.cpu.yml run --rm classical-ml
```
