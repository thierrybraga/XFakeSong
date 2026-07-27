# Benchmark XFakeSong — resumo

- Ambiente: Linux 6.18.33.2-microsoft-standard-WSL2 · Python 3.11.15 · TF 2.21.0 · GPU=True
- Dispositivo: ✓ NVIDIA GeForce RTX 3060 · CC 8.6 · Tensor Cores · FP16
- Dataset: **benchmark_audio_raw_balanced_15k_academic_v2** — 15572 amostras (teste held-out: 2336 → {'real': 1133, 'fake': 1203})

| Arquitetura | Status | Conv. | Acur. | EER | AUC | min-tDCF | Lat.(ms) | Params |
|---|---|---|---|---|---|---|---|---|
| SpectrogramTransformer | ok | ✅ | 96,23\% | 3,64\% | 0,993 | 0,0967 | 50.58 | 85304834 |