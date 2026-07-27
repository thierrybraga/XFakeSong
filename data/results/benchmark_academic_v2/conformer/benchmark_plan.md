# Plano de Benchmark

- Preset: `single:Conformer`
- Perfil de dispositivo: `gpu`
- Dataset: `benchmark_audio_raw_balanced_15k_academic_v2`
- Amostras: `15572`
- SNRs: `[30, 20, 10]`
- API probe: `False`

## Hiperparâmetros Efetivos

| Arquitetura | Tipo | Treino | Batch | LR | Ajuste |
|---|---|---:|---:|---:|---|
| Conformer | neural | 100 | 32 | 0.0001 | gpu_vram_safe_cap |