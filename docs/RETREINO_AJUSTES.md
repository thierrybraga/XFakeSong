# Retreino com Ajustes — pós `clean_benchmark_full_20260626`

Documento de rastreio dos ajustes de hiperparâmetros aplicados após o
diagnóstico do último benchmark (14 modelos). Os ajustes já estão **aplicados no
código**; o retreino precisa ser executado em máquina com GPU + dataset.

## Resumo do diagnóstico

| Modelo | Sintoma observado | Evidência |
| --- | --- | --- |
| RawGAT-ST | Overfitting/divergência | `val_acc` cai após época 4; `val_loss` 0.39→1.85; pior limpo (0.833) |
| AASIST | Subajuste | `val_acc` travada ~0.92; recall colapsa a 0.29 @10dB |
| Ensemble | Colapso de robustez | acc 0.50 e recall ~0 @10dB (prediz tudo "real") |
| RandomForest | Overfitting | `mean_train_score=1.0` no tuning; robustez 0.98→0.68 |
| SVM | Robustez fraca | 0.966→0.672 @10dB |
| Hybrid CNN-Transformer | Robustez moderada | 0.973→0.785 @10dB |
| EfficientNet-LSTM | Acurácia limpa baixa | 0.929 (robusto, mas baixo) |
| MultiscaleCNN | Instabilidade de treino | `val_loss=NaN` épocas 4–8 (recuperou) |

Mantidos sem alteração (sólidos e robustos): **Conformer, HuBERT,
SpectrogramTransformer, WavLM, RawNet2**.
**Sonic Sleuth** (1.0 perfeito) — auditar vazamento antes de confiar
(`scripts/audit_dataset_leakage.py`), não retreinado por ora.

## Ajustes aplicados

| Modelo | Arquivo | Mudança |
| --- | --- | --- |
| RawGAT-ST | `registry.py` | dropout 0.2→0.35; l2 5e-4→1e-3; clip 0.8→0.5; aug 0.3→0.4; patience 18→25 |
| RawGAT-ST | `rawgat_st.py` | LR 1e-4→5e-5; `global_clipnorm` 1.0→0.7 |
| AASIST | `registry.py` | l2 5e-4→2e-4; aug 0.25→0.35; patience 20→25 |
| AASIST | `aasist.py` | LR 1e-4→3e-4; `weight_decay` 0.01→1e-3 |
| Ensemble | `registry.py` | `augmentation_strength` 0.3→0.45; patience 15→20 |
| Hybrid CNN-Transformer | `registry.py` | dropout 0.1→0.2; stochastic-depth 0.1→0.15; aug 0.3→0.4 |
| EfficientNet-LSTM | `registry.py` | dropout 0.3→0.25; patience 15→20 |
| MultiscaleCNN | `multiscale_cnn.py` | Adam `clipnorm=1.0` (anti-NaN) |
| RandomForest | `random_forest.py` | grid regularizado: max_depth sem `None`/30; `min_samples_leaf` [2,4,8]; `min_samples_split` [5,10,20] |

Augmentation de ruído no treino (controlada por `use_augmentation`/`snr_range_db`
= (5,40) em `app/core/config/settings.py`) já cobre a faixa de robustez avaliada
(10/20/30 dB) — é a principal alavanca para Ensemble, SVM, RandomForest e AASIST.
A calibração de temperatura e o threshold automático (`auto_calibrate_temperature`)
recalibram a decisão do Ensemble pós-treino.

## Como retreinar (na máquina com GPU)

```bash
# Linux / WSL2 / Docker GPU
bash scripts/retrain_ajustado.sh

# Windows
scripts\retrain_ajustado.bat
```

Roda apenas os 8 modelos ajustados (um por vez, `--resume`, 120 épocas, SNR
30/20/10) em `results/retune_ajustado_<data>/`. Os modelos não ajustados não são
retreinados.

## Verificação (antes de promover)

1. `python scripts/consolidate_results.py --results results/retune_ajustado_<data>`
2. `python scripts/validate_artifacts.py --results results/retune_ajustado_<data>`
3. Comparar `accuracy`/`f1`/`eer` e a curva de robustez (10 dB) com o baseline.
4. Só então sincronizar para `app/models/benchmark_final`:
   `python scripts/sync_completed_benchmark_artifacts.py --results results/retune_ajustado_<data>`

> Importante: promova um modelo só se ele melhorar (ou empatar) o baseline,
> especialmente a robustez a 10 dB. Caso contrário, mantenha o artefato anterior.

## Qualidade de treino (P1) — status

Ajustes de qualidade aplicados para o retreino dos novos modelos:

1. **Augmentation de ruído casada com o teste — JÁ ATIVO.** O benchmark treina
   redes com `use_augmentation` (ruído SNR + SpecAug) e os clássicos (SVM/RF)
   com `classical_noise_augmentation` (default True), anexando cópias do treino
   com AWGN nos SNRs avaliados (`benchmarks/runner.py`). A faixa global é
   `snr_range_db=(5,40)` (`settings.py`), cobrindo 10/20/30 dB.
2. **Calibração de temperatura/threshold sob ruído — APLICADO.** `settings.py`
   ganhou `calibrate_under_noise` (default True) + `calibration_snr_db=[20,10]`;
   o `ModelTrainer` (`_build_calibration_set`) calibra temperatura, EER e OOD
   num val que inclui cópias com AWGN — corrige o ponto de operação (ex.: colapso
   de recall do Ensemble) antes medido só em áudio limpo.
3. **Early stopping `val_loss` + `restore_best_weights` — JÁ ATIVO.** Callbacks do
   `ModelTrainer` usam `monitor="val_loss"`, `restore_best_weights=True` e
   paciência da config/registry (RawGAT-ST/AASIST já com paciência maior).
4. **Split disjunto por falante (tier `large`) — APLICADO.**
   `run_models_sequential.py` ganhou `--speaker-split`/`--group-split` (repassados
   ao `run_benchmark.py`) e os scripts `retrain_ajustado.sh/.bat` já passam
   `--speaker-split` (requer `.npz` tier large com `speaker_ids`; cai para o
   estratificado com aviso se ausente).
