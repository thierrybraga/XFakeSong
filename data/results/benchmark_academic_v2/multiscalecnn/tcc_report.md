# Relatório de Benchmark para TCC - XFakeSong

## 1. Dataset

- Nome: `benchmark_audio_raw_balanced_15k_academic_v2`
- Total de amostras exportadas: `15572`
- Amostras no teste held-out: `2336`
- Shape de entrada bruto: `[80000, 1]`
- Balanceamento no teste: `{'real': 1133, 'fake': 1203}`
- Caminho de origem: `data/datasets/splits`

## 2. Ambiente de Execução

- Plataforma: `Linux 6.18.33.2-microsoft-standard-WSL2`
- Python: `3.11.15`
- TensorFlow: `2.21.0`
- GPU ativa: `True`
- Dispositivo: `✓ NVIDIA GeForce RTX 3060 · CC 8.6 · Tensor Cores · FP16`

## 3. Configuração Global do Benchmark

- Arquiteturas: `MultiscaleCNN`
- Épocas por arquitetura neural: `100`
- Batch size: `32`
- Semente: `42`
- Testes de robustez SNR: `[30, 20, 10]`
- Medições de latência por arquitetura: `30`
- API probe: `False`

## 4. Resultados Numéricos

| Arquitetura | Status | Acurácia | AUC-ROC | EER | min-tDCF | Latência ms | Params |
|---|---:|---:|---:|---:|---:|---:|---:|
| MultiscaleCNN | ok | 0,9902 | 0,9995 | 0,0107 | 0,0232 | 60,69 | 23707438 |

## 5. Gráficos Agregados

![Curvas ROC](figures/roc.png)

![Matrizes de confusão](figures/confusion_matrices.png)

![Distribuição de scores](figures/score_distributions.png)

![Eficiência](figures/eficiencia.png)

![Convergência](figures/convergencia.png)

![Robustez](figures/robustez.png)

## 6. Hiperparâmetros e Artefatos por Arquitetura

### MultiscaleCNN

- Status: `ok`
- Tipo: `neural`
- Shape preparado: `[100, 80]`
- Treino executado: `48`
- Tempo total: `1734.5` s
- Artefato do modelo: `/app/data/models/bench_multiscalecnn.keras`

Hiperparâmetros/configuração de treino:

```json
{
  "batch_size": 32,
  "epochs": 100,
  "learning_rate": 0.001,
  "lr_is_explicit": true,
  "verbose": 0,
  "progress_log_interval": 1,
  "progress_label": "MultiscaleCNN",
  "validation_split": 0.2,
  "test_split": 0.1,
  "early_stopping": false,
  "early_stopping_patience": 15,
  "reduce_lr_on_plateau": true,
  "reduce_lr_patience": 8,
  "available_architectures": [
    "aasist",
    "conformer",
    "efficientnet_lstm",
    "ensemble",
    "multiscale_cnn",
    "rawgat_st",
    "spectrogram_transformer"
  ],
  "optimizer": "AdamW",
  "loss_function": "binary_crossentropy",
  "metrics": [
    "accuracy",
    "precision",
    "recall",
    "f1"
  ],
  "use_augmentation": false,
  "augmentation_config": {
    "noise_factor": 0.1,
    "snr_range_db": [
      5.0,
      40.0
    ],
    "time_stretch_factor": 0.1,
    "pitch_shift_steps": 2
  },
  "use_class_weighting": true,
  "auto_calibrate_temperature": true,
  "calibration_min_samples": 50,
  "calibrate_under_noise": false,
  "calibration_snr_db": [
    20,
    10
  ],
  "use_swa": false,
  "swa_start_epoch": -1,
  "swa_freq": 1,
  "use_mixup": false,
  "mixup_alpha": 0.2,
  "compute_ood_threshold": true,
  "ood_quantile": 0.95,
  "export_onnx": false,
  "export_onnx_int8": false,
  "use_mixed_precision": true,
  "checkpoint_path": "/app/data/results/benchmark_academic_v2/multiscalecnn/architectures/multiscalecnn/models/best_checkpoint.weights.h5",
  "checkpoint_best": true,
  "best_checkpoint_path": "/app/data/results/benchmark_academic_v2/multiscalecnn/architectures/multiscalecnn/models/best_checkpoint.weights.h5",
  "waveform_noise_protocol": {
    "evaluation_domain": "waveform",
    "frontend_after_noise": true,
    "training_augmentation_domain": "waveform",
    "train_aug_snr_db": [
      30,
      20,
      10
    ],
    "train_noise_copies": 1,
    "waveform_noise_batch_size": 64,
    "assigned_snr_counts": {
      "10": 3633,
      "20": 3633,
      "30": 3633
    },
    "clean_train_samples": 10899,
    "fit_train_samples": 21798,
    "input_type": "spectrogram",
    "original_shape": [
      80000,
      1
    ],
    "prepared_shape": [
      100,
      80
    ],
    "train_crop_strategy": null,
    "eval_crop_strategy": null,
    "eval_num_crops": 1
  },
  "epochs_budget": 100,
  "epochs_executed": 48
}
```

![Matriz de confusão — MultiscaleCNN](architectures/multiscalecnn/confusion_matrix.png)

![ROC — MultiscaleCNN](architectures/multiscalecnn/roc.png)

![Scores — MultiscaleCNN](architectures/multiscalecnn/score_distribution.png)

![Convergência — MultiscaleCNN](architectures/multiscalecnn/convergence.png)

- Métricas completas: `architectures/multiscalecnn/metrics.json`
- Predições limpas: `architectures/multiscalecnn/predictions_clean.csv`
- Predições sob ruído: `architectures/multiscalecnn/predictions_robustness.csv`
- Robustez: `architectures/multiscalecnn/robustness.csv`
