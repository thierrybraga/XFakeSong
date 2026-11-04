# training

Pipeline de treinamento.

## Responsabilidade

Fornece trainer, metricas, augmentations, otimizadores, split seguro, tuning,
SWA e pruning.

## Quando usar

Use para mudar comportamento de treino, calibracao, metricas ou salvamento de
artefatos. Mudancas aqui costumam exigir testes unitarios e smoke quando tocam
TensorFlow real.

## Arquivos

- `trainer.py`: `ModelTrainer`, orquestrador de treino e salvamento.
- `callbacks.py`: callbacks Keras do treino (checkpoint retomável,
  histórico persistente, guarda de colapso, `val_eer`, progresso).
- `calibration.py`: calibração pós-treino (ruído, temperatura, OOD, EER).
- `artifacts.py`: `.keras` de inferência e `input_contract`.
- `secure_training_pipeline.py`: splits e protecoes contra leakage.
- `augmentation.py`, `rawboost.py`, `spec_augment.py`: augmentations.
- `metrics.py`: metricas anti-spoofing e classificacao; EER de validacao
  (`validation_eer`) sem TensorFlow.
- `optimization.py`: otimizadores e schedules.
- `hyperparameter_defaults.py`, `hyperparameter_tuning.py`: defaults e busca.
- `swa_callback.py`, `magnitude_pruning.py`: tecnicas opcionais.
