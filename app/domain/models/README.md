# models

Modelos de dominio, arquiteturas e pipelines de ML.

## Responsabilidade

Agrupa entidades persistidas, arquiteturas de deteccao, treinamento e helpers de
inferencia.

## Quando usar

Use para evoluir arquiteturas, configuracoes de treino, metricas, entidades
persistidas e carregadores de inferencia que pertencem ao dominio ML.

## Pastas e arquivos

- `architectures/`: 14 arquiteturas e registry/factory.
- `training/`: treinamento, augmentation, metricas, otimizadores e tuning.
- `inference/`: exportacao/execucao ONNX.
- `base_model.py`, `user.py`, `training_job.py`, `voice_profile.py`, `analysis.py`: modelos persistidos.
