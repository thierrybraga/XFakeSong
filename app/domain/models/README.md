# models

Modelos de ML do dominio: arquiteturas e pipelines de treino/inferencia.

## Responsabilidade

Agrupa arquiteturas de deteccao, treinamento e helpers de inferencia. As
entidades persistidas (ORM) ficam em `app/core/db/models/`.

## Quando usar

Use para evoluir arquiteturas, configuracoes de treino, metricas e carregadores
de inferencia que pertencem ao dominio ML.

## Pastas e arquivos

- `architectures/`: 14 arquiteturas e registry/factory.
- `training/`: treinamento, augmentation, metricas, otimizadores e tuning.
- `inference/`: exportacao/execucao ONNX.
