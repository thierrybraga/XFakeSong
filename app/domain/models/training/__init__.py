"""Módulo de Treinamento de Modelos.

Sem import eager aqui: `augmentation.py`/`optimization.py`/`trainer.py`
importam `tensorflow` no topo do arquivo, mas `metrics.py` (MetricsCalculator)
é puro numpy/sklearn e é usado por `benchmarks/evaluate.py` para avaliar
QUALQUER arquitetura, incl. SVM/RandomForest sem TF instalado (ambiente
Docker "classical-ml"). Um import eager de ModelTrainer/AudioAugmenter aqui
forçaria TensorFlow mesmo para esse caminho puramente clássico.

Todo consumidor real já importa direto do submódulo:
`from app.domain.models.training.trainer import ModelTrainer`,
`from app.domain.models.training.metrics import MetricsCalculator`, etc.
"""
