# detection

Submodulo de inferencia e preparacao de entrada.

## Responsabilidade

Carrega artefatos, resolve contratos de entrada, prepara tensores/features e
executa predicoes por TensorFlow, PyTorch SSL, ONNX ou scikit-learn.

## Quando usar

Use para alterar o caminho de inferencia real. Mudancas aqui exigem testes de
paridade treino-inferencia.

## Arquivos

- `model_loader.py`: descoberta e carregamento lazy de artefatos.
- `feature_preparer.py`: resolve `input_contract` e prepara raw/spectrogram/tabular.
- `predictor.py`: pos-processamento, thresholds, temperatura, OOD e TTA.
- `audio_preprocessing.py`: front-end LFCC/log-mel compartilhado.
- `utils.py`: helpers locais do submodulo.
