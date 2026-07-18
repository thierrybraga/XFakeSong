# inference

Helpers de inferencia e exportacao.

## Responsabilidade

Suporta exportacao e execucao ONNX para modelos treinados.

## Quando usar

Use quando a preocupacao for portabilidade/execucao de artefatos, nao o fluxo de
predicao completo. O fluxo principal fica em `domain/services/detection`.

## Arquivos

- `onnx_export.py`: exportacao, quantizacao e sessao ONNX.
