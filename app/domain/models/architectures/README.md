# architectures

Arquiteturas de deteccao de deepfake.

## Responsabilidade

Implementa os modelos neurais e classicos expostos pelo factory/registry.

## Quando usar

Use ao adicionar uma arquitetura ou alterar seu contrato de entrada. Qualquer
mudanca deve preservar `create_model(input_shape, num_classes, **kwargs)` ou
registrar claramente uma excecao.

## Arquivos principais

- `factory.py`: criacao por nome.
- `registry.py`: metadados, aliases e requisitos de entrada.
- `layers.py`: camadas compartilhadas.
- `*_cnn*`, `aasist.py`, `rawgat_st.py`, `rawnet2.py`, `wavlm.py`, `hubert.py`: arquiteturas.
- `svm.py`, `random_forest.py`, `classical_ml_helpers.py`: modelos classicos.
