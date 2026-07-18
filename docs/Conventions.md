# Convencoes

## Camadas

| Camada | Pode depender de | Nao deve depender de |
| --- | --- | --- |
| `domain` | `core.contracts`, `utils`, libs de ML/DSP encapsuladas | `interfaces` |
| `core` | libs de infra, `utils` | regras de negocio especificas |
| `interfaces` | `domain`, `core`, frameworks | detalhes de persistencia interna sem servico |

## Contratos de modelo

Todo artefato treinado deve preservar `_config.json` com `input_contract`:

- `input_type`;
- `input_shape`;
- `sample_rate`;
- `feature_frontend`;
- parametros de front-end;
- `temperature`;
- `eer_threshold`;
- `ood_threshold` quando disponivel.

## Documentacao

- Cada pasta fonte principal deve ter `README.md`.
- Mudancas arquiteturais devem atualizar `Architecture.md`, `FolderStructure.md`
  ou `RefactoringReport.md`.
- Mermaid deve ser usado para fluxos e dependencias.
