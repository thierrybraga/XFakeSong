# contracts

Contratos abstratos canonicos do sistema.

## Responsabilidade

Define os tipos e interfaces que desacoplam dominio, infraestrutura e
adaptadores de entrada.

## Quando usar

Use esta pasta para declarar interfaces, DTOs transversais e objetos de
resultado compartilhados entre camadas.

## Arquivos

- `audio.py`: `AudioData`, `AudioFeatures`, `FeatureType` e contratos de audio.
- `base.py`: `ProcessingResult`, `ProcessingStatus` e interfaces base.
- `services.py`: contratos de servicos, repositorios e metadados.
