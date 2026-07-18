# features

Sistema de features acusticas.

## Responsabilidade

Implementa extratores, adapters, registry, exportacao e modelos de dados para
caracteristicas acusticas.

## Quando usar

Use para adicionar ou evoluir features acusticas. Para features novas, implemente
o contrato `IFeatureExtractor`, registre no `FeatureExtractorRegistry` e exponha
por adapter quando necessario.

## Arquivos e pastas

- `extractor_registry.py`: registry canonico de extratores.
- `interfaces.py` e `types.py`: reexports dos contratos canonicos.
- `adapters/`: wrappers que adaptam extratores ao contrato.
- `extractors/`: implementacoes por familia acustica.
- `models/`: estruturas de dados de features.
- `exporters/`: exportacao de features.
- `benchmark_frontend.py`: front-end usado para paridade com benchmark.
