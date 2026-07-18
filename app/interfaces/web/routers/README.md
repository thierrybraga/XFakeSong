# routers

Rotas FastAPI por area funcional.

## Responsabilidade

Mapeia HTTP para servicos de dominio e schemas Pydantic.

## Quando usar

Use para adicionar endpoints, parametros, rate limits e traducao de erros HTTP.

## Arquivos

- `detection.py`: analise single, multi-modelo, incerteza e descoberta.
- `training.py`: treino, status, cross-validation e ONNX.
- `features.py`: extracao e tipos de features.
- `datasets.py`: criacao, upload e remocao de datasets.
- `history.py`: historico de analises.
- `voice_profiles.py`: perfis de voz.
- `system.py`: saude, feedback, versao e info.
