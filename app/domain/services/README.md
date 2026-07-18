# services

Casos de uso e servicos de dominio.

## Responsabilidade

Orquestra fluxos como deteccao, treinamento, upload, extracao de features,
perfil de voz e visualizacao forense.

## Quando usar

Use quando uma operacao combina entidades, extratores, modelos, validacoes e
persistencia. Evite colocar logica de UI, request HTTP ou componentes Gradio
nesta pasta.

## Arquivos importantes

- `detection_service.py`: caso de uso de deteccao single, segmentada e multi-modelo.
- `training_service.py`: treino, cross-validation e exportacao.
- `feature_extraction_service.py`: fachada de extracao de features.
- `upload_service.py`: validacao e armazenamento de audio.
- `voice_profile_service.py`: perfis de voz e amostras associadas.
- `forensic_visualization.py`: graficos e analises forenses.
- `detection/`: submodulo de inferencia detalhada.
- `feature_extraction/`: core, loader, tipos e validadores de extracao.
