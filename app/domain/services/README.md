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
- `voice_profile_service.py`: perfis de voz e amostras associadas; o treino
  orquestra as etapas de `voice_profile/` (`features.py`: vetor de 144
  features e fakes sintéticos; `models.py`: um modelo por família de
  arquitetura; `training.py`: split, treino Keras/clássico e EER).
- `interactive_training.py`: treino do assistente do Gradio (dados, front-end,
  treino, avaliação, gravação do modelo); emite eventos de progresso que a aba
  só renderiza.
- `training_jobs.py`: ciclo de vida dos jobs de treino/K-fold disparados pela API
  (criação, execução em background, status com cache TTL); o router só valida e
  delega.
- `forensic_measures.py`: medidas da analise forense (formantes, HNR,
  jitter/shimmer, F0, DET/EER, radar); os graficos ficam em
  `app/interfaces/gradio/presenters/forensic_plots.py`.
- `detection/`: submodulo de inferencia detalhada.
- `feature_extraction/`: core, loader, tipos e validadores de extracao.
- `dataset/`: aquisicao (`download.py`) e pre-processamento (`preprocessing.py`)
  de datasets — logica das CLIs `tools/dataset/download_datasets.py` e
  `preprocess_dataset.py` e da aba de datasets do Gradio.
