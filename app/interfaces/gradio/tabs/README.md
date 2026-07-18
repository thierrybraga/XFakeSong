# tabs

Abas funcionais da interface Gradio.

## Responsabilidade

Cada arquivo monta uma area da UI e seus callbacks.

## Quando usar

Use para mudar fluxos visuais. Se uma funcao de callback crescer demais, extraia
logica para `domain/services` ou para um helper de UI em `gradio/utils`.

## Arquivos

- `dashboard.py`: KPIs e saude do sistema.
- `detection.py`: analise de audio e inferencia.
- `forensic_analysis.py`: investigacao e graficos.
- `training.py` e `training_wizard.py`: treino guiado.
- `dataset_management.py`: gestao de datasets.
- `features.py`: extracao de features pela UI.
- `history.py`: historico de analises.
- `voice_profiles.py`: perfis de voz.
- `optimization.py`: tuning e otimizacao.
