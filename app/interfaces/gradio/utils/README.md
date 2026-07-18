# utils

Helpers especificos da interface Gradio.

## Responsabilidade

Agrupa componentes visuais reutilizaveis, graficos, internacionalizacao,
notificacoes e helpers de hiperparametros.

## Quando usar

Use para codigo de UI compartilhado por multiplas abas. Nao coloque regras de
dominio aqui.

## Arquivos

- `components.py`: componentes padronizados.
- `plotting.py`: helpers Matplotlib/Plotly e lock de servico.
- `notifications.py`: adaptacao do feedback para UI.
- `i18n.py`: textos e idioma.
- `hyperparameters.py`, `tuning_charts.py`: UI de tuning.
- `training_wizard_presenter.py`: cards de modelos e navegacao visual do
  wizard de treinamento.
