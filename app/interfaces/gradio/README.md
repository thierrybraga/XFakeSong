# gradio

Interface visual Gradio.

## Responsabilidade

Monta a aplicacao visual com abas de painel, deteccao, investigacao, treinamento
e gerenciamento.

## Quando usar

Use para alterar componentes, callbacks, layout e integracao da UI com servicos
de dominio.

## Arquivos e pastas

- `app.py`: montagem principal do `gr.Blocks`.
- `schema_patch.py`: compatibilidade Gradio/Pydantic/Starlette.
- `tabs/`: telas funcionais da UI.
- `utils/`: helpers de UI, graficos, i18n e notificacoes.
