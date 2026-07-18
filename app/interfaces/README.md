# interfaces

Adaptadores de entrada do sistema.

## Responsabilidade

Traduz interacoes externas para chamadas aos servicos de dominio. Inclui UI
Gradio, API FastAPI e CLI.

## Quando usar

Use para alterar experiencia de usuario, rotas HTTP, schemas de request/response
ou menus CLI. Regras de negocio devem permanecer em `app.domain`.

## Modulos

- `gradio/`: interface visual e abas.
- `web/`: FastAPI, routers, schemas, templates e assets.
- `cli/`: menus interativos e contexto de execucao.

## Dependencias

Interfaces podem depender de `core`, `domain`, `dependencies.py` e bibliotecas
de framework. O sentido inverso nao deve acontecer.
