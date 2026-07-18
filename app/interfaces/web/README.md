# web

Adaptador HTTP FastAPI.

## Responsabilidade

Expoe a API REST, pagina inicial, assets estaticos e montagem opcional do Gradio.

## Quando usar

Use para alterar contratos HTTP, rotas, templates e bootstrap web.

## Arquivos e pastas

- `main_fastapi.py`: app unificada FastAPI + Gradio.
- `routers/`: endpoints por area funcional.
- `schemas/`: modelos Pydantic de request/response.
- `static/`: CSS/JS servidos pela API.
- `templates/`: templates Jinja2.
