# web

Adaptador HTTP FastAPI.

## Responsabilidade

Expoe a API REST e monta a interface visual Gradio. A raiz redireciona para a
UI canônica; em modo API-only, redireciona para a documentação Swagger.

## Quando usar

Use para alterar contratos HTTP, redirecionamentos e bootstrap da aplicação web.

## Arquivos e pastas

- `main_fastapi.py`: app unificada FastAPI + Gradio e rotas de redirecionamento.
- `routers/`: endpoints por area funcional.
- `schemas/`: modelos Pydantic de request/response.
