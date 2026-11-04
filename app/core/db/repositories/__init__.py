"""Repositórios: as consultas SQLAlchemy das tabelas operacionais.

Routers, abas e serviços pedem dados por aqui em vez de montar consultas
(`app/interfaces/` não importa SQLAlchemy). Cada método abre e fecha a própria
sessão, como o ``ExperimentStore``; entidades devolvidas saem destacadas da
sessão, com as colunas já carregadas. Instâncias únicas em
``app.dependencies``. Importe pelo submódulo.
"""
