"""Infraestrutura transversal do SpeechChecker.

Configuração (`config/`), persistência (`db/`), autenticação (`auth/`),
contratos SOLID (`contracts/`), segurança HTTP, GPU e logging.

Este pacote não re-exporta nomes: importe sempre do submódulo, por exemplo
``from app.core.contracts.base import ProcessingResult``. Um ``__init__``
com re-exports carregava FastAPI, numpy e todos os contratos a cada
``import app.core.<qualquer coisa>``.
"""
