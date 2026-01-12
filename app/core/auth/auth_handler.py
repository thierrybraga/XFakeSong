from fastapi import Security, HTTPException, status
from fastapi.security import APIKeyHeader
import os
import logging

logger = logging.getLogger(__name__)

API_KEY_NAME = "X-API-Key"
api_key_header = APIKeyHeader(name=API_KEY_NAME, auto_error=False)


async def get_api_key(api_key_header: str = Security(api_key_header)):
    """
    Valida a API Key para proteger endpoints sensíveis.
    """
    SERVER_API_KEY = os.getenv("XFAKESONG_API_KEY")

    # Se não houver chave configurada no servidor, assumimos modo
    # desenvolvimento. Mas logamos um aviso.
    if not SERVER_API_KEY:
        logger.warning(
            "XFAKESONG_API_KEY não configurada. "
            "Autenticação desabilitada (INSEGURO)."
        )
        return None

    if api_key_header == SERVER_API_KEY:
        return api_key_header

    raise HTTPException(
        status_code=status.HTTP_403_FORBIDDEN,
        detail="Credenciais inválidas ou ausentes."
    )
