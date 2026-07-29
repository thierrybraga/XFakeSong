import logging
import os
import secrets

from fastapi import HTTPException, Security, status
from fastapi.security import APIKeyHeader

logger = logging.getLogger(__name__)

API_KEY_NAME = "X-API-Key"
api_key_header = APIKeyHeader(name=API_KEY_NAME, auto_error=False)


def _env_flag(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def is_production() -> bool:
    """Indica se o processo está executando no perfil de produção."""
    return os.getenv("DEEPFAKE_ENV", "development").strip().lower() == "production"


def insecure_development_allowed() -> bool:
    """Mantém o bypass local somente quando habilitado explicitamente."""
    return not is_production() and _env_flag("XFAKE_ALLOW_INSECURE_DEV", False)


def get_gradio_auth() -> tuple[str, str] | None:
    """Retorna credenciais do Gradio e falha fechado em produção."""
    username = os.getenv("XFAKE_GRADIO_USERNAME", "").strip()
    password = os.getenv("XFAKE_GRADIO_PASSWORD", "")
    if bool(username) != bool(password):
        raise RuntimeError(
            "XFAKE_GRADIO_USERNAME e XFAKE_GRADIO_PASSWORD devem ser "
            "configurados em conjunto."
        )
    if username and password:
        return username, password
    if is_production():
        raise RuntimeError("Autenticação do Gradio é obrigatória em produção.")
    return None


async def get_api_key(
    api_key_header: str | None = Security(api_key_header),
) -> str:
    """Valida a API Key para proteger endpoints sensíveis."""
    server_api_key = os.getenv("XFAKESONG_API_KEY", "")

    if not server_api_key:
        if insecure_development_allowed():
            logger.warning(
                "API sem autenticação por opção explícita de desenvolvimento."
            )
            return "insecure-development"
        logger.warning("XFAKESONG_API_KEY não configurada; acesso protegido recusado.")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Autenticação do servidor não configurada.",
        )

    if api_key_header and secrets.compare_digest(api_key_header, server_api_key):
        return api_key_header

    raise HTTPException(
        status_code=status.HTTP_403_FORBIDDEN,
        detail="Credenciais inválidas ou ausentes.",
    )
