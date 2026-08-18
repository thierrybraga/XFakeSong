"""Configuração de segurança centralizada.

- CORS configurável via env
- Rate limiting global
- Sanitização robusta de filenames (URL-encoding, directory traversal)
- Trusted host protection
"""

import os
import re
import urllib.parse

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from slowapi import Limiter
from slowapi.util import get_remote_address

from app.core.auth.auth_handler import is_production

# Carregar variáveis de ambiente
load_dotenv()

# Inicializar Limiter para Rate Limiting
limiter = Limiter(key_func=get_remote_address)


def setup_security(app: FastAPI):
    """Configura middlewares e falha fechado quando o perfil é produção."""
    production = is_production()

    # 1. CORS — restrito em produção, aberto apenas no desenvolvimento.
    origins_str = os.getenv("ALLOWED_ORIGINS", "" if production else "*").strip()
    if production and origins_str == "*":
        raise RuntimeError("ALLOWED_ORIGINS='*' não é permitido em produção.")
    if origins_str == "*":
        allow_origins = ["*"]
    else:
        allow_origins = [
            origin.strip() for origin in origins_str.split(",") if origin.strip()
        ]

    app.add_middleware(
        CORSMiddleware,
        allow_origins=allow_origins,
        allow_credentials=bool(allow_origins) and origins_str != "*",
        allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
        allow_headers=["Content-Type", "Authorization", "X-API-Key", "X-Request-ID"],
    )

    # 2. Trusted Host — protege contra DNS Rebinding.
    allowed_hosts = os.getenv("ALLOWED_HOSTS", "" if production else "*").strip()
    if production and not allowed_hosts:
        raise RuntimeError("ALLOWED_HOSTS deve ser configurado em produção.")
    if allowed_hosts != "*":
        hosts = {h.strip() for h in allowed_hosts.split(",") if h.strip()}
        hosts.update({"127.0.0.1", "localhost", "::1"})
        app.add_middleware(
            TrustedHostMiddleware,
            allowed_hosts=sorted(hosts),
        )

    # 3. Rate Limiting
    app.state.limiter = limiter


def sanitize_filename(filename: str) -> str:
    """Retorna um nome de arquivo seguro contra path traversal e injection.

    1. Decodifica URL-encoding (%2f, %2e, etc.)
    2. Extrai apenas o basename (remove diretórios)
    3. Remove caracteres perigosos
    4. Limita comprimento
    """
    if not filename:
        return "unnamed"

    # Decodificar URL encoding (previne bypass via %2f, %2e)
    decoded = urllib.parse.unquote(filename)

    # Extrair apenas o nome do arquivo (remove paths)
    basename = os.path.basename(decoded)

    # Remover directory traversal patterns
    basename = basename.replace("..", "").replace("/", "").replace("\\", "")

    # Remover caracteres não-seguros (manter alphanum, -, _, .)
    basename = re.sub(r"[^\w\-.]", "_", basename)

    # Remover pontos iniciais (previne arquivos ocultos)
    basename = basename.lstrip(".")

    # Limitar comprimento
    if len(basename) > 200:
        name, ext = os.path.splitext(basename)
        basename = name[: 200 - len(ext)] + ext

    return basename or "unnamed"
