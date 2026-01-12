import os
from slowapi import Limiter
from slowapi.util import get_remote_address
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi import FastAPI
from dotenv import load_dotenv

# Carregar variáveis de ambiente
load_dotenv()

# Inicializar Limiter para Rate Limiting
limiter = Limiter(key_func=get_remote_address)

def setup_security(app: FastAPI):
    """Configura middlewares e configurações de segurança."""
    
    # 1. CORS
    # Em produção, ALLOWED_ORIGINS deve ser uma lista separada por vírgula de domínios confiáveis
    origins_str = os.getenv("ALLOWED_ORIGINS", "*")
    if origins_str == "*":
        allow_origins = ["*"]
    else:
        allow_origins = [origin.strip() for origin in origins_str.split(",")]

    app.add_middleware(
        CORSMiddleware,
        allow_origins=allow_origins,
        allow_credentials=True,
        allow_methods=["GET", "POST", "OPTIONS"], # Restringir métodos se possível
        allow_headers=["*"],
    )

    # 2. Trusted Host
    # Protege contra ataques de Host Header (DNS Rebinding)
    allowed_hosts = os.getenv("ALLOWED_HOSTS", "*")
    if allowed_hosts != "*":
        app.add_middleware(
            TrustedHostMiddleware, 
            allowed_hosts=[host.strip() for host in allowed_hosts.split(",")]
        )
    
    # 3. Rate Limiting
    app.state.limiter = limiter
    # Nota: O middleware do slowapi deve ser adicionado manualmente ou via add_exception_handler no main

def sanitize_filename(filename: str) -> str:
    """Retorna um nome de arquivo seguro."""
    return os.path.basename(filename).replace("..", "").replace("/", "").replace("\\", "")
