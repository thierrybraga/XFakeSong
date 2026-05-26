import os
import sys
from contextlib import asynccontextmanager
from pathlib import Path

import gradio as gr
from fastapi import FastAPI
from slowapi import _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded

from app.core.db_setup import init_db
from app.core.exceptions import setup_exception_handlers
from app.core.gpu import describe_gpu_setup, setup_gpu
from app.core.middleware import setup_middleware
from app.core.security import limiter, setup_security
from app.core.version_check import check_versions
# API.3: importa ALL_ROUTERS (inclui voice_profiles que faltava antes!).
from app.routers import ALL_ROUTERS
from gradio_app import demo as gradio_demo

# BUG.Render.2: verifica versões críticas — detecta incompatibilidade
# gradio<4.31 + starlette>=0.36 antes do TypeError em runtime.
check_versions(strict=False)

# GPU.2: configura TF cedo (memory_growth + mixed precision em GPUs Tensor
# Cores). Idempotente — funciona com Gradio também (mesmo lock global).
_gpu_info = setup_gpu()
import logging as _l
_l.getLogger(__name__).info(f"GPU setup: {describe_gpu_setup()}")

# Adicionar raiz ao path
sys.path.insert(0, str(Path(__file__).parent.parent))


# API.8: lifespan moderno substitui on_event("startup") deprecado
@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    yield


# Inicializar App
app = FastAPI(
    title="XFakeSong API",
    description=(
        "API para detecção de deepfakes de áudio. "
        "Inclui detecção single + multi-model fusion + uncertainty (MC Dropout), "
        "extração de features, treinamento + K-fold CV, gestão de datasets "
        "e perfis de voz."
    ),
    version="1.1.0",
    docs_url="/api/docs",
    redoc_url="/api/redoc",
    openapi_url="/api/openapi.json",
    lifespan=lifespan,
)

# Configurar Segurança (CORS, TrustedHost, RateLimiting) + Middleware
setup_security(app)
setup_middleware(app)
setup_exception_handlers(app)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)


# API.3: incluir TODOS os routers via lista única (inclui voice_profiles)
for r in ALL_ROUTERS:
    app.include_router(r)

# Montar Gradio
# O Gradio será servido na raiz "/" ou em "/ui" conforme preferência.
# O usuário pediu "todo sistema", então faz sentido ter a UI como principal
# visualização, mas a API acessível.
# Vou montar o Gradio na raiz "/" para manter a experiência de usuário atual,
# e a API estará em "/api/v1".

# Configurar diretórios permitidos para evitar erros de acesso a arquivos
# temporários (ERR_ABORTED/403)
allowed_paths = [os.path.abspath(".")]
app = gr.mount_gradio_app(
    app,
    gradio_demo,
    path="/",
    allowed_paths=allowed_paths
)


if __name__ == "__main__":
    import uvicorn
    # Ler porta do ambiente ou usar padrão
    port = int(os.getenv("PORT", 7861))
    uvicorn.run(app, host="0.0.0.0", port=port)
