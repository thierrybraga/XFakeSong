import logging as _l
import os
import sys
from contextlib import asynccontextmanager
from pathlib import Path

# === Compatibilidade huggingface_hub ===
# Gradio v4 ainda importa HfFolder, removido em huggingface_hub recente.
try:
    from huggingface_hub import HfFolder  # noqa: F401
except ImportError:
    import huggingface_hub

    class _HfFolder:
        """Shim para HfFolder removido em huggingface_hub >= 0.16."""

        pass

    huggingface_hub.HfFolder = _HfFolder

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from slowapi import _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded

from app.core.db_setup import init_db
from app.core.exceptions import setup_exception_handlers
from app.core.feedback import configure_logging
from app.core.gpu import describe_gpu_setup, setup_gpu
from app.core.middleware import setup_middleware
from app.core.performance import configure_runtime_environment
from app.core.security import limiter, setup_security
from app.core.version_check import check_versions

# API.3: importa ALL_ROUTERS (inclui voice_profiles que faltava antes!).
from app.routers import ALL_ROUTERS

configure_logging(level=_l.INFO, log_file="system.log", force=False)
configure_runtime_environment()


def _running_under_pytest() -> bool:
    return "pytest" in sys.modules or "PYTEST_CURRENT_TEST" in os.environ


# BUG.Render.2: verifica versões críticas — detecta incompatibilidade
# gradio<4.31 + starlette>=0.36 antes do TypeError em runtime.
check_versions(strict=False)

# GPU.2: configura TF cedo em runtime real. Em pytest isso torna qualquer
# import da API lento e carrega modelos antes dos mocks de dependência.
if not _running_under_pytest():
    _gpu_info = setup_gpu()
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

app.mount(
    "/static",
    StaticFiles(directory="app/static"),
    name="static",
)
templates = Jinja2Templates(directory="app/templates")


@app.get("/", response_class=HTMLResponse, include_in_schema=False)
async def index(request: Request):
    return templates.TemplateResponse(
        "index.html",
        {"request": request},
    )


@app.get("/loading", response_class=HTMLResponse, include_in_schema=False)
async def loading(request: Request):
    return templates.TemplateResponse(
        "pages/loading.html",
        {"request": request},
    )


# API.3: incluir TODOS os routers via lista única (inclui voice_profiles)
for r in ALL_ROUTERS:
    app.include_router(r)

if not _running_under_pytest():
    import gradio as gr  # noqa: E402,I001
    from gradio_app import demo as gradio_demo  # noqa: E402

    # Configurar diretórios permitidos para evitar erros de acesso a arquivos
    # temporários (ERR_ABORTED/403)
    allowed_paths = [os.path.abspath(".")]
    app = gr.mount_gradio_app(
        app,
        gradio_demo,
        path="/gradio",
        allowed_paths=allowed_paths,
    )


if __name__ == "__main__":
    import uvicorn

    # Ler porta do ambiente ou usar padrão
    port = int(os.getenv("PORT", 7861))
    uvicorn.run(app, host="0.0.0.0", port=port)
