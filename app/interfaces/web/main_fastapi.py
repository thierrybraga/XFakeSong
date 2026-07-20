import logging as _l
import os
import sys
from contextlib import asynccontextmanager
from pathlib import Path

from app.core.hf_compat import ensure_hf_folder_shim

ensure_hf_folder_shim()  # antes de qualquer import que precise de Gradio

from fastapi import FastAPI, Request  # noqa: E402
from fastapi.responses import HTMLResponse  # noqa: E402
from fastapi.staticfiles import StaticFiles  # noqa: E402
from fastapi.templating import Jinja2Templates  # noqa: E402
from slowapi import _rate_limit_exceeded_handler  # noqa: E402
from slowapi.errors import RateLimitExceeded  # noqa: E402

from app.core.db.setup import init_db  # noqa: E402
from app.core.exceptions import setup_exception_handlers  # noqa: E402
from app.core.feedback import configure_logging  # noqa: E402
from app.core.gpu import describe_gpu_setup, setup_gpu  # noqa: E402
from app.core.middleware import setup_middleware  # noqa: E402
from app.core.performance import configure_runtime_environment  # noqa: E402
from app.core.security import limiter, setup_security  # noqa: E402
from app.core.version_check import check_versions  # noqa: E402

# API.3: importa ALL_ROUTERS (inclui voice_profiles que faltava antes!).
from app.interfaces.web.routers import ALL_ROUTERS  # noqa: E402

from app.core.bootstrap import OperationalPaths  # noqa: E402

configure_logging(
    level=_l.INFO,
    log_file=str(OperationalPaths.resolve().logs / "system.log"),
    force=False,
)
configure_runtime_environment()


def _running_under_pytest() -> bool:
    return "pytest" in sys.modules or "PYTEST_CURRENT_TEST" in os.environ


def _env_flag(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _api_only_mode() -> bool:
    return _env_flag("XFAKE_API_ONLY") or _env_flag("XFAKE_SKIP_GRADIO")


# BUG.Render.2: verifica versões críticas — detecta incompatibilidade
# gradio<4.31 + starlette>=0.36 antes do TypeError em runtime.
check_versions(strict=False)

# GPU.2: configura TF cedo em runtime real. Em pytest isso torna qualquer
# import da API lento e carrega modelos antes dos mocks de dependência.
if not _running_under_pytest() and not _api_only_mode():
    _gpu_info = setup_gpu()
    _l.getLogger(__name__).info(f"GPU setup: {describe_gpu_setup()}")

# Adicionar raiz ao path
# (app/interfaces/web/main_fastapi.py -> web -> interfaces -> app -> raiz)
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))


# API.8: lifespan moderno substitui on_event("startup") deprecado
@asynccontextmanager
async def lifespan(app: FastAPI):
    from app.core.bootstrap import bootstrap_application
    from app.core.db.experiment_store import experiment_store

    bootstrap_application()
    experiment_store.record_system_snapshot("api_startup")
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

_WEB_DIR = Path(__file__).resolve().parent
_STATIC_DIR = _WEB_DIR / "static"
_TEMPLATES_DIR = _WEB_DIR / "templates"

app.mount(
    "/static",
    StaticFiles(directory=str(_STATIC_DIR)),
    name="static",
)
templates = Jinja2Templates(directory=str(_TEMPLATES_DIR))


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

if not _running_under_pytest() and not _api_only_mode():
    import gradio as gr  # noqa: E402,I001
    from app.interfaces.gradio.app import demo as gradio_demo  # noqa: E402

    # PROD.3: allowed_paths precisa incluir TODOS os diretórios que o Gradio
    # vai servir arquivos. Restrito a só "." dá 403 (ERR_ABORTED) em:
    #   - /tmp/gradio (uploads + cache, default temp dir)
    #   - /tmp (fallback geral, cobre numba/matplotlib/hf caches também)
    # Sem isto, drag-and-drop de áudio falha silenciosamente no browser.
    allowed_paths = [
        os.path.abspath("."),
        os.environ.get("GRADIO_TEMP_DIR", "/tmp/gradio"),
        "/tmp",
    ]
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
