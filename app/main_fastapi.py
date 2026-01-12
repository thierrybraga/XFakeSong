from app.core.db_setup import init_db
from gradio_app import demo as gradio_demo
from app.routers import (
    system, detection, features, training, history, datasets
)
import gradio as gr
from fastapi import FastAPI
from slowapi import _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from app.core.security import setup_security, limiter
import os
import sys
from pathlib import Path

# Adicionar raiz ao path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Inicializar App
app = FastAPI(
    title="XfakeSong API",
    description="API para detecção de deepfakes de áudio e análise forense.",
    version="1.0.0"
)

# Configurar Segurança (CORS, TrustedHost, RateLimiting)
setup_security(app)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)


@app.on_event("startup")
def on_startup():
    init_db()


# Incluir Routers
app.include_router(system.router)
app.include_router(detection.router)
app.include_router(features.router)
app.include_router(training.router)
app.include_router(history.router)
app.include_router(datasets.router)

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
