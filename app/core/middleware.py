"""Middleware centralizado para a API XFakeSong.

Inclui:
- Request ID tracking para rastreabilidade
- Error handling padronizado (RFC 7807 Problem Details)
- Logging estruturado de requests
- Limites de tamanho de upload
"""

import logging
import time
import uuid
from contextvars import ContextVar
from typing import Optional

from fastapi import FastAPI, Request, Response
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

logger = logging.getLogger(__name__)

# Context var para Request ID (acessível em qualquer lugar da request)
request_id_ctx: ContextVar[Optional[str]] = ContextVar("request_id", default=None)


def get_request_id() -> Optional[str]:
    """Retorna o request ID da request atual (thread-safe via contextvars)."""
    return request_id_ctx.get()


class RequestIDMiddleware(BaseHTTPMiddleware):
    """Adiciona X-Request-ID a cada request para rastreabilidade."""

    async def dispatch(self, request: Request, call_next):
        # Aceitar request ID do cliente ou gerar novo
        req_id = request.headers.get("X-Request-ID", str(uuid.uuid4())[:12])
        request_id_ctx.set(req_id)
        request.state.request_id = req_id

        response = await call_next(request)
        response.headers["X-Request-ID"] = req_id
        return response


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """Loga cada request com métricas de tempo."""

    async def dispatch(self, request: Request, call_next):
        start = time.perf_counter()
        response = await call_next(request)
        elapsed_ms = (time.perf_counter() - start) * 1000

        req_id = getattr(request.state, "request_id", "?")
        logger.info(
            "request_completed",
            extra={
                "request_id": req_id,
                "method": request.method,
                "path": request.url.path,
                "status": response.status_code,
                "elapsed_ms": round(elapsed_ms, 2),
                "client": request.client.host if request.client else "unknown",
            },
        )
        return response


class MaxUploadSizeMiddleware(BaseHTTPMiddleware):
    """Rejeita uploads que excedam o limite configurado."""

    def __init__(self, app, max_size_mb: int = 100):
        super().__init__(app)
        self.max_size = max_size_mb * 1024 * 1024

    async def dispatch(self, request: Request, call_next):
        content_length = request.headers.get("content-length")
        if content_length and int(content_length) > self.max_size:
            return JSONResponse(
                status_code=413,
                content={
                    "type": "about:blank",
                    "title": "Payload Too Large",
                    "status": 413,
                    "detail": (
                        f"Upload excede o limite de "
                        f"{self.max_size // (1024 * 1024)}MB."
                    ),
                },
            )
        return await call_next(request)


def setup_middleware(app: FastAPI):
    """Registra todos os middlewares na aplicação."""
    # Ordem importa: último adicionado = primeiro executado
    app.add_middleware(RequestLoggingMiddleware)
    app.add_middleware(RequestIDMiddleware)
    app.add_middleware(MaxUploadSizeMiddleware, max_size_mb=100)
