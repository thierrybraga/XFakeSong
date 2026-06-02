"""Middleware centralizado e leve para a API XFakeSong.

O middleware combina em uma passagem:
- Request ID tracking via contextvars
- Limite de upload por Content-Length
- Header X-Request-ID em toda resposta
- Logging estruturado com filtro de ruído para assets/filas da UI
"""

from __future__ import annotations

import logging
import os
import time
import uuid
from contextvars import ContextVar
from typing import Any, Callable, Optional

from fastapi import FastAPI
from starlette.datastructures import MutableHeaders
from starlette.types import ASGIApp, Message, Receive, Scope, Send

logger = logging.getLogger(__name__)

request_id_ctx: ContextVar[Optional[str]] = ContextVar("request_id", default=None)

_DEFAULT_NOISY_PREFIXES = (
    "/static/",
    "/assets/",
    "/favicon",
    "/gradio/queue/",
    "/gradio/file=",
    "/api/v1/system/bootstrap",
    "/api/v1/system/feedback",
)


def get_request_id() -> Optional[str]:
    """Retorna o request ID da request atual."""

    return request_id_ctx.get()


def _env_bool(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        return default


def _client_host(scope: Scope) -> str:
    client = scope.get("client")
    if isinstance(client, tuple) and client:
        return str(client[0])
    return "unknown"


def _header_value(scope: Scope, name: bytes) -> Optional[str]:
    for key, value in scope.get("headers") or []:
        if key.lower() == name:
            return value.decode("latin-1", errors="ignore")
    return None


def _should_log_request(path: str, status: int, elapsed_ms: float) -> bool:
    if _env_bool("XFAKE_LOG_EVERY_REQUEST", False):
        return True
    if status >= 400:
        return True
    slow_ms = _env_int("XFAKE_SLOW_REQUEST_MS", 1500)
    if elapsed_ms >= slow_ms:
        return True
    if path.startswith("/api/") and not any(
        path.startswith(prefix) for prefix in _DEFAULT_NOISY_PREFIXES
    ):
        return True
    return False


class SystemASGIMiddleware:
    """ASGI middleware único para reduzir overhead por request."""

    def __init__(self, app: ASGIApp, max_size_mb: int = 100):
        self.app = app
        self.max_size = max_size_mb * 1024 * 1024

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        req_id = _header_value(scope, b"x-request-id") or str(uuid.uuid4())[:12]
        token = request_id_ctx.set(req_id)
        start = time.perf_counter()
        status_code = 500
        path = str(scope.get("path") or "")
        method = str(scope.get("method") or "")

        content_length = _header_value(scope, b"content-length")
        if content_length:
            try:
                if int(content_length) > self.max_size:
                    await self._send_payload_too_large(send, req_id)
                    self._log_request(
                        req_id=req_id,
                        method=method,
                        path=path,
                        status=413,
                        elapsed_ms=(time.perf_counter() - start) * 1000,
                        scope=scope,
                    )
                    request_id_ctx.reset(token)
                    return
            except ValueError:
                pass

        async def send_wrapper(message: Message) -> None:
            nonlocal status_code
            if message["type"] == "http.response.start":
                status_code = int(message.get("status", 500))
                headers = MutableHeaders(scope=message)
                headers["X-Request-ID"] = req_id
            await send(message)

        try:
            await self.app(scope, receive, send_wrapper)
        finally:
            elapsed_ms = (time.perf_counter() - start) * 1000
            self._log_request(
                req_id=req_id,
                method=method,
                path=path,
                status=status_code,
                elapsed_ms=elapsed_ms,
                scope=scope,
            )
            request_id_ctx.reset(token)

    async def _send_payload_too_large(self, send: Send, req_id: str) -> None:
        body = (
            b'{"type":"about:blank","title":"Payload Too Large",'
            b'"status":413,"detail":"Upload excede o limite configurado."}'
        )
        await send(
            {
                "type": "http.response.start",
                "status": 413,
                "headers": [
                    (b"content-type", b"application/json"),
                    (b"x-request-id", req_id.encode("latin-1")),
                ],
            }
        )
        await send({"type": "http.response.body", "body": body})

    def _log_request(
        self,
        *,
        req_id: str,
        method: str,
        path: str,
        status: int,
        elapsed_ms: float,
        scope: Scope,
    ) -> None:
        elapsed = round(elapsed_ms, 2)
        if not _should_log_request(path, status, elapsed):
            return
        log_fn: Callable[..., Any] = logger.info
        if status >= 500:
            log_fn = logger.error
        elif status >= 400:
            log_fn = logger.warning
        log_fn(
            "request_completed",
            extra={
                "request_id": req_id,
                "method": method,
                "path": path,
                "status": status,
                "elapsed_ms": elapsed,
                "client": _client_host(scope),
            },
        )


def setup_middleware(app: FastAPI) -> None:
    """Registra middleware de sistema.

    Ordem importa: o último adicionado pelo Starlette roda primeiro. Mantemos
    este middleware como camada única para reduzir overhead e ruído.
    """

    max_upload_mb = _env_int("XFAKE_MAX_UPLOAD_MB", 100)
    app.add_middleware(SystemASGIMiddleware, max_size_mb=max_upload_mb)


__all__ = ["SystemASGIMiddleware", "get_request_id", "setup_middleware"]
