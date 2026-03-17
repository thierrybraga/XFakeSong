"""Sistema de exceções padronizado e error handling global.

Implementa RFC 7807 Problem Details para respostas de erro consistentes.
Todas as exceções de domínio herdam de AppError para tratamento uniforme.
"""

import logging
from typing import Any, Dict, Optional

from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from app.core.middleware import get_request_id

logger = logging.getLogger(__name__)


# ── Exceções de Domínio ────────────────────────────────────────────────

class AppError(Exception):
    """Exceção base da aplicação com suporte a RFC 7807."""

    def __init__(
        self,
        detail: str,
        status_code: int = 500,
        error_code: str = "INTERNAL_ERROR",
        extra: Optional[Dict[str, Any]] = None,
    ):
        self.detail = detail
        self.status_code = status_code
        self.error_code = error_code
        self.extra = extra or {}
        super().__init__(detail)


class NotFoundError(AppError):
    """Recurso não encontrado (404)."""
    def __init__(self, resource: str, identifier: Any = None):
        detail = f"{resource} não encontrado"
        if identifier is not None:
            detail += f": {identifier}"
        super().__init__(detail, status_code=404, error_code="NOT_FOUND")


class ValidationError(AppError):
    """Erro de validação de dados (422)."""
    def __init__(self, detail: str, field: str = None):
        extra = {"field": field} if field else {}
        super().__init__(detail, status_code=400, error_code="VALIDATION_ERROR", extra=extra)


class ConflictError(AppError):
    """Conflito de recurso (409)."""
    def __init__(self, detail: str):
        super().__init__(detail, status_code=409, error_code="CONFLICT")


class ServiceUnavailableError(AppError):
    """Serviço indisponível (503)."""
    def __init__(self, service: str):
        super().__init__(
            f"Serviço '{service}' indisponível",
            status_code=503,
            error_code="SERVICE_UNAVAILABLE",
        )


class AudioProcessingError(AppError):
    """Erro no processamento de áudio (400)."""
    def __init__(self, detail: str):
        super().__init__(detail, status_code=400, error_code="AUDIO_PROCESSING_ERROR")


class ModelNotFoundError(NotFoundError):
    """Modelo ML não encontrado."""
    def __init__(self, model_name: str):
        super().__init__("Modelo", model_name)
        self.error_code = "MODEL_NOT_FOUND"


class DatasetNotFoundError(NotFoundError):
    """Dataset não encontrado."""
    def __init__(self, dataset_name: str):
        super().__init__("Dataset", dataset_name)
        self.error_code = "DATASET_NOT_FOUND"


class ProfileNotFoundError(NotFoundError):
    """Perfil de voz não encontrado."""
    def __init__(self, profile_id: int):
        super().__init__("Perfil de voz", profile_id)
        self.error_code = "PROFILE_NOT_FOUND"


class TrainingError(AppError):
    """Erro durante treinamento (400)."""
    def __init__(self, detail: str):
        super().__init__(detail, status_code=400, error_code="TRAINING_ERROR")


class FileTooLargeError(AppError):
    """Arquivo excede tamanho máximo (413)."""
    def __init__(self, max_size_mb: int):
        super().__init__(
            f"Arquivo excede o limite de {max_size_mb}MB",
            status_code=413,
            error_code="FILE_TOO_LARGE",
        )


class UnsupportedFormatError(AppError):
    """Formato de arquivo não suportado (415)."""
    def __init__(self, fmt: str, supported: list):
        super().__init__(
            f"Formato '{fmt}' não suportado. Use: {', '.join(supported)}",
            status_code=400,
            error_code="UNSUPPORTED_FORMAT",
        )


# ── RFC 7807 Problem Details ──────────────────────────────────────────

def _problem_detail(
    status: int,
    title: str,
    detail: str,
    error_code: str = None,
    extra: Dict[str, Any] = None,
) -> Dict[str, Any]:
    """Monta response body no formato RFC 7807."""
    body = {
        "type": "about:blank",
        "title": title,
        "status": status,
        "detail": detail,
    }
    req_id = get_request_id()
    if req_id:
        body["request_id"] = req_id
    if error_code:
        body["error_code"] = error_code
    if extra:
        body.update(extra)
    return body


# ── Handlers Globais ──────────────────────────────────────────────────

def _handle_app_error(request: Request, exc: AppError) -> JSONResponse:
    """Handler para exceções de domínio."""
    if exc.status_code >= 500:
        logger.error(f"[{exc.error_code}] {exc.detail}", exc_info=True)
    else:
        logger.warning(f"[{exc.error_code}] {exc.detail}")

    return JSONResponse(
        status_code=exc.status_code,
        content=_problem_detail(
            status=exc.status_code,
            title=exc.error_code.replace("_", " ").title(),
            detail=exc.detail,
            error_code=exc.error_code,
            extra=exc.extra if exc.extra else None,
        ),
    )


def _handle_http_exception(request: Request, exc: HTTPException) -> JSONResponse:
    """Handler para HTTPException do FastAPI."""
    return JSONResponse(
        status_code=exc.status_code,
        content=_problem_detail(
            status=exc.status_code,
            title="HTTP Error",
            detail=str(exc.detail),
        ),
    )


def _handle_validation_error(request: Request, exc: RequestValidationError) -> JSONResponse:
    """Handler para erros de validação do Pydantic."""
    errors = []
    for err in exc.errors():
        loc = " → ".join(str(loc_part) for loc_part in err.get("loc", []))
        errors.append({"field": loc, "message": err.get("msg", "")})

    return JSONResponse(
        status_code=422,
        content=_problem_detail(
            status=422,
            title="Validation Error",
            detail="Os dados enviados contêm erros de validação.",
            error_code="VALIDATION_ERROR",
            extra={"errors": errors},
        ),
    )


def _handle_unhandled_exception(request: Request, exc: Exception) -> JSONResponse:
    """Handler para exceções não tratadas — nunca expõe detalhes internos."""
    logger.critical(
        f"Unhandled exception on {request.method} {request.url.path}: {exc}",
        exc_info=True,
    )
    return JSONResponse(
        status_code=500,
        content=_problem_detail(
            status=500,
            title="Internal Server Error",
            detail="Ocorreu um erro interno. Tente novamente mais tarde.",
            error_code="INTERNAL_ERROR",
        ),
    )


def setup_exception_handlers(app: FastAPI):
    """Registra todos os exception handlers na aplicação."""
    app.add_exception_handler(AppError, _handle_app_error)
    app.add_exception_handler(HTTPException, _handle_http_exception)
    app.add_exception_handler(RequestValidationError, _handle_validation_error)
    app.add_exception_handler(Exception, _handle_unhandled_exception)
