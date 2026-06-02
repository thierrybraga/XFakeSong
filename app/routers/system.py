"""Endpoints de sistema: health check, status, bootstrap, version, info.

Health check verifica DB, modelos e storage de forma real.
"""

import logging
import platform
import subprocess
import sys
import time
from functools import lru_cache

from fastapi import APIRouter, Request

from app.core.database import check_database_health
from app.core.feedback import (
    clear_feedback_events,
    get_feedback_events,
    get_feedback_summary,
    mark_feedback_read,
)
from app.core.performance import get_resource_snapshot
from app.core.security import limiter
from app.schemas.api_models import (
    HealthCheckResponse,
    SystemStatus,
    SystemVersionResponse,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/system", tags=["System"])

# Timestamp de início para calcular uptime
_start_time = time.monotonic()


@lru_cache(maxsize=1)
def _get_git_sha() -> str:
    """Retorna o SHA curto do commit atual ou 'unknown'."""
    try:
        out = (
            subprocess.check_output(
                ["git", "rev-parse", "--short", "HEAD"],
                stderr=subprocess.DEVNULL,
                timeout=2,
            )
            .decode()
            .strip()
        )
        return out or "unknown"
    except Exception:
        return "unknown"


def _safe_pkg_version(pkg_name: str) -> str | None:
    """Tenta obter a versão instalada de um pacote sem importar tudo dele."""
    try:
        from importlib.metadata import PackageNotFoundError, version

        try:
            return version(pkg_name)
        except PackageNotFoundError:
            return None
    except Exception:
        return None


@router.get(
    "/status",
    response_model=SystemStatus,
    summary="Status geral do sistema",
)
@limiter.limit("30/minute")
async def get_system_status(request: Request):
    return SystemStatus(
        status="operational",
        message="System is running correctly",
        version="1.1.0",
        active_services=[
            "detection",
            "features",
            "training",
            "voice_profiles",
            "forensic_analysis",
        ],
    )


@router.get(
    "/health",
    response_model=HealthCheckResponse,
    summary="Health check detalhado com verificação de DB e modelos",
)
@limiter.limit("60/minute")
async def health_check(request: Request):
    """Verifica saúde do banco de dados, modelos e storage."""
    db_ok = check_database_health()

    # Verificar modelos carregados
    models_count = 0
    try:
        from app.dependencies import get_detection_service

        svc = get_detection_service()
        models_count = len(svc.get_available_models())
    except Exception as e:
        # API.9: log do erro (em vez de silenciar) — útil para diagnóstico
        logger.debug(f"Não foi possível contar modelos no health check: {e}")

    # Verificar storage
    from pathlib import Path

    storage_ok = Path("app/models").exists() or Path("models").exists()

    uptime = time.monotonic() - _start_time
    overall = "healthy" if db_ok else "degraded"

    return HealthCheckResponse(
        status=overall,
        database="ok" if db_ok else "error",
        models_loaded=models_count,
        storage_available=storage_ok,
        uptime_seconds=round(uptime, 2),
    )


@router.get("/bootstrap", summary="Bootstrap endpoint")
@limiter.limit("10/minute")
async def bootstrap(request: Request):
    return {"status": "ok"}


@router.get(
    "/feedback",
    summary="Eventos recentes de feedback, logs e notificações",
)
@limiter.limit("60/minute")
async def get_feedback(request: Request, limit: int = 50, mark_read: bool = False):
    """Retorna o buffer central de feedback usado por terminal, API e UI."""

    safe_limit = min(max(limit, 1), 200)
    events = [event.as_dict() for event in get_feedback_events(limit=safe_limit)]
    summary = get_feedback_summary()
    if mark_read:
        mark_feedback_read()
        summary = get_feedback_summary()
    return {
        "status": "ok",
        "summary": summary,
        "events": events,
    }


@router.post(
    "/feedback/read",
    summary="Marca todos os eventos de feedback como lidos",
)
@limiter.limit("30/minute")
async def mark_feedback_as_read(request: Request):
    mark_feedback_read()
    return {"status": "ok", "summary": get_feedback_summary()}


@router.post(
    "/feedback/clear",
    summary="Limpa o histórico em memória de feedback",
)
@limiter.limit("10/minute")
async def clear_feedback(request: Request):
    removed = clear_feedback_events()
    return {
        "status": "ok",
        "removed": removed,
        "summary": get_feedback_summary(),
    }


# ── API.10: Versão + info ──────────────────────────────────────────────


@router.get(
    "/version",
    response_model=SystemVersionResponse,
    summary="Versões das dependências críticas (debug em produção)",
)
@limiter.limit("60/minute")
async def get_version(request: Request):
    """Útil para debug em produção: versão do app, Python, TF/Keras, git SHA."""
    return SystemVersionResponse(
        version="1.1.0",
        python_version=sys.version.split()[0],
        tensorflow_version=_safe_pkg_version("tensorflow"),
        keras_version=_safe_pkg_version("keras"),
        gradio_version=_safe_pkg_version("gradio"),
        sklearn_version=_safe_pkg_version("scikit-learn"),
        git_sha=_get_git_sha(),
        platform=f"{platform.system()} {platform.release()} ({platform.machine()})",
    )


@router.get(
    "/info",
    summary="Sumário consolidado do estado da aplicação",
)
@limiter.limit("30/minute")
async def get_info(request: Request):
    """Snapshot consolidado: status + versões + contagens.

    Útil para um dashboard de monitoring fazer 1 só request.
    """
    db_ok = check_database_health()

    models_count = 0
    architectures = []
    default_model = None
    try:
        from app.dependencies import get_detection_service

        svc = get_detection_service()
        models_count = len(svc.get_available_models())
        architectures = svc.get_available_architectures()
        default_model = svc.default_model
    except Exception as e:
        logger.debug(f"Não foi possível obter info de detecção: {e}")

    uptime = time.monotonic() - _start_time

    return {
        "status": "operational" if db_ok else "degraded",
        "uptime_seconds": round(uptime, 2),
        "version": "1.1.0",
        "git_sha": _get_git_sha(),
        "python": sys.version.split()[0],
        "platform": platform.system(),
        "database": {"available": db_ok},
        "models": {
            "count": models_count,
            "default": default_model,
            "architectures_supported": architectures,
        },
        "endpoints": {
            "docs": "/api/docs",
            "redoc": "/api/redoc",
            "openapi": "/api/openapi.json",
            "feedback": "/api/v1/system/feedback",
        },
        "feedback": get_feedback_summary(),
        "performance": get_resource_snapshot(),
    }
