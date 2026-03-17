"""Endpoints de sistema: health check, status, bootstrap.

Health check verifica DB, modelos e storage de forma real.
"""

import logging
import time

from fastapi import APIRouter, Request

from app.core.database import check_database_health
from app.core.security import limiter
from app.schemas.api_models import HealthCheckResponse, SystemStatus

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/system", tags=["System"])

# Timestamp de início para calcular uptime
_start_time = time.monotonic()


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
            "detection", "features", "training",
            "voice_profiles", "forensic_analysis",
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
    except Exception:
        pass

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
