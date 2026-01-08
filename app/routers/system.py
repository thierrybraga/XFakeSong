from fastapi import APIRouter
from app.schemas.api_models import SystemStatus

router = APIRouter(prefix="/api/v1/system", tags=["System"])


@router.get("/status", response_model=SystemStatus)
async def get_system_status():
    return SystemStatus(
        status="operational",
        message="System is running correctly",
        active_services=["detection", "features", "training"]
    )


@router.get("/health")
async def health_check():
    return {"status": "ok"}
