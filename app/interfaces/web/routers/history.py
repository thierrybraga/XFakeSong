"""Endpoints de histórico de análises.

Melhorias: rate limiting, paginação com total, error handling consistente.
"""

import logging

from fastapi import APIRouter, Depends, Query, Request

from app.core.auth.auth_handler import get_api_key
from app.core.db.repositories.analysis_results import AnalysisResultRepository
from app.core.exceptions import NotFoundError
from app.core.security import limiter
from app.dependencies import get_analysis_result_repository
from app.interfaces.web.schemas.api_models import HistoryItem, HistoryListResponse

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/api/v1/history",
    tags=["History"],
    dependencies=[Depends(get_api_key)],
)


def _to_history_item(item) -> HistoryItem:
    return HistoryItem(
        id=item.id,
        filename=item.filename or "unknown",
        is_fake=item.is_fake,
        confidence=item.confidence,
        model_name=item.model_name or "unknown",
        created_at=(item.created_at.isoformat() if item.created_at else ""),
    )


@router.get(
    "/",
    response_model=HistoryListResponse,
    summary="Lista histórico de análises com paginação",
)
@limiter.limit("30/minute")
async def list_history(
    request: Request,
    limit: int = Query(50, ge=1, le=500, description="Itens por página"),
    offset: int = Query(0, ge=0, description="Offset para paginação"),
    history: AnalysisResultRepository = Depends(get_analysis_result_repository),
):
    results, total = history.list_page(limit=limit, offset=offset)

    return HistoryListResponse(
        items=[_to_history_item(item) for item in results],
        total=total,
        limit=limit,
        offset=offset,
    )


@router.get(
    "/{analysis_id}",
    response_model=HistoryItem,
    summary="Detalhes de uma análise",
)
@limiter.limit("30/minute")
async def get_history_item(
    request: Request,
    analysis_id: int,
    history: AnalysisResultRepository = Depends(get_analysis_result_repository),
):
    item = history.get(analysis_id)
    if not item:
        raise NotFoundError("Análise", analysis_id)

    return _to_history_item(item)


@router.delete(
    "/{analysis_id}",
    summary="Remove uma análise do histórico",
)
@limiter.limit("10/minute")
async def delete_history_item(
    request: Request,
    analysis_id: int,
    history: AnalysisResultRepository = Depends(get_analysis_result_repository),
):
    if not history.delete(analysis_id):
        raise NotFoundError("Análise", analysis_id)

    return {"message": "Análise removida com sucesso"}
