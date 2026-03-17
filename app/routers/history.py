"""Endpoints de histórico de análises.

Melhorias: rate limiting, paginação com total, error handling consistente.
"""

import logging

from fastapi import APIRouter, Depends, Query, Request
from sqlalchemy import func
from sqlalchemy.orm import Session

from app.core.database import get_db
from app.core.exceptions import NotFoundError
from app.core.security import limiter
from app.domain.models.analysis import AnalysisResult
from app.schemas.api_models import HistoryItem, HistoryListResponse

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/history", tags=["History"])


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
    db: Session = Depends(get_db),
):
    # Total para paginação
    total = db.query(func.count(AnalysisResult.id)).scalar() or 0

    results = (
        db.query(AnalysisResult)
        .order_by(AnalysisResult.created_at.desc())
        .offset(offset)
        .limit(limit)
        .all()
    )

    items = [
        HistoryItem(
            id=item.id,
            filename=item.filename or "unknown",
            is_fake=item.is_fake,
            confidence=item.confidence,
            model_name=item.model_name or "unknown",
            created_at=(
                item.created_at.isoformat() if item.created_at else ""
            ),
        )
        for item in results
    ]

    return HistoryListResponse(
        items=items,
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
    db: Session = Depends(get_db),
):
    item = db.query(AnalysisResult).filter_by(id=analysis_id).first()
    if not item:
        raise NotFoundError("Análise", analysis_id)

    return HistoryItem(
        id=item.id,
        filename=item.filename or "unknown",
        is_fake=item.is_fake,
        confidence=item.confidence,
        model_name=item.model_name or "unknown",
        created_at=item.created_at.isoformat() if item.created_at else "",
    )


@router.delete(
    "/{analysis_id}",
    summary="Remove uma análise do histórico",
)
@limiter.limit("10/minute")
async def delete_history_item(
    request: Request,
    analysis_id: int,
    db: Session = Depends(get_db),
):
    item = db.query(AnalysisResult).filter_by(id=analysis_id).first()
    if not item:
        raise NotFoundError("Análise", analysis_id)

    db.delete(item)
    db.commit()
    return {"message": "Análise removida com sucesso"}
