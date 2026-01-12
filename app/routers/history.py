from fastapi import APIRouter, HTTPException
from typing import List
from app.schemas.api_models import HistoryItem
from app.core.db_setup import get_flask_app
from app.domain.models.analysis import AnalysisResult

router = APIRouter(prefix="/api/v1/history", tags=["History"])


@router.get("/", response_model=List[HistoryItem])
async def list_history(limit: int = 50, offset: int = 0):
    try:
        flask_app = get_flask_app()
        with flask_app.app_context():
            # Query com SQLAlchemy
            query = AnalysisResult.query.order_by(
                AnalysisResult.created_at.desc()
            )
            query = query.offset(offset).limit(limit)
            results = query.all()

            history_items = []
            for item in results:
                history_items.append(HistoryItem(
                    id=item.id,
                    filename=item.filename or "unknown",
                    is_fake=item.is_fake,
                    confidence=item.confidence,
                    model_name=item.model_name or "unknown",
                    created_at=(
                        item.created_at.isoformat() if item.created_at else ""
                    )
                ))

            return history_items
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Erro ao acessar banco de dados: {str(e)}"
        )


@router.get("/{analysis_id}", response_model=HistoryItem)
async def get_history_item(analysis_id: int):
    try:
        flask_app = get_flask_app()
        with flask_app.app_context():
            item = AnalysisResult.query.get(analysis_id)
            if not item:
                raise HTTPException(
                    status_code=404, detail="Análise não encontrada"
                )

            return HistoryItem(
                id=item.id,
                filename=item.filename or "unknown",
                is_fake=item.is_fake,
                confidence=item.confidence,
                model_name=item.model_name or "unknown",
                created_at=(
                    item.created_at.isoformat() if item.created_at else ""
                )
            )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Erro ao acessar banco de dados: {str(e)}"
        )


@router.delete("/{analysis_id}")
async def delete_history_item(analysis_id: int):
    try:
        flask_app = get_flask_app()
        with flask_app.app_context():
            item = AnalysisResult.query.get(analysis_id)
            if not item:
                raise HTTPException(
                    status_code=404, detail="Análise não encontrada"
                )

            item.delete()
            return {"message": "Análise removida com sucesso"}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Erro ao acessar banco de dados: {str(e)}"
        )
