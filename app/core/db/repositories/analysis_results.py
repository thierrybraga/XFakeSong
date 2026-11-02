"""Histórico de análises (``analysis_results``).

Consumido pelo router ``/api/v1/history``, pela aba Histórico e pelo painel.
"""

from __future__ import annotations

from datetime import datetime
from typing import Callable

from sqlalchemy import func
from sqlalchemy.orm import Session

from app.core.db.models.analysis import AnalysisResult
from app.core.db.session import SessionLocal


class AnalysisResultRepository:
    """Leitura e remoção de análises, sempre da mais recente para a mais antiga."""

    def __init__(self, session_factory: Callable[[], Session] | None = None) -> None:
        self._session_factory = session_factory or SessionLocal

    def list_page(
        self, *, limit: int, offset: int = 0
    ) -> tuple[list[AnalysisResult], int]:
        """Página de análises e o total da tabela, lidos na mesma sessão."""
        with self._session_factory() as db:
            total = db.query(func.count(AnalysisResult.id)).scalar() or 0
            items = (
                db.query(AnalysisResult)
                .order_by(AnalysisResult.created_at.desc())
                .offset(offset)
                .limit(limit)
                .all()
            )
            return items, total

    def list_recent(self, limit: int | None = None) -> list[AnalysisResult]:
        """Análises mais recentes; ``limit=None`` devolve todas."""
        with self._session_factory() as db:
            query = db.query(AnalysisResult).order_by(AnalysisResult.created_at.desc())
            if limit is not None:
                query = query.limit(limit)
            return query.all()

    def search(
        self,
        *,
        model_name: str | None = None,
        is_fake: bool | None = None,
        filename_contains: str | None = None,
        limit: int = 500,
    ) -> list[AnalysisResult]:
        """Filtra por modelo, veredito e trecho do nome do arquivo (sem caixa)."""
        with self._session_factory() as db:
            query = db.query(AnalysisResult).order_by(AnalysisResult.created_at.desc())
            if model_name is not None:
                query = query.filter(AnalysisResult.model_name == model_name)
            if is_fake is not None:
                query = query.filter(AnalysisResult.is_fake == is_fake)
            if filename_contains:
                query = query.filter(
                    AnalysisResult.filename.ilike(f"%{filename_contains}%")
                )
            return query.limit(limit).all()

    def distinct_model_names(self) -> list[str]:
        """Nomes de modelo presentes no histórico, sem vazios nem repetição."""
        with self._session_factory() as db:
            rows = db.query(AnalysisResult.model_name).distinct().all()
            return [row[0] for row in rows if row[0]]

    def count_since(self, cutoff: datetime) -> int:
        with self._session_factory() as db:
            count = (
                db.query(func.count(AnalysisResult.id))
                .filter(AnalysisResult.created_at >= cutoff)
                .scalar()
            )
            return int(count or 0)

    def get(self, analysis_id: int) -> AnalysisResult | None:
        with self._session_factory() as db:
            return db.get(AnalysisResult, analysis_id)

    def delete(self, analysis_id: int) -> bool:
        """Remove uma análise; ``False`` quando o ID não existe."""
        with self._session_factory() as db:
            item = db.get(AnalysisResult, analysis_id)
            if item is None:
                return False
            db.delete(item)
            db.commit()
            return True

    def delete_all(self) -> int:
        """Esvazia o histórico e devolve quantas linhas foram removidas."""
        with self._session_factory() as db:
            removed = db.query(AnalysisResult).delete()
            db.commit()
            return removed
