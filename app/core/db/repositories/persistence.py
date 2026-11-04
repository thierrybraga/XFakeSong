"""Estado da persistência para o ``GET /api/v1/system/info``."""

from __future__ import annotations

from typing import Callable

from sqlalchemy import text
from sqlalchemy.orm import Session

from app.core.db.models.experiment import (
    ConfigurationEntry,
    ExperimentRun,
    MetricRecord,
    ModelRun,
    SystemSnapshot,
)
from app.core.db.session import SessionLocal


class PersistenceStatsRepository:
    def __init__(self, session_factory: Callable[[], Session] | None = None) -> None:
        self._session_factory = session_factory or SessionLocal

    def schema_version(self) -> int | None:
        """``PRAGMA user_version`` gravado por ``init_db`` (só SQLite)."""
        with self._session_factory() as db:
            return db.execute(text("PRAGMA user_version")).scalar()

    def table_counts(self) -> dict[str, int]:
        """Linhas das tabelas do ``ExperimentStore``."""
        with self._session_factory() as db:
            return {
                "experiments": db.query(ExperimentRun).count(),
                "model_runs": db.query(ModelRun).count(),
                "metrics": db.query(MetricRecord).count(),
                "configurations": db.query(ConfigurationEntry).count(),
                "system_snapshots": db.query(SystemSnapshot).count(),
            }
