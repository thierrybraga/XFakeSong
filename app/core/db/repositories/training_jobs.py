"""Jobs de treino e validação cruzada disparados pela API (``training_jobs``).

O ciclo de vida (estados, progresso, cache) é do
``app.domain.services.training_jobs.TrainingJobService``; aqui fica só a
persistência.
"""

from __future__ import annotations

from typing import Any, Callable

from sqlalchemy.orm import Session

from app.core.db.models.training_job import TrainingJob
from app.core.db.session import SessionLocal


class TrainingJobRepository:
    def __init__(self, session_factory: Callable[[], Session] | None = None) -> None:
        self._session_factory = session_factory or SessionLocal

    def create(self, **fields: Any) -> TrainingJob:
        """Persiste um job novo e o devolve com os valores gravados."""
        with self._session_factory() as db:
            job = TrainingJob(**fields)
            db.add(job)
            db.commit()
            db.refresh(job)
            return job

    def get(self, job_id: str) -> TrainingJob | None:
        with self._session_factory() as db:
            return db.query(TrainingJob).filter_by(job_id=job_id).first()

    def update(self, job_id: str, **fields: Any) -> TrainingJob | None:
        """Aplica ``fields`` e confirma; ``None`` quando o job não existe."""
        with self._session_factory() as db:
            job = db.query(TrainingJob).filter_by(job_id=job_id).first()
            if job is None:
                return None
            for name, value in fields.items():
                setattr(job, name, value)
            db.commit()
            db.refresh(job)
            return job
