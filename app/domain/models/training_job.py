from sqlalchemy import JSON, Integer, String
from sqlalchemy.orm import Mapped, mapped_column

from app.domain.models.base_model import BaseModel


class TrainingJob(BaseModel):
    __tablename__ = "training_jobs"

    job_id: Mapped[str] = mapped_column(String(64), unique=True, index=True, nullable=False)
    status: Mapped[str] = mapped_column(String(32), default="pending", nullable=False)
    message: Mapped[str | None] = mapped_column(String(500), default="", nullable=True)
    progress: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    architecture: Mapped[str | None] = mapped_column(String(100), nullable=True)
    model_name: Mapped[str | None] = mapped_column(String(200), nullable=True)
    dataset_path: Mapped[str | None] = mapped_column(String(500), nullable=True)
    parameters: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    metrics: Mapped[dict | None] = mapped_column(JSON, nullable=True)

    def __repr__(self):
        return f"<TrainingJob {self.job_id} {self.status} {self.progress}%>"

