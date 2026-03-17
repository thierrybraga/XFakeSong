from sqlalchemy import JSON, Boolean, Float, Index, Integer, String
from sqlalchemy.orm import Mapped, mapped_column

from app.domain.models.base_model import BaseModel


class AnalysisResult(BaseModel):
    __tablename__ = 'analysis_results'

    filename: Mapped[str | None] = mapped_column(String(255), nullable=True, index=True)
    is_fake: Mapped[bool] = mapped_column(Boolean, nullable=False)
    confidence: Mapped[float] = mapped_column(Float, nullable=False)
    model_name: Mapped[str | None] = mapped_column(String(100), nullable=True, index=True)
    duration_seconds: Mapped[float | None] = mapped_column(Float, nullable=True)
    sample_rate: Mapped[int | None] = mapped_column(Integer, nullable=True)

    __table_args__ = (
        Index('ix_analysis_created_at', 'created_at'),
    )

    # Armazenar detalhes complexos como JSON
    # No SQLite, JSON é armazenado como TEXT, mas SQLAlchemy abstrai isso
    details: Mapped[dict | None] = mapped_column(JSON, nullable=True)

    def __repr__(self):
        return f"<AnalysisResult {self.id} - {'FAKE' if self.is_fake else 'REAL'} ({self.confidence:.2f})>"
