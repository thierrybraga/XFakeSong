from app.extensions import db
from app.domain.models.base_model import BaseModel


class AnalysisResult(BaseModel):
    __tablename__ = 'analysis_results'

    filename = db.Column(db.String(255), nullable=True)
    is_fake = db.Column(db.Boolean, nullable=False)
    confidence = db.Column(db.Float, nullable=False)
    model_name = db.Column(db.String(100), nullable=True)
    duration_seconds = db.Column(db.Float, nullable=True)
    sample_rate = db.Column(db.Integer, nullable=True)

    # Armazenar detalhes complexos como JSON
    # No SQLite, JSON é armazenado como TEXT, mas SQLAlchemy abstrai isso
    details = db.Column(db.JSON, nullable=True)

    def __repr__(self):
        return f"<AnalysisResult {self.id} - {'FAKE' if self.is_fake else 'REAL'} ({self.confidence:.2f})>"
