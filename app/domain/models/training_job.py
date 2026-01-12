from app.extensions import db
from app.domain.models.base_model import BaseModel


class TrainingJob(BaseModel):
    __tablename__ = "training_jobs"

    job_id = db.Column(db.String(64), unique=True, index=True, nullable=False)
    status = db.Column(db.String(32), default="pending", nullable=False)
    message = db.Column(db.String(500), default="", nullable=True)
    progress = db.Column(db.Integer, default=0, nullable=False)  # 0–100
    architecture = db.Column(db.String(100), nullable=True)
    model_name = db.Column(db.String(200), nullable=True)
    dataset_path = db.Column(db.String(500), nullable=True)
    parameters = db.Column(db.JSON, nullable=True)
    metrics = db.Column(db.JSON, nullable=True)

    def __repr__(self):
        return f"<TrainingJob {self.job_id} {self.status} {self.progress}%>"

