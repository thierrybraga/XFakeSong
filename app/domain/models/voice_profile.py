"""Modelo ORM para Perfis de Voz personalizados."""

from sqlalchemy import JSON, Float, Integer, String
from sqlalchemy.orm import Mapped, mapped_column

from app.domain.models.base_model import BaseModel


class VoiceProfile(BaseModel):
    """Perfil de voz com dataset customizado e modelo treinado."""

    __tablename__ = "voice_profiles"

    # Dados pessoais
    name: Mapped[str] = mapped_column(String(100), nullable=False)
    telegram_id: Mapped[str | None] = mapped_column(
        String(100), nullable=True, index=True
    )
    phone: Mapped[str | None] = mapped_column(String(20), nullable=True)
    email: Mapped[str | None] = mapped_column(String(120), nullable=True, index=True)

    # Dataset e modelo
    dataset_dir: Mapped[str] = mapped_column(String(500), nullable=False)
    model_path: Mapped[str | None] = mapped_column(String(500), nullable=True)
    scaler_path: Mapped[str | None] = mapped_column(String(500), nullable=True)

    # Configuração do modelo
    architecture: Mapped[str | None] = mapped_column(
        String(100), default="sonic_sleuth"
    )
    feature_config: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    training_config: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    training_metrics: Mapped[dict | None] = mapped_column(JSON, nullable=True)

    # Status: created, collecting, training, ready, error
    status: Mapped[str] = mapped_column(String(30), default="created")
    num_samples: Mapped[int] = mapped_column(Integer, default=0)
    total_duration_seconds: Mapped[float] = mapped_column(Float, default=0.0)

    # Descrição
    description: Mapped[str | None] = mapped_column(String(500), nullable=True)

    def __repr__(self):
        return f"<VoiceProfile {self.id}: {self.name} [{self.status}]>"
