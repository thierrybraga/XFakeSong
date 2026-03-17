from sqlalchemy import JSON, Boolean, String
from sqlalchemy.orm import Mapped, mapped_column

from app.domain.models.base_model import BaseModel


class ArchitectureConfig(BaseModel):
    __tablename__ = 'architecture_configs'

    architecture_name: Mapped[str] = mapped_column(
        String(100), nullable=False, index=True
    )
    variant_name: Mapped[str] = mapped_column(String(100), default="default")
    description: Mapped[str | None] = mapped_column(String(500))
    # Armazenamos os parâmetros como JSON no banco para flexibilidade (listas,
    # booleanos, ints)
    parameters: Mapped[dict] = mapped_column(JSON, nullable=False)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)

    def __repr__(self):
        return f"<ArchitectureConfig {self.architecture_name}:{self.variant_name}>"
