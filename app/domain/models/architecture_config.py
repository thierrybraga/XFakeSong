from app.extensions import db
from app.domain.models.base_model import BaseModel


class ArchitectureConfig(BaseModel):
    __tablename__ = 'architecture_configs'

    architecture_name = db.Column(db.String(100), nullable=False, index=True)
    variant_name = db.Column(db.String(100), default="default")
    description = db.Column(db.String(500))
    # Armazenamos os parâmetros como JSON no banco para flexibilidade (listas,
    # booleanos, ints)
    parameters = db.Column(db.JSON, nullable=False)
    is_active = db.Column(db.Boolean, default=True)

    def __repr__(self):
        return f"<ArchitectureConfig {self.architecture_name}:{self.variant_name}>"
