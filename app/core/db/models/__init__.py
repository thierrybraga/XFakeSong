"""Entidades ORM (SQLAlchemy) persistidas em ``data/app.db``.

Persistência é infraestrutura: as entidades vivem em ``app.core.db`` junto
da sessão, e o domínio as consome — nunca o contrário. Antes ficavam em
``app/domain/models/`` e ``core.db.setup`` precisava importar o domínio para
criar as tabelas (ciclo ``core.db`` ↔ ``domain``).

Importar este pacote registra todas as tabelas no ``Base.metadata``; é isso
que ``init_db`` e ``tools/ops/audit_database_schema.py`` fazem.
"""

from .analysis import AnalysisResult
from .architecture_config import ArchitectureConfig
from .base import BaseModel
from .experiment import (
    ArtifactRecord,
    ConfigurationEntry,
    ExperimentRun,
    MetricRecord,
    ModelRun,
    SystemSnapshot,
)
from .training_job import TrainingJob
from .voice_profile import VoiceProfile

__all__ = [
    "AnalysisResult",
    "ArchitectureConfig",
    "ArtifactRecord",
    "BaseModel",
    "ConfigurationEntry",
    "ExperimentRun",
    "MetricRecord",
    "ModelRun",
    "SystemSnapshot",
    "TrainingJob",
    "VoiceProfile",
]
