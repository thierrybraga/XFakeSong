import logging

from app.core.database import Base, SessionLocal, engine

# Importar modelos para garantir que o SQLAlchemy os conheça antes do
# create_all
from app.domain.models import (  # noqa: F401
    AnalysisResult,
    ArchitectureConfig,
    TrainingJob,
    User,
)
from app.domain.models.voice_profile import VoiceProfile  # noqa: F401

logger = logging.getLogger(__name__)


def seed_architectures():
    """Popula o banco com as configurações padrão do registry se estiver vazio."""
    db = SessionLocal()
    try:
        # Importação tardia para evitar ciclo com registry -> db_setup
        from app.domain.models.architectures.registry import architecture_registry

        # Verificar se já existem configs
        if db.query(ArchitectureConfig).first():
            return

        logger.info(
            "Populando banco de dados com configurações de arquitetura padrão...")

        architectures = architecture_registry.get_all_architectures()

        for name, info in architectures.items():
            # Configuração Padrão (Default)
            default_config = ArchitectureConfig(
                architecture_name=name,
                variant_name="default",
                description=info.description,
                parameters=info.default_params,
                is_active=True
            )
            db.add(default_config)

        db.commit()
        logger.info(f"{len(architectures)} arquiteturas semeadas com sucesso.")

    except Exception as e:
        logger.error(f"Erro ao semear arquiteturas: {e}")
        db.rollback()
    finally:
        db.close()


def init_db():
    """Inicializa o banco de dados (Cria tabelas se não existirem)."""
    try:
        Base.metadata.create_all(bind=engine)
        seed_architectures()
        logger.info(
            "Banco de dados SQLite inicializado e tabelas criadas.")
    except Exception as e:
        logger.error(f"Erro ao inicializar banco de dados: {e}")
