import logging
from app.main_startup import create_app
from app.extensions import db
# Importar modelos para garantir que o SQLAlchemy os conheça antes do
# create_all
from app.domain.models import AnalysisResult, ArchitectureConfig, User

logger = logging.getLogger(__name__)


def seed_architectures():
    """Popula o banco com as configurações padrão do registry se estiver vazio."""
    try:
        # Importação tardia para evitar ciclo com registry -> db_setup
        from app.domain.models.architectures.registry import architecture_registry

        # Verificar se já existem configs
        if ArchitectureConfig.query.first():
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
            db.session.add(default_config)

            # Variantes (se houver, e se quisermos criar configs explícitas para elas)
            # Por enquanto, vamos criar apenas a default baseada no registry.
            # Se variants tiverem parâmetros diferentes hardcoded no código,
            # o registry atual não expõe isso facilmente (apenas supported_variants list).
            # Assumiremos que o usuário criará configs para variantes via UI
            # depois.

        db.session.commit()
        logger.info(f"{len(architectures)} arquiteturas semeadas com sucesso.")

    except Exception as e:
        logger.error(f"Erro ao semear arquiteturas: {e}")
        db.session.rollback()


def init_db():
    """Inicializa o banco de dados (Cria tabelas se não existirem)."""
    app = create_app()
    with app.app_context():
        try:
            db.create_all()
            seed_architectures()
            logger.info(
                "Banco de dados SQLite inicializado e tabelas criadas.")
        except Exception as e:
            logger.error(f"Erro ao inicializar banco de dados: {e}")


def get_flask_app():
    return create_app()
