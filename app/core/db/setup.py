import logging

from sqlalchemy import text

from app.core.db.session import Base, engine

logger = logging.getLogger(__name__)
SCHEMA_VERSION = 2


def _load_orm_models() -> None:
    """Registra as entidades ORM no ``Base.metadata`` antes do ``create_all``.

    Importação tardia: quem só precisa de uma sessão (``app.core.db.session``)
    não paga o custo de carregar todas as entidades.
    """
    from app.core.db import models  # noqa: F401


def init_db(*, raise_on_error: bool = False) -> bool:
    """Inicializa/migra o schema e informa se a operação foi concluída.

    Só schema: a semeadura de ``architecture_configs`` depende do registry do
    domínio e é feita pela composição (``app.dependencies.bootstrap_application``).
    """
    try:
        _load_orm_models()
        Base.metadata.create_all(bind=engine)
        if engine.dialect.name == "sqlite":
            with engine.begin() as connection:
                connection.execute(text(f"PRAGMA user_version={SCHEMA_VERSION}"))
        logger.info("Banco de dados SQLite inicializado e tabelas criadas.")
        return True
    except Exception as e:
        logger.error(f"Erro ao inicializar banco de dados: {e}")
        if raise_on_error:
            raise
        return False
