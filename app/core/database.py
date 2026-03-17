"""Configuração centralizada do banco de dados.

Pool de conexões configurável via variáveis de ambiente.
Session management com context managers seguros.
"""

import logging
import os
from contextlib import contextmanager
from pathlib import Path

from sqlalchemy import create_engine, event, text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

logger = logging.getLogger(__name__)

# ── Configuração ───────────────────────────────────────────────────────

BASE_DIR = Path(__file__).parent.parent.parent
DB_PATH = os.getenv("DATABASE_URL", f"sqlite:///{BASE_DIR}/app.db")

# Pool configurável via env
POOL_SIZE = int(os.getenv("DB_POOL_SIZE", "5"))
MAX_OVERFLOW = int(os.getenv("DB_MAX_OVERFLOW", "10"))
POOL_TIMEOUT = int(os.getenv("DB_POOL_TIMEOUT", "30"))

_engine_kwargs = {}
if "sqlite" in DB_PATH:
    _engine_kwargs["connect_args"] = {"check_same_thread": False}
else:
    _engine_kwargs["pool_size"] = POOL_SIZE
    _engine_kwargs["max_overflow"] = MAX_OVERFLOW
    _engine_kwargs["pool_timeout"] = POOL_TIMEOUT
    _engine_kwargs["pool_pre_ping"] = True  # Detecta conexões mortas

engine = create_engine(DB_PATH, **_engine_kwargs)

# Ativar WAL mode e foreign keys no SQLite para melhor concorrência
if "sqlite" in DB_PATH:
    @event.listens_for(engine, "connect")
    def _set_sqlite_pragma(dbapi_conn, connection_record):
        cursor = dbapi_conn.cursor()
        cursor.execute("PRAGMA journal_mode=WAL")
        cursor.execute("PRAGMA foreign_keys=ON")
        cursor.execute("PRAGMA busy_timeout=5000")
        cursor.close()

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


# ── Dependencies ───────────────────────────────────────────────────────

def get_db():
    """Dependency para FastAPI (request lifecycle) — auto-close."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


@contextmanager
def get_background_db():
    """Context manager para sessões fora do request lifecycle.

    Commit automático ao sair sem exceção; rollback em caso de erro.
    Uso:
        with get_background_db() as db:
            db.query(Model).filter_by(...).first()
    """
    db = SessionLocal()
    try:
        yield db
        db.commit()
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()


@contextmanager
def get_readonly_db():
    """Session somente-leitura (sem commit). Útil para consultas."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def check_database_health() -> bool:
    """Verifica se o banco está acessível."""
    try:
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        return True
    except Exception as e:
        logger.error(f"Database health check failed: {e}")
        return False
