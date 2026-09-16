"""
Database connection and session management for SupportDNA Live Conversation Memory.
Supports PostgreSQL as the primary database with an automatic fallback to SQLite
for environments where PostgreSQL is not yet started or during isolated tests.
"""

import os
import logging
from contextlib import contextmanager
from typing import Generator
from sqlalchemy import create_engine, event
from sqlalchemy.engine import Engine
from sqlalchemy.orm import declarative_base, sessionmaker, Session

logger = logging.getLogger("supportdna.database")

Base = declarative_base()

_engine: Engine | None = None
_SessionFactory: sessionmaker | None = None


def get_database_url() -> str:
    """
    Construct the database connection URL from environment variables.
    Conceptually expects:
      POSTGRES_HOST
      POSTGRES_PORT
      POSTGRES_DB
      POSTGRES_USER
      POSTGRES_PASSWORD
    Or a direct DATABASE_URL override.
    """
    db_url = os.getenv("DATABASE_URL")
    if db_url:
        return db_url

    pg_host = os.getenv("POSTGRES_HOST")
    pg_port = os.getenv("POSTGRES_PORT", "5432")
    pg_db = os.getenv("POSTGRES_DB", "supportdna")
    pg_user = os.getenv("POSTGRES_USER", "postgres")
    pg_password = os.getenv("POSTGRES_PASSWORD", "")

    if pg_host:
        if pg_password:
            return f"postgresql+psycopg2://{pg_user}:{pg_password}@{pg_host}:{pg_port}/{pg_db}"
        else:
            return f"postgresql+psycopg2://{pg_user}@{pg_host}:{pg_port}/{pg_db}"

    # Default fallback when no PostgreSQL environment is provided
    os.makedirs("data", exist_ok=True)
    return "sqlite:///data/supportdna_live.db"


def get_engine(url: str | None = None, force_new: bool = False) -> Engine:
    """
    Get or create the SQLAlchemy engine.
    Attempts PostgreSQL connection if configured; falls back gracefully to SQLite
    if the PostgreSQL instance is unreachable or not started.
    """
    global _engine, _SessionFactory
    if _engine is not None and not force_new and url is None:
        return _engine

    target_url = url or get_database_url()

    try:
        if target_url.startswith("postgresql"):
            # Attempt to connect with short connect_args timeout to detect if server is running
            test_engine = create_engine(
                target_url,
                pool_pre_ping=True,
                connect_args={"connect_timeout": 3}
            )
            # Test connection
            with test_engine.connect() as conn:
                pass
            _engine = test_engine
            logger.info(f"[Database] Successfully connected to PostgreSQL at {target_url.split('@')[-1]}")
        else:
            # SQLite configuration
            test_engine = create_engine(
                target_url,
                connect_args={"check_same_thread": False}
            )
            # Enable SQLite foreign key constraints
            @event.listens_for(test_engine, "connect")
            def set_sqlite_pragma(dbapi_connection, connection_record):
                cursor = dbapi_connection.cursor()
                cursor.execute("PRAGMA foreign_keys=ON")
                cursor.close()

            _engine = test_engine
            logger.info(f"[Database] Using SQLite storage at {target_url}")
    except Exception as e:
        if target_url.startswith("postgresql"):
            logger.warning(
                f"[Database] Could not connect to PostgreSQL ({e}). "
                "Falling back to local SQLite at sqlite:///data/supportdna_live.db for live conversation memory."
            )
            os.makedirs("data", exist_ok=True)
            fallback_url = "sqlite:///data/supportdna_live.db"
            _engine = create_engine(
                fallback_url,
                connect_args={"check_same_thread": False}
            )
            @event.listens_for(_engine, "connect")
            def set_sqlite_pragma(dbapi_connection, connection_record):
                cursor = dbapi_connection.cursor()
                cursor.execute("PRAGMA foreign_keys=ON")
                cursor.close()
        else:
            raise e

    _SessionFactory = sessionmaker(bind=_engine, autocommit=False, autoflush=False)
    return _engine


def get_session_factory() -> sessionmaker:
    """Get the session maker, initializing engine if needed."""
    global _SessionFactory
    if _SessionFactory is None:
        get_engine()
    return _SessionFactory


@contextmanager
def get_db_session() -> Generator[Session, None, None]:
    """Provide a transactional scope around a series of operations."""
    factory = get_session_factory()
    session = factory()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def init_db(engine: Engine | None = None) -> None:
    """Initialize database tables."""
    target_engine = engine or get_engine()
    # Import models to ensure they are registered with Base.metadata
    from src.database import models  # noqa: F401
    Base.metadata.create_all(bind=target_engine)
    logger.info("[Database] Schema tables verified and initialized.")
