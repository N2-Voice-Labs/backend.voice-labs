"""Database configuration for durable control-plane state."""

import os
from collections.abc import Iterator
from typing import Any

from sqlalchemy import create_engine
from sqlalchemy.orm import Session, declarative_base, sessionmaker
from sqlalchemy.pool import StaticPool

DATABASE_URL = os.getenv("PLATFORM_DATABASE_URL", "sqlite:///./voice_labs.db")
engine_options: dict[str, Any] = (
    {"connect_args": {"check_same_thread": False}}
    if DATABASE_URL.startswith("sqlite")
    else {}
)
if DATABASE_URL.startswith("postgresql"):
    # Handlers use a synchronous Session inside async routes, so a row-lock wait blocks the whole
    # event loop. Fail the waiting statement instead of hanging the Platform indefinitely.
    lock_timeout = os.getenv("PLATFORM_DB_LOCK_TIMEOUT", "15s")
    engine_options["connect_args"] = {"options": f"-c lock_timeout={lock_timeout}"}
if DATABASE_URL in {"sqlite://", "sqlite:///:memory:"}:
    engine_options["poolclass"] = StaticPool
engine = create_engine(DATABASE_URL, **engine_options)
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)
Base = declarative_base()


def get_db() -> Iterator[Session]:
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
