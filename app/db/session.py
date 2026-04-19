from __future__ import annotations

from collections.abc import Callable

from sqlalchemy import create_engine
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, sessionmaker

from app.db.models import Base


class Database:
    def __init__(self, database_url: str) -> None:
        connect_args = {"check_same_thread": False} if database_url.startswith("sqlite") else {}
        self._engine: Engine = create_engine(database_url, connect_args=connect_args)
        self._session_factory = sessionmaker(bind=self._engine, autoflush=False, expire_on_commit=False)

    def create_tables(self) -> None:
        Base.metadata.create_all(self._engine)

    def session_factory(self) -> Callable[[], Session]:
        return self._session_factory

    def dispose(self) -> None:
        self._engine.dispose()
