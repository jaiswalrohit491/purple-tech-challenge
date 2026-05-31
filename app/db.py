import os
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.pool import NullPool

from .config import settings

# In test mode each pytest-asyncio test gets its own event loop. A pooled engine
# created at import time gets bound to the first loop and breaks for subsequent
# tests with "Task pending" errors. NullPool sidesteps it by opening a fresh
# connection per session.
_engine_kwargs: dict = {"pool_pre_ping": True, "future": True}
if os.getenv("TEST_MODE") == "1":
    _engine_kwargs["poolclass"] = NullPool
else:
    _engine_kwargs.update(pool_size=10, max_overflow=20)

engine = create_async_engine(settings.database_url, **_engine_kwargs)

SessionLocal = async_sessionmaker(engine, expire_on_commit=False, class_=AsyncSession)


SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS events (
  event_id      UUID PRIMARY KEY,
  store_id      TEXT NOT NULL,
  camera_id     TEXT NOT NULL,
  visitor_id    TEXT NOT NULL,
  event_type    TEXT NOT NULL,
  ts            TIMESTAMPTZ NOT NULL,
  zone_id       TEXT,
  dwell_ms      INTEGER NOT NULL DEFAULT 0,
  is_staff      BOOLEAN NOT NULL DEFAULT FALSE,
  confidence    DOUBLE PRECISION NOT NULL,
  metadata      JSONB NOT NULL DEFAULT '{}'::jsonb,
  ingested_at   TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS ix_events_store_ts      ON events(store_id, ts);
CREATE INDEX IF NOT EXISTS ix_events_store_visitor ON events(store_id, visitor_id);
CREATE INDEX IF NOT EXISTS ix_events_type_ts       ON events(event_type, ts);
CREATE INDEX IF NOT EXISTS ix_events_zone          ON events(store_id, zone_id, ts);

CREATE TABLE IF NOT EXISTS pos_transactions (
  store_id      TEXT NOT NULL,
  txn_id        TEXT PRIMARY KEY,
  ts            TIMESTAMPTZ NOT NULL,
  amount_inr    NUMERIC NOT NULL,
  brand_name    TEXT
);
ALTER TABLE pos_transactions ADD COLUMN IF NOT EXISTS brand_name TEXT;
CREATE INDEX IF NOT EXISTS ix_pos_store_ts ON pos_transactions(store_id, ts);
CREATE INDEX IF NOT EXISTS ix_pos_brand ON pos_transactions(store_id, brand_name, ts);
"""


async def init_db() -> None:
    from sqlalchemy import text

    async with engine.begin() as conn:
        for stmt in SCHEMA_SQL.split(";"):
            if stmt.strip():
                await conn.execute(text(stmt))


@asynccontextmanager
async def get_session() -> AsyncIterator[AsyncSession]:
    async with SessionLocal() as session:
        yield session


async def session_dep() -> AsyncIterator[AsyncSession]:
    async with SessionLocal() as session:
        yield session
