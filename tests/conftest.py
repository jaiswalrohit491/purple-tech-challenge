"""Shared fixtures for the test suite.

Tests run against the *same* Postgres instance the API uses, with a TRUNCATE
between tests. We use FastAPI's ASGITransport so requests don't go out over
the network — the app and the test share a process and a transaction loop.
"""
from __future__ import annotations

import os

# Must be set before app.db is imported so the engine uses NullPool.
os.environ.setdefault("TEST_MODE", "1")

import uuid  # noqa: E402
from datetime import datetime, timedelta, timezone  # noqa: E402

import pytest_asyncio  # noqa: E402
from httpx import ASGITransport, AsyncClient  # noqa: E402
from sqlalchemy import text  # noqa: E402

from app.db import SessionLocal, init_db  # noqa: E402
from app.main import app  # noqa: E402


@pytest_asyncio.fixture(autouse=True, scope="function")
async def _clean_db():
    """Wipe events + pos_transactions before each test."""
    await init_db()
    async with SessionLocal() as s:
        await s.execute(text("TRUNCATE events"))
        await s.execute(text("TRUNCATE pos_transactions"))
        await s.commit()
    yield


@pytest_asyncio.fixture
async def client():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as c:
        yield c


# ---------- factories ----------

DEFAULT_STORE = "STORE_TEST_01"
DEFAULT_CAM_ENTRY = "CAM_ENTRY_01"
DEFAULT_CAM_FLOOR = "CAM_FLOOR_01"
DEFAULT_CAM_BILL = "CAM_BILL_01"


def make_event(
    *,
    event_type: str,
    visitor_id: str,
    ts: datetime | None = None,
    store_id: str = DEFAULT_STORE,
    camera_id: str = DEFAULT_CAM_ENTRY,
    zone_id: str | None = None,
    dwell_ms: int = 0,
    is_staff: bool = False,
    confidence: float = 0.9,
    queue_depth: int | None = None,
    session_seq: int | None = None,
    event_id: str | None = None,
) -> dict:
    """Build a fully-formed event dict matching the production schema."""
    ts = ts or datetime.now(timezone.utc)
    return {
        "event_id": event_id or str(uuid.uuid4()),
        "store_id": store_id,
        "camera_id": camera_id,
        "visitor_id": visitor_id,
        "event_type": event_type,
        "timestamp": ts.astimezone(timezone.utc).isoformat().replace("+00:00", "Z"),
        "zone_id": zone_id,
        "dwell_ms": dwell_ms,
        "is_staff": is_staff,
        "confidence": confidence,
        "metadata": {
            "queue_depth": queue_depth,
            "sku_zone": zone_id,
            "session_seq": session_seq,
        },
    }


async def insert_pos(store_id: str, ts: datetime, amount: float = 500.0) -> str:
    """Insert a POS row directly into the DB. Returns the txn_id."""
    txn_id = f"TXN_{uuid.uuid4().hex[:10]}"
    async with SessionLocal() as s:
        await s.execute(
            text(
                """
                INSERT INTO pos_transactions (store_id, txn_id, ts, amount_inr)
                VALUES (:sid, :tid, :ts, :amt)
                """
            ),
            {"sid": store_id, "tid": txn_id, "ts": ts, "amt": amount},
        )
        await s.commit()
    return txn_id


def today_at(minutes_ago: int = 30, offset_seconds: int = 0) -> datetime:
    """A point in time within today's UTC window, in the past.

    `minutes_ago` is the base offset; `offset_seconds` is added on top so a
    test loop can stagger event timestamps (e.g. `today_at(10, i)` produces
    five distinct times 1 second apart). The earlier signature had a silent
    `_hour_ignored` arg that swallowed the second positional, which caused
    test loops like `today_at(10, i)` to emit five identical timestamps —
    masking time-ordering bugs.
    """
    now = datetime.now(timezone.utc)
    candidate = now - timedelta(minutes=minutes_ago) + timedelta(seconds=offset_seconds)
    # Clamp to today's UTC day so window queries (00:00 -> now) include it.
    day_start = now.replace(hour=0, minute=0, second=0, microsecond=0)
    if candidate < day_start:
        candidate = day_start + timedelta(seconds=1)
    return candidate
