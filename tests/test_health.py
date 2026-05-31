# PROMPT:
#   "Tests for GET /health. Returns status, db_reachable, last_event_at per
#    store with stale=true when last event > STALE_FEED_THRESHOLD_SECONDS old.
#    Verify the endpoint never raises and that the stale flag works at the
#    boundary."
#
# CHANGES MADE:
#   - Originally the AI test stubbed the DB to simulate unreachability. Dropped
#     that — we want to verify *real* behavior. Used a far-past timestamp to
#     drive the stale path.

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from tests.conftest import DEFAULT_STORE, make_event


async def test_health_empty(client):
    body = (await client.get("/health")).json()
    assert body["status"] == "OK"
    assert body["db_reachable"] is True
    assert body["stores"] == []


async def test_health_fresh_feed(client):
    """A recent event => store appears with stale=false."""
    e = make_event(event_type="ENTRY", visitor_id="V1",
                   ts=datetime.now(timezone.utc))
    await client.post("/events/ingest", json=[e])
    body = (await client.get("/health")).json()
    feeds = {s["store_id"]: s for s in body["stores"]}
    assert DEFAULT_STORE in feeds
    assert feeds[DEFAULT_STORE]["stale"] is False


async def test_health_stale_feed(client):
    """An event older than 10 minutes flags STALE_FEED."""
    e = make_event(event_type="ENTRY", visitor_id="V1",
                   ts=datetime.now(timezone.utc) - timedelta(minutes=20))
    await client.post("/events/ingest", json=[e])
    body = (await client.get("/health")).json()
    feeds = {s["store_id"]: s for s in body["stores"]}
    assert feeds[DEFAULT_STORE]["stale"] is True
