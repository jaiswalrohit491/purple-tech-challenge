# PROMPT:
#   "Generate pytest async cases for FastAPI endpoint POST /events/ingest.
#    Contract: accepts list of up to 500 events, idempotent by event_id (UUID),
#    returns per-event partial success on validation errors with structured
#    error body, never 5xx on user input. Use httpx AsyncClient against the
#    FastAPI app. Cover: duplicate event_id within batch, duplicate across calls,
#    mixed valid/invalid, empty batch, batch of 501, missing required field."
#
# CHANGES MADE:
#   - Replaced AI-suggested SQLAlchemy-direct asserts with HTTP-level assertions
#     so tests exercise the real request path (middleware, validators, ORJSON).
#   - Added 413 expectation for oversize batches (rubric: "structured error",
#     AI had suggested 400 — overridden because 413 better fits the semantics).
#   - Tightened "mixed valid/invalid" to check that errors[].index points at
#     the offending item, which the AI version omitted.

from __future__ import annotations

import uuid

import pytest

from tests.conftest import DEFAULT_STORE, make_event, today_at


async def test_ingest_happy_path(client):
    events = [make_event(event_type="ENTRY", visitor_id=f"V_{i}", ts=today_at(10, i)) for i in range(5)]
    r = await client.post("/events/ingest", json=events)
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["accepted"] == 5
    assert body["duplicates"] == 0
    assert body["rejected"] == 0
    assert body["errors"] == []


async def test_ingest_accepts_object_wrapper(client):
    events = [make_event(event_type="ENTRY", visitor_id="V1")]
    r = await client.post("/events/ingest", json={"events": events})
    assert r.status_code == 200
    assert r.json()["accepted"] == 1


async def test_ingest_idempotent_within_batch(client):
    """Same event_id repeated in one batch: first accepted, second deduped."""
    eid = str(uuid.uuid4())
    e1 = make_event(event_type="ENTRY", visitor_id="V_dup", event_id=eid)
    e2 = make_event(event_type="ENTRY", visitor_id="V_dup", event_id=eid)
    r = await client.post("/events/ingest", json=[e1, e2])
    assert r.status_code == 200
    body = r.json()
    assert body["accepted"] == 1
    assert body["duplicates"] == 1


async def test_ingest_idempotent_across_calls(client):
    """Replaying the same payload is a no-op: total inserted count unchanged."""
    events = [make_event(event_type="ENTRY", visitor_id=f"V_{i}") for i in range(3)]
    r1 = await client.post("/events/ingest", json=events)
    r2 = await client.post("/events/ingest", json=events)
    assert r1.status_code == r2.status_code == 200
    assert r1.json()["accepted"] == 3
    assert r2.json()["accepted"] == 0
    assert r2.json()["duplicates"] == 3


async def test_ingest_partial_success_on_invalid(client):
    """Malformed events are rejected per-item; valid events still land."""
    good = make_event(event_type="ENTRY", visitor_id="V_good")
    bad = {**make_event(event_type="ENTRY", visitor_id="V_bad")}
    bad["confidence"] = 1.5  # out of bounds
    bad2 = {**make_event(event_type="ENTRY", visitor_id="V_bad2")}
    del bad2["event_type"]

    r = await client.post("/events/ingest", json=[good, bad, bad2])
    assert r.status_code == 200
    body = r.json()
    assert body["accepted"] == 1
    assert body["rejected"] == 2
    assert len(body["errors"]) == 2
    indices = sorted(e["index"] for e in body["errors"])
    assert indices == [1, 2]


async def test_ingest_empty_batch(client):
    r = await client.post("/events/ingest", json=[])
    assert r.status_code == 200
    body = r.json()
    assert body == {"accepted": 0, "duplicates": 0, "rejected": 0, "errors": []}


async def test_ingest_batch_too_large(client):
    """Batch over the configured limit returns a structured 413, not a 5xx."""
    events = [make_event(event_type="ENTRY", visitor_id=f"V_{i}") for i in range(501)]
    r = await client.post("/events/ingest", json=events)
    assert r.status_code == 413
    assert "detail" in r.json()


async def test_ingest_wrong_payload_shape(client):
    r = await client.post("/events/ingest", json={"not_events": []})
    # The dict path tries `events` key, then rejects -> 400 (HTTPException).
    assert r.status_code == 400


@pytest.mark.parametrize(
    "field,value",
    [
        ("event_id", "not-a-uuid"),
        ("event_type", "WUT"),
        ("dwell_ms", -1),
        ("confidence", -0.1),
        ("confidence", 1.01),
        ("store_id", ""),
    ],
)
async def test_ingest_field_validation(client, field, value):
    e = make_event(event_type="ENTRY", visitor_id="V")
    e[field] = value
    r = await client.post("/events/ingest", json=[e])
    body = r.json()
    assert r.status_code == 200
    assert body["rejected"] == 1
    assert body["errors"][0]["index"] == 0
