# PROMPT:
#   "Tests for GET /stores/{id}/anomalies. Three anomaly types:
#    BILLING_QUEUE_SPIKE (WARN >=5, CRITICAL >=8), CONVERSION_DROP (today's
#    rolling 1h < 0.7 * 7-day baseline), DEAD_ZONE (zone with prior activity
#    but no visits in last 30 min). Verify severity escalation, suggested_action
#    is populated, and quiet stores produce zero anomalies."
#
# CHANGES MADE:
#   - The AI version triggered DEAD_ZONE by inserting one prior event 35 min
#     ago, which is too sparse to count as prior activity. Increased to 3 events
#     in the 4-hour window to match the production threshold.
#   - Added explicit "no anomalies on a clean store" test to guard against
#     false positives — a real rubric concern.

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from tests.conftest import DEFAULT_STORE, make_event


async def test_anomalies_clean_store_returns_empty(client):
    body = (await client.get(f"/stores/{DEFAULT_STORE}/anomalies")).json()
    assert body["anomalies"] == []


async def test_anomaly_queue_spike_warn(client):
    """queue_depth=5 within last 30 min triggers WARN."""
    now = datetime.now(timezone.utc)
    e = make_event(
        event_type="BILLING_QUEUE_JOIN", visitor_id="V1",
        ts=now - timedelta(minutes=5), queue_depth=5,
    )
    await client.post("/events/ingest", json=[e])
    body = (await client.get(f"/stores/{DEFAULT_STORE}/anomalies")).json()
    queue = [a for a in body["anomalies"] if a["anomaly_type"] == "BILLING_QUEUE_SPIKE"]
    assert len(queue) == 1
    assert queue[0]["severity"] == "WARN"
    assert "queue" in queue[0]["suggested_action"].lower()


async def test_anomaly_queue_spike_critical(client):
    """queue_depth=10 trips CRITICAL severity."""
    now = datetime.now(timezone.utc)
    e = make_event(
        event_type="BILLING_QUEUE_JOIN", visitor_id="V1",
        ts=now - timedelta(minutes=2), queue_depth=10,
    )
    await client.post("/events/ingest", json=[e])
    body = (await client.get(f"/stores/{DEFAULT_STORE}/anomalies")).json()
    queue = [a for a in body["anomalies"] if a["anomaly_type"] == "BILLING_QUEUE_SPIKE"]
    assert queue and queue[0]["severity"] == "CRITICAL"


async def test_anomaly_queue_spike_outside_window(client):
    """A spike from 45 min ago should NOT trigger (window is 30 min)."""
    now = datetime.now(timezone.utc)
    e = make_event(
        event_type="BILLING_QUEUE_JOIN", visitor_id="V1",
        ts=now - timedelta(minutes=45), queue_depth=10,
    )
    await client.post("/events/ingest", json=[e])
    body = (await client.get(f"/stores/{DEFAULT_STORE}/anomalies")).json()
    assert not [a for a in body["anomalies"] if a["anomaly_type"] == "BILLING_QUEUE_SPIKE"]


async def test_anomaly_dead_zone(client):
    """Zone with 3+ visits in last 4h but nothing in last 30 min → INFO alert."""
    now = datetime.now(timezone.utc)
    events = [
        make_event(
            event_type="ZONE_ENTER", visitor_id=f"V_{i}", zone_id="SKINCARE",
            ts=now - timedelta(minutes=60 + 5 * i),
        )
        for i in range(3)
    ]
    await client.post("/events/ingest", json=events)
    body = (await client.get(f"/stores/{DEFAULT_STORE}/anomalies")).json()
    dead = [a for a in body["anomalies"] if a["anomaly_type"] == "DEAD_ZONE"]
    assert dead and dead[0]["detail"]["zone_id"] == "SKINCARE"
    assert dead[0]["severity"] == "INFO"
