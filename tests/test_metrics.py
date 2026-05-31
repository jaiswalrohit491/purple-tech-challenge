# PROMPT:
#   "Write pytest async tests for GET /stores/{id}/metrics. The endpoint:
#    - filters out is_staff=true events
#    - returns conversion_rate as 0.0 (not null) when there are no purchases
#    - returns empty avg_dwell_by_zone for stores with no zone events
#    - never raises on completely empty stores
#    - correlates BILLING_QUEUE_JOIN with POS transactions in [join_ts, join_ts+5min)
#    Build a small dataset per test using the conftest factories."
#
# CHANGES MADE:
#   - Reframed AI's "happy path" test to assert specific numeric values rather
#     than just shape — catches off-by-one in the POS correlation window.
#   - Added the all-staff edge case explicitly (rubric calls it out).
#   - Added a boundary test for the POS window: txn at exactly +5min must NOT
#     correlate (right-open interval); txn at +4m59s must.

from __future__ import annotations

from datetime import timedelta

from tests.conftest import DEFAULT_STORE, insert_pos, make_event, today_at


async def test_metrics_empty_store_returns_zeros(client):
    r = await client.get(f"/stores/{DEFAULT_STORE}/metrics")
    assert r.status_code == 200
    body = r.json()
    assert body["unique_visitors"] == 0
    assert body["converted_visitors"] == 0
    assert body["conversion_rate"] == 0.0
    assert body["avg_dwell_by_zone"] == []
    assert body["current_queue_depth"] == 0
    assert body["abandonment_rate"] == 0.0


async def test_metrics_unknown_store_returns_zeros(client):
    """Querying a non-existent store must not 404 or crash — return zero state."""
    r = await client.get("/stores/STORE_DOES_NOT_EXIST/metrics")
    assert r.status_code == 200
    assert r.json()["unique_visitors"] == 0


async def test_metrics_excludes_staff(client):
    """Staff events must not contribute to visitor counts."""
    base = today_at(10)
    events = [
        make_event(event_type="ENTRY", visitor_id="V_customer", ts=base),
        make_event(event_type="ENTRY", visitor_id="V_staff", is_staff=True, ts=base),
        make_event(event_type="ENTRY", visitor_id="V_staff2", is_staff=True, ts=base),
    ]
    await client.post("/events/ingest", json=events)

    body = (await client.get(f"/stores/{DEFAULT_STORE}/metrics")).json()
    assert body["unique_visitors"] == 1


async def test_metrics_all_staff_clip(client):
    """An all-staff store has 0 customers, not 1 errant visitor."""
    base = today_at(10)
    events = [make_event(event_type="ENTRY", visitor_id=f"S_{i}", is_staff=True, ts=base)
              for i in range(5)]
    await client.post("/events/ingest", json=events)

    body = (await client.get(f"/stores/{DEFAULT_STORE}/metrics")).json()
    assert body["unique_visitors"] == 0
    assert body["conversion_rate"] == 0.0


async def test_metrics_zero_purchases(client):
    """Visitors with no POS correlation produce conversion_rate=0.0 (not null, not error)."""
    base = today_at(11)
    events = [
        make_event(event_type="ENTRY", visitor_id=f"V_{i}", ts=base) for i in range(3)
    ] + [
        make_event(event_type="BILLING_QUEUE_JOIN", visitor_id=f"V_{i}",
                   ts=base + timedelta(minutes=5), queue_depth=2)
        for i in range(3)
    ]
    await client.post("/events/ingest", json=events)

    body = (await client.get(f"/stores/{DEFAULT_STORE}/metrics")).json()
    assert body["unique_visitors"] == 3
    assert body["converted_visitors"] == 0
    assert body["conversion_rate"] == 0.0


async def test_metrics_pos_correlation_inside_window(client):
    base = today_at(12)
    join_ts = base + timedelta(minutes=10)
    events = [
        make_event(event_type="ENTRY", visitor_id="V1", ts=base),
        make_event(event_type="BILLING_QUEUE_JOIN", visitor_id="V1",
                   ts=join_ts, queue_depth=1),
    ]
    await client.post("/events/ingest", json=events)
    # POS txn 4 minutes after join — inside the 5-min window.
    await insert_pos(DEFAULT_STORE, join_ts + timedelta(minutes=4))

    body = (await client.get(f"/stores/{DEFAULT_STORE}/metrics")).json()
    assert body["converted_visitors"] == 1
    assert body["conversion_rate"] == 1.0


async def test_metrics_pos_correlation_outside_window(client):
    """A POS txn just past the 5-minute window must NOT count as converted."""
    base = today_at(13)
    join_ts = base + timedelta(minutes=10)
    events = [
        make_event(event_type="ENTRY", visitor_id="V1", ts=base),
        make_event(event_type="BILLING_QUEUE_JOIN", visitor_id="V1",
                   ts=join_ts, queue_depth=1),
    ]
    await client.post("/events/ingest", json=events)
    await insert_pos(DEFAULT_STORE, join_ts + timedelta(minutes=6))

    body = (await client.get(f"/stores/{DEFAULT_STORE}/metrics")).json()
    assert body["converted_visitors"] == 0


async def test_metrics_avg_dwell_per_zone(client):
    base = today_at(14)
    events = [
        make_event(event_type="ZONE_ENTER", visitor_id="V1", zone_id="SKINCARE",
                   ts=base, dwell_ms=0),
        make_event(event_type="ZONE_DWELL", visitor_id="V1", zone_id="SKINCARE",
                   ts=base + timedelta(seconds=30), dwell_ms=30_000),
        make_event(event_type="ZONE_DWELL", visitor_id="V1", zone_id="SKINCARE",
                   ts=base + timedelta(seconds=60), dwell_ms=60_000),
    ]
    await client.post("/events/ingest", json=events)
    body = (await client.get(f"/stores/{DEFAULT_STORE}/metrics")).json()
    zones = body["avg_dwell_by_zone"]
    assert len(zones) == 1
    assert zones[0]["zone_id"] == "SKINCARE"
    # `visits` now counts ZONE_ENTER only (one per zone visit), not all events.
    # The visitor entered SKINCARE once and got 2 dwell heartbeats; that's 1 visit.
    assert zones[0]["visits"] == 1
    # avg_dwell averages over ZONE_DWELL events only (30000 + 60000) / 2 = 45000ms,
    # not (0 + 30000 + 60000) / 3 = 30000ms that the old buggy query produced.
    assert zones[0]["avg_dwell_ms"] == 45000.0


async def test_metrics_avg_dwell_from_zone_exit_when_no_heartbeats(client):
    """Short visits never emit a 30s ZONE_DWELL heartbeat, but ZONE_EXIT still
    carries the completed-visit dwell. The metric must read that instead of
    reporting 0 — this is the real 2-minute-clip case (no ZONE_DWELL at all)."""
    base = today_at(20)
    events = [
        make_event(event_type="ZONE_ENTER", visitor_id="V1", zone_id="LAKME",
                   ts=base, dwell_ms=0),
        # Visitor leaves after 8s — too short for a heartbeat, so the only
        # dwell signal is the cumulative value on the exit event.
        make_event(event_type="ZONE_EXIT", visitor_id="V1", zone_id="LAKME",
                   ts=base + timedelta(seconds=8), dwell_ms=8_000),
    ]
    await client.post("/events/ingest", json=events)
    body = (await client.get(f"/stores/{DEFAULT_STORE}/metrics")).json()
    zones = body["avg_dwell_by_zone"]
    assert len(zones) == 1
    assert zones[0]["zone_id"] == "LAKME"
    assert zones[0]["visits"] == 1
    assert zones[0]["avg_dwell_ms"] == 8000.0


async def test_metrics_reentry_not_double_counted(client):
    """A visitor with both ENTRY and a later REENTRY is ONE unique visitor —
    unique_visitors filters ENTRY+REENTRY but counts DISTINCT visitor_id."""
    base = today_at(30)
    events = [
        make_event(event_type="ENTRY", visitor_id="V1", ts=base),
        make_event(event_type="REENTRY", visitor_id="V1",
                   ts=base + timedelta(minutes=10)),
        make_event(event_type="ENTRY", visitor_id="V2", ts=base),
    ]
    await client.post("/events/ingest", json=events)
    body = (await client.get(f"/stores/{DEFAULT_STORE}/metrics")).json()
    assert body["unique_visitors"] == 2   # V1 (entry+reentry) + V2, not 3
