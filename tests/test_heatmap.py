# PROMPT:
#   "Tests for GET /stores/{id}/heatmap. Verify: visit_score and dwell_score
#    are 0-100 scaled to the busiest zone, data_confidence is 'LOW' when fewer
#    than 20 distinct non-staff visitors in window, 'OK' otherwise, and the
#    response handles empty stores by returning zones=[]."
#
# CHANGES MADE:
#   - Added an explicit normalization check: the top-visit zone must be 100.0,
#     and a half-traffic zone must be ~50.0. AI version only checked >0.

from __future__ import annotations

from datetime import timedelta

from tests.conftest import DEFAULT_STORE, make_event, today_at


async def test_heatmap_empty(client):
    body = (await client.get(f"/stores/{DEFAULT_STORE}/heatmap")).json()
    assert body["zones"] == []
    assert body["data_confidence"] == "LOW"


async def test_heatmap_normalization(client):
    """Busiest zone => visit_score == 100; quieter zones scaled proportionally."""
    base = today_at(10)
    events = []
    # SKINCARE: 4 visits / HAIRCARE: 2 visits / BILLING: 1 visit
    for i, (zone, count) in enumerate([("SKINCARE", 4), ("HAIRCARE", 2), ("BILLING", 1)]):
        for j in range(count):
            events.append(make_event(
                event_type="ZONE_ENTER", visitor_id=f"V_{zone}_{j}",
                zone_id=zone, ts=base + timedelta(seconds=j),
            ))
    await client.post("/events/ingest", json=events)

    body = (await client.get(f"/stores/{DEFAULT_STORE}/heatmap")).json()
    zones = {z["zone_id"]: z for z in body["zones"]}
    assert zones["SKINCARE"]["visit_score"] == 100.0
    assert zones["HAIRCARE"]["visit_score"] == 50.0
    assert zones["BILLING"]["visit_score"] == 25.0


async def test_heatmap_data_confidence_threshold(client):
    """20+ distinct visitors flips data_confidence to OK."""
    base = today_at(10)
    events = [
        make_event(event_type="ZONE_ENTER", visitor_id=f"V_{i}",
                   zone_id="SKINCARE", ts=base + timedelta(seconds=i))
        for i in range(25)
    ]
    await client.post("/events/ingest", json=events)
    body = (await client.get(f"/stores/{DEFAULT_STORE}/heatmap")).json()
    assert body["data_confidence"] == "OK"


async def test_heatmap_excludes_staff(client):
    base = today_at(10)
    events = [
        make_event(event_type="ZONE_ENTER", visitor_id=f"S_{i}", is_staff=True,
                   zone_id="SKINCARE", ts=base) for i in range(5)
    ]
    await client.post("/events/ingest", json=events)
    body = (await client.get(f"/stores/{DEFAULT_STORE}/heatmap")).json()
    assert body["zones"] == []
