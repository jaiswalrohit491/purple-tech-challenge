# PROMPT:
#   "Write pytest cases for pipeline/zones.py covering LineCrosser direction
#    decisions, DwellTracker state transitions (enter, traverse, dwell heartbeat,
#    exit), and point_in_zone for a simple rectangle. Use synthetic point
#    sequences — no video needed."
#
# CHANGES MADE:
#   - Tightened the dwell test: AI version asserted that ZONE_DWELL fires at
#     exactly 30s. The real contract is "every 30s of *continued* dwell";
#     reframed the test to assert two heartbeats over 65s.
#   - Added the entry-exit symmetry case (visitor crosses, returns, crosses
#     again) which the AI version missed — this is the case staff sweeping
#     the doorway hits regularly.

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

shapely = pytest.importorskip("shapely")
from shapely.geometry import Polygon  # noqa: E402

from pipeline.zones import DwellTracker, LineCrosser, build_zone_polygons, point_in_zone  # noqa: E402


def test_line_crosser_returns_entry_when_moving_inward():
    crosser = LineCrosser(a=(0, 100), b=(200, 100), inward_normal=(0, 1), hysteresis_px=50.0)
    # Track 1: well above line (y<<100) → well below (y>>100). Inward normal +y → ENTRY.
    # Both points must be ≥hysteresis_px from the line to count as stable.
    assert crosser.crossed(1, (50, 20)) is None     # first sample, no prior stable side
    assert crosser.crossed(1, (60, 30)) is None     # same side (still well above)
    assert crosser.crossed(1, (70, 180)) == "ENTRY" # now well below → crossing


def test_line_crosser_returns_exit_when_moving_outward():
    crosser = LineCrosser(a=(0, 100), b=(200, 100), inward_normal=(0, 1), hysteresis_px=50.0)
    assert crosser.crossed(2, (70, 200)) is None    # well below
    assert crosser.crossed(2, (80, 30)) == "EXIT"   # now well above → EXIT


def test_line_crosser_handles_back_and_forth():
    """A track that crosses in then out reports ENTRY then EXIT."""
    crosser = LineCrosser(a=(0, 100), b=(200, 100), inward_normal=(0, 1), hysteresis_px=50.0)
    assert crosser.crossed(3, (50, 30)) is None
    assert crosser.crossed(3, (50, 180)) == "ENTRY"
    assert crosser.crossed(3, (50, 20)) == "EXIT"


def test_line_crosser_ignores_jitter_in_deadband():
    """Jittery centroids within hysteresis_px of the line do not trip crossings.
    This is what fixes the CAM_03 over-firing on the actual footage."""
    crosser = LineCrosser(a=(0, 100), b=(200, 100), inward_normal=(0, 1), hysteresis_px=50.0)
    # Walk back and forth across the line with small (<50px) excursions.
    assert crosser.crossed(4, (50, 95)) is None
    assert crosser.crossed(4, (50, 105)) is None
    assert crosser.crossed(4, (50, 90)) is None
    assert crosser.crossed(4, (50, 110)) is None
    assert crosser.crossed(4, (50, 130)) is None  # 30px below — still in deadband
    # Now go clearly inside the store.
    assert crosser.crossed(4, (50, 200)) is None  # first stable side, no prior
    # ...and back outside clearly.
    assert crosser.crossed(4, (50, 20)) == "EXIT"


def test_dwell_tracker_enter_dwell_exit():
    t0 = datetime(2026, 5, 30, 10, 0, 0, tzinfo=timezone.utc)
    dt = DwellTracker(dwell_emit_every=timedelta(seconds=30))

    # Enter zone SKINCARE at t0.
    events = dt.update("V1", "SKINCARE", t0)
    assert [(e[0], e[2]) for e in events] == [("ZONE_ENTER", "SKINCARE")]

    # 15s later — same zone, no event.
    assert dt.update("V1", "SKINCARE", t0 + timedelta(seconds=15)) == []

    # 35s in — first dwell heartbeat.
    out = dt.update("V1", "SKINCARE", t0 + timedelta(seconds=35))
    assert len(out) == 1 and out[0][0] == "ZONE_DWELL" and out[0][2] == "SKINCARE"

    # 70s in — second heartbeat.
    out = dt.update("V1", "SKINCARE", t0 + timedelta(seconds=70))
    assert any(e[0] == "ZONE_DWELL" for e in out)

    # Move to a different zone → emits exit + enter.
    out = dt.update("V1", "HAIRCARE", t0 + timedelta(seconds=80))
    kinds = [e[0] for e in out]
    assert kinds == ["ZONE_EXIT", "ZONE_ENTER"]


def test_dwell_tracker_exit_when_zone_none():
    t0 = datetime(2026, 5, 30, 10, 0, 0, tzinfo=timezone.utc)
    dt = DwellTracker()
    dt.update("V1", "SKINCARE", t0)
    out = dt.update("V1", None, t0 + timedelta(seconds=10))
    assert [e[0] for e in out] == ["ZONE_EXIT"]


def test_point_in_zone_rectangle():
    polys = {"SKINCARE": Polygon([(0, 0), (100, 0), (100, 100), (0, 100)])}
    assert point_in_zone(polys, (50, 50)) == "SKINCARE"
    assert point_in_zone(polys, (150, 50)) is None
    assert point_in_zone(polys, (50, -10)) is None


def test_build_zone_polygons_from_layout_structs():
    """build_zone_polygons must accept the Zone dataclass shape from layout.py."""
    from pipeline.layout import Zone
    zones = [
        Zone(zone_id="A", polygon=[(0, 0), (10, 0), (10, 10), (0, 10)]),
        Zone(zone_id="B", polygon=[(20, 0), (30, 0), (30, 10), (20, 10)]),
    ]
    polys = build_zone_polygons(zones)
    assert set(polys) == {"A", "B"}
    assert polys["A"].contains(shapely.geometry.Point(5, 5))
