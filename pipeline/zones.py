"""Geometric primitives for zone membership, line crossing, and dwell tracking.

Why we don't use the full shapely STRtree: each frame has at most ~30 detections,
each store has ~5 zones, so a naive point-in-polygon over `shapely.Polygon` is
fast enough and simpler to debug. If a future store has hundreds of zones, swap
this for STRtree-based indexing in one place.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta

from shapely.geometry import Point, Polygon


def build_zone_polygons(zone_defs) -> dict[str, Polygon]:
    """Materialise shapely polygons from the layout structs (cached per camera)."""
    return {z.zone_id: Polygon(z.polygon) for z in zone_defs}


def point_in_zone(zones: dict[str, Polygon], xy: tuple[float, float]) -> str | None:
    """Return the first zone that contains (x, y); None if none does.
    The bottom-center of a bbox is the most stable reference for floor location.
    """
    p = Point(xy)
    for zone_id, poly in zones.items():
        if poly.contains(p):
            return zone_id
    return None


# ---------- line crossing ----------

@dataclass
class LineCrosser:
    """Detects when a track's centroid crosses the entry line, with hysteresis.

    Plain side-flip detection produces dozens of false crossings when a
    visitor stands near the line and the YOLO bbox center jitters by a few
    pixels per frame. Real crossings, by contrast, happen with continuous
    motion — the centroid is far from the line on side A, then far on side B.

    Hysteresis enforces that: a track's `stable_side` is only updated when its
    centroid is at least `hysteresis_px` pixels away from the line. A crossing
    fires when the stable_side flips, never on jitter near the line itself.
    """
    a: tuple[float, float]
    b: tuple[float, float]
    inward_normal: tuple[float, float] = (0.0, 1.0)
    hysteresis_px: float = 50.0
    _stable_side: dict[int, int] = field(default_factory=dict)

    def _signed_distance(self, p: tuple[float, float]) -> float:
        """Signed perpendicular distance from p to the line (a, b).

        Sign convention matches `inside_sign = (lx*ny - ly*nx) > 0` (see
        `crossed`), so positive values lie on the side picked out by the
        inward_normal direction. Implemented as the cross product
        (a→b) × (a→p) divided by |a→b|.
        """
        ax, ay = self.a
        bx, by = self.b
        px, py = p
        dx, dy = bx - ax, by - ay
        length = (dx * dx + dy * dy) ** 0.5 or 1.0
        return (dx * (py - ay) - dy * (px - ax)) / length

    def crossed(self, track_id: int, p: tuple[float, float]) -> str | None:
        d = self._signed_distance(p)
        if abs(d) < self.hysteresis_px:
            # Inside the dead-band around the line — can't be considered stable.
            return None
        new_side = 1 if d > 0 else -1
        prev_stable = self._stable_side.get(track_id)
        self._stable_side[track_id] = new_side
        if prev_stable is None or prev_stable == new_side:
            return None

        # Resolve which side is "inside".
        lx, ly = self.b[0] - self.a[0], self.b[1] - self.a[1]
        nx, ny = self.inward_normal
        inside_sign = 1 if (lx * ny - ly * nx) > 0 else -1
        moving_inside = new_side == inside_sign
        return "ENTRY" if moving_inside else "EXIT"


# ---------- per-visitor dwell tracker ----------

@dataclass
class DwellTracker:
    """Maintains current-zone state per visitor and emits ZONE_ENTER / ZONE_EXIT /
    ZONE_DWELL transitions. Caller drives it once per frame via `update()`.

    Time math is in *event clock* (i.e. timestamps derived from clip + frame
    offset), not wall clock. The dwell threshold is fixed at 30s by the brief.
    """
    dwell_emit_every: timedelta = timedelta(seconds=30)
    _state: dict[str, dict] = field(default_factory=dict)

    def update(self, visitor_id: str, zone_id: str | None, now: datetime):
        """Returns a list of (event_type, dwell_ms, zone_id) tuples to emit."""
        out = []
        st = self._state.get(visitor_id)

        if st is None:
            if zone_id is not None:
                self._state[visitor_id] = {"zone": zone_id, "entered_at": now, "last_dwell": now}
                out.append(("ZONE_ENTER", 0, zone_id))
            return out

        prev_zone = st["zone"]
        if zone_id != prev_zone:
            # Leaving previous zone.
            dwell_ms = int((now - st["entered_at"]).total_seconds() * 1000)
            out.append(("ZONE_EXIT", dwell_ms, prev_zone))
            if zone_id is not None:
                out.append(("ZONE_ENTER", 0, zone_id))
                self._state[visitor_id] = {"zone": zone_id, "entered_at": now, "last_dwell": now}
            else:
                self._state.pop(visitor_id, None)
            return out

        # Same zone — check dwell heartbeat.
        if now - st["last_dwell"] >= self.dwell_emit_every:
            dwell_ms = int((now - st["entered_at"]).total_seconds() * 1000)
            out.append(("ZONE_DWELL", dwell_ms, prev_zone))
            st["last_dwell"] = now
        return out

    def force_exit(self, visitor_id: str, now: datetime):
        """Emit a final ZONE_EXIT (e.g. when track is lost or session ends)."""
        st = self._state.pop(visitor_id, None)
        if not st:
            return []
        dwell_ms = int((now - st["entered_at"]).total_seconds() * 1000)
        return [("ZONE_EXIT", dwell_ms, st["zone"])]
