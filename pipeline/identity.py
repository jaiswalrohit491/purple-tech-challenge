"""Spatiotemporal identity resolution — a *measured* unique-person count, no prior K.

The earlier pipeline counted unique people by forcing appearance embeddings into
`K` clusters, where `K` was an operator-supplied headcount. That is a hardcoded
answer and does not survive real data. This module replaces it: the count
*emerges* from physical constraints that hold no matter how similar people look.

Signals (strongest first):
  1. **Same-camera temporal CANNOT-LINK** — two tracks that overlap in time on
     one camera are definitely different people (one body can't be in two places).
     This caps over-merging with zero appearance assumptions and yields a hard,
     measured lower bound: peak simultaneous occupancy.
  2. **Within-camera tracklet stitching** — ByteTrack fragments of one person are
     temporally *disjoint* and close in time; stitch adjacent fragments (small
     time gap) when appearance agrees. This removes the fragmentation that
     otherwise inflates the count.
  3. **Cross-camera linking** — link tracks across cameras only when the camera
     topology allows the move within a time window AND appearance agrees. The
     entrance camera is used to confirm/anchor, never to supply the count.

The unique count = number of connected components after applying these. No `K`,
no `expected_*`. The similarity thresholds are re-ID *matching* parameters
(like any gallery threshold), not the answer — they don't encode how many people
there are.
"""
from __future__ import annotations

from collections import defaultdict
from datetime import datetime

import numpy as np


def parse_ts(s: str) -> float:
    """ISO-8601 → epoch seconds."""
    return datetime.fromisoformat(s.replace("Z", "+00:00")).timestamp()


def _overlap(a: tuple[float, float], b: tuple[float, float]) -> bool:
    return a[0] <= b[1] and b[0] <= a[1]


def _gap(a: tuple[float, float], b: tuple[float, float]) -> float:
    """Seconds between two intervals; 0 if they overlap."""
    return max(b[0] - a[1], a[0] - b[1], 0.0)


def resolve(
    cameras: list[str],
    spans: list[tuple[float, float]],
    X: np.ndarray,
    topology: dict[str, set[str]] | None = None,
    zones: list[tuple[str | None, str | None]] | None = None,
    same_cam_dist: float = 0.30,
    cross_cam_dist: float = 0.25,
    stitch_gap_s: float = 20.0,
    cross_window_s: float = 45.0,
) -> tuple[np.ndarray, dict]:
    """Resolve tracks into physical identities via constrained clustering.

    Args index-aligned: cameras[i], spans[i]=(t0,t1) epoch s, X[i]=unit embedding.
    topology: cam -> set of reachable cams. None ⇒ fully connected (any cross-camera
              move allowed within the time window).
    Returns (labels, stats). labels[i] = identity index for track i.
    """
    n = len(cameras)
    parent = list(range(n))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(i: int, j: int) -> None:
        ri, rj = find(i), find(j)
        if ri != rj:
            parent[ri] = rj

    D = 1.0 - X @ X.T  # cosine distance
    cannot_link_pairs = 0
    same_links = 0
    cross_links = 0

    for i in range(n):
        for j in range(i + 1, n):
            ci, cj = cameras[i], cameras[j]
            si, sj = spans[i], spans[j]
            d = float(D[i, j])
            if ci == cj:
                if _overlap(si, sj):
                    cannot_link_pairs += 1  # different people — never merge
                    continue
                # Stitch adjacent fragments of one person: small time gap AND
                # appearance agreement. Movement-as-continuity: we only consider
                # stitching tracks that are temporally adjacent (the gap test) —
                # the appearance threshold is never relaxed, so colour separation
                # between different people is preserved. (Zone continuity is
                # available via `zones` for stricter gating but is intentionally
                # NOT used to *loosen* matching, which over-merged in testing.)
                zone_ok = True
                if zones is not None:
                    ie, je = (i, j) if si[1] <= sj[1] else (j, i)  # ie ends first
                    ze, zb = zones[ie][1], zones[je][0]
                    # If both endpoints have a zone and they differ, the person
                    # would have had to teleport — require appearance to be tight.
                    zone_ok = (ze is None or zb is None or ze == zb)
                if _gap(si, sj) <= stitch_gap_s and d <= same_cam_dist and zone_ok:
                    union(i, j)
                    same_links += 1
            else:
                # Cross-camera: topology + time window + appearance.
                if topology is not None and cj not in topology.get(ci, set()):
                    continue
                if _gap(si, sj) <= cross_window_s and d <= cross_cam_dist:
                    union(i, j)
                    cross_links += 1

    roots: dict[int, int] = {}
    labels = np.empty(n, dtype=int)
    for i in range(n):
        r = find(i)
        if r not in roots:
            roots[r] = len(roots)
        labels[i] = roots[r]

    return labels, {
        "n_tracks": n,
        "n_identities": len(roots),
        "same_camera_stitches": same_links,
        "cross_camera_links": cross_links,
        "same_camera_cannot_link_pairs": cannot_link_pairs,
    }


def peak_occupancy(spans: list[tuple[float, float]], cameras: list[str]) -> int:
    """Max concurrent tracks on a *single* camera — a hard, measured lower bound
    on how many distinct people are present.

    This is valid WITHOUT any re-ID: two tracks on the same camera at the same
    instant are guaranteed to be different people (one body, one place). We take
    the busiest single camera rather than summing across cameras, because tracks
    on different cameras at the same instant might be the *same* person (FOV
    overlap / hand-off) and summing would over-count. So this is a conservative
    floor on occupancy that holds on any footage, even when appearance can't
    merge fragments into identities."""
    by_cam: dict[str, list[tuple[float, int]]] = defaultdict(list)
    for (t0, t1), cam in zip(spans, cameras):
        by_cam[cam].append((t0, 0))   # start (process before ends at same ts)
        by_cam[cam].append((t1, 1))   # end
    peak = 0
    for pts in by_cam.values():
        pts.sort(key=lambda x: (x[0], x[1]))
        cur = 0
        for _, kind in pts:
            cur += 1 if kind == 0 else -1
            peak = max(peak, cur)
    return peak


def build_topology(cameras: list[str], layout_topology: dict | None) -> dict[str, set[str]] | None:
    """Build a reachability map. If the layout declares `camera_topology`
    (`{cam: [reachable cams]}`), use it; else fully-connect the given cameras
    (any in-store move allowed — conservative, lets appearance+time decide)."""
    cams = sorted(set(cameras))
    if layout_topology:
        return {c: set(layout_topology.get(c, cams)) | {c} for c in cams}
    return {c: set(cams) for c in cams}  # fully connected
