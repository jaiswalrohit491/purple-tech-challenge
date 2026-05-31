# PROMPT:
#   "Unit-test pipeline/identity.py — the spatiotemporal identity resolver that
#    replaced the hardcoded-K clustering. Cover the constraints that define
#    'uniqueness': same-camera temporal cannot-link, tracklet stitching, the
#    stitch time-gap, the zone (movement) constraint, topology- and time-gated
#    cross-camera linking, peak-occupancy as a per-camera lower bound, and
#    build_topology defaults. Pure unit tests, no DB."
#
# CHANGES MADE:
#   - Used orthogonal unit vectors for "different appearance" and identical
#     vectors for "same look", so cosine distance is exactly 0 or 1 and the
#     thresholds are unambiguous.
#   - Added the adversarial case the whole redesign hinges on: two identical-
#     looking people CONCURRENT on one camera must stay 2 (cannot-link), while
#     the same two sequential must collapse to 1 (stitch).

from __future__ import annotations

import numpy as np

from pipeline.identity import (
    build_topology,
    choose_count_source,
    peak_occupancy,
    resolve,
)

SAME = np.array([1.0, 0.0, 0.0])      # identical appearance
OTHER = np.array([0.0, 1.0, 0.0])     # orthogonal -> cosine distance 1.0


def test_cannot_link_concurrent_same_camera():
    """Two identical-looking tracks overlapping in time on ONE camera are
    different people — never merged, regardless of appearance."""
    cams = ["CAM_02", "CAM_02"]
    spans = [(0.0, 30.0), (5.0, 25.0)]            # overlap
    X = np.vstack([SAME, SAME])
    _, stats = resolve(cams, spans, X, same_cam_dist=0.9)
    assert stats["n_identities"] == 2
    assert stats["same_camera_cannot_link_pairs"] == 1


def test_stitch_sequential_same_camera():
    """Same look, NON-overlapping, small gap -> one person (stitched)."""
    cams = ["CAM_02", "CAM_02"]
    spans = [(0.0, 10.0), (12.0, 20.0)]           # 2s gap
    X = np.vstack([SAME, SAME])
    _, stats = resolve(cams, spans, X, same_cam_dist=0.5, stitch_gap_s=20.0)
    assert stats["n_identities"] == 1


def test_stitch_blocked_by_large_time_gap():
    cams = ["CAM_02", "CAM_02"]
    spans = [(0.0, 10.0), (200.0, 210.0)]         # 190s gap > stitch_gap
    X = np.vstack([SAME, SAME])
    _, stats = resolve(cams, spans, X, same_cam_dist=0.5, stitch_gap_s=20.0)
    assert stats["n_identities"] == 2


def test_stitch_blocked_by_different_appearance():
    cams = ["CAM_02", "CAM_02"]
    spans = [(0.0, 10.0), (12.0, 20.0)]
    X = np.vstack([SAME, OTHER])                  # cosine distance 1.0 > thr
    _, stats = resolve(cams, spans, X, same_cam_dist=0.5, stitch_gap_s=20.0)
    assert stats["n_identities"] == 2


def test_zone_constraint_blocks_teleport_stitch():
    """Movement: a track ending in zone A and the next starting in zone B (no
    shared endpoint) must NOT stitch even with identical appearance."""
    cams = ["CAM_02", "CAM_02"]
    spans = [(0.0, 10.0), (12.0, 20.0)]
    X = np.vstack([SAME, SAME])
    zones = [(None, "MAYBELLINE"), ("LAKME", None)]   # ends MAYBELLINE, starts LAKME
    _, stats = resolve(cams, spans, X, zones=zones, same_cam_dist=0.9, stitch_gap_s=20.0)
    assert stats["n_identities"] == 2
    # Same zone at the seam -> allowed to stitch.
    zones_ok = [(None, "LAKME"), ("LAKME", None)]
    _, stats2 = resolve(cams, spans, X, zones=zones_ok, same_cam_dist=0.9, stitch_gap_s=20.0)
    assert stats2["n_identities"] == 1


def test_cross_camera_link_respects_topology():
    cams = ["CAM_01", "CAM_05"]
    spans = [(0.0, 10.0), (12.0, 20.0)]
    X = np.vstack([SAME, SAME])
    topo_ok = {"CAM_01": {"CAM_05"}, "CAM_05": {"CAM_01"}}
    _, s_ok = resolve(cams, spans, X, topology=topo_ok, cross_cam_dist=0.5, cross_window_s=45.0)
    assert s_ok["n_identities"] == 1 and s_ok["cross_camera_links"] == 1
    topo_no = {"CAM_01": {"CAM_01"}, "CAM_05": {"CAM_05"}}     # not reachable
    _, s_no = resolve(cams, spans, X, topology=topo_no, cross_cam_dist=0.5, cross_window_s=45.0)
    assert s_no["n_identities"] == 2


def test_cross_camera_blocked_by_time_window():
    cams = ["CAM_01", "CAM_05"]
    spans = [(0.0, 10.0), (500.0, 510.0)]         # 490s gap > cross_window
    X = np.vstack([SAME, SAME])
    _, stats = resolve(cams, spans, X, topology=None, cross_cam_dist=0.5, cross_window_s=45.0)
    assert stats["n_identities"] == 2


def test_peak_occupancy_is_per_camera_not_summed():
    """3 concurrent on CAM_A + 2 concurrent on CAM_B at the same instant ->
    occupancy is 3 (busiest single camera), not 5 (summing double-counts FOV)."""
    cams = ["CAM_A", "CAM_A", "CAM_A", "CAM_B", "CAM_B"]
    spans = [(0, 10), (1, 9), (2, 8), (0, 10), (1, 9)]
    labels = np.arange(5)
    assert peak_occupancy(spans, cams) == 3


def test_build_topology_default_and_explicit():
    cams = ["CAM_01", "CAM_02", "CAM_05"]
    full = build_topology(cams, None)             # fully connected
    assert full["CAM_01"] == {"CAM_01", "CAM_02", "CAM_05"}
    explicit = build_topology(cams, {"CAM_01": ["CAM_02"]})
    assert explicit["CAM_01"] == {"CAM_01", "CAM_02"}   # self always included


def test_resolve_empty_input():
    labels, stats = resolve([], [], np.empty((0, 3)))
    assert stats["n_identities"] == 0
    assert len(labels) == 0


def test_count_source_prefers_entry_gate():
    """Entry-gate footfall is authoritative whenever it has data; appearance is
    used only as a fallback when the gate feed is insufficient."""
    assert choose_count_source(gate_entries=5) == "entry_gate"
    assert choose_count_source(gate_entries=1) == "entry_gate"
    assert choose_count_source(gate_entries=0) == "appearance"          # this clip
    # threshold is configurable, but the gate always wins once it's met
    assert choose_count_source(gate_entries=2, min_gate_entries=3) == "appearance"
    assert choose_count_source(gate_entries=3, min_gate_entries=3) == "entry_gate"
