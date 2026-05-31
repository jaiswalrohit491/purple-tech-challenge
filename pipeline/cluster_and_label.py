"""Resolve ByteTrack tracks across in-store cameras into physical persons and
label each staff/customer — with NO operator headcount prior. The unique count
is measured, not supplied.

The pipeline:
  1. For each `CAM_XX_trackYYY.jpg` crop, compute an appearance signature via
     pipeline/reid.py. Default backend is a torso HSV colour histogram — the cue
     that separates a grey shirt / tan bag / dark uniform (deep ResNet/OSNet
     backends are opt-in via REID_BACKEND). A crop-quality gate drops occluded
     boxes first.
  2. Classify each track staff/customer by distance to the auto-built
     staff_gallery (the dark-uniform group); threshold derived from gallery
     cohesion.
  3. Within each role, pipeline/identity.py resolves distinct people by
     CONSTRAINED SPATIOTEMPORAL clustering on the signature: same-camera
     temporal cannot-link (concurrent tracks = different people) + tracklet
     stitching of adjacent fragments + topology-gated cross-camera links. The
     identity count EMERGES — no K, no expected_* prior.
  4. Output a single merged JSONL where visitor_id is the canonical person ID
     (CUSTOMER_01, STAFF_03, …). One synthetic ENTRY per resolved customer feeds
     /metrics.unique_visitors.

Usage:
    python -m pipeline.cluster_and_label \\
        --events-dir /events --cameras CAM_01 CAM_02 CAM_05 \\
        --crops-dirs /events/track_crops/CAM_01 /events/track_crops/CAM_02 /events/track_crops/CAM_05 \\
        --staff-gallery /data/staff_gallery --layout /data/store_layout.json \\
        --same-cam-dist 0.45 --cross-cam-dist 0.45 \\
        --out /events/ST1008_merged.jsonl
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

TRACK_RE = re.compile(r"(CAM_\d+)_track(\d+)")
VISITOR_RE = re.compile(r"(CAM_\d+)#(\d+)")


def gallery_embeddings(gallery_dir: Path, embedder) -> np.ndarray | None:
    """All gallery crop embeddings, stacked. None if the dir is empty/missing."""
    if not gallery_dir.is_dir():
        return None
    embs = []
    for f in sorted(gallery_dir.iterdir()):
        if f.suffix.lower() in (".jpg", ".jpeg", ".png"):
            v = embedder.embed_path(f)
            if v is not None:
                embs.append(v)
    return np.vstack(embs) if embs else None


def gallery_centroid(gallery_dir: Path, embedder) -> np.ndarray | None:
    G = gallery_embeddings(gallery_dir, embedder)
    if G is None:
        return None
    c = G.mean(axis=0)
    return c / (np.linalg.norm(c) or 1.0)


def _crop_quality_ok(path: Path, min_aspect: float, min_short_side: int) -> bool:
    """Crop-quality gate for appearance-based identity.

    A YOLO person box for a standing/walking person is portrait-oriented
    (taller than wide). A near-square or landscape crop means the person is
    heavily occluded, partially out of frame, or seen top-down — its ResNet
    embedding is unreliable and tends to land in the ambiguous distance band,
    producing spurious identities (e.g. a top-down billing-counter blob that
    gets labelled "customer" and fabricates a billing-queue visitor).

    We require: height/width >= `min_aspect` AND the short side is at least
    `min_short_side` px. This is a standard re-ID input filter, not a
    per-track tuning knob — it drops the same class of degenerate crops on any
    store.
    """
    from PIL import Image
    try:
        with Image.open(path) as img:
            w, h = img.size
    except Exception:
        return False
    if w <= 0 or h <= 0:
        return False
    return (h / w) >= min_aspect and min(w, h) >= min_short_side


def parse_crops(
    crops_dirs: list[Path],
    min_aspect: float = 1.3,
    min_short_side: int = 60,
) -> tuple[list[tuple[str, int, Path]], list[str]]:
    """Return (kept tracks, dropped crop names). Tracks whose representative
    crop fails the quality gate are excluded from clustering entirely; their
    events fall through as unmatched downstream."""
    out: list[tuple[str, int, Path]] = []
    dropped: list[str] = []
    for d in crops_dirs:
        if not d.is_dir():
            continue
        for f in sorted(d.glob("*.jpg")):
            m = TRACK_RE.match(f.name)
            if not m:
                continue
            if not _crop_quality_ok(f, min_aspect, min_short_side):
                dropped.append(f.name)
                continue
            out.append((m.group(1), int(m.group(2)), f))
    return out, dropped


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--events-dir", type=Path, default=Path("/events"))
    p.add_argument("--cameras", nargs="+", default=["CAM_01", "CAM_02", "CAM_05"])
    p.add_argument("--crops-dirs", nargs="+", type=Path, required=True)
    p.add_argument("--staff-gallery", type=Path, default=Path("/data/staff_gallery"))
    p.add_argument("--customer-gallery", type=Path, default=Path("/data/customer_gallery"))
    p.add_argument("--staff-threshold", type=float, default=-1.0,
                   help="Cosine distance threshold against the staff gallery. "
                        "Negative (default) = derive it data-drivenly from the "
                        "gallery's own cohesion, so it adapts to the active "
                        "embedding backend. Used only when --customer-gallery is missing.")
    p.add_argument("--layout", type=Path, default=None,
                   help="Layout JSON; read for optional camera_topology (store geometry).")
    p.add_argument("--same-cam-dist", type=float, default=0.30,
                   help="Cosine-distance threshold to stitch adjacent same-camera "
                        "tracklets (re-ID matching param, not a count).")
    p.add_argument("--cross-cam-dist", type=float, default=0.25,
                   help="Cosine-distance threshold to link tracks across cameras.")
    p.add_argument("--stitch-gap", type=float, default=20.0,
                   help="Max seconds between same-camera tracklets to stitch them.")
    p.add_argument("--cross-window", type=float, default=45.0,
                   help="Max seconds between tracks on different cameras to link them.")
    p.add_argument("--reentry-gap", type=float, default=60.0,
                   help="If a resolved customer is absent from all cameras longer "
                        "than this many seconds and returns, emit a REENTRY event "
                        "(same visitor_id, no double-count).")
    p.add_argument("--entry-camera", default=None,
                   help="Entrance camera id (line-crossing footfall). The unique "
                        "count comes from here when it has entries; appearance is "
                        "the fallback. Default: auto-detect from layout (view=ENTRY).")
    p.add_argument("--min-gate-entries", type=int, default=1,
                   help="Min entry-gate crossings to treat the gate as the "
                        "authoritative count source (else fall back to appearance).")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--store-id", default="STORE_001")
    p.add_argument("--min-crop-aspect", type=float, default=1.3,
                   help="Crop-quality gate: drop tracks whose representative "
                        "crop has height/width below this (occluded / partial / "
                        "top-down boxes). Set 0 to disable.")
    p.add_argument("--min-crop-short-side", type=int, default=60,
                   help="Crop-quality gate: drop tracks whose crop short side "
                        "is below this many pixels.")
    args = p.parse_args()

    from .reid import get_embedder
    embedder = get_embedder()
    if not embedder.available:
        print("no embedding backend available (torch missing)", file=sys.stderr)
        return 1
    print(f"embedding backend: {embedder.backend} (dim={embedder.dim})", file=sys.stderr)

    tracks, dropped_crops = parse_crops(
        args.crops_dirs,
        min_aspect=args.min_crop_aspect,
        min_short_side=args.min_crop_short_side,
    )
    if dropped_crops:
        print(f"crop-quality gate dropped {len(dropped_crops)} low-quality "
              f"tracks (aspect<{args.min_crop_aspect}, short-side<"
              f"{args.min_crop_short_side}px): {', '.join(sorted(dropped_crops))}",
              file=sys.stderr)
    if not tracks:
        print("no crops found", file=sys.stderr)
        return 1
    print(f"embedding {len(tracks)} tracks…", file=sys.stderr)
    X = np.vstack([embedder.embed_path(t[2]) for t in tracks])

    staff_G = gallery_embeddings(args.staff_gallery, embedder)
    cust_c = gallery_centroid(args.customer_gallery, embedder)
    if staff_G is None:
        print("staff gallery empty — cannot label", file=sys.stderr)
        return 1
    staff_c = staff_G.mean(axis=0)
    staff_c = staff_c / (np.linalg.norm(staff_c) or 1.0)
    has_customer_gallery = cust_c is not None

    # Distances to gallery (1 - cosine).
    d_staff = 1.0 - X @ staff_c
    if has_customer_gallery:
        d_cust = 1.0 - X @ cust_c
        is_customer_track = d_cust < d_staff
    else:
        # No customer gallery — pure-data, uniform-only classification: tracks
        # within `staff_threshold` of the auto-built staff gallery are staff;
        # everything else is a customer. The threshold is derived from the
        # gallery's OWN cohesion (mean + 2·std of gallery-to-centroid distance)
        # so it adapts to the active embedding backend rather than being a
        # magic number tuned for one model. An explicit --staff-threshold >= 0
        # overrides it.
        if args.staff_threshold >= 0:
            staff_threshold = args.staff_threshold
        else:
            gd = 1.0 - staff_G @ staff_c
            staff_threshold = float(gd.mean() + 2.0 * gd.std()) if len(gd) > 1 else 0.3
        print(f"staff distance threshold: {staff_threshold:.3f} "
              f"({'explicit' if args.staff_threshold >= 0 else 'auto from gallery cohesion'})",
              file=sys.stderr)
        is_customer_track = d_staff > staff_threshold

    customer_idxs = np.where(is_customer_track)[0]
    staff_idxs = np.where(~is_customer_track)[0]
    print(f"preliminary: {len(staff_idxs)} staff tracks, {len(customer_idxs)} customer tracks",
          file=sys.stderr)

    # ---- Resolve physical identities per role via spatiotemporal constrained
    # clustering (pipeline/identity.py). The count EMERGES — no operator K, no
    # expected_* prior. Same-camera temporal overlap is a hard cannot-link;
    # within-camera fragments are stitched; cross-camera links are topology- and
    # time-gated. See identity.py for the full rationale.
    from . import identity as idmod

    # Per-track time spans (epoch seconds) + first/last zone, from the
    # per-camera event streams. Zones feed the movement-continuity bonus.
    span_map: dict[tuple[str, int], tuple[float, float]] = {}
    zone_map: dict[tuple[str, int], dict] = {}
    for cam in args.cameras:
        ef = args.events_dir / f"{args.store_id}_{cam}.jsonl"
        if not ef.exists():
            continue
        with ef.open() as f:
            for line in f:
                if not line.strip():
                    continue
                e = json.loads(line)
                mm = VISITOR_RE.match(e.get("visitor_id", ""))
                if not mm:
                    continue
                k = (mm.group(1), int(mm.group(2)))
                ts = idmod.parse_ts(e["timestamp"])
                lo, hi = span_map.get(k, (ts, ts))
                span_map[k] = (min(lo, ts), max(hi, ts))
                z = e.get("zone_id")
                zi = zone_map.get(k)
                if zi is None:
                    zone_map[k] = {"fts": ts, "fz": z, "lts": ts, "lz": z}
                else:
                    if ts <= zi["fts"]:
                        zi["fts"], zi["fz"] = ts, z
                    if ts >= zi["lts"]:
                        zi["lts"], zi["lz"] = ts, z

    # Camera topology (physical reachability) from the layout; fully-connected
    # default if not declared. This is store geometry, not a count.
    layout_topo = None
    if args.layout and args.layout.exists():
        try:
            ld = json.loads(args.layout.read_text())
            stores = ld.get("stores", ld if isinstance(ld, list) else [])
            for s in stores:
                if s.get("store_id") == args.store_id or len(stores) == 1:
                    layout_topo = s.get("camera_topology")
                    # Auto-detect the entrance camera (view==ENTRY / has entry_line)
                    # unless one was passed explicitly.
                    if not args.entry_camera:
                        for c in s.get("cameras", []):
                            if c.get("view") == "ENTRY" or c.get("entry_line"):
                                args.entry_camera = c["camera_id"]
                                break
                    break
        except Exception:
            layout_topo = None
    topo = idmod.build_topology([t[0] for t in tracks], layout_topo)

    canonical_per_track: dict[int, tuple[str, bool]] = {}
    resolve_stats: dict[str, dict] = {}

    def _resolve(idxs: np.ndarray, role: str, is_staff: bool) -> int:
        if not len(idxs):
            return 0
        cams = [tracks[i][0] for i in idxs]
        spans = [span_map.get((tracks[i][0], tracks[i][1]), (0.0, 0.0)) for i in idxs]
        zlist = []
        for i in idxs:
            zi = zone_map.get((tracks[i][0], tracks[i][1]))
            zlist.append((zi["fz"], zi["lz"]) if zi else (None, None))
        labels, stats = idmod.resolve(
            cams, spans, X[idxs], topology=topo, zones=zlist,
            same_cam_dist=args.same_cam_dist, cross_cam_dist=args.cross_cam_dist,
            stitch_gap_s=args.stitch_gap, cross_window_s=args.cross_window,
        )
        for i, lab in zip(idxs, labels):
            canonical_per_track[int(i)] = (f"{role}_{int(lab) + 1:02d}", is_staff)
        stats["peak_occupancy"] = idmod.peak_occupancy(spans, cams)
        resolve_stats[role] = stats
        return stats["n_identities"]

    _resolve(staff_idxs, "STAFF", True)
    _resolve(customer_idxs, "CUSTOMER", False)

    # Map (camera, track_id) -> canonical.
    track_map: dict[tuple[str, int], tuple[str, bool]] = {}
    for idx, (cam, tid, _) in enumerate(tracks):
        track_map[(cam, tid)] = canonical_per_track[idx]

    # Rewrite events.
    out_events: list[dict] = []
    unmatched = 0
    for cam in args.cameras:
        ef = args.events_dir / f"{args.store_id}_{cam}.jsonl"
        if not ef.exists():
            continue
        with ef.open() as f:
            for line in f:
                if not line.strip():
                    continue
                e = json.loads(line)
                m = VISITOR_RE.match(e["visitor_id"])
                if not m:
                    out_events.append(e); continue
                key = (m.group(1), int(m.group(2)))
                if key not in track_map:
                    unmatched += 1
                    continue
                canon, is_staff = track_map[key]
                e["visitor_id"] = canon
                e["is_staff"] = is_staff
                # Stamp the source track into metadata for auditing.
                meta = dict(e.get("metadata") or {})
                meta["source_track"] = f"{key[0]}#{key[1]}"
                e["metadata"] = meta
                out_events.append(e)

    import uuid as _uuid
    from . import identity as _idmod

    # ---- Visitor count: ENTRY GATE is authoritative; appearance is fallback ----
    # Footfall through the entrance line (CAM with view==ENTRY / an entry_line) is
    # the canonical, higher-confidence count and ALWAYS wins when it has data.
    # Appearance-resolved identities are used to count ONLY when the gate feed is
    # insufficient (e.g. this clip: customers already inside, zero entries
    # captured). See identity.choose_count_source.
    earliest: dict[str, dict] = {}
    for e in out_events:
        canon = e["visitor_id"]
        if canon not in earliest or e["timestamp"] < earliest[canon]["timestamp"]:
            earliest[canon] = e

    # Gate entries: customer inward ENTRY crossings on the entry camera, one per
    # gate track (re-crossings handled as re-entry downstream).
    gate_entries: list[dict] = []
    if args.entry_camera:
        gef = args.events_dir / f"{args.store_id}_{args.entry_camera}.jsonl"
        if gef.exists():
            seen_track: dict[str, str] = {}
            for line in gef.read_text().splitlines():
                if not line.strip():
                    continue
                ge = json.loads(line)
                if ge.get("event_type") != "ENTRY" or ge.get("is_staff"):
                    continue
                tk = ge["visitor_id"]
                if tk not in seen_track or ge["timestamp"] < seen_track[tk]:
                    seen_track[tk] = ge["timestamp"]
            gate_entries = [{"track": tk, "timestamp": ts} for tk, ts in seen_track.items()]
            gate_entries.sort(key=lambda g: g["timestamp"])

    count_source = _idmod.choose_count_source(len(gate_entries), args.min_gate_entries)

    entry_events = []
    if count_source == "entry_gate":
        # Authoritative footfall: one ENTRY per gate crossing. Link each crossing
        # to the in-store appearance identity that first appears AFTER it (so the
        # funnel still chains entry→zones→billing); unmatched crossings are real
        # visitors who simply weren't re-identified in-store.
        instore = sorted(
            ((e["timestamp"], canon) for canon, e in earliest.items() if not e.get("is_staff")),
        )
        used: set[str] = set()
        for n, g in enumerate(gate_entries, 1):
            vid = None
            for ts, canon in instore:
                if canon not in used and ts >= g["timestamp"]:
                    vid, _ = canon, used.add(canon)
                    break
            vid = vid or f"VISITOR_{n:02d}"
            ev0 = earliest.get(vid, next(iter(earliest.values())))
            entry_events.append({
                "event_id": str(_uuid.uuid4()), "store_id": ev0["store_id"],
                "camera_id": args.entry_camera, "visitor_id": vid, "event_type": "ENTRY",
                "timestamp": g["timestamp"], "zone_id": None, "dwell_ms": 0,
                "is_staff": False, "confidence": 0.9,
                "metadata": {"queue_depth": None, "sku_zone": None, "session_seq": 0,
                             "synthetic": True, "source": "entry_gate",
                             "gate_track": g["track"]},
            })
    else:
        # Fallback: appearance-resolved identities (gate feed insufficient).
        for canon, e in earliest.items():
            if e.get("is_staff"):
                continue
            entry_events.append({
                "event_id": str(_uuid.uuid4()), "store_id": e["store_id"],
                "camera_id": e["camera_id"], "visitor_id": canon, "event_type": "ENTRY",
                "timestamp": e["timestamp"], "zone_id": None, "dwell_ms": 0,
                "is_staff": False, "confidence": 0.9,
                "metadata": {"queue_depth": None, "sku_zone": None, "session_seq": 0,
                             "synthetic": True, "source": "appearance_fallback",
                             "reason": "entry gate had no usable entries"},
            })
    out_events.extend(entry_events)

    # Re-entry: a resolved customer who disappears from all cameras for longer
    # than --reentry-gap and then reappears gets a REENTRY event at the moment
    # of return, REUSING the same visitor_id (so the funnel / unique_visitors do
    # not double-count — REENTRY and ENTRY both map to one person). This is the
    # measured analogue of "left the floor and came back".
    reentry_events = []
    by_canon: dict[str, list[dict]] = defaultdict(list)
    for e in out_events:
        if e["event_type"] in ("ENTRY", "REENTRY"):
            continue
        if not e.get("is_staff"):
            by_canon[e["visitor_id"]].append(e)
    for canon, evs in by_canon.items():
        evs = sorted(evs, key=lambda e: e["timestamp"])
        prev = None
        for e in evs:
            ts = _idmod.parse_ts(e["timestamp"])
            if prev is not None and (ts - prev) > args.reentry_gap:
                reentry_events.append({
                    "event_id": str(_uuid.uuid4()),
                    "store_id": e["store_id"], "camera_id": e["camera_id"],
                    "visitor_id": canon, "event_type": "REENTRY",
                    "timestamp": e["timestamp"], "zone_id": None, "dwell_ms": 0,
                    "is_staff": False, "confidence": 0.9,
                    "metadata": {"queue_depth": None, "sku_zone": None,
                                 "session_seq": 0, "synthetic": True,
                                 "reason": f"reappeared after >{int(args.reentry_gap)}s absence"},
                })
            prev = ts
    out_events.extend(reentry_events)
    out_events.sort(key=lambda e: e["timestamp"])

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w") as f:
        for e in out_events:
            f.write(json.dumps(e, separators=(",", ":")) + "\n")

    # Summary.
    canon_role = {}
    for e in out_events:
        canon_role[e["visitor_id"]] = "STAFF" if e["is_staff"] else "CUSTOMER"
    customers = sorted([c for c, r in canon_role.items() if r == "CUSTOMER"])
    staff = sorted([c for c, r in canon_role.items() if r == "STAFF"])

    cust_occ = resolve_stats.get("CUSTOMER", {}).get("peak_occupancy", 0)
    staff_occ = resolve_stats.get("STAFF", {}).get("peak_occupancy", 0)
    print(json.dumps({
        "tracks_in": len(tracks),
        "crops_dropped_low_quality": len(dropped_crops),
        "unmatched_events": unmatched,
        "events_out": len(out_events),
        "synthetic_entries": len(entry_events),
        "reentry_events": len(reentry_events),
        "visitor_count_source": count_source,
        "gate_entries": len(gate_entries),
        "entry_camera": args.entry_camera,
        "n_customers": len(customers),
        "n_staff": len(staff),
        "peak_customer_occupancy": cust_occ,
        "peak_staff_occupancy": staff_occ,
        "resolve_stats": resolve_stats,
        "customers": customers,
        "staff": staff,
        "out": str(args.out),
    }, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
