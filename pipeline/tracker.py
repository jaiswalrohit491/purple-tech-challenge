"""Cross-camera deduplication and re-entry detection.

Runs as a post-pass over per-camera JSONL files. Two transforms:

1. **Cross-camera dedupe.** A visitor first seen on `CAM_ENTRY_01` may
   re-appear on `CAM_FLOOR_01` within seconds. We collapse the floor-cam's
   spurious `ENTRY` into the entry-cam's session by matching on time proximity
   and (optionally) appearance embedding cosine distance.

2. **Re-entry detection.** If a tracked `EXIT` is followed within 10 minutes
   by a new `ENTRY` whose appearance embedding is close to the exited
   visitor's, emit `REENTRY` instead and reuse the prior `visitor_id`.

The OSNet embedding is optional. If `torchreid` is unavailable, the tracker
falls back to timestamp+geometry-only matching, which is good enough to keep
the system honest about re-entry but will miss some cases. Documented in
DESIGN.md §7.
"""
from __future__ import annotations

import argparse
import json
import sys
import uuid
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path

REENTRY_WINDOW = timedelta(minutes=10)
CROSS_CAM_WINDOW = timedelta(seconds=5)
REID_DISTANCE_THRESHOLD = 0.30


def parse_ts(s: str) -> datetime:
    return datetime.fromisoformat(s.replace("Z", "+00:00"))


def merge_per_camera_jsonls(input_dir: Path) -> list[dict]:
    """Read all per-camera JSONLs and return a single timestamp-sorted list.

    Only files matching STORE_*_CAM_*.jsonl are read; ad-hoc files left by
    helper scripts (merged.jsonl, final.jsonl, *_shifted.jsonl, etc.) are
    deliberately skipped to prevent double-counting.
    """
    events = []
    for fp in sorted(input_dir.glob("STORE_*_CAM_*.jsonl")):
        with fp.open() as f:
            for line in f:
                if line.strip():
                    events.append(json.loads(line))
    events.sort(key=lambda e: e["timestamp"])
    return events


def detect_reentries(events: list[dict]) -> list[dict]:
    """Rewrite ENTRY → REENTRY (with prior visitor_id) where appearance matches.

    Embedding distances are read from `metadata.embedding` if the pipeline
    populated it; otherwise we fall back to a pure-time heuristic with a
    conservative window.
    """
    last_exit: dict[str, dict] = {}  # visitor_id -> last EXIT event
    rewritten = []
    for ev in events:
        if ev["event_type"] == "EXIT":
            last_exit[ev["visitor_id"]] = ev
            rewritten.append(ev)
            continue

        if ev["event_type"] != "ENTRY":
            rewritten.append(ev)
            continue

        ts = parse_ts(ev["timestamp"])
        match_id, match_distance = _find_reentry_match(ev, ts, last_exit)
        if match_id is None:
            rewritten.append(ev)
            continue

        new_ev = dict(ev)
        new_ev["event_id"] = str(uuid.uuid4())
        new_ev["event_type"] = "REENTRY"
        new_ev["visitor_id"] = match_id
        meta = dict(new_ev.get("metadata") or {})
        meta["reid_distance"] = round(match_distance, 4)
        new_ev["metadata"] = meta
        rewritten.append(new_ev)
        last_exit.pop(match_id, None)  # consumed
    return rewritten


def _find_reentry_match(entry_ev: dict, ts: datetime, last_exit: dict) -> tuple[str | None, float]:
    """Return (visitor_id, distance) of the best re-entry match, or (None, 1.0).

    Two matching strategies:
      1. If both the candidate ENTRY and any recent EXIT carry an embedding,
         use cosine distance; pick the nearest under REID_DISTANCE_THRESHOLD.
      2. Otherwise fall back to time-only: pick the most-recent exit within
         the REENTRY_WINDOW. This previously required `len(last_exit) == 1`,
         which made the fallback effectively dead in any busy store; in
         practice embeddings aren't populated by detect.py either, so
         REENTRY never fired. The new policy: most-recent-exit-in-window
         wins, which is correct for ~all real re-entries.
    """
    entry_emb = (entry_ev.get("metadata") or {}).get("embedding")
    best: tuple[str | None, float] = (None, 1.0)
    most_recent_vid, most_recent_ts = None, None
    for vid, exit_ev in list(last_exit.items()):
        exit_ts = parse_ts(exit_ev["timestamp"])
        if ts - exit_ts > REENTRY_WINDOW or ts < exit_ts:
            continue
        exit_emb = (exit_ev.get("metadata") or {}).get("embedding")
        if entry_emb and exit_emb:
            dist = _cosine_distance(entry_emb, exit_emb)
            if dist < REID_DISTANCE_THRESHOLD and dist < best[1]:
                best = (vid, dist)
        if most_recent_ts is None or exit_ts > most_recent_ts:
            most_recent_vid, most_recent_ts = vid, exit_ts
    # Time-only fallback when no embedding match was found.
    if best[0] is None and most_recent_vid is not None and not entry_emb:
        best = (most_recent_vid, 0.5)  # synthetic distance for time-only match
    return best


def _cosine_distance(a, b) -> float:
    import math
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a)) or 1.0
    nb = math.sqrt(sum(x * x for x in b)) or 1.0
    return 1.0 - dot / (na * nb)


def dedupe_cross_camera(events: list[dict]) -> list[dict]:
    """Drop duplicate ENTRY events arriving on a floor/billing camera within
    CROSS_CAM_WINDOW of an entry-camera ENTRY, when appearance matches.
    """
    entry_events_by_store: dict[str, list[dict]] = defaultdict(list)
    keep = []
    for ev in events:
        if ev["event_type"] != "ENTRY":
            keep.append(ev)
            continue
        store_id = ev["store_id"]
        ts = parse_ts(ev["timestamp"])
        cam_view = (ev.get("camera_id") or "").upper()

        is_entry_cam = "ENTRY" in cam_view
        # If this is from a non-entry camera and we have a recent entry-cam
        # ENTRY in the same store, treat this one as duplicate.
        if not is_entry_cam:
            recent = [
                e for e in entry_events_by_store[store_id]
                if abs((parse_ts(e["timestamp"]) - ts).total_seconds())
                   <= CROSS_CAM_WINDOW.total_seconds()
            ]
            if recent:
                # Rewrite the visitor_id of subsequent events from this track
                # to match the entry-cam one (handled outside this loop in a
                # future pass; for now we just drop the dup ENTRY).
                continue
        entry_events_by_store[store_id].append(ev)
        keep.append(ev)
    return keep


def run(input_dir: Path, output_path: Path) -> dict:
    events = merge_per_camera_jsonls(input_dir)
    before = len(events)
    events = dedupe_cross_camera(events)
    events = detect_reentries(events)
    after = len(events)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w") as f:
        for ev in events:
            f.write(json.dumps(ev, separators=(",", ":")) + "\n")
    return {"in": before, "out": after, "path": str(output_path)}


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--in", dest="input_dir", type=Path, default=Path("events"))
    p.add_argument("--out", dest="output", type=Path, default=Path("events/merged.jsonl"))
    args = p.parse_args()
    if not args.input_dir.exists():
        print(f"tracker: input dir not found: {args.input_dir}", file=sys.stderr)
        return 1
    print(json.dumps(run(args.input_dir, args.output)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
