"""Shift timestamps in a JSONL of events so the latest event lands ~1 minute ago.

The CCTV clips have OSD timestamps from when they were captured (e.g.
2026-10-04T20:11), which usually doesn't match today's UTC calendar day. The
API's read endpoints filter events to today's window, so events outside that
window are silently excluded from metrics and the dashboard.

This utility preserves relative timing between events but slides the whole
batch to "now minus 1 minute" so:
  - all events fall in today's UTC day window
  - the most recent event triggers a fresh /health response
  - the dashboard sees activity

Usage:
    python -m pipeline.shift_to_now events/final.jsonl > events/final.shifted.jsonl
    python -m pipeline.shift_to_now events/*.jsonl --out events/final.shifted.jsonl

Idempotency note: the script regenerates event_ids when --regenerate-ids is
passed. Otherwise event_ids are preserved (so replaying the shifted file twice
is still a no-op).
"""
from __future__ import annotations

import argparse
import json
import sys
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path


def parse_ts(s: str) -> datetime:
    return datetime.fromisoformat(s.replace("Z", "+00:00"))


def iso(ts: datetime) -> str:
    return ts.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def shift(paths: list[Path], out_path: Path | None, regenerate_ids: bool,
          anchor_offset_seconds: int) -> dict:
    events = []
    for p in paths:
        with p.open() as f:
            for line in f:
                if line.strip():
                    events.append(json.loads(line))

    if not events:
        return {"events": 0}

    max_ts = max(parse_ts(e["timestamp"]) for e in events)
    target_max = datetime.now(timezone.utc) - timedelta(seconds=anchor_offset_seconds)
    delta = target_max - max_ts

    for e in events:
        e["timestamp"] = iso(parse_ts(e["timestamp"]) + delta)
        if regenerate_ids:
            e["event_id"] = str(uuid.uuid4())

    events.sort(key=lambda e: e["timestamp"])

    if out_path:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w") as f:
            for e in events:
                f.write(json.dumps(e, separators=(",", ":")) + "\n")
        return {"events": len(events), "delta_seconds": int(delta.total_seconds()),
                "out": str(out_path)}
    for e in events:
        print(json.dumps(e, separators=(",", ":")))
    return {"events": len(events), "delta_seconds": int(delta.total_seconds())}


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("inputs", nargs="+", type=Path)
    p.add_argument("--out", type=Path, default=None,
                   help="If provided, write shifted events here; else stream to stdout.")
    p.add_argument("--regenerate-ids", action="store_true",
                   help="Mint fresh event_ids (useful if re-replaying into a non-truncated DB).")
    p.add_argument("--anchor-offset-seconds", type=int, default=60,
                   help="Place the latest event this many seconds in the past.")
    args = p.parse_args()

    missing = [pp for pp in args.inputs if not pp.exists()]
    if missing:
        print(f"shift_to_now: missing files: {missing}", file=sys.stderr)
        return 1

    result = shift(args.inputs, args.out, args.regenerate_ids, args.anchor_offset_seconds)
    print(json.dumps(result), file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
