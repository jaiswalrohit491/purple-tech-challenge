"""Replay a JSONL of events into the Store Intelligence API.

Usage:
    python -m pipeline.replay events/final.jsonl
    python -m pipeline.replay --url http://localhost:8000 data/sample_events.jsonl

Designed to be idempotent — re-running it is a no-op because the API
deduplicates by event_id. Used both by run.sh and by the test suite.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import httpx


def chunks(seq, n):
    buf = []
    for item in seq:
        buf.append(item)
        if len(buf) >= n:
            yield buf
            buf = []
    if buf:
        yield buf


def replay(path: Path, url: str, batch_size: int = 500, max_retries: int = 5) -> dict:
    totals = {"accepted": 0, "duplicates": 0, "rejected": 0, "batches": 0}
    with path.open() as f:
        events = [json.loads(line) for line in f if line.strip()]

    if not events:
        print(f"replay: no events in {path}", file=sys.stderr)
        return totals

    with httpx.Client(timeout=30.0) as client:
        for batch in chunks(events, batch_size):
            for attempt in range(max_retries):
                try:
                    r = client.post(f"{url}/events/ingest", json=batch)
                    if 400 <= r.status_code < 500:
                        # 4xx is a permanent client error — retrying won't help.
                        # Log and move on to the next batch instead of stalling.
                        print(f"replay: batch rejected {r.status_code}: {r.text[:200]}",
                              file=sys.stderr)
                        totals["rejected"] += len(batch)
                        break
                    if r.status_code >= 500:
                        raise httpx.HTTPError(f"server {r.status_code}")
                    r.raise_for_status()
                    body = r.json()
                    totals["accepted"] += body.get("accepted", 0)
                    totals["duplicates"] += body.get("duplicates", 0)
                    totals["rejected"] += body.get("rejected", 0)
                    totals["batches"] += 1
                    break
                except (httpx.HTTPError, httpx.TransportError) as exc:
                    if attempt == max_retries - 1:
                        print(f"replay: giving up on batch: {exc}", file=sys.stderr)
                        raise
                    backoff = 0.5 * (2 ** attempt)
                    time.sleep(backoff)
    return totals


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("path", type=Path)
    p.add_argument("--url", default="http://localhost:8000")
    p.add_argument("--batch-size", type=int, default=500)
    args = p.parse_args()

    if not args.path.exists():
        print(f"replay: file not found: {args.path}", file=sys.stderr)
        return 1

    totals = replay(args.path, args.url, args.batch_size)
    print(json.dumps(totals))
    return 0


if __name__ == "__main__":
    sys.exit(main())
