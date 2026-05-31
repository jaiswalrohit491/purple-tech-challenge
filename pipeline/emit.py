"""Event builder + JSONL writer.

Every event in the pipeline goes through `make_event` so the schema stays
exactly aligned with `app/models.Event`. The build step validates locally
(via Pydantic) and refuses to emit non-conforming events — bad detections
are logged, not silently dropped.
"""
from __future__ import annotations

import json
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import IO

from pydantic import ValidationError

from app.models import Event


def make_event(
    *,
    store_id: str,
    camera_id: str,
    visitor_id: str,
    event_type: str,
    timestamp: datetime,
    zone_id: str | None = None,
    dwell_ms: int = 0,
    is_staff: bool = False,
    confidence: float,
    queue_depth: int | None = None,
    sku_zone: str | None = None,
    session_seq: int | None = None,
) -> dict | None:
    """Return a serialisable event dict, or None if the inputs don't validate."""
    payload = {
        "event_id": str(uuid.uuid4()),
        "store_id": store_id,
        "camera_id": camera_id,
        "visitor_id": visitor_id,
        "event_type": event_type,
        "timestamp": timestamp.astimezone(timezone.utc).isoformat().replace("+00:00", "Z"),
        "zone_id": zone_id,
        "dwell_ms": int(dwell_ms),
        "is_staff": is_staff,
        "confidence": float(confidence),
        "metadata": {
            "queue_depth": queue_depth,
            "sku_zone": sku_zone,
            "session_seq": session_seq,
        },
    }
    try:
        Event.model_validate(payload)
    except ValidationError as ve:
        print(f"emit: dropping invalid event ({ve.errors()[0]['msg']}): {payload}",
              file=sys.stderr)
        return None
    return payload


class JsonlWriter:
    """Append-only JSONL writer with a flush-per-event option for live tail.

    Used as a context manager:
        with JsonlWriter(path, live=True) as w:
            w.write(event)
    """

    def __init__(self, path: str | Path, live: bool = False):
        self.path = Path(path)
        self.live = live
        self._fp: IO | None = None

    def __enter__(self) -> "JsonlWriter":
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fp = self.path.open("a")
        return self

    def __exit__(self, *_a) -> None:
        if self._fp:
            self._fp.flush()
            self._fp.close()

    def write(self, event: dict) -> None:
        assert self._fp is not None, "JsonlWriter used outside `with` block"
        self._fp.write(json.dumps(event, separators=(",", ":")) + "\n")
        if self.live:
            self._fp.flush()
