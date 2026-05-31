"""POS↔visitor correlation + BILLING_QUEUE_ABANDON emission.

This module is the seam between the detection events (already on disk as
JSONL) and the POS CSV. Two outputs:

1. **POS transactions, pushed straight to the DB**, so the API's metrics and
   funnel queries can join on `pos_transactions`. We bypass the API for this
   because (a) it's static data and (b) the API doesn't expose a POS-ingest
   endpoint (intentional — it's not customer-facing).

2. **BILLING_QUEUE_ABANDON events appended to the JSONL stream.** A visitor
   who emitted `BILLING_QUEUE_JOIN` without a corresponding POS row within
   the 5-minute window gets an abandon event timestamped at their first
   zone-exit from billing (best heuristic available offline).
"""
from __future__ import annotations

import argparse
import asyncio
import csv
import json
import sys
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path

from sqlalchemy import text

from app.config import settings
from app.db import SessionLocal, init_db


def parse_ts(s: str) -> datetime:
    return datetime.fromisoformat(s.replace("Z", "+00:00"))


def _parse_pos_row(r: dict) -> dict | None:
    """Map a row from the Brigade POS export to our schema.

    The brief showed a simplified schema (`store_id, transaction_id, timestamp,
    basket_value_inr`); the actual export has 38 columns including
    `order_id`/`invoice_number`/`order_date`/`order_time`/`total_amount`. This
    function accepts either shape so the pipeline works with both the example
    sample_events POS and the real Brigade export.
    """
    store_id = (r.get("store_id") or "").strip()
    if not store_id:
        return None

    # Composite ID: order_id + sku to ensure uniqueness across line items.
    txn_id = (r.get("transaction_id") or r.get("txn_id")
              or r.get("invoice_number") or r.get("order_id") or "").strip()
    sku = (r.get("sku") or "").strip()
    if sku and txn_id:
        txn_id = f"{txn_id}_{sku}"
    if not txn_id:
        return None

    ts_str = (r.get("timestamp") or r.get("ts") or "").strip()
    if not ts_str:
        d = (r.get("order_date") or "").strip()
        t = (r.get("order_time") or "").strip()
        if not (d and t):
            return None
        # Brigade default is DD-MM-YYYY but some exports use YYYY-MM-DD.
        # Detect by first segment length.
        parts = d.split("-")
        if len(parts) != 3:
            return None
        if len(parts[0]) == 4:        # YYYY-MM-DD
            year, month, day = parts
        else:                         # DD-MM-YYYY
            day, month, year = parts
        # Normalise time to HH:MM:SS so fromisoformat works on every Python
        # version (3.10 chokes on HH:MM-only strings).
        time_parts = t.split(":")
        if len(time_parts) == 2:
            t = f"{t}:00"
        elif len(time_parts) == 1 and t.isdigit():
            t = f"{t}:00:00"
        try:
            # Local time is Asia/Kolkata; convert to UTC by subtracting 5h30m.
            local = datetime.fromisoformat(f"{year}-{month}-{day}T{t}")
        except (ValueError, IndexError):
            return None
        ts = (local - timedelta(hours=5, minutes=30)).replace(tzinfo=timezone.utc)
    else:
        try:
            ts = parse_ts(ts_str)
        except ValueError:
            return None

    amount_str = (r.get("basket_value_inr") or r.get("amount_inr")
                  or r.get("total_amount") or r.get("NMV") or "0").strip()
    try:
        amount = float(amount_str)
    except (TypeError, ValueError):
        amount = 0.0

    brand = (r.get("brand_name") or "").strip() or None
    return {"store_id": store_id, "txn_id": txn_id, "ts": ts,
            "amount_inr": amount, "brand_name": brand}


async def load_pos(csv_path: Path, store_id_filter: str | None = None,
                   ts_shift_seconds: int = 0) -> int:
    """Bulk-load POS CSV into the DB. Idempotent via PK.

    Args:
        csv_path: path to the POS CSV.
        store_id_filter: if set, only load rows for this store_id.
        ts_shift_seconds: shift every timestamp by this delta (used when the
            event stream has been shifted via `pipeline/shift_to_now` so POS
            correlation lines up).
    """
    await init_db()
    rows = []
    with csv_path.open() as f:
        reader = csv.DictReader(f, skipinitialspace=True)
        for r in reader:
            parsed = _parse_pos_row(r)
            if not parsed:
                continue
            if store_id_filter and parsed["store_id"] != store_id_filter:
                continue
            if ts_shift_seconds:
                parsed["ts"] = parsed["ts"] + timedelta(seconds=ts_shift_seconds)
            rows.append(parsed)
    if not rows:
        return 0
    async with SessionLocal() as s:
        await s.execute(
            text("""
                INSERT INTO pos_transactions (store_id, txn_id, ts, amount_inr, brand_name)
                VALUES (:store_id, :txn_id, :ts, :amount_inr, :brand_name)
                ON CONFLICT (txn_id) DO NOTHING
            """),
            rows,
        )
        await s.commit()
    return len(rows)


def emit_abandons(events: list[dict], pos_rows: list[dict]) -> list[dict]:
    """Append BILLING_QUEUE_ABANDON events to the stream where applicable.

    A visitor abandons if:
      - they emitted BILLING_QUEUE_JOIN at ts T
      - no POS txn in their store within [T, T + 5 minutes)
      - they emitted a ZONE_EXIT from a billing zone after T (the moment we
        use for the abandon timestamp)

    The strict 5-minute window matches the brief's definition of a converted
    visitor; using the wider `pos_correlation_window_seconds` (30 min) would
    cause the abandon detector to attribute *any* same-store POS to the
    visitor, under-emitting abandons in busy stores.
    """
    window = timedelta(seconds=settings.pos_strict_window_seconds)
    by_store_pos = {}
    for r in pos_rows:
        by_store_pos.setdefault(r["store_id"], []).append(r["ts"])
    for v in by_store_pos.values():
        v.sort()

    appended = list(events)
    joins = [e for e in events if e["event_type"] == "BILLING_QUEUE_JOIN"]
    for join in joins:
        store_id = join["store_id"]
        visitor_id = join["visitor_id"]
        join_ts = parse_ts(join["timestamp"])
        if _has_pos_in_window(by_store_pos.get(store_id, []), join_ts, window):
            continue

        # Find the next BILLING zone exit for this visitor after the join.
        exit_ts = _next_billing_zone_exit(events, visitor_id, join_ts)
        if exit_ts is None:
            continue

        appended.append({
            "event_id": str(uuid.uuid4()),
            "store_id": store_id,
            "camera_id": join["camera_id"],
            "visitor_id": visitor_id,
            "event_type": "BILLING_QUEUE_ABANDON",
            "timestamp": exit_ts.isoformat().replace("+00:00", "Z"),
            "zone_id": join.get("zone_id"),
            "dwell_ms": int((exit_ts - join_ts).total_seconds() * 1000),
            "is_staff": join.get("is_staff", False),
            "confidence": join.get("confidence", 0.8),
            "metadata": {
                "queue_depth": (join.get("metadata") or {}).get("queue_depth"),
                "reason": "no_pos_within_window",
                "join_event_id": join["event_id"],
            },
        })
    appended.sort(key=lambda e: e["timestamp"])
    return appended


def _has_pos_in_window(sorted_ts: list[datetime], join_ts: datetime, window: timedelta) -> bool:
    if not sorted_ts:
        return False
    end = join_ts + window
    # Linear scan is fine — POS rows per store fit in memory easily.
    for t in sorted_ts:
        if t < join_ts:
            continue
        if t >= end:
            return False
        return True
    return False


def _next_billing_zone_exit(events: list[dict], visitor_id: str, after: datetime) -> datetime | None:
    for ev in events:
        if ev["visitor_id"] != visitor_id:
            continue
        if ev["event_type"] != "ZONE_EXIT":
            continue
        zone = (ev.get("zone_id") or "").upper()
        if "BILL" not in zone:
            continue
        ts = parse_ts(ev["timestamp"])
        if ts > after:
            return ts
    return None


def _load_jsonl(path: Path) -> list[dict]:
    with path.open() as f:
        return [json.loads(line) for line in f if line.strip()]


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--pos", type=Path, required=True)
    p.add_argument("--in", dest="input", type=Path, required=True)
    p.add_argument("--out", dest="output", type=Path, required=True)
    p.add_argument("--store-id", default=None,
                   help="If set, only load POS rows for this store_id.")
    p.add_argument("--ts-shift-seconds", type=int, default=0,
                   help="Shift every POS timestamp by this delta. Use the value "
                        "reported by pipeline.shift_to_now so event-side and "
                        "POS-side shifts agree.")
    args = p.parse_args()

    if not args.pos.exists() or not args.input.exists():
        print("correlate: input file missing", file=sys.stderr)
        return 1

    # Load POS and read it back in one event loop to keep SQLAlchemy's pool happy.
    async def _do_load_and_fetch():
        n = await load_pos(args.pos, args.store_id, args.ts_shift_seconds)
        async with SessionLocal() as s:
            r = await s.execute(text("SELECT store_id, ts FROM pos_transactions"))
            return n, [{"store_id": row.store_id, "ts": row.ts} for row in r]
    pos_count, pos_rows = asyncio.run(_do_load_and_fetch())
    events = _load_jsonl(args.input)

    out_events = emit_abandons(events, pos_rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w") as f:
        for ev in out_events:
            f.write(json.dumps(ev, separators=(",", ":")) + "\n")
    print(json.dumps({
        "pos_loaded": pos_count,
        "events_in": len(events),
        "events_out": len(out_events),
        "abandons_added": len(out_events) - len(events),
    }))
    return 0


if __name__ == "__main__":
    sys.exit(main())
