from datetime import datetime
from typing import Any

from pydantic import ValidationError
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from .config import settings
from .logging_setup import log
from .models import Event, IngestError, IngestResponse


_INSERT_SQL = text(
    """
    INSERT INTO events (
        event_id, store_id, camera_id, visitor_id, event_type, ts,
        zone_id, dwell_ms, is_staff, confidence, metadata
    )
    VALUES (
        :event_id, :store_id, :camera_id, :visitor_id, :event_type, :ts,
        :zone_id, :dwell_ms, :is_staff, :confidence, CAST(:metadata AS JSONB)
    )
    ON CONFLICT (event_id) DO NOTHING
    RETURNING event_id
    """
)


async def ingest_events(session: AsyncSession, raw_events: list[dict[str, Any]]) -> IngestResponse:
    """Ingest a batch of raw event dicts.

    Behavior contract (matches challenge spec):
      - Batches up to settings.ingest_max_batch enforced by caller (route).
      - Each event validated independently; invalid items collected, not 5xx.
      - Deduplication by event_id PK (ON CONFLICT DO NOTHING).
      - Returns counts + per-item error list with original batch index.
    """
    accepted = 0
    duplicates = 0
    rejected = 0
    errors: list[IngestError] = []

    import json
    for idx, raw in enumerate(raw_events):
        try:
            ev = Event.model_validate(raw)
        except ValidationError as ve:
            rejected += 1
            errors.append(
                IngestError(
                    index=idx,
                    event_id=str(raw.get("event_id")) if isinstance(raw, dict) else None,
                    error=_first_error(ve),
                )
            )
            continue

        result = await session.execute(
            _INSERT_SQL,
            {
                "event_id": str(ev.event_id),
                "store_id": ev.store_id,
                "camera_id": ev.camera_id,
                "visitor_id": ev.visitor_id,
                "event_type": ev.event_type,
                "ts": ev.timestamp,
                "zone_id": ev.zone_id,
                "dwell_ms": ev.dwell_ms,
                "is_staff": ev.is_staff,
                "confidence": ev.confidence,
                "metadata": json.dumps(ev.metadata.model_dump(exclude_none=False)),
            },
        )
        row = result.first()
        if row is None:
            duplicates += 1
        else:
            accepted += 1

    await session.commit()
    log.info(
        "ingest.batch",
        event_count=len(raw_events),
        accepted=accepted,
        duplicates=duplicates,
        rejected=rejected,
    )
    return IngestResponse(
        accepted=accepted, duplicates=duplicates, rejected=rejected, errors=errors
    )


def _first_error(ve: ValidationError) -> str:
    errs = ve.errors()
    if not errs:
        return "validation failed"
    err = errs[0]
    loc = ".".join(str(p) for p in err.get("loc", ()))
    return f"{loc}: {err.get('msg', 'invalid')}"


def utc_day_bounds(now: datetime) -> tuple[datetime, datetime]:
    start = now.replace(hour=0, minute=0, second=0, microsecond=0)
    end = start.replace(hour=23, minute=59, second=59, microsecond=999_000)
    return start, end


# Re-export for convenience
__all__ = ["ingest_events", "utc_day_bounds"]
