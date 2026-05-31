from datetime import datetime, timezone

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from .config import settings
from .models import FunnelStage, StoreFunnel


async def compute_funnel(session: AsyncSession, store_id: str, now: datetime | None = None) -> StoreFunnel:
    """Session-grouped funnel. Re-entries collapse because pipeline reuses visitor_id."""
    now = now or datetime.now(timezone.utc)
    window_start = now.replace(hour=0, minute=0, second=0, microsecond=0)
    window_end = now
    params = {"sid": store_id, "ws": window_start, "we": window_end,
              "win": str(settings.pos_correlation_window_seconds)}

    rows = (
        await session.execute(
            text(
                """
                WITH sessions AS (
                    SELECT
                        visitor_id,
                        BOOL_OR(event_type IN ('ENTRY','REENTRY'))            AS has_entry,
                        BOOL_OR(event_type IN ('ZONE_ENTER','BILLING_QUEUE_JOIN')) AS has_zone,
                        BOOL_OR(event_type = 'BILLING_QUEUE_JOIN')            AS has_billing,
                        MIN(CASE WHEN event_type = 'BILLING_QUEUE_JOIN'
                                 THEN ts END)                                 AS billing_ts
                    FROM events
                    WHERE store_id = :sid AND ts >= :ws AND ts < :we
                      AND is_staff = FALSE
                    GROUP BY visitor_id
                )
                SELECT
                    COUNT(*) FILTER (WHERE has_entry)                                       AS entries,
                    COUNT(*) FILTER (WHERE has_entry AND has_zone)                          AS zone_visits,
                    COUNT(*) FILTER (WHERE has_entry AND has_zone AND has_billing)          AS billings,
                    -- A session counts as PURCHASE only if the brief's strict
                    -- definition holds: a POS row in the same store within 5
                    -- minutes of BILLING_QUEUE_JOIN. The wider POS window in
                    -- `pos_correlation_window_seconds` is the "potential"
                    -- attribution window in /metrics; the funnel uses the
                    -- conservative 5 minutes to avoid double-counting POS
                    -- across visitors when traffic is dense.
                    COUNT(*) FILTER (
                        WHERE has_entry AND has_zone AND has_billing AND EXISTS (
                            SELECT 1 FROM pos_transactions p
                            WHERE p.store_id = :sid
                              AND p.ts >= sessions.billing_ts
                              AND p.ts <  sessions.billing_ts + INTERVAL '5 minutes'
                        )
                    )                                                                       AS purchases
                FROM sessions
                """
            ),
            params,
        )
    ).first()

    entries = int(rows.entries or 0) if rows else 0
    zone_visits = int(rows.zone_visits or 0) if rows else 0
    billings = int(rows.billings or 0) if rows else 0
    purchases = int(rows.purchases or 0) if rows else 0

    stages_raw = [
        ("ENTRY", entries),
        ("ZONE_VISIT", zone_visits),
        ("BILLING_QUEUE", billings),
        ("PURCHASE", purchases),
    ]
    stages: list[FunnelStage] = []
    for i, (name, count) in enumerate(stages_raw):
        if i == 0:
            dropoff = 0.0
        else:
            prev = stages_raw[i - 1][1]
            dropoff = round(1.0 - (count / prev), 4) if prev else 0.0
        stages.append(FunnelStage(stage=name, visitors=count, dropoff_pct=dropoff))

    return StoreFunnel(
        store_id=store_id, window_start=window_start, window_end=window_end, stages=stages
    )
