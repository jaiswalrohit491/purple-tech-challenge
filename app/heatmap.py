from datetime import datetime, timezone

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from .models import HeatmapZone, StoreHeatmap


async def compute_heatmap(session: AsyncSession, store_id: str, now: datetime | None = None) -> StoreHeatmap:
    now = now or datetime.now(timezone.utc)
    window_start = now.replace(hour=0, minute=0, second=0, microsecond=0)
    window_end = now

    rows = (
        await session.execute(
            text(
                """
                SELECT
                    zone_id,
                    COUNT(*) FILTER (WHERE event_type IN ('ZONE_ENTER','BILLING_QUEUE_JOIN')) AS visits,
                    -- Dwell is carried by ZONE_EXIT (cumulative dwell of a
                    -- completed visit) and ZONE_DWELL (30s heartbeats during a
                    -- long visit). Average over both so short clips that never
                    -- produce a heartbeat still report real dwell. ZONE_ENTER
                    -- is dwell=0 at entry and is excluded so it can't skew the mean.
                    AVG(dwell_ms) FILTER (WHERE event_type IN ('ZONE_DWELL', 'ZONE_EXIT')
                                          AND dwell_ms > 0)::float AS avg_dwell
                FROM events
                WHERE store_id = :sid AND ts >= :ws AND ts < :we
                  AND is_staff = FALSE
                  AND zone_id IS NOT NULL
                GROUP BY zone_id
                ORDER BY zone_id
                """
            ),
            {"sid": store_id, "ws": window_start, "we": window_end},
        )
    ).all()

    distinct_visitors_row = (
        await session.execute(
            text(
                """
                SELECT COUNT(DISTINCT visitor_id) AS n
                FROM events
                WHERE store_id = :sid AND ts >= :ws AND ts < :we AND is_staff = FALSE
                """
            ),
            {"sid": store_id, "ws": window_start, "we": window_end},
        )
    ).first()
    distinct_visitors = int(distinct_visitors_row.n or 0) if distinct_visitors_row else 0

    if not rows:
        return StoreHeatmap(
            store_id=store_id,
            window_start=window_start,
            window_end=window_end,
            zones=[],
            data_confidence="LOW" if distinct_visitors < 20 else "OK",
        )

    max_visits = max((int(r.visits or 0) for r in rows), default=0) or 1
    max_dwell = max((float(r.avg_dwell or 0.0) for r in rows), default=0.0) or 1.0

    zones = [
        HeatmapZone(
            zone_id=r.zone_id,
            visit_count=int(r.visits or 0),
            avg_dwell_ms=float(r.avg_dwell or 0.0),
            visit_score=round((int(r.visits or 0) / max_visits) * 100, 2),
            dwell_score=round((float(r.avg_dwell or 0.0) / max_dwell) * 100, 2),
        )
        for r in rows
    ]

    return StoreHeatmap(
        store_id=store_id,
        window_start=window_start,
        window_end=window_end,
        zones=zones,
        data_confidence="LOW" if distinct_visitors < 20 else "OK",
    )
