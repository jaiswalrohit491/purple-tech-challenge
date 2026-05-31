from datetime import datetime, timedelta, timezone

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from .config import settings
from .models import Anomaly, StoreAnomalies


async def detect_anomalies(session: AsyncSession, store_id: str, now: datetime | None = None) -> StoreAnomalies:
    now = now or datetime.now(timezone.utc)
    anomalies: list[Anomaly] = []

    # 1. BILLING_QUEUE_SPIKE — max queue_depth in last 30 min
    spike_row = (
        await session.execute(
            text(
                """
                SELECT
                    COALESCE(MAX((metadata->>'queue_depth')::int), 0) AS max_depth,
                    MAX(ts) AS last_ts
                FROM events
                WHERE store_id = :sid AND event_type = 'BILLING_QUEUE_JOIN'
                  AND ts >= :recent
                """
            ),
            {"sid": store_id, "recent": now - timedelta(minutes=30)},
        )
    ).first()
    if spike_row and spike_row.max_depth:
        depth = int(spike_row.max_depth)
        if depth >= settings.queue_critical_depth:
            anomalies.append(
                Anomaly(
                    anomaly_type="BILLING_QUEUE_SPIKE",
                    severity="CRITICAL",
                    detected_at=spike_row.last_ts or now,
                    detail={"queue_depth": depth, "threshold": settings.queue_critical_depth},
                    suggested_action="Open additional billing counter immediately",
                )
            )
        elif depth >= settings.queue_warn_depth:
            anomalies.append(
                Anomaly(
                    anomaly_type="BILLING_QUEUE_SPIKE",
                    severity="WARN",
                    detected_at=spike_row.last_ts or now,
                    detail={"queue_depth": depth, "threshold": settings.queue_warn_depth},
                    suggested_action="Monitor billing queue; prepare to open another counter",
                )
            )

    # 2. CONVERSION_DROP — today's rolling 1h vs baseline of the SAME hour
    # of day across the last 7 days. Filtering by EXTRACT(HOUR ...) avoids
    # comparing dead-of-night traffic against peak-hour averages.
    conv_row = (
        await session.execute(
            text(
                """
                WITH today AS (
                    SELECT
                        COUNT(DISTINCT visitor_id) FILTER (
                            WHERE event_type IN ('ENTRY','REENTRY') AND is_staff = FALSE
                        ) AS visitors,
                        COUNT(DISTINCT visitor_id) FILTER (
                            WHERE event_type = 'BILLING_QUEUE_JOIN' AND is_staff = FALSE
                        ) AS billed
                    FROM events
                    WHERE store_id = :sid AND ts >= :hour_ago AND ts < :now_ts
                ),
                baseline AS (
                    SELECT
                        COUNT(DISTINCT visitor_id) FILTER (
                            WHERE event_type IN ('ENTRY','REENTRY') AND is_staff = FALSE
                        ) AS visitors,
                        COUNT(DISTINCT visitor_id) FILTER (
                            WHERE event_type = 'BILLING_QUEUE_JOIN' AND is_staff = FALSE
                        ) AS billed
                    FROM events
                    WHERE store_id = :sid AND ts >= :base_start AND ts < :base_end
                      AND EXTRACT(HOUR FROM ts AT TIME ZONE 'UTC') = :hour_of_day
                )
                SELECT today.visitors AS t_visitors, today.billed AS t_billed,
                       baseline.visitors AS b_visitors, baseline.billed AS b_billed
                FROM today, baseline
                """
            ),
            {
                "sid": store_id,
                "hour_ago": now - timedelta(hours=1),
                "now_ts": now,
                "base_start": now - timedelta(days=7),
                "base_end": now - timedelta(hours=1),
                "hour_of_day": now.hour,
            },
        )
    ).first()
    if conv_row and conv_row.t_visitors and conv_row.b_visitors:
        today_rate = (conv_row.t_billed or 0) / conv_row.t_visitors
        base_rate = (conv_row.b_billed or 0) / conv_row.b_visitors
        if base_rate > 0 and today_rate < settings.conversion_drop_ratio * base_rate:
            anomalies.append(
                Anomaly(
                    anomaly_type="CONVERSION_DROP",
                    severity="WARN",
                    detected_at=now,
                    detail={
                        "today_rate": round(today_rate, 4),
                        "baseline_rate": round(base_rate, 4),
                    },
                    suggested_action="Investigate floor staffing, displays, or queue dynamics",
                )
            )

    # 3. DEAD_ZONE — zones with prior activity but none in last 30 min.
    # Use ALL historical events for the baseline (not just last 4h) so the
    # alert doesn't vanish once the original burst ages out of the window.
    # The `total >= 3` floor still guards against zones that genuinely
    # have never been busy.
    dead_rows = (
        await session.execute(
            text(
                """
                WITH zone_baseline AS (
                    SELECT zone_id, MAX(ts) AS last_visit, COUNT(*) AS total
                    FROM events
                    WHERE store_id = :sid AND zone_id IS NOT NULL
                      AND event_type IN ('ZONE_ENTER', 'ZONE_DWELL')
                    GROUP BY zone_id
                )
                SELECT zone_id, last_visit, total
                FROM zone_baseline
                WHERE last_visit < :stale_cutoff AND total >= 3
                """
            ),
            {
                "sid": store_id,
                "stale_cutoff": now - timedelta(seconds=settings.dead_zone_window_seconds),
            },
        )
    ).all()
    for r in dead_rows:
        anomalies.append(
            Anomaly(
                anomaly_type="DEAD_ZONE",
                severity="INFO",
                detected_at=now,
                detail={"zone_id": r.zone_id, "last_visit": r.last_visit.isoformat()},
                suggested_action="Check camera health and zone merchandising",
            )
        )

    return StoreAnomalies(store_id=store_id, anomalies=anomalies)
