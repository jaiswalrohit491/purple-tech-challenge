from datetime import datetime, timedelta, timezone

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from .config import settings
from .models import StoreMetrics, ZoneDwell


async def compute_metrics(session: AsyncSession, store_id: str, now: datetime | None = None) -> StoreMetrics:
    """Today's metrics for a store, non-staff only, never returns null fields."""
    now = now or datetime.now(timezone.utc)
    window_start = now.replace(hour=0, minute=0, second=0, microsecond=0)
    window_end = now

    params = {"sid": store_id, "ws": window_start, "we": window_end}

    # Unique non-staff visitors with at least one ENTRY (or REENTRY counted as same visitor)
    unique_row = (
        await session.execute(
            text(
                """
                SELECT COUNT(DISTINCT visitor_id) AS n
                FROM events
                WHERE store_id = :sid AND ts >= :ws AND ts < :we
                  AND is_staff = FALSE
                  AND event_type IN ('ENTRY', 'REENTRY')
                """
            ),
            params,
        )
    ).first()
    unique_visitors = int(unique_row.n or 0) if unique_row else 0

    # Staff count: distinct visitor_ids with is_staff=true in the same window.
    # Not part of customer metrics but useful for /metrics consumers that want
    # the full population breakdown (e.g. "2 customers, 5 staff").
    staff_row = (
        await session.execute(
            text(
                """
                SELECT COUNT(DISTINCT visitor_id) AS n
                FROM events
                WHERE store_id = :sid AND ts >= :ws AND ts < :we
                  AND is_staff = TRUE
                """
            ),
            params,
        )
    ).first()
    staff_count = int(staff_row.n or 0) if staff_row else 0

    # CONFIRMED conversion (brief's strict definition): a visitor was at the
    # billing till AND a POS row fires in the same store within 5 minutes.
    # This is direct evidence — the customer was queueing, then a sale happened.
    # No brand match required; the 5-min window is what the brief calls out.
    converted_row = (
        await session.execute(
            text(
                """
                WITH bj AS (
                    SELECT DISTINCT visitor_id, MIN(ts) AS ts
                    FROM events
                    WHERE store_id = :sid AND ts >= :ws AND ts < :we
                      AND is_staff = FALSE
                      AND event_type = 'BILLING_QUEUE_JOIN'
                    GROUP BY visitor_id
                )
                SELECT COUNT(DISTINCT bj.visitor_id) AS n
                FROM bj
                JOIN pos_transactions p
                  ON p.store_id = :sid
                 AND p.ts >= bj.ts
                 AND p.ts <  bj.ts + INTERVAL '5 minutes'
                """
            ),
            params,
        )
    ).first()
    converted_visitors = int(converted_row.n or 0) if converted_row else 0
    conversion_rate = (converted_visitors / unique_visitors) if unique_visitors else 0.0

    # POTENTIAL conversion (inferred signal, NOT direct evidence): a customer
    # browsed a brand zone, and that brand sold within `pos_correlation_window`
    # in the same store. Useful when the clip ends before the customer reaches
    # the till. Always >= conversion_rate. Labeled "potential" deliberately —
    # this is correlation, not causation. Other customers (not in footage) may
    # be the actual buyers.
    potential_row = (
        await session.execute(
            text(
                """
                WITH visits AS (
                    SELECT DISTINCT visitor_id, ts, zone_id
                    FROM events
                    WHERE store_id = :sid AND ts >= :ws AND ts < :we
                      AND is_staff = FALSE
                      AND event_type IN ('ZONE_ENTER', 'ZONE_DWELL', 'BILLING_QUEUE_JOIN')
                )
                SELECT COUNT(DISTINCT visits.visitor_id) AS n
                FROM visits
                JOIN pos_transactions p
                  ON p.store_id = :sid
                 AND p.ts >= visits.ts
                 AND p.ts <  visits.ts + (:win || ' seconds')::interval
                 AND (
                       -- Customer at the till: any POS in window is plausible.
                       visits.zone_id IN ('BILLING_QUEUE', 'CASH_COUNTER')
                       -- Brand-zone match (normalised lower-alphanumeric).
                       OR (p.brand_name IS NOT NULL AND visits.zone_id IS NOT NULL
                           AND REGEXP_REPLACE(LOWER(p.brand_name), '[^a-z0-9]', '', 'g')
                             = REGEXP_REPLACE(LOWER(visits.zone_id), '[^a-z0-9]', '', 'g'))
                     )
                """
            ),
            {**params, "win": str(settings.pos_correlation_window_seconds)},
        )
    ).first()
    potential_converted = int(potential_row.n or 0) if potential_row else 0
    potential_conversion_rate = (
        potential_converted / unique_visitors if unique_visitors else 0.0
    )

    # VERIFIED purchase (operator's preferred stronger signal): the visitor
    # satisfies a per-visit trajectory: ENTRY (or REENTRY) before the BILLING
    # → EXIT after the BILLING with the gap ≤5 min → POS between bill and
    # exit. We pair each BILLING with the NEXT exit (not the global MAX
    # exit), so visitors with multiple bill-exit cycles in one day are still
    # detected. For clips that don't capture entries/exits — like the 2-min
    # ST1008 clip — this is correctly 0; you cannot verify a purchase
    # without seeing the visitor leave.
    verified_row = (
        await session.execute(
            text(
                """
                WITH bill_exit AS (
                    SELECT
                        b.visitor_id,
                        b.ts AS bill_ts,
                        (SELECT MIN(e.ts) FROM events e
                          WHERE e.visitor_id = b.visitor_id
                            AND e.store_id = b.store_id
                            AND e.event_type = 'EXIT'
                            AND e.ts > b.ts) AS exit_ts
                    FROM events b
                    WHERE b.store_id = :sid AND b.ts >= :ws AND b.ts < :we
                      AND b.is_staff = FALSE
                      AND b.event_type = 'BILLING_QUEUE_JOIN'
                ),
                trajectories AS (
                    SELECT be.visitor_id, be.bill_ts, be.exit_ts
                    FROM bill_exit be
                    WHERE be.exit_ts IS NOT NULL
                      AND be.exit_ts < be.bill_ts + INTERVAL '5 minutes'
                      AND EXISTS (
                          SELECT 1 FROM events e
                          WHERE e.visitor_id = be.visitor_id
                            AND e.store_id = :sid
                            AND e.event_type IN ('ENTRY','REENTRY')
                            AND e.ts <= be.bill_ts
                      )
                )
                SELECT COUNT(DISTINCT t.visitor_id) AS n
                FROM trajectories t
                WHERE EXISTS (
                    SELECT 1 FROM pos_transactions p
                    WHERE p.store_id = :sid
                      AND p.ts >= t.bill_ts
                      AND p.ts <= t.exit_ts + INTERVAL '2 minutes'
                )
                """
            ),
            params,
        )
    ).first()
    verified_purchases = int(verified_row.n or 0) if verified_row else 0
    verified_purchase_rate = (
        verified_purchases / unique_visitors if unique_visitors else 0.0
    )

    # Avg dwell per zone. Dwell time is carried by two event types:
    #   - ZONE_EXIT: the cumulative dwell at the moment the visit ends
    #     (entry→exit). This is the completed-visit dwell and the most
    #     accurate single measurement per visit.
    #   - ZONE_DWELL: a 30s heartbeat emitted *during* a long visit. It only
    #     appears for visits that last ≥30s without a track break, so on short
    #     or fragmented clips there may be none at all.
    # We average over BOTH (dwell_ms > 0) so the metric still reports real
    # numbers when a clip is too short to produce any ZONE_DWELL heartbeats —
    # the common case for the provided 2-minute footage. ZONE_ENTER has
    # dwell_ms=0 by construction and is excluded so it can't halve the mean.
    # `visits` counts ZONE_ENTER only (one per visit).
    zone_rows = (
        await session.execute(
            text(
                """
                SELECT
                    zone_id,
                    AVG(dwell_ms) FILTER (WHERE event_type IN ('ZONE_DWELL', 'ZONE_EXIT')
                                          AND dwell_ms > 0)::float AS avg_dwell,
                    COUNT(*) FILTER (WHERE event_type = 'ZONE_ENTER') AS visits
                FROM events
                WHERE store_id = :sid AND ts >= :ws AND ts < :we
                  AND is_staff = FALSE
                  AND event_type IN ('ZONE_ENTER', 'ZONE_DWELL', 'ZONE_EXIT')
                  AND zone_id IS NOT NULL
                GROUP BY zone_id
                HAVING COUNT(*) FILTER (WHERE event_type = 'ZONE_ENTER') > 0
                ORDER BY zone_id
                """
            ),
            params,
        )
    ).all()
    avg_dwell_by_zone = [
        ZoneDwell(zone_id=r.zone_id, avg_dwell_ms=float(r.avg_dwell or 0.0), visits=int(r.visits))
        for r in zone_rows
    ]

    # Current queue depth = max queue_depth from latest BILLING_QUEUE_JOIN in last 5 min
    queue_row = (
        await session.execute(
            text(
                """
                SELECT COALESCE(MAX((metadata->>'queue_depth')::int), 0) AS depth
                FROM events
                WHERE store_id = :sid AND ts >= :recent AND ts < :we
                  AND event_type = 'BILLING_QUEUE_JOIN'
                """
            ),
            {**params, "recent": window_end - timedelta(minutes=5)},
        )
    ).first()
    current_queue_depth = int(queue_row.depth or 0) if queue_row else 0

    # Abandonment rate: BILLING_QUEUE_ABANDON / BILLING_QUEUE_JOIN
    abandon_row = (
        await session.execute(
            text(
                """
                SELECT
                  SUM(CASE WHEN event_type='BILLING_QUEUE_JOIN'    THEN 1 ELSE 0 END) AS joins,
                  SUM(CASE WHEN event_type='BILLING_QUEUE_ABANDON' THEN 1 ELSE 0 END) AS abandons
                FROM events
                WHERE store_id = :sid AND ts >= :ws AND ts < :we AND is_staff = FALSE
                """
            ),
            params,
        )
    ).first()
    joins = int((abandon_row.joins if abandon_row else 0) or 0)
    abandons = int((abandon_row.abandons if abandon_row else 0) or 0)
    abandonment_rate = (abandons / joins) if joins else 0.0

    return StoreMetrics(
        store_id=store_id,
        window_start=window_start,
        window_end=window_end,
        unique_visitors=unique_visitors,
        staff_count=staff_count,
        converted_visitors=converted_visitors,
        conversion_rate=round(conversion_rate, 4),
        verified_purchases=verified_purchases,
        verified_purchase_rate=round(verified_purchase_rate, 4),
        potential_converted=potential_converted,
        potential_conversion_rate=round(potential_conversion_rate, 4),
        avg_dwell_by_zone=avg_dwell_by_zone,
        current_queue_depth=current_queue_depth,
        abandonment_rate=round(abandonment_rate, 4),
    )
