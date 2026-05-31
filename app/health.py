from datetime import datetime, timezone

from sqlalchemy import text
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.ext.asyncio import AsyncSession

from .config import settings
from .models import HealthResponse, StoreFeedHealth

VERSION = "0.1.0"


async def compute_health(session: AsyncSession) -> HealthResponse:
    db_reachable = True
    feeds: list[StoreFeedHealth] = []
    now = datetime.now(timezone.utc)

    try:
        rows = (
            await session.execute(
                text(
                    """
                    SELECT store_id, MAX(ts) AS last_ts
                    FROM events
                    GROUP BY store_id
                    ORDER BY store_id
                    """
                )
            )
        ).all()
        for r in rows:
            last = r.last_ts
            stale = bool(last) and (now - last).total_seconds() > settings.stale_feed_threshold_seconds
            feeds.append(StoreFeedHealth(store_id=r.store_id, last_event_at=last, stale=stale))
    except SQLAlchemyError:
        db_reachable = False

    return HealthResponse(
        status="OK" if db_reachable else "DEGRADED",
        version=VERSION,
        db_reachable=db_reachable,
        stores=feeds,
    )
