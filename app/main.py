from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated, Any

from fastapi import Body, Depends, FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse
from sqlalchemy.exc import OperationalError, SQLAlchemyError
from sqlalchemy.ext.asyncio import AsyncSession

from .anomalies import detect_anomalies
from .config import settings
from .db import init_db, session_dep
from .funnel import compute_funnel
from .health import compute_health
from .heatmap import compute_heatmap
from .ingestion import ingest_events
from .logging_setup import TraceIdMiddleware, configure_logging, current_trace_id, log
from .metrics import compute_metrics
from .models import (
    HealthResponse,
    IngestResponse,
    StoreAnomalies,
    StoreFunnel,
    StoreHeatmap,
    StoreMetrics,
)


@asynccontextmanager
async def lifespan(_app: FastAPI):
    configure_logging()
    try:
        await init_db()
        log.info("startup.db_ready")
    except Exception as exc:
        log.error("startup.db_init_failed", error=str(exc))
    yield


app = FastAPI(
    title="Apex Retail Store Intelligence",
    version="0.1.0",
    lifespan=lifespan,
)
app.add_middleware(TraceIdMiddleware)


SessionDep = Annotated[AsyncSession, Depends(session_dep)]


@app.exception_handler(OperationalError)
async def db_unavailable_handler(_request: Request, exc: OperationalError):
    log.error("db.unavailable", error=str(exc))
    return JSONResponse(
        status_code=503,
        content={
            "error": "DB_UNAVAILABLE",
            "message": "Storage backend is unreachable. Retry shortly.",
            "trace_id": current_trace_id(),
        },
    )


@app.exception_handler(SQLAlchemyError)
async def db_generic_handler(_request: Request, exc: SQLAlchemyError):
    log.error("db.error", error=str(exc))
    return JSONResponse(
        status_code=503,
        content={
            "error": "DB_ERROR",
            "message": "Storage backend returned an error.",
            "trace_id": current_trace_id(),
        },
    )


@app.get("/health", response_model=HealthResponse)
async def health(session: SessionDep) -> HealthResponse:
    return await compute_health(session)


@app.post("/events/ingest", response_model=IngestResponse)
async def ingest(session: SessionDep, payload: Any = Body(...)) -> IngestResponse:
    # Accept either a bare list or {"events": [...]} for tooling flexibility.
    if isinstance(payload, dict) and "events" in payload:
        events = payload["events"]
    elif isinstance(payload, list):
        events = payload
    else:
        raise HTTPException(status_code=400, detail="Expected JSON list or {events: [...]}.")

    if not isinstance(events, list):
        raise HTTPException(status_code=400, detail="events must be a list")

    if len(events) > settings.ingest_max_batch:
        raise HTTPException(
            status_code=413,
            detail=f"Batch too large: {len(events)} > max {settings.ingest_max_batch}",
        )

    return await ingest_events(session, events)


@app.get("/stores/{store_id}/metrics", response_model=StoreMetrics)
async def store_metrics(store_id: str, session: SessionDep) -> StoreMetrics:
    return await compute_metrics(session, store_id)


@app.get("/stores/{store_id}/funnel", response_model=StoreFunnel)
async def store_funnel(store_id: str, session: SessionDep) -> StoreFunnel:
    return await compute_funnel(session, store_id)


@app.get("/stores/{store_id}/heatmap", response_model=StoreHeatmap)
async def store_heatmap(store_id: str, session: SessionDep) -> StoreHeatmap:
    return await compute_heatmap(session, store_id)


@app.get("/stores/{store_id}/anomalies", response_model=StoreAnomalies)
async def store_anomalies(store_id: str, session: SessionDep) -> StoreAnomalies:
    return await detect_anomalies(session, store_id)


_DASHBOARD_HTML_PATH = Path(__file__).resolve().parent.parent / "dashboard" / "web.html"


@app.get("/dashboard", response_class=HTMLResponse, include_in_schema=False)
async def dashboard() -> HTMLResponse:
    """Single-page web dashboard. Polls /health, /metrics, /anomalies every 2s."""
    if not _DASHBOARD_HTML_PATH.exists():
        return HTMLResponse("<h1>Dashboard asset missing</h1>", status_code=500)
    return HTMLResponse(_DASHBOARD_HTML_PATH.read_text())
