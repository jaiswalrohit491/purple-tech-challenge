import logging
import sys
import time
import uuid
from contextvars import ContextVar

import structlog
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request

from .config import settings

_trace_id: ContextVar[str] = ContextVar("trace_id", default="-")


def _add_trace_id(_logger, _method, event_dict):
    event_dict["trace_id"] = _trace_id.get()
    return event_dict


def configure_logging() -> None:
    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=getattr(logging, settings.log_level.upper(), logging.INFO),
    )
    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            _add_trace_id,
            structlog.processors.TimeStamper(fmt="iso", utc=True),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.JSONRenderer(),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(
            getattr(logging, settings.log_level.upper(), logging.INFO)
        ),
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=True,
    )


log = structlog.get_logger("app")


class TraceIdMiddleware(BaseHTTPMiddleware):
    """Stamp each request with a trace_id and emit a structured access log line."""

    async def dispatch(self, request: Request, call_next):
        trace = request.headers.get("x-trace-id") or uuid.uuid4().hex
        token = _trace_id.set(trace)
        start = time.perf_counter()
        store_id = request.path_params.get("store_id") if request.path_params else None
        try:
            response = await call_next(request)
            latency_ms = round((time.perf_counter() - start) * 1000, 2)
            log.info(
                "request.completed",
                method=request.method,
                path=request.url.path,
                status_code=response.status_code,
                latency_ms=latency_ms,
                store_id=store_id,
            )
            response.headers["x-trace-id"] = trace
            return response
        except Exception as exc:
            latency_ms = round((time.perf_counter() - start) * 1000, 2)
            log.error(
                "request.failed",
                method=request.method,
                path=request.url.path,
                latency_ms=latency_ms,
                error=str(exc),
            )
            raise
        finally:
            _trace_id.reset(token)


def current_trace_id() -> str:
    return _trace_id.get()
