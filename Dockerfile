# API-only image. The detection pipeline runs on the host (it needs OpenCV +
# Torch + clip files) and POSTs events into this container via /events/ingest.
# Keeping the API image lean speeds up rebuilds and dodges GPU/torch portability.
FROM python:3.12-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /srv

RUN apt-get update && apt-get install -y --no-install-recommends \
        curl \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml ./
RUN pip install --upgrade pip \
    && pip install \
        "fastapi>=0.115" \
        "uvicorn[standard]>=0.32" \
        "pydantic>=2.9" \
        "pydantic-settings>=2.6" \
        "sqlalchemy[asyncio]>=2.0.36" \
        "asyncpg>=0.30" \
        "structlog>=24.4" \
        "httpx>=0.27" \
        "orjson>=3.10" \
        "pytest>=8.3" \
        "pytest-asyncio>=0.24" \
        "pytest-cov>=5.0" \
        "shapely>=2.0" \
        "rich>=13.9" \
        "opencv-python-headless>=4.10" \
        "pillow>=10.4" \
        "anthropic>=0.39" \
    && pip install --extra-index-url https://download.pytorch.org/whl/cpu \
        "torch>=2.4" "torchvision>=0.19" \
    && pip install "ultralytics>=8.3" \
    && pip uninstall -y opencv-python opencv-python-headless \
    && pip install --no-cache-dir "opencv-python-headless>=4.10"

COPY app ./app
COPY pipeline ./pipeline
COPY dashboard ./dashboard
COPY tests ./tests

EXPOSE 8000

HEALTHCHECK --interval=10s --timeout=3s --start-period=5s --retries=3 \
    CMD curl -fsS http://localhost:8000/health || exit 1

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--proxy-headers"]
