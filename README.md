# Apex Retail — Store Intelligence

End-to-end pipeline that turns raw CCTV footage into a live offline-store
analytics API. Detection → structured events → REST API → live dashboard.

## Quick start (5 commands)

```bash
git clone <repo-url> store-intelligence && cd store-intelligence
docker compose up -d --build               # starts Postgres + API
SHIFT=1 ./pipeline/run.sh                  # auto-bootstrap (see below) + replay
open http://localhost:8000/dashboard       # live web dashboard
open http://localhost:8000/docs            # OpenAPI UI
```

That's it. No manual schema migration, no manual labels, no extra services.

The `SHIFT=1` flag rebases the clip's timestamps to "today" so the API's
day-window queries (which look at `ts >= today_00:00 UTC`) see the events.
Drop it in production where events arrive in real-time.

## What `run.sh` actually does (auto-bootstrap)

1. **`store_layout.json` missing?** `auto_setup.py` writes a coarse default
   (one whole-frame zone per camera, sensible roles by camera number).
2. Per-camera YOLO + ByteTrack detection emits raw JSONL events.
3. `extract_track_crops.py` saves one representative crop per ByteTrack track.
4. **Staff gallery empty?** `auto_setup.py` clusters the crop embeddings
   (ResNet50 + DBSCAN), picks the largest visually-coherent cluster as the
   staff uniform group, and writes its 10 most-central crops as the gallery.
   **No manual labels — staff are identified from the footage itself.**
5. `cluster_and_label.py` re-classifies every track against the gallery,
   merges cross-camera identities, and writes the canonical merged stream.
6. (Optional `SHIFT=1`) shift timestamps to today.
7. `correlate.py` loads POS rows and emits abandon events.
8. `replay.py` POSTs the final stream into `/events/ingest`.

## Live dashboards (Part E)

Two interchangeable dashboards, both wired to the same endpoints and both
refreshing every 2 seconds:

```bash
# Web — open in any browser, no install needed
open http://localhost:8000/dashboard

# Terminal — runs inside the API container, no host install needed
docker compose exec api python -m dashboard.tui
```

The web page shows a per-store grid (visitors, conversion, queue depth,
abandonment, feed status) plus an active-anomalies tail. The TUI shows the
same data with a colour-coded queue column (white → yellow at ≥5 → red at
≥8) and an anomalies panel that respects severity.

### Don't have the dataset yet? Demo the dashboards with synthetic traffic:

```bash
docker compose exec api python -m pipeline.demo_seed \
    --stores 3 --duration 60 --rate 3 --queue-spike-at 20
```

The seeder simulates entries, zone transitions, billing-queue joins (with a
forced spike at t=20s), abandonments, and exits. Watch either dashboard while
it runs — visitor counts climb, the queue spike turns red, an anomaly
appears in the active-anomalies panel.

## What you get

| Endpoint | What it returns |
| --- | --- |
| `GET  /health` | Service status, last event per store, `STALE_FEED` flag |
| `POST /events/ingest` | Idempotent batch ingest (≤500 events per call) |
| `GET  /stores/{id}/metrics` | Today's visitors, conversion, dwell, queue, abandonment |
| `GET  /stores/{id}/funnel` | Entry → Zone → Billing → Purchase with drop-off % |
| `GET  /stores/{id}/heatmap` | Per-zone visits + dwell, normalized 0–100 |
| `GET  /stores/{id}/anomalies` | Active queue / conversion / dead-zone alerts |

## Detection pipeline

The detection pipeline runs on the **host** (it needs OpenCV, Torch, and the
clip files) and emits events that get POSTed into the API container. This
keeps the API image lean and avoids GPU portability headaches.

```bash
# Install pipeline dependencies (once)
pip install -e ".[pipeline]"

# Run end-to-end against all stores in data/
./pipeline/run.sh

# Or replay an existing JSONL of events
python -m pipeline.replay events/final.jsonl
python -m pipeline.replay data/sample_events.jsonl  # the dataset sample
```

Per-camera output lands in `events/<store>_<camera>.jsonl`. The post-passes
(`pipeline/tracker.py` for cross-camera dedupe + Re-ID, `pipeline/correlate.py`
for POS correlation) merge into `events/final.jsonl`, which is the JSON that
gets replayed into the API.

## Sanity check

After `docker compose up`, run the acceptance gate script to verify everything
end-to-end (health check, ingest, idempotency, all 4 store endpoints):

```bash
./verify_gate.sh
```

It prints **GATE: OK** on success. This is exactly what reviewers run on a
fresh clone. **Note**: the gate tears down the DB volume and seeds 3 synthetic
`VIS_smoke_*` events to prove the ingest path. To restore the real
ST1008 analytics after running it, re-run `./pipeline/run.sh` (which is
idempotent — gallery and track-crop work is cached).

## Tests

`pytest` is bundled into the API image, so the suite runs in-place:

```bash
docker compose exec api pytest --cov=app --cov-report=term -q
```

49 tests, 78%+ statement coverage. The suite covers idempotency
(within-batch and across-calls), per-event partial-success on validation
errors, oversized batches, empty-store and all-staff filtering, zero-purchase
conversion handling, the POS-correlation 5-minute window boundary,
re-entry funnel correctness, heatmap normalization, anomaly severity
escalation, and graceful degradation when the database is unreachable. Each
test file leads with the AI prompt used to draft it plus the edits applied
after — see `tests/test_*.py`.

## Configuration

All tunables are environment variables (see `app/config.py`):

| Var | Default | Meaning |
| --- | --- | --- |
| `DATABASE_URL` | `postgresql+asyncpg://apex:apex@db:5432/apex` | Postgres DSN |
| `LOG_LEVEL` | `INFO` | structlog level |
| `STALE_FEED_THRESHOLD_SECONDS` | `600` | When `/health` flags a store stale |
| `QUEUE_WARN_DEPTH` | `5` | Anomaly threshold |
| `QUEUE_CRITICAL_DEPTH` | `8` | Anomaly threshold |
| `POS_CORRELATION_WINDOW_SECONDS` | `1800` | Window for `potential_conversion_rate` (loose, brand-aware) |
| `POS_STRICT_WINDOW_SECONDS` | `300` | Window for funnel PURCHASE + abandon detection (brief's definition) |
| `STAFF_GALLERY_DIR` | `/data/staff_gallery` | Where auto-built gallery lives |
| `STAFF_DISTANCE_THRESHOLD` | `0.30` | Cosine distance for staff classification |
| `NUM_STAFF` / `NUM_CUSTOMERS` | `0` (auto-K) | Identity-merge K hint; `0` = let DBSCAN decide |

## Docs

- `docs/DESIGN.md` — architecture, schema rationale, AI-assisted decisions, known limitations
- `docs/CHOICES.md` — three explicit decisions with full reasoning
