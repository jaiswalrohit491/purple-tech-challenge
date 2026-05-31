#!/usr/bin/env bash
# Acceptance gate sanity check. Run from repo root.
# Mirrors what reviewers will run on a fresh clone.

set -euo pipefail

API="http://localhost:8000"
STORE_ID="${STORE_ID:-ST1008}"

step() { printf '\n\033[1;34m▸ %s\033[0m\n' "$*"; }
fail() { printf '\033[1;31m✗ %s\033[0m\n' "$*"; exit 1; }

step "Tearing down any previous stack"
docker compose down -v 2>/dev/null || true

step "Building and starting docker compose"
docker compose up -d --build

step "Waiting for API to come up"
for i in $(seq 1 30); do
    if curl -fsS "$API/health" >/dev/null 2>&1; then
        echo "  ready after ${i}s"
        break
    fi
    sleep 1
    [ "$i" = "30" ] && fail "API did not become healthy in 30s"
done

step "GET /health"
curl -fsS "$API/health" | python3 -m json.tool || fail "/health invalid JSON"

step "Generating a minimal synthetic event batch (smoke ingest)"
python3 - <<'PY' > /tmp/_smoke_events.json
import json, uuid
from datetime import datetime, timezone, timedelta
base = datetime.now(timezone.utc) - timedelta(minutes=5)
events = []
for i in range(3):
    events.append({
        "event_id": str(uuid.uuid4()),
        "store_id": "ST1008",
        "camera_id": "CAM_ENTRY_01",
        "visitor_id": f"VIS_smoke_{i}",
        "event_type": "ENTRY",
        "timestamp": (base + timedelta(seconds=i*10)).isoformat().replace("+00:00","Z"),
        "zone_id": None,
        "dwell_ms": 0,
        "is_staff": False,
        "confidence": 0.9,
        "metadata": {"session_seq": 1}
    })
print(json.dumps(events))
PY

step "POST /events/ingest (smoke)"
RESP=$(curl -fsS -X POST "$API/events/ingest" -H "content-type: application/json" -d @/tmp/_smoke_events.json)
echo "$RESP" | python3 -m json.tool
echo "$RESP" | python3 -c "import json,sys; r=json.load(sys.stdin); assert r['accepted']>=3, r" || fail "ingest did not accept events"

step "Re-POST same payload (idempotency)"
RESP2=$(curl -fsS -X POST "$API/events/ingest" -H "content-type: application/json" -d @/tmp/_smoke_events.json)
echo "$RESP2" | python3 -m json.tool
echo "$RESP2" | python3 -c "import json,sys; r=json.load(sys.stdin); assert r['duplicates']>=3, r" || fail "idempotency broken"

step "GET /stores/$STORE_ID/metrics"
curl -fsS "$API/stores/$STORE_ID/metrics" | python3 -m json.tool

step "GET /stores/$STORE_ID/funnel"
curl -fsS "$API/stores/$STORE_ID/funnel" | python3 -m json.tool

step "GET /stores/$STORE_ID/heatmap"
curl -fsS "$API/stores/$STORE_ID/heatmap" | python3 -m json.tool

step "GET /stores/$STORE_ID/anomalies"
curl -fsS "$API/stores/$STORE_ID/anomalies" | python3 -m json.tool

printf '\n\033[1;32mGATE: OK\033[0m\n'
printf '\n\033[2mNote: this script tore down the DB and seeded 3 synthetic VIS_smoke_*\n'
printf 'events to prove the pipe end-to-end. To restore real ST1008 data, run:\n'
printf '  ./pipeline/run.sh\n\033[0m\n'
