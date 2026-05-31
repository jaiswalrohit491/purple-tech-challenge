#!/usr/bin/env bash
# End-to-end pipeline driver.
#
# Cold-start behaviour:
#   1. If `data/store_layout.json` is missing → `auto_setup.py` writes a coarse
#      default (per-camera whole-frame zone) so the rest of the pipeline can run.
#   2. Per-camera detection (`detect.py`) emits raw JSONL events. With an empty
#      gallery this initial pass tags every track as `is_staff=false`; the
#      classification is fixed in step 5.
#   3. `extract_track_crops.py` saves one representative crop per ByteTrack
#      track on each in-store camera.
#   4. If `data/staff_gallery/` is empty → `auto_setup.py` clusters the crop
#      embeddings (ResNet50 + DBSCAN), picks the largest visually-coherent
#      cluster as the staff uniform group, and writes its 10 most-central crops
#      as the gallery. No manual labels required.
#   5. `cluster_and_label.py` re-classifies every track against the
#      (auto-built) gallery, merges cross-camera identities, synthesises one
#      `ENTRY` per canonical customer, and writes `events/$STORE_merged.jsonl`.
#   6. Optional `shift_to_now`: shifts timestamps so old footage lands in
#      today's window (set `SHIFT=1` env). Useful for demos.
#   7. `correlate.py` loads POS rows + appends `BILLING_QUEUE_ABANDON` events.
#   8. `replay.py` POSTs the final stream into `/events/ingest`.

set -euo pipefail

DATA_DIR="${DATA_DIR:-data}"
EVENTS_DIR="${EVENTS_DIR:-events}"
API_URL="${API_URL:-http://localhost:8000}"
LAYOUT="${LAYOUT:-$DATA_DIR/store_layout.json}"
POS="${POS:-$DATA_DIR/Brigade_Bangalore_10_April_26 (1)bc6219c.csv}"
STAFF_GALLERY_DIR="${STAFF_GALLERY_DIR:-$DATA_DIR/staff_gallery}"
SHIFT="${SHIFT:-0}"

mkdir -p "$EVENTS_DIR"

step() { printf '\n▸ %s\n' "$*"; }

# ---------- 1. layout ----------
if [ ! -f "$LAYOUT" ]; then
    step "no layout — auto-generating coarse default"
    python3 -m pipeline.auto_setup --layout "$LAYOUT" --footage "$DATA_DIR/footage" \
        --gallery "$STAFF_GALLERY_DIR" \
        --crops-dirs "$EVENTS_DIR/track_crops/CAM_01" "$EVENTS_DIR/track_crops/CAM_02" "$EVENTS_DIR/track_crops/CAM_05" \
        --force 2>&1 | tail -5 || true
fi

# ---------- 2. per-camera detection ----------
mapfile -t TUPLES < <(python3 -c "
import json, sys
data = json.load(open('$LAYOUT'))
stores = data.get('stores') or data
for s in stores:
    cs = s.get('clip_start_utc', '2026-01-01T00:00:00Z')
    for c in s.get('cameras', []):
        if not c.get('clip_path'):
            continue
        print(f\"{s['store_id']}|{c['camera_id']}|{c['clip_path']}|{cs}\")
")

if [ "${#TUPLES[@]}" -eq 0 ]; then
    echo "run.sh: no cameras with clip_path in layout" >&2; exit 1
fi

for row in "${TUPLES[@]}"; do
    IFS='|' read -r STORE CAM CLIP_REL CS <<<"$row"
    CLIP="$DATA_DIR/$CLIP_REL"
    if [ ! -f "$CLIP" ]; then
        echo "  skipping $STORE/$CAM (no $CLIP)" >&2; continue
    fi
    OUT="$EVENTS_DIR/${STORE}_${CAM}.jsonl"
    step "detecting $STORE/$CAM ($CLIP) → $OUT"
    : > "$OUT"
    STAFF_GALLERY_DIR="$STAFF_GALLERY_DIR" python3 -m pipeline.detect \
        --store "$STORE" --camera "$CAM" --clip "$CLIP" \
        --layout "$LAYOUT" --clip-start "$CS" --out "$OUT"
done

# Discover the store_id for the merge step (single-store layouts).
STORE_ID=$(python3 -c "
import json; print(json.load(open('$LAYOUT'))['stores'][0]['store_id'])")

# Unique-person count is MEASURED by the spatiotemporal identity engine
# (pipeline/identity.py) — no operator headcount prior, no K. Camera geometry
# (`camera_topology`) is read from the layout to gate cross-camera links.

# ---------- 3. extract crops for in-store cameras ----------
for CAM in CAM_01 CAM_02 CAM_05; do
    CROPS_DIR="$EVENTS_DIR/track_crops/$CAM"
    if [ ! -d "$CROPS_DIR" ] || [ -z "$(ls -A "$CROPS_DIR" 2>/dev/null)" ]; then
        step "extracting track crops on $CAM"
        python3 -m pipeline.extract_track_crops --layout "$LAYOUT" \
            --store "$STORE_ID" --camera "$CAM" --data-dir "$DATA_DIR" \
            --out "$CROPS_DIR" 2>&1 | tail -2
    fi
done

# ---------- 4. auto-build staff gallery (if empty) ----------
if [ ! -d "$STAFF_GALLERY_DIR" ] || [ -z "$(ls "$STAFF_GALLERY_DIR"/*.jpg 2>/dev/null)" ]; then
    step "auto-discovering staff cluster → $STAFF_GALLERY_DIR"
    python3 -m pipeline.auto_setup --layout "$LAYOUT" --footage "$DATA_DIR/footage" \
        --gallery "$STAFF_GALLERY_DIR" \
        --crops-dirs "$EVENTS_DIR/track_crops/CAM_01" "$EVENTS_DIR/track_crops/CAM_02" "$EVENTS_DIR/track_crops/CAM_05" \
        --force 2>&1 | tail -10
fi

# ---------- 5. cross-camera person merge + label ----------
step "cross-camera merge + classify"
python3 -m pipeline.cluster_and_label \
    --events-dir "$EVENTS_DIR" \
    --cameras CAM_01 CAM_02 CAM_05 \
    --crops-dirs "$EVENTS_DIR/track_crops/CAM_01" "$EVENTS_DIR/track_crops/CAM_02" "$EVENTS_DIR/track_crops/CAM_05" \
    --staff-gallery "$STAFF_GALLERY_DIR" \
    --customer-gallery "$DATA_DIR/nonexistent_customer_gallery" \
    --layout "$LAYOUT" \
    --same-cam-dist "${SAME_CAM_DIST:-0.45}" --cross-cam-dist "${CROSS_CAM_DIST:-0.45}" \
    --stitch-gap "${STITCH_GAP:-60}" --cross-window "${CROSS_WINDOW:-90}" \
    --store-id "$STORE_ID" \
    --out "$EVENTS_DIR/${STORE_ID}_merged.jsonl" 2>&1 | tail -16

MERGED="$EVENTS_DIR/${STORE_ID}_merged.jsonl"

# ---------- 6. (optional) shift timestamps to today ----------
INGEST_SOURCE="$MERGED"
SHIFT_DELTA=0
if [ "$SHIFT" = "1" ]; then
    step "shifting timestamps to today (SHIFT=1)"
    SHIFT_OUT=$(python3 -m pipeline.shift_to_now "$MERGED" \
        --out "$EVENTS_DIR/${STORE_ID}_merged.shifted.jsonl" --regenerate-ids 2>&1 || true)
    SHIFT_DELTA=$(python3 -c "
import json, sys
try:
    d = json.loads(sys.argv[1])
    print(d.get('delta_seconds', 0))
except Exception:
    print(0)" "$SHIFT_OUT")
    if [ -f "$EVENTS_DIR/${STORE_ID}_merged.shifted.jsonl" ]; then
        INGEST_SOURCE="$EVENTS_DIR/${STORE_ID}_merged.shifted.jsonl"
    else
        echo "  shift produced no output (empty merged stream?); falling back to unshifted" >&2
    fi
fi

# ---------- 7. POS correlation (loads POS into DB; emits abandons) ----------
if [ -f "$POS" ]; then
    step "POS correlation + abandons"
    python3 -m pipeline.correlate --pos "$POS" --in "$INGEST_SOURCE" \
        --out "$EVENTS_DIR/${STORE_ID}_final.jsonl" --store-id "$STORE_ID" \
        --ts-shift-seconds "$SHIFT_DELTA" 2>&1 | tail -2
    FINAL="$EVENTS_DIR/${STORE_ID}_final.jsonl"
else
    step "no POS file at $POS — skipping correlation"
    cp "$INGEST_SOURCE" "$EVENTS_DIR/${STORE_ID}_final.jsonl"
    FINAL="$EVENTS_DIR/${STORE_ID}_final.jsonl"
fi

# ---------- 8. replay into API ----------
step "replay into API"
python3 -m pipeline.replay --url "$API_URL" "$FINAL" 2>&1 | tail -2

echo
echo "✓ done.  store_id=$STORE_ID  events=$FINAL"
echo "  open $API_URL/dashboard  or  python -m dashboard.tui"
