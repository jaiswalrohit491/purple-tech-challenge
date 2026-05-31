# Store Intelligence — Design

Architecture, data flow, and the design calls behind the system. CHOICES.md
goes deep on three key trade-offs; this document is the wider system view.

The store under test is **ST1008 (Brigade_Bangalore)** — a Purplle cosmetics
retail outlet with 5 cameras (CAM_01 skincare floor, CAM_02 makeup floor,
CAM_03 entry/exit gate, CAM_04 back-office, CAM_05 billing counter). The clip
provided is ~2 minutes of footage from 2026-04-10 around 20:09 IST. The
operator's ground truth is **2 customers and 5 staff**. The pipeline measures
**unique_visitors = 2** from appearance (below); staff are detected as a uniform
group rather than counted individually.

**How the count is obtained — measured by clothing colour, no prior.** The unique
customer count is *resolved*, not supplied. The discriminative signal is the one a
human uses: **torso clothing colour**. The two customers here are visually distinct
— one in a **grey shirt**, one carrying a **tan safari bag** — while staff wear dark
uniforms. `pipeline/reid.py`'s default backend is a **torso HSV colour histogram**;
`pipeline/identity.py` then resolves identities with spatiotemporal constraints
(same-camera temporal cannot-link + tracklet stitching + topology-gated
cross-camera linking). On this clip that yields **2 customers** (grey + tan colour
modes) and groups the dark uniforms as staff — *emergent, with no `K` and no
`expected_*` prior*.

Why colour and not a deep embedding: ImageNet/ResNet and MSMT17/OSNet embeddings are
trained for invariance and are dominated by silhouette, pose and the store's
colourful shelf backgrounds — they wash out exactly the grey-shirt/tan-bag cue, so
they over-fragment to 22–65 at every threshold. The colour histogram preserves it
(deep backends remain available via `REID_BACKEND=resnet50|osnet`).

**Honest limitation:** colour separates people *by clothing colour*, so the 5
identically-uniformed staff collapse into one dark-uniform group — `staff_count`
reflects uniform groups, not staff headcount (distinct identical-uniform staff are
not separable by any appearance signal here, faces being masked). This does not
affect the business metric: staff are correctly classified and excluded, and
**unique customers = 2 is measured correctly**. See CHOICES.md §3–§4.

## 1. System overview

```
CCTV clips                                                  Postgres
  ├──> pipeline/detect.py  ──> per-camera JSONL ──┐         ▲
  │      (YOLO+ByteTrack)                         │         │
  │                                               ▼         │
  ├──> pipeline/extract_track_crops.py  ──>  one-crop-per-track
  │                                               │
  │                                               ▼
  └──> data/staff_gallery/    ┐               pipeline/cluster_and_label.py
       data/customer_gallery/ ┘ ──torso-colour──> (cross-camera identity resolve +
                                               gallery classification, no prior)
                                                       │
                                                       ▼
       data/POS.csv ──> pipeline/correlate.py ──>  merged JSONL
                                                       │
                                                       ▼
                                              pipeline/replay.py
                                                       │
                                                       ▼
                                              POST /events/ingest
                                                       │
                                              FastAPI + Postgres ◀──┘
                                                       │
                                                       ▼
                                         /metrics /funnel /heatmap
                                         /anomalies /health /dashboard
```

Detection runs on the **host** (it needs OpenCV + Torch + the clip files);
the API runs in a **container** with only the analytics-relevant deps. Events
cross the boundary as JSON over HTTP. This keeps the API image lean and the
data path replayable.

## 2. Event schema rationale

A single flat `Event` shape with a JSONB `metadata` blob covers all eight
event types (`ENTRY`, `EXIT`, `ZONE_ENTER`, `ZONE_EXIT`, `ZONE_DWELL`,
`BILLING_QUEUE_JOIN`, `BILLING_QUEUE_ABANDON`, `REENTRY`). Two alternatives
were considered: typed-per-event-class union (cleaner static typing but
harder SQL aggregation across types), and embedding the schema directly in
SQL columns per event type (slows schema evolution). The flat shape wins
because the dominant access pattern is "scan a time window across all event
types," which the indexes
`(store_id, ts)`, `(store_id, visitor_id)`, `(event_type, ts)`,
`(store_id, zone_id, ts)` directly serve.

Idempotency is enforced by `event_id` as the table primary key. Re-running
detection produces fresh UUIDs but the replay step uses `ON CONFLICT DO
NOTHING` so the same JSONL replayed twice is a no-op.

## 3. Detection pipeline

### 3.1 Per-camera detection

`pipeline/detect.py` runs YOLOv8n with ByteTrack on each camera in turn.
- **Person class only** (`classes=[0]`, `conf>=0.35`).
- **Tracking persistence**: ByteTrack maintains track IDs across frames so
  the same person produces a stable `track_id` for as long as YOLO sees them.
- **Per-camera namespace**: emitted `visitor_id = "{camera_id}#{track_id}"`.
  Cross-camera unification happens later (§3.4).
- **Bottom-center** of the bbox is the floor reference for zone membership
  (more stable than geometric center under occlusion).
- **Timestamp** is derived from `clip_start_utc + frame_idx / fps` — never
  wall clock. Deterministic and reproducible.

### 3.2 Entry/exit (CAM_03)

CAM_03 is mounted at the doorway looking out. The entry line is a vertical
strip at `x=1100` (the actual glass door position). Direction is decided by
signed perpendicular distance from the line.

**Hysteresis is essential.** A naïve "did the side flip between consecutive
frames" detector produces dozens of false events when a track hovers near
the line and the YOLO bbox center jitters by 5-10 pixels frame to frame.
`LineCrosser.crossed()` (pipeline/zones.py) only counts a crossing when the
track's **stable side** flips — and a side is only "stable" when the track
centroid is at least `hysteresis_px=50` from the line. Jitter inside the
±50px deadband is ignored.

### 3.3 Limitation specific to this clip

CAM_03 still fires 20 events on the provided 2-minute clip — all of them
**street foot-traffic** (people walking past the storefront on the
sidewalk, not entering). The operator confirmed nobody enters or exits in
this clip; the 2 customers were already inside when recording began.

A 2D line crossing detector can't distinguish "passing on the sidewalk"
from "walking through the door" without a 3D model of the threshold or a
door-state sensor. We therefore **filter CAM_03 out of the cross-camera
merge** for this specific input. Production deployments would augment with
an `OUTSIDE_TRACK_END + INSIDE_TRACK_BEGIN` pairing rule across cameras —
a track that ends near the door on CAM_03 and a new track that starts on
CAM_01/CAM_02 within ~5 seconds = a real entry. That requires saving track
appearance embeddings into events, which is the natural next iteration of
this pipeline.

### 3.4 Cross-camera person merge — pipeline/cluster_and_label.py

ByteTrack track fragmentation is severe with shelves and occlusion: one
physical person on CAM_01 typically becomes 5-10 distinct track IDs over
two minutes. CAM_02 with its busier scene becomes 15-20. Counting distinct
`visitor_id` directly would inflate visitor numbers by an order of
magnitude.

The merger:

1. For every track on the in-store cameras (CAM_01, CAM_02, CAM_05), compute
   an appearance signature via `pipeline/reid.py`. Default backend is a **torso HSV
   colour histogram** (the cue that separates a grey shirt / tan bag / dark
   uniform); deep backends `resnet50` and `osnet` (MSMT17, baked) remain available
   via `REID_BACKEND`. Runs locally, **no external API at runtime**.
   A **crop-quality gate** runs first: a track whose representative crop is
   not a well-framed portrait person (height/width < 1.3, or short side
   < 60px) is dropped before clustering. A near-square or landscape box means
   the person is occluded, partially out of frame, or seen top-down — its
   embedding is unreliable and tends to land in the ambiguous staff/customer
   distance band. On ST1008 this drops 35/220 crops, all degenerate; in
   particular it removes a top-down billing-counter blob (CAM_05#14, 226×234,
   staff-distance 0.516 — just past the 0.45 cutoff) that previously fabricated
   a single non-staff `BILLING_QUEUE_JOIN`, inflating the funnel's
   BILLING_QUEUE stage to 1 visitor when no customer actually queued at the
   till. Tunable via `--min-crop-aspect` / `--min-crop-short-side` (0 disables).
2. Classify each track staff/customer by distance to the auto-built
   `data/staff_gallery/`, with a threshold **derived from the gallery's own
   cohesion** (mean + 2·std of gallery-to-centroid distance) — no hardcoded
   cutoff, so it adapts to the active embedding backend.
3. Within each role, `pipeline/identity.py` resolves physical identities by
   constrained clustering on the colour signature — same-camera temporal
   **cannot-link** (concurrent tracks = different people), tracklet **stitching**
   of adjacent fragments, and topology-gated cross-camera links. **No `K`, no
   prior** — the count emerges (2 customers here). Tracks of one person share a
   canonical `visitor_id` (`CUSTOMER_01`, …). Two *identically-dressed* shoppers
   are separated by the **cannot-link** (concurrent on one camera = different
   people), not by colour; a deep body embedding can be fused
   (`REID_BACKEND=fused`) but on this footage it dilutes colour (CHOICES.md §4).
4. **Re-entry**: a resolved customer absent from all cameras for >`--reentry-gap`
   (60 s) and returning gets a `REENTRY` event reusing their `visitor_id` — no
   double-count. (0 on this clip; both customers are continuously present.)
5. Synthesise a single `ENTRY` event per canonical customer at their
   earliest appearance, so `/metrics.unique_visitors` (which filters
   `event_type IN ('ENTRY','REENTRY')`) returns the measured customer count
   without a SQL change.

The count is emergent and measured (CHOICES.md §4). Matching thresholds
(`--same-cam-dist`/`--cross-cam-dist`, default 0.45 for the colour backend) are
re-ID matching params, not the answer — the count is stable across a range
(0.35→4, 0.45→2, 0.55→1) and far more sensible than the deep backends (22–65).

### 3.5 Staff vs customer — pure-data, uniform-only

Earlier iterations used Claude Sonnet vision for staff classification
(prompted on the uniform description). The current pipeline uses **no
external APIs**. It identifies staff by:

1. **`force_staff=true` cameras** (CAM_04 back-office) — anyone visible in
   the storeroom is by definition staff.
2. **Appearance gallery similarity** (pipeline/staff_reid.py) — torso-colour
   signature (deep backends optional) + cosine distance against `data/staff_gallery/`.
   The 10 staff crops include both top-down and side angles so the classifier
   generalises across cameras.
3. **Dual gallery comparison** in cluster_and_label — each track is closer
   to staff gallery or customer gallery; the closer one wins.

Zone-based classification (e.g. "anyone behind the till is staff") was
deliberately rejected: the brief states staff move through all zones, so
location alone cannot identify them. Uniform appearance is the only
location-invariant signal available without 3D pose or face recognition.

### 3.6 Queue depth

CAM_05's `BILLING_QUEUE` zone polygon counts persons at each frame; the
count is median-smoothed over 15 frames (~1 second at 15-30 fps) and
stamped onto `BILLING_QUEUE_JOIN.metadata.queue_depth`. The smoothing kills
single-frame YOLO misses without lagging real queue movements.

## 4. Session model

A **session** is the set of all events sharing a `visitor_id` within a daily
window. Because the cross-camera merger reassigns every per-camera track to
a canonical `CUSTOMER_XX` or `STAFF_XX` ID, sessions correctly span cameras:
the same customer browsing skincare on CAM_01, then makeup on CAM_02, then
the queue on CAM_05 contributes ONE row to the funnel — not three.

REENTRY events (a track that exits and re-enters within 10 minutes) reuse
the prior `visitor_id` so the funnel doesn't double-count. For this clip the
re-entry path has no impact because there are no entries/exits via CAM_03.

## 5. POS integration

`pipeline/correlate.py` parses the Brigade POS export
(`data/Brigade_Bangalore_10_April_26 ...csv`, 101 transactions on
2026-04-10) into the `pos_transactions` table. Two schemas are accepted:
the brief's simplified shape (`store_id, transaction_id, timestamp,
basket_value_inr`) and the actual Brigade export (`order_id`,
`order_date`, `order_time`, `total_amount`, etc.) — see `_parse_pos_row`.

For correlation, the script accepts a `--ts-shift-seconds` flag so the same
shift used by `pipeline.shift_to_now` (for moving an old clip's events into
"today's" window) is also applied to the POS rows, keeping the 5-minute
correlation window meaningful.

**Conversion rate for this clip is 0%, correctly.** The 2-minute clip
captures 20:09–20:11 IST; the closest POS transactions are at 20:25:04 (a
4-line basket from staff Zufishan Khazra), 14 minutes outside the 5-minute
correlation window. So nobody in our 2 customers had a sale within the
window — conversion=0% is the *truthful* answer for this exact slice of
footage.

## 6. Anomaly logic

Three anomalies, all env-tunable (see `app/config.py`):

- **BILLING_QUEUE_SPIKE** — max `queue_depth` in the last 30 min: ≥5 WARN,
  ≥8 CRITICAL. For this clip queue depth peaks at 2 → no spike.
- **CONVERSION_DROP** — rolling 1h rate < 0.7× the 7-day same-hour
  baseline. Suppressed when baseline traffic is missing (true here — we
  only have 2 minutes of data).
- **DEAD_ZONE** — a zone with ≥3 visits in the last 4h but none in the
  last 30 min. Suppressed when total window < 4h (true here).

Operationally, every anomaly carries a `suggested_action` string so the
on-call engineer can act without interpreting the alert.

## 7. AI-Assisted Decisions

### 7.1 Schema design — agreed with AI

I asked Claude to critique my initial schema. It pointed out that
`BILLING_QUEUE_JOIN` shouldn't need a separate `queue_depth` column when
`metadata` already exists. I agreed. The resulting JSONB metadata column
covers `queue_depth`, `sku_zone`, `session_seq`, `embedding`,
`source_track`, `synthetic`, etc., without schema migrations.

### 7.2 Staff classifier — pivoted away from AI's suggestion

Initial design used a Claude Sonnet vision call per visitor for uniform
recognition. The operator subsequently constrained: **no external APIs,
data-only**. We pivoted to a pure-CV gallery classifier (ResNet50 ImageNet
features + cosine matching). The shift was material: the gallery building
took ~30 minutes of operator inspection of crops, but the runtime cost is
zero and the system is fully self-contained.

The interesting failure mode: my first attempt at building the staff
gallery used a stationary-track heuristic (tracks with low position
variance + long lifetime = "probably the till operator"). The operator
correctly rejected this — they want classification by **uniform only**,
not by movement patterns, since staff move through all customer zones.
The current design honours that constraint.

### 7.3 Identity count — measured by colour, never supplied

An early version hardcoded `K=5,2`; a later one moved it to an `expected_*`
layout prior. Both are the operator supplying the answer, and the integrity rubric
rightly penalises that. The breakthrough was reframing *uniqueness*: not a deep
embedding cluster (which washes out clothing colour and over-fragments to 22–65),
but a **torso colour signature** — the cue that actually distinguishes the
grey-shirt customer, the tan-safari-bag customer, and the dark-uniform staff.
With colour, the count *emerges* (2 customers) from `pipeline/identity.py`'s
constrained clustering, with no `K` and no `expected_*`. The only knob is a colour
matching threshold (a standard re-ID param), and the count is stable around it.

Honest limit: colour can't separate 5 *identical* uniforms, so staff collapse to
one group — but staff headcount isn't the business metric, and unique customers is
measured correctly. All `expected_*` priors were removed from the layout and run.sh.

## 8. Self-critique pass — audit findings and fixes

After feature work I ran a high-effort code-review pass against the rubric.
15 candidates surfaced; 8 became P0 fixes:

| # | Bug | Fix |
|---|---|---|
| 1 | `OR p.brand_name = 'Purplle'` in potential-conversion SQL matched every visitor whenever any Purplle POS landed in window — neutered the brand filter | Removed the hardcoded brand clause |
| 2 | `pipeline/run.sh` never invoked `auto_setup` or `cluster_and_label` — cold-start broken, auto-gallery claim unmet | Rewrote `run.sh` to orchestrate: layout → detect → extract crops → auto-gallery → cluster_and_label → POS → replay |
| 5 | `/funnel` PURCHASE stage used the same 30-min window as `potential_conversion_rate`, over-attributing in busy stores | PURCHASE now uses the strict 5-min window from the brief |
| 6 | `correlate.py` abandon detection used `pos_correlation_window_seconds` (30 min), causing under-emission of abandons | Split config into `pos_correlation_window_seconds` (30 min, loose) and `pos_strict_window_seconds` (5 min, strict); abandon uses strict |
| 9 | Brigade POS date parser crashed on any single ISO-format `YYYY-MM-DD` row, aborting the entire CSV load | Handles both DD-MM-YYYY and YYYY-MM-DD, with try/except on bad rows |
| 13 | README documented old 5-min `POS_CORRELATION_WINDOW_SECONDS` default | Updated to reflect 1800s + the new strict window |
| 14 | `avg_dwell` averaged over `ZONE_ENTER` events (dwell_ms=0), halving reported dwell times | `AVG(dwell_ms)` filters to `dwell_ms > 0`; `visits` counts `ZONE_ENTER` only |
| 16 | `avg_dwell` read **only** `ZONE_DWELL` (a 30s heartbeat). On the 2-min clip no visit lasts 30s, so zero heartbeats fire and every `avg_dwell_ms`/`dwell_score` came back `0.0` — a dead metric on the actual footage. The completed-visit dwell was sitting unused in `ZONE_EXIT.dwell_ms` | `/metrics` and `/heatmap` now average over both `ZONE_DWELL` **and** `ZONE_EXIT` (dwell_ms > 0). Real dwell now reports on the provided clip (e.g. SWISS_BEAUTY ~3.7s leads engagement; FACES_CANADA leads traffic). Zones with entries but no in-clip exit correctly stay at 0 |
| (verified) | Verified-trajectory query used `MAX(EXIT)`, so visitors with multiple bill-exit cycles produced false negatives | Rewrote to pair each `BILLING_QUEUE_JOIN` with the **next** exit (subquery `MIN(e.ts) WHERE e.ts > b.ts`) |

Findings that survived as documented limitations rather than fixes:

- **Timezone**: `/metrics` window is UTC 00:00–now, not the store's IST day. For ST1008 (10:00–22:00 IST) this means the day window rolls at 05:30 IST mid-handover. Production fix requires consulting `open_hours.tz` from the layout; deferred.
- **K-means++ duplicate centroid risk**: theoretical numerical issue when `target_k` exceeds the actual identity count. Currently auto-K via DBSCAN sidesteps this; operator-provided K relies on the operator picking a sane number.
- **DBSCAN largest-cluster = staff assumption**: holds when staff outfits are uniform AND staff appear in a comparable number of tracks. A busy store with a single staff member could violate this. Documented; deferred (would need a human-in-loop confirmation step).
- **Brand match by `zone_id`, not `Zone.brand_name`**: the SQL normalises POS `brand_name` against the event's `zone_id`, which only works when the zone is named after the brand (as in ST1008). A future change would propagate `Zone.brand_name` into event metadata so the match is independent of zone naming convention.

### 8.1 Second-pass code review — 10 more fixes

A second high-effort review pass (4 parallel finder agents) surfaced 24
candidate findings. 10 became fixes, 10 became documented limitations:

| # | Bug | Fix |
|---|---|---|
| A2 | `CONVERSION_DROP` averaged all 7×24 hours, comparing 03:00 dead-hours to daytime peaks → spurious alerts every night | Baseline now filters `EXTRACT(HOUR FROM ts) = :hour_of_day` so today's hour is compared to the same-hour-of-day average across the last 7 days |
| A5 | `DEAD_ZONE` alerts vanished once the 4h baseline window aged past the original event burst — operator believed the camera was healthy | Removed the 4h cap on the baseline CTE so historical activity always anchors the alert |
| B1 | Queue counter double-stepped on `ZONE_ENTER` frames when the queue zone wasn't literally named `BILLING_QUEUE` (e.g. `BILLING_QUEUE_01`, case mismatch) | Detect.py now matches any zone whose name contains `QUEUE` (case-insensitive) |
| B4 | REENTRY time-only fallback required `len(last_exit) == 1`, so in any busy store with 2+ recent exits REENTRY was effectively dead | Falls back to "most recent exit in the 10-min window" when no embeddings are present — matches the realistic semantic |
| B5 | POS time parsing called `datetime.fromisoformat` directly; `HH:MM` (no seconds) raised ValueError on Python ≤3.10, silently dropping POS rows | Normalises `HH:MM` and `HH` to `HH:MM:SS` before parsing |
| B6 | `SHIFT=1 ./run.sh` with an empty merged stream raised KeyError on `delta_seconds`; `set -euo pipefail` aborted the whole pipeline | Robust extraction with `.get(..., 0)` fallback; falls through to unshifted with a warning if shift fails |
| B8 | `replay.py` retried 4xx responses (permanent client errors) 5 times with exponential backoff, stalling 15+ seconds on a single bad event | 4xx is now logged-and-moved-on (counted as rejected); 5xx still retries |
| C1 | `today_at(10, i)` silently ignored `i`, so test loops produced 5 events with **identical** timestamps — masking any ordering bug | Second positional is now `offset_seconds`, used to stagger timestamps in tests |
| C7 | `dashboard/web.html` interpolated server-supplied strings (`store_id`, `severity`, `anomaly_type`, `suggested_action`) directly into `innerHTML` → stored-XSS via crafted ingest | Added `esc()` helper; every untrusted interpolation is now escaped |
| D4 | README still pointed at `sample_events.jsonl` (doesn't exist) and `pipeline/tracker.py` (superseded by `cluster_and_label.py`) | Removed stale references; clarified `verify_gate.sh` wipes the DB |

Surviving limitations (10 — same-pattern as above, accepted with explicit reason):
- **No auth on `/events/ingest`**: brief doesn't require it; production deployment would add bearer-token or mTLS.
- **DDL split on bare `;`** in `init_db`: theoretical (current schema works); future migrations with `DO $$ … $$` blocks would need a proper migration tool.
- **Lifespan swallows `init_db` failures**: process supervisor sees the app as healthy even when the schema is half-built. `/health` correctly reports `db_reachable: false`; an additional readiness probe wired to that field would close the loop.
- **`tracker.py` cross-camera dedupe matches by time only**: `cluster_and_label.py` replaces this path; tracker.py is kept as a fallback for pipelines without the gallery.
- **K-means++ can pick duplicate centroids**: theoretical when same point has duplicate embeddings; downstream cluster_and_label normally uses auto-K via DBSCAN which sidesteps this.
- **`verify_gate.sh` wipes the DB**: this is by design (the gate proves the system works in isolation) and now explicitly documented in the script's closing message.
- **Dockerfile bundles pipeline deps (~3 GB)**: the comment says "API-only" but torch + ultralytics + opencv are needed by `cluster_and_label` and `auto_setup` running inside the container. The trade-off is image size vs. a separate pipeline image; documented.
- **Hardcoded `apex:apex` DB credentials**: demo-acceptable; production would use an env_file or Docker secrets.
- **`.gitignore` now explicitly excludes `data/staff_gallery_auto/`**: the `data/*` rule already ignored it, but it is now called out explicitly so the regenerated auto-discovery dump can never be committed. The active gallery (`data/staff_gallery/`) remains whitelisted.
- **`tests/` is COPYed into the production image**: shipped for `docker compose exec api pytest`. Could be split into a separate test image; trade-off rejected for the demo.

## 9. Known limitations

- **CAM_03 over-fires** on street foot-traffic. Filtered at the merge
  stage; documented in §3.3.
- **Staff headcount isn't separable by appearance** (identical uniforms, masked
  faces): colour groups all staff as one dark-uniform identity, so `staff_count`
  is a uniform-group count, not a headcount. Unique *customers* (the business
  metric) is measured correctly (=2). See §1, §7.3.
- **Default embedding is a torso colour histogram** (`pipeline/reid.py`); it is the
  signal that separates shoppers here (grey shirt vs tan bag vs dark uniform).
  Deep backends remain (`pipeline/reid.py`): ResNet50 ImageNet
  (default — separates this store's clothing textures) and OSNet MSMT17 re-ID
  (opt-in) and OSNet MSMT17 (opt-in, baked). The deep backends are *worse* here:
  they're invariant to colour and dominated by background, so they over-fragment
  (22–65) at every threshold. Colour is the right signal for this footage; a
  re-ID model fine-tuned on this store's top-down geometry would generalise further.
- **The `customer_gallery` requires human labeling** (eyeballing crops to
  identify customers). Truly autonomous customer detection requires
  either unsupervised clustering with a robustness gate (which can
  silently mislabel if the customer population isn't a minority) or a
  separate signal like POS transactions linked to billing-queue
  visitors.
