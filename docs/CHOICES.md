# Choices

Design decisions documented as the challenge requires: options considered,
what an LLM suggested, what I chose, **why**, and what would change the call.

## 1. Detection model — YOLOv8n + ByteTrack

**Options considered.**
- YOLOv8n with the built-in ByteTrack tracker (Ultralytics one-liner).
- RT-DETR with a separate StrongSORT pipeline.
- MediaPipe Pose (person presence only, no proper tracking).

**What AI suggested.** Claude leaned toward RT-DETR for its transformer-based
accuracy on crowded scenes and recommended adding torchreid OSNet on top of
StrongSORT for the Re-ID surface. The reasoning was correct on accuracy but
underweighted three real costs in a 48-hour window: model-export friction
(no ByteTrack integration out of the box), CPU inference latency on 1080p15
clips, and the absence of a community-vetted "RT-DETR + StrongSORT" recipe.

**What I chose.** YOLOv8n with Ultralytics' built-in ByteTrack. Ultralytics
exposes tracking as a single argument (`model.track(..., persist=True)`),
which means we get tracker integration for free and can spend the saved time
on the harder problems — Re-ID for the `REENTRY` event type and cross-camera
deduplication. We still use ResNet50 embeddings, but as a *post-pass* over
the tracker output rather than as part of the live tracker loop.

**Why.** The grading rubric rewards correct events, schema compliance, and
edge-case handling — not raw detection mAP. A slightly less accurate tracker
that produces clean, consistent visitor IDs scores higher than a perfect
detector with a hand-rolled tracker we don't have time to validate.

**What would change my mind.** A second day of headroom, or a clip with
denser crowds (>15 people per frame) where ByteTrack's IoU-based association
breaks down. In production with GPU budget, RT-DETR + StrongSORT is the
right call.

## 2. Event schema — flat shape with JSONB metadata

**Options considered.**
- Flat `Event` with a JSONB `metadata` field for type-specific extras (chosen).
- Discriminated union of typed event classes (`EntryEvent`, `DwellEvent`, …)
  with a per-event-type table per the parlance of event sourcing.

**What AI suggested.** Both approaches got airtime. Claude initially proposed
the discriminated union for "stronger typing" but conceded that the SQL
surface fights it — funnel and metrics queries aggregate *across* event
types in the same window, so a typed union forces UNION ALL queries or
materialized views.

**What I chose.** One flat table, one Pydantic model, JSONB metadata for the
fields that vary by event type (`queue_depth` only meaningful for
`BILLING_QUEUE_JOIN`, `sku_zone` only for zone events, `session_seq`
populated for ordering, `embedding`/`source_track`/`synthetic` populated by
the cross-camera merger).

**Why.** Three reasons. (a) Schema evolution is cheap: adding a new event
type means adding it to a `Literal` and writing one query. (b) The events
table indexes do real work — `(store_id, ts)` serves four endpoints.
(c) The Pydantic model's per-field validators catch the semantic errors a
SQL constraint cannot (e.g. `zone_id` required for zone events but null for
entry/exit).

**What would change my mind.** A second consumer of the event stream with
materially different access patterns (e.g. a streaming analytics layer that
projects per-event-type aggregates). At that point per-type tables earn
their keep.

## 3. Staff classification — fully autonomous (pivoted twice: VLM → manual gallery → auto-discovery)

**Options considered.**
- **Claude Sonnet vision call** per visitor with a prompt describing the
  Purplle uniform. Cached by `visitor_id`.
- **OSNet / torchreid** with a small staff gallery — designed for person
  re-ID, would generalise well across angles.
- **ResNet50 ImageNet** features + cosine match against a manually-built
  staff gallery (chosen). Generic image-similarity, no Re-ID fine-tuning.
- **HSV color histogram** match against a uniform color profile.
- **Zone-based** — anyone behind the till is staff. **Rejected** because
  the brief states staff move through all customer zones.

**What AI suggested.** Initial Claude review recommended the VLM path —
zero training data, adapts to any uniform via prompt, ~$0.005/visitor with
caching. Solid for a production system; the prompt approach is also easy to
update when uniforms change.

**Operational note**: the production code path in `pipeline/run.sh` uses
**single-gallery mode** (staff gallery only — customers are defined as "not
matching staff"). The `cluster_and_label.py` script also supports a
dual-gallery mode with an explicit `customer_gallery/`, but the run.sh
orchestration deliberately passes a non-existent customer-gallery path so
the simpler single-gallery branch runs. The dual-gallery code is kept for
operators who want to seed a customer cluster from a handful of confirmed
crops; not the default because it needs manual labels which the auto path
sidesteps.

**What I chose, after TWO pivots.** Fully autonomous discovery:
1. Per-camera YOLO+ByteTrack produces tracks.
2. `pipeline/extract_track_crops.py` saves one representative crop per track.
3. **`pipeline/auto_setup.py` runs DBSCAN** (cosine distance, eps=0.30,
   min_samples=4) on the ResNet50 embeddings of all crops. The largest
   visually-coherent cluster is by definition the uniform group — staff —
   because customers are visually diverse and staff aren't.
4. The 10 crops closest to that cluster's centroid become the auto-built
   `data/staff_gallery/`.
5. Future visitors are classified by cosine distance to that gallery.

**Why.** Four reasons. (a) **Operator constraint**: "no external APIs"
ruled out the VLM. (b) **Operator constraint v2**: "don't ask me to label
crops manually — derive staff from dress code." DBSCAN-on-uniform satisfies
that. (c) **Reproducibility**: the gallery + threshold + crops are all on
disk, so a reviewer can re-run the classifier and get the same answer; a
VLM's behaviour can drift between versions. (d) **No per-visitor inference
cost** — important at 40 stores × hundreds of visitors per hour.

The first manual-gallery version produced only 1 staff hit because the 4
seed crops were all CAM_05 top-down views; oblique CAM_01/02 angles read
as too far in embedding space. Expanding to 10 seeds across angles fixed
that locally but didn't scale. The DBSCAN auto-discovery fixed it
permanently: the gallery is whatever clusters in the footage itself.

**What would change my mind.** A store where the "largest visually-coherent
cluster" is actually customers (e.g., a uniformed school visit). The
auto-discovery would mis-identify; we'd need either a human-in-the-loop
confirmation step or a separate weak-supervision signal (e.g., POS
salesperson IDs as anchors — see DESIGN.md §8 for the production roadmap).

## 4. K-means K — operator-provided, not auto-inferred

**Options considered.**
- **Operator-provided K** — pass `--num-staff` and `--num-customers` as CLI
  parameters reflecting the ground truth (chosen).
- **Auto K via elbow/silhouette** — fit k-means for K=2..15 and pick the
  inflection point on the inertia curve.
- **DBSCAN with a distance threshold** — emergent cluster count.

**What AI suggested.** The auto-K path. Claude argued correctly that
operator-provided K is a partial cheat — it carries information the
classifier itself doesn't compute, so the system is less reproducible on a
different store or clip.

**What I chose.** Both paths supported in `pipeline/cluster_and_label.py`:

- **`--num-staff 0 --num-customers 0`**: DBSCAN within each role group
  (eps=0.20, min_samples=2) infers K from the data. No operator input.
- **`--num-staff N --num-customers M`**: K-means with operator-supplied K.

Default in `run.sh` is auto-K (`NUM_STAFF=0 NUM_CUSTOMERS=0`).

**Why both.** Three reasons. (a) **The silhouette signal is poor on this
data**: most staff wear identical uniforms; ResNet50 embeddings of the same
uniform from different angles cluster very tightly. With our 2-min clip,
DBSCAN at eps=0.20 over-fragments (48 staff identities); at eps=0.30 it
under-merges (1 staff identity). The "right" eps is data-dependent.
(b) **Operator-provided K is a clean override** when ground truth is
known — e.g. a retail manager who knows there are 5 people on shift.
(c) **Architecturally** both paths exist; what the system can compute
autonomously is the *classification* (staff vs customer), not necessarily
the *count of distinct staff identities*. The classification is what
matters for `unique_visitors` (the brief's North Star).

**What would change my mind.** A real person-Re-ID model (OSNet fine-tuned
on retail footage) would tighten the within-uniform embedding distances so
auto-K's silhouette becomes informative. Then the operator override goes
away. For the demo, ResNet50 ImageNet features are good enough for
class-level (staff vs customer) but not identity-level.

## 5. Conversion as a three-tier evidence ladder, not a single number

**Options considered.**
- Single `conversion_rate` per the brief's definition (BILLING_QUEUE_JOIN +
  POS within 5 min).
- Brand-aware single rate: customer's zone visits joined to POS by
  matching brand within 30 min.
- A three-tier ladder making evidence quality explicit.

**What AI suggested.** The single-rate path — match the rubric verbatim,
return 0 when the data doesn't support it, move on.

**What I chose, after operator pushback.** A three-tier ladder, returned
together by `/metrics`:

| Tier | Definition | Failure mode it guards against |
|---|---|---|
| `verified_purchase_rate` | Full trajectory: ENTRY → BILLING → next-EXIT within 5 min, POS between bill and exit | Customer loitered near the till without paying; system over-credits |
| `conversion_rate` (brief) | BILLING_QUEUE_JOIN + same-store POS within 5 min | Same as above but accepted by the rubric |
| `potential_conversion_rate` | Brand-zone visit + same-brand POS within 30 min in same store | Clip ends before customer reaches the till; correlational only |

Always `verified ≤ confirmed ≤ potential`. The dashboard headlines
`verified` (most defensible). `/metrics` returns all three so reviewers
can inspect the evidence ladder.

**Why.** Two reasons. (a) **Honesty matters more than a high number.** For
the ST1008 2-min clip, the verified and confirmed rates are both 0
(nobody's full ENTRY+BILL+EXIT trajectory is captured; nobody's billing
event has a POS within 5 min). The 100% potential rate is a
correlation-only signal, clearly labelled. Reporting 100% as if it were
confirmed would have failed an operator review. (b) **Different stakeholders
ask different questions.** A store manager wants to know who *probably*
converted (potential, brand-aware). A finance auditor wants confirmed
purchases only (verified). One number can't serve both.

**What would change my mind.** A POS schema that linked transactions to
visitor IDs directly (e.g., a loyalty-card check-in at the till). Then
attribution becomes exact and the three tiers collapse to one.

## 6. Pipeline orchestration — auto-bootstrap in run.sh

**Options considered.**
- Manual setup: operator runs `auto_setup`, then `detect`, then
  `cluster_and_label`, then `correlate`, then `replay` by hand.
- Make-style with explicit targets and dependencies.
- One `run.sh` that detects what's missing and bootstraps.

**What AI suggested.** Make-style with explicit dependencies. Robust but
adds a build-system concept and a Makefile to debug.

**What I chose.** `pipeline/run.sh` is a single Bash script with cold-start
detection at each step:

```
layout missing       → auto_setup writes default
detection JSONLs     → always re-run (cheap to redo)
track crops missing  → extract_track_crops runs
staff gallery empty  → auto_setup clusters crops
                       → cluster_and_label re-classifies
optional shift       → SHIFT=1 rebases timestamps to today
POS CSV present      → correlate loads + emits abandons
                       → replay POSTs to /events/ingest
```

**Why.** Three reasons. (a) **The acceptance gate is the bar.** Reviewers
have 10 minutes per submission per the framework PDF. The bootstrap MUST
work without setup steps the reviewer has to learn. (b) **Bash is honest
about what it does.** A reviewer can `cat pipeline/run.sh` and see the
pipeline; no hidden Make graph. (c) **Idempotent enough.** Track crops
aren't re-extracted if present; layout isn't overwritten; gallery isn't
rebuilt if non-empty. Re-running `./pipeline/run.sh` after a config tweak
re-does only the affected steps.

**What would change my mind.** A multi-stage pipeline with shared state
across runs and partial failures (e.g., process 40 stores nightly, retry
on failure per-store). At that point a real workflow engine (Airflow,
Dagster, Prefect) earns its place.
