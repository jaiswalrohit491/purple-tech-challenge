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
deduplication. Appearance embeddings (ResNet50 by default, OSNet re-ID opt-in — see §3) run as
a *post-pass* over the tracker output rather than as part of the live tracker loop.

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
- **OSNet (x1.0) trained on MSMT17** — a real person-re-ID network (chosen).
  Weights baked into the image at build from the HuggingFace mirror; ResNet50
  ImageNet is the automatic fallback if they're absent.
- **ResNet50 ImageNet** features + cosine match (the earlier cut, now the
  fallback). Generic image-similarity, no Re-ID fine-tuning — same-person
  fragments drift apart, so identity counts over-fragment.
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

**What I chose, after THREE pivots.** Fully autonomous discovery:
1. Per-camera YOLO+ByteTrack produces tracks.
2. `pipeline/extract_track_crops.py` saves one representative crop per track;
   a crop-quality gate (portrait aspect ≥1.3, short side ≥60px) drops
   occluded/partial boxes before embedding.
3. **`pipeline/reid.py` embeds each crop** — default is a **torso HSV colour
   histogram** (separates grey shirt / tan bag / dark uniform); deep backends
   `resnet50` and `osnet` (MSMT17, baked) are opt-in via `REID_BACKEND`.
4. **`pipeline/auto_setup.py` runs DBSCAN** on those embeddings with a
   **data-driven eps** (median k-th-nearest-neighbour distance) so the
   threshold adapts to whichever backend is live instead of a magic number.
   The largest visually-coherent cluster is taken as the uniform group; its
   10 most-central crops become `data/staff_gallery/`.
5. `pipeline/cluster_and_label.py` classifies staff/customer by gallery distance,
   then `pipeline/identity.py` resolves identities by **constrained spatiotemporal
   clustering on the colour signature** (same-camera cannot-link + tracklet
   stitching + topology-gated cross-camera links). **No `K`, no prior** — the
   count emerges (2 customers). See §4.

**Embedding backend.** Default is a **torso HSV colour histogram** — it captures
the grey-shirt/tan-bag/dark-uniform cue that a human uses. Deep backends
(`REID_BACKEND=resnet50|osnet`, OSNet weights baked) are kept as options, but on
this footage they over-fragment (22–65) because they're colour-invariant and
dominated by silhouette/shelf-background. Colour both separates the two customers
and merges each one's fragments.

**Result on this clip.** Colour resolves **2 customers** (grey + tan modes),
emergent and prior-free, while the 5 dark-uniformed staff collapse into one group.
That staff collapse is the honest limit: identical uniforms (and masked faces)
mean distinct staff *identities* aren't separable by any appearance signal — so
`staff_count` is a uniform-group count, not a headcount. It doesn't affect the
business metric: unique *customers* is measured correctly.

## 4. Defining uniqueness — torso colour, measured count (no prior)

**Count-source priority (entry gate > appearance).** The unique-visitor count is
taken from **entry-gate line-crossings** (CAM with `view==ENTRY`) whenever the gate
has entries — it's the canonical, higher-confidence footfall signal and always
wins. The appearance engine below is the **fallback**, used to count only when the
gate feed is insufficient (this clip: 0 entries, customers already inside). See
`identity.choose_count_source` and DESIGN.md §3.2b. The rest of this section is the
fallback's design.

**The question that fixed this:** *how are we even defining uniqueness?* The
earlier pipeline defined a person as a cluster of tracks whose **deep whole-body
embedding** matched. That is the wrong signature: ResNet/OSNet embeddings are
trained for invariance and dominated by silhouette, pose and the colourful shelf
backgrounds, so they discard the very cue that distinguishes shoppers — clothing
colour. Result: they neither separated the two customers nor merged each one's
fragments, and the count was either an injected prior (=2) or fragmentation noise
(22–65), stable at *no* threshold.

**Options considered.**
- **Torso colour histogram** (HSV of the central torso band) — captures the
  grey-shirt / tan-bag / dark-uniform distinction directly (chosen, default).
- Deep re-ID embedding (ResNet/OSNet) — kept as opt-in backends.
- Operator headcount prior (`K` / `expected_*`) — **removed**: it supplied the
  answer and could not survive real data.

**What I chose.** Uniqueness = a **torso colour signature**, resolved into physical
identities by `pipeline/identity.py` with spatiotemporal constraints (same-camera
temporal cannot-link, tracklet stitching, topology-gated cross-camera links). The
count **emerges**: on this clip, 2 distinct non-dark colour modes → **2 customers**;
the dark uniforms group as staff. Verified stable: matching-distance 0.35→4,
0.45→2, 0.55→1 — a small, sane range, versus the deep backends' 22–65 at every
setting. The 0.45 threshold is a re-ID *matching* parameter, not the count.

**Why this is genuinely prior-free.** No `K`, no `expected_*` anywhere (both deleted
from the layout and `run.sh`). The number of customers is computed from the
footage's colour structure; it varies with the input. The only operator config is
`camera_topology` (store geometry) and the colour matching threshold (a standard
re-ID knob) — neither encodes how many people there are.

**Honest limitation.** Colour groups by *clothing colour*, so 5 identically-uniformed
staff collapse to one identity — distinct staff headcount is not recoverable from
appearance here (identical uniforms, masked faces). `staff_count` is therefore a
uniform-group count. The business metric — unique *customers* — is measured
correctly (=2).

**Why not also fuse a deep body embedding (the obvious "add physical appearance")?**
I built it (`REID_BACKEND=fused`, colour ⊕ ResNet, weighted). On this footage it
makes things *worse*: the deep features aren't discriminative here (masked faces,
top-down, everyone reads as a generic "person"), so `d_deep` between the two
customers is ~0.2 and fusing it **dilutes** the colour gap and merges them (2→1).
It's kept for diverse/real data or a fine-tuned re-ID model, off by default.

**So how are two *identically-dressed* shoppers told apart?** Not by appearance —
by **movement**. `pipeline/identity.py`'s same-camera temporal **cannot-link** makes
two tracks that are on one camera *at the same time* provably different people
(one body, one place). Unit-tested: identical-looking + concurrent → 2 identities;
identical-looking + sequential → 1 (stitched). For two identically-dressed people
at *different* times, only a discriminative deep/re-ID signal could separate them —
the documented hard case. Cross-camera links are topology- and time-gated, and
zone continuity is used as a *constraint* (no teleport stitches), not a loosener.

**Re-entry.** A resolved customer absent from all cameras longer than
`--reentry-gap` (60 s) and returning gets a `REENTRY` event reusing the same
`visitor_id`, so the funnel/`unique_visitors` never double-count. (0 on this clip —
both customers are continuously present in the 2-minute window.)

**What would change my mind.** A re-ID model fine-tuned on this store's top-down
geometry, or per-frame body keypoints/colour-by-region, would let even
same-colour people be separated and would generalise to crowded, colour-diverse
stores where a single torso colour is less discriminative.

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
