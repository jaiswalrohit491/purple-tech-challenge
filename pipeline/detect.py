"""Per-camera detection runner.

Usage:
    python -m pipeline.detect \\
        --store STORE_BLR_002 \\
        --camera CAM_ENTRY_01 \\
        --clip data/STORE_BLR_002/entry.mp4 \\
        --layout data/store_layout.json \\
        --clip-start 2026-03-03T10:00:00Z \\
        --out events/STORE_BLR_002_entry.jsonl

Pipeline per frame:
    1. YOLOv8n (person class only) with ByteTrack via Ultralytics.
    2. For each tracked detection, compute frame timestamp = clip_start + frame_idx/fps.
    3. Bottom-center of bbox is the floor reference.
    4. If the camera has an entry line, drive LineCrosser → emit ENTRY/EXIT.
    5. If the camera has zones, drive DwellTracker → emit ZONE_*.
    6. For billing camera, count people in queue zone → stamp queue_depth on
       any BILLING_QUEUE_JOIN.
    7. Lazily classify staff on a representative crop, cache per visitor.

The runner emits *raw* events with a per-camera visitor_id namespace
(`{camera_id}#{track_id}`). The post-pass (`pipeline.tracker`) collapses these
across cameras and emits REENTRY where appropriate.
"""
from __future__ import annotations

import argparse
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

from .emit import JsonlWriter, make_event
from .layout import Camera, Store, load_layout
from .queue_depth import QueueDepthCounter
from .staff_reid import GalleryStaffClassifier
from .zones import DwellTracker, LineCrosser, build_zone_polygons, point_in_zone


def parse_iso(s: str) -> datetime:
    return datetime.fromisoformat(s.replace("Z", "+00:00"))


def run_camera(
    *,
    store: Store,
    camera: Camera,
    clip_path: Path,
    clip_start: datetime,
    out_writer: JsonlWriter,
    weights: str = "yolov8n.pt",
    person_conf: float = 0.35,
    skip_frames: int = 0,
    realtime: bool = False,
    ingest_url: str | None = None,
) -> dict:
    """Process one clip from one camera. Returns counters."""
    try:
        import cv2
        from ultralytics import YOLO
    except ImportError as e:
        print(f"detect: pipeline deps missing ({e}). "
              f"Install with `pip install -e \".[pipeline]\"`", file=sys.stderr)
        return {"emitted": 0, "error": "missing-deps"}

    cap = cv2.VideoCapture(str(clip_path))
    if not cap.isOpened():
        return {"emitted": 0, "error": f"cannot open {clip_path}"}
    fps = cap.get(cv2.CAP_PROP_FPS) or 15.0

    model = YOLO(weights)
    zone_polys = build_zone_polygons(camera.zones)
    queue_zone = next((p for zid, p in zone_polys.items() if "QUEUE" in zid.upper()), None)
    queue_counter = QueueDepthCounter(queue_zone) if queue_zone else None

    crosser = (
        LineCrosser(a=camera.entry_line[0], b=camera.entry_line[1],
                    inward_normal=camera.entry_normal_inward or (0.0, 1.0))
        if camera.entry_line else None
    )
    dwell = DwellTracker()
    # Pure-data, uniform-only staff classifier. Gallery is whatever is in
    # /data/staff_gallery/ — see pipeline/staff_reid.py for the embedding +
    # cosine matching logic. No external API calls.
    import os
    staff = GalleryStaffClassifier(
        gallery_dir=os.getenv("STAFF_GALLERY_DIR", "/data/staff_gallery"),
        distance_threshold=float(os.getenv("STAFF_DISTANCE_THRESHOLD", "0.30")),
    )
    session_seq: dict[str, int] = {}
    crop_seen: set[str] = set()
    emitted = 0

    # Use the Ultralytics streaming tracker API.
    results = model.track(
        source=str(clip_path),
        stream=True,
        classes=[0],          # person only
        conf=person_conf,
        tracker="bytetrack.yaml",
        persist=True,
        verbose=False,
    )

    sender = _build_sender(realtime, ingest_url)

    for frame_idx, r in enumerate(results):
        if skip_frames and frame_idx % (skip_frames + 1) != 0:
            continue
        ts = clip_start + timedelta(seconds=frame_idx / fps)

        bottoms_this_frame: list[tuple[float, float]] = []
        boxes = r.boxes
        if boxes is None or boxes.id is None:
            continue

        for box in boxes:
            track_id = int(box.id.item())
            visitor_id = f"{camera.camera_id}#{track_id}"
            session_seq[visitor_id] = session_seq.get(visitor_id, 0) + 1

            x1, y1, x2, y2 = (float(v) for v in box.xyxy[0].tolist())
            bottom = ((x1 + x2) / 2.0, y2)
            bottoms_this_frame.append(bottom)
            det_conf = float(box.conf.item()) if box.conf is not None else 0.5

            # Staff classification: one crop per visitor, lazy.
            # If the camera is configured as staff-only (e.g. back office),
            # short-circuit to is_staff=True without spending a VLM call.
            is_staff = False
            staff_conf = 0.0
            if camera.force_staff:
                is_staff = True
                staff_conf = 0.99
            elif visitor_id not in crop_seen and (y2 - y1) >= 160 and (x2 - x1) >= 80:
                crop_seen.add(visitor_id)
                crop_jpeg = _encode_crop(r.orig_img, x1, y1, x2, y2)
                verdict = staff.classify(visitor_id, crop_jpeg)
                is_staff = verdict.is_staff
                staff_conf = verdict.confidence

            # 1. Line crossing — entry / exit.
            if crosser is not None:
                kind = crosser.crossed(track_id, bottom)
                if kind in ("ENTRY", "EXIT"):
                    ev = make_event(
                        store_id=store.store_id, camera_id=camera.camera_id,
                        visitor_id=visitor_id, event_type=kind, timestamp=ts,
                        is_staff=is_staff, confidence=min(det_conf, 0.99),
                        session_seq=session_seq[visitor_id],
                    )
                    if ev:
                        out_writer.write(ev); sender(ev); emitted += 1

            # 2. Zone transitions — ZONE_ENTER / ZONE_EXIT / ZONE_DWELL.
            zone_id = point_in_zone(zone_polys, bottom) if zone_polys else None
            for kind, dwell_ms, zid in dwell.update(visitor_id, zone_id, ts):
                qd = None
                if kind == "ZONE_ENTER" and queue_counter is not None and "BILL" in (zid or "").upper():
                    # Treat zone entry into billing queue as BILLING_QUEUE_JOIN.
                    qd = queue_counter.step(bottoms_this_frame)
                    join_ev = make_event(
                        store_id=store.store_id, camera_id=camera.camera_id,
                        visitor_id=visitor_id, event_type="BILLING_QUEUE_JOIN",
                        timestamp=ts, zone_id=zid, is_staff=is_staff,
                        confidence=min(det_conf, 0.99), queue_depth=qd,
                        sku_zone=zid, session_seq=session_seq[visitor_id],
                    )
                    if join_ev:
                        out_writer.write(join_ev); sender(join_ev); emitted += 1
                    continue

                ev = make_event(
                    store_id=store.store_id, camera_id=camera.camera_id,
                    visitor_id=visitor_id, event_type=kind, timestamp=ts,
                    zone_id=zid, dwell_ms=dwell_ms, is_staff=is_staff,
                    confidence=min(det_conf, 0.99),
                    sku_zone=zid, session_seq=session_seq[visitor_id],
                )
                if ev:
                    out_writer.write(ev); sender(ev); emitted += 1

        # Keep the queue counter's smoothing window primed on every billing
        # frame where a ZONE_ENTER did NOT already call step(). We detect that
        # by looking up the queue zone by any case-insensitive name match
        # containing "QUEUE", not the literal string "BILLING_QUEUE" — earlier
        # versions double-stepped on every ZONE_ENTER when the operator named
        # the zone differently.
        queue_zone_name = next((zid for zid in zone_polys if "QUEUE" in zid.upper()), None)
        if queue_counter and queue_zone_name is None:
            queue_counter.step(bottoms_this_frame)

    cap.release()
    return {"emitted": emitted, "store": store.store_id, "camera": camera.camera_id}


def _encode_crop(orig_img, x1: float, y1: float, x2: float, y2: float) -> bytes | None:
    try:
        import cv2
        crop = orig_img[int(max(0, y1)):int(y2), int(max(0, x1)):int(x2)]
        if crop.size == 0:
            return None
        ok, buf = cv2.imencode(".jpg", crop, [cv2.IMWRITE_JPEG_QUALITY, 80])
        return bytes(buf) if ok else None
    except Exception:
        return None


def _build_sender(realtime: bool, ingest_url: str | None):
    """Return a function (event_dict) -> None that POSTs in realtime mode, else noop."""
    if not realtime or not ingest_url:
        return lambda _ev: None
    import httpx
    client = httpx.Client(timeout=5.0, base_url=ingest_url)
    def send(ev):
        try:
            client.post("/events/ingest", json=[ev])
        except Exception as e:
            print(f"detect: realtime send failed: {e}", file=sys.stderr)
    return send


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--store", required=True)
    p.add_argument("--camera", required=True)
    p.add_argument("--clip", required=True, type=Path)
    p.add_argument("--layout", required=True, type=Path)
    p.add_argument("--clip-start", required=True, help="ISO-8601 UTC start time of clip")
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--weights", default="yolov8n.pt")
    p.add_argument("--conf", type=float, default=0.35)
    p.add_argument("--skip-frames", type=int, default=0,
                   help="Process 1 of every (skip+1) frames; 0 = every frame.")
    p.add_argument("--realtime", action="store_true",
                   help="POST each event to --ingest-url as it's emitted.")
    p.add_argument("--ingest-url", default="http://localhost:8000")
    args = p.parse_args()

    layout = load_layout(args.layout)
    store = layout.get(args.store)
    if not store:
        print(f"detect: store {args.store} not in layout", file=sys.stderr); return 1
    camera = store.camera(args.camera)
    if not camera:
        print(f"detect: camera {args.camera} not in store {args.store}", file=sys.stderr); return 1

    with JsonlWriter(args.out, live=args.realtime) as w:
        result = run_camera(
            store=store, camera=camera, clip_path=args.clip,
            clip_start=parse_iso(args.clip_start),
            out_writer=w, weights=args.weights, person_conf=args.conf,
            skip_frames=args.skip_frames,
            realtime=args.realtime, ingest_url=args.ingest_url,
        )
    import json as _json
    print(_json.dumps(result))
    return 0 if "error" not in result else 1


if __name__ == "__main__":
    sys.exit(main())
