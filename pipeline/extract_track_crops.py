"""Save one representative crop per YOLO+ByteTrack track.

No trajectory heuristic, no zone heuristic, no location filter. We just want
to *see* every unique person the model tracked, so we can pick out the staff
by their uniform alone.

For each tracked person, we save a single mid-track crop to
`<out-dir>/<camera>_track<id>.jpg`. The operator (or a downstream classifier)
then decides which tracks are staff by looking at clothing alone.

Usage:
    python -m pipeline.extract_track_crops \\
        --layout /data/store_layout.json --camera CAM_05 \\
        --out /events/track_crops/CAM_05
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

from .layout import load_layout


def run(layout_path: Path, store_id: str, camera_id: str, data_dir: Path,
        out_dir: Path, conf: float = 0.3, min_box_h: int = 120) -> dict:
    try:
        import cv2
        from ultralytics import YOLO
    except ImportError as e:
        return {"error": f"missing deps: {e}"}

    layout = load_layout(layout_path)
    store = layout.get(store_id)
    cam = store.camera(camera_id) if store else None
    if not cam or not cam.clip_path:
        return {"error": f"camera {camera_id} not in layout or has no clip_path"}
    clip_path = data_dir / cam.clip_path
    if not clip_path.exists():
        return {"error": f"no clip at {clip_path}"}

    # Collect samples per track: (frame_idx, x1, y1, x2, y2, det_conf).
    samples: dict[int, list[tuple[int, float, float, float, float, float]]] = defaultdict(list)
    model = YOLO("yolov8n.pt")
    results = model.track(
        source=str(clip_path), stream=True, classes=[0], conf=conf,
        tracker="bytetrack.yaml", persist=True, verbose=False,
    )
    for fi, r in enumerate(results):
        boxes = r.boxes
        if boxes is None or boxes.id is None:
            continue
        for box in boxes:
            tid = int(box.id.item())
            x1, y1, x2, y2 = (float(v) for v in box.xyxy[0].tolist())
            if (y2 - y1) < min_box_h:
                continue
            dc = float(box.conf.item()) if box.conf is not None else 0.5
            samples[tid].append((fi, x1, y1, x2, y2, dc))

    out_dir.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(clip_path))
    saved = []
    for tid, pts in sorted(samples.items()):
        if not pts:
            continue
        # Pick the sample with the largest bbox area * confidence — most
        # likely to give us a clean, well-framed crop of the person.
        scored = [(p[4] - p[2]) * (p[3] - p[1]) * p[5] for p in pts]
        best = pts[scored.index(max(scored))]
        fi, x1, y1, x2, y2, dc = best
        cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
        ok, frame = cap.read()
        if not ok:
            continue
        mx, my = int((x2 - x1) * 0.08), int((y2 - y1) * 0.04)
        cx1 = max(0, int(x1) - mx); cy1 = max(0, int(y1) - my)
        cx2 = min(frame.shape[1] - 1, int(x2) + mx)
        cy2 = min(frame.shape[0] - 1, int(y2) + my)
        crop = frame[cy1:cy2, cx1:cx2]
        if crop.size == 0:
            continue
        name = f"{camera_id}_track{tid:03d}.jpg"
        cv2.imwrite(str(out_dir / name), crop, [cv2.IMWRITE_JPEG_QUALITY, 85])
        saved.append({
            "name": name, "track_id": tid, "frames": len(pts),
            "best_frame": fi, "best_conf": round(dc, 3),
            "bbox": [int(x1), int(y1), int(x2), int(y2)],
        })
    cap.release()

    return {
        "camera_id": camera_id, "tracks_total": len(samples),
        "crops_saved": len(saved), "out_dir": str(out_dir),
        "saved": saved,
    }


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--layout", required=True, type=Path)
    p.add_argument("--store", default="STORE_001")
    p.add_argument("--camera", required=True)
    p.add_argument("--data-dir", type=Path, default=Path("/data"))
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--conf", type=float, default=0.3)
    p.add_argument("--min-box-h", type=int, default=120)
    args = p.parse_args()

    r = run(args.layout, args.store, args.camera, args.data_dir, args.out,
            args.conf, args.min_box_h)
    print(json.dumps(r, indent=2))
    return 0 if "error" not in r else 1


if __name__ == "__main__":
    sys.exit(main())
