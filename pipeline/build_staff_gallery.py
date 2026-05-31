"""Auto-extract candidate staff crops from CCTV footage.

The heuristic: **staff stay; customers transit.** A track that persists for
most of the clip's duration AND whose centroid hardly moves is overwhelmingly
likely to be a staff member at a fixed station (billing till, workstation,
etc.). This is a clean, defensible signal that requires no labels, no model,
and survives the brief's constraint that staff *also* move through customer
zones — because we only need to identify them *once*, on whichever camera
catches them stationary.

Output is written to `data/staff_gallery/<camera>_track<id>_<n>.jpg` — three
evenly-spaced crops per qualifying track. After running, eyeball the saved
images and delete any that aren't staff. Then the gallery classifier
(`pipeline/staff_reid.py`) generalises across cameras by appearance.

Usage:
    # Default: cam 5 (billing), the most reliable seed source.
    python -m pipeline.build_staff_gallery --layout data/store_layout.json --camera CAM_05

    # Override the dwell / variance thresholds if the default is too strict.
    python -m pipeline.build_staff_gallery --layout data/store_layout.json --camera CAM_05 \\
        --min-frames-frac 0.4 --max-stddev-px 80
"""
from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path
from statistics import stdev

from .layout import load_layout


def run(layout_path: Path, store_id: str, camera_id: str, data_dir: Path,
        gallery_dir: Path, min_frames_frac: float, max_stddev_px: float,
        crops_per_track: int) -> dict:
    try:
        import cv2
        from ultralytics import YOLO
    except ImportError as e:
        print(f"build_staff_gallery: pipeline deps missing ({e})", file=sys.stderr)
        return {"error": "missing-deps"}

    layout = load_layout(layout_path)
    store = layout.get(store_id)
    if not store:
        return {"error": f"store {store_id} not in layout"}
    cam = store.camera(camera_id)
    if not cam or not cam.clip_path:
        return {"error": f"camera {camera_id} has no clip_path"}

    clip_path = data_dir / cam.clip_path
    if not clip_path.exists():
        return {"error": f"clip not found at {clip_path}"}

    cap = cv2.VideoCapture(str(clip_path))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    min_frames = int(total_frames * min_frames_frac)

    model = YOLO("yolov8n.pt")

    # Per-track stats: list of (frame_idx, cx, cy, x1, y1, x2, y2).
    samples: dict[int, list[tuple[int, float, float, float, float, float, float]]] = defaultdict(list)

    results = model.track(
        source=str(clip_path), stream=True, classes=[0], conf=0.3,
        tracker="bytetrack.yaml", persist=True, verbose=False,
    )
    for frame_idx, r in enumerate(results):
        boxes = r.boxes
        if boxes is None or boxes.id is None:
            continue
        for box in boxes:
            tid = int(box.id.item())
            x1, y1, x2, y2 = (float(v) for v in box.xyxy[0].tolist())
            cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
            samples[tid].append((frame_idx, cx, cy, x1, y1, x2, y2))

    # Filter to stationary-long-dwell tracks.
    candidates: list[tuple[int, dict]] = []
    for tid, pts in samples.items():
        if len(pts) < min_frames:
            continue
        xs = [p[1] for p in pts]
        ys = [p[2] for p in pts]
        sx = stdev(xs) if len(xs) > 1 else 0.0
        sy = stdev(ys) if len(ys) > 1 else 0.0
        if sx > max_stddev_px or sy > max_stddev_px:
            continue
        candidates.append((tid, {
            "frames": len(pts), "stddev_x": round(sx, 1), "stddev_y": round(sy, 1),
            "mean_cx": round(sum(xs) / len(xs), 1),
            "mean_cy": round(sum(ys) / len(ys), 1),
            "pts": pts,
        }))

    if not candidates:
        return {
            "candidates": 0,
            "hint": "No tracks met the dwell + stationarity threshold. "
                    "Try `--min-frames-frac 0.3` or `--max-stddev-px 100`.",
            "total_tracks": len(samples),
            "long_tracks": sum(1 for p in samples.values() if len(p) >= min_frames),
        }

    # Save crops.
    gallery_dir.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(clip_path))
    saved = []
    for tid, meta in candidates:
        pts = meta["pts"]
        # Pick crops_per_track evenly-spaced samples from this track.
        idxs = [pts[int(i * (len(pts) - 1) / max(crops_per_track - 1, 1))]
                for i in range(crops_per_track)]
        for n, (fi, cx, cy, x1, y1, x2, y2) in enumerate(idxs):
            cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
            ok, frame = cap.read()
            if not ok:
                continue
            # Light margin so the box doesn't clip clothing edges.
            mx, my = int((x2 - x1) * 0.08), int((y2 - y1) * 0.04)
            cx1 = max(0, int(x1) - mx); cy1 = max(0, int(y1) - my)
            cx2 = min(frame.shape[1] - 1, int(x2) + mx)
            cy2 = min(frame.shape[0] - 1, int(y2) + my)
            crop = frame[cy1:cy2, cx1:cx2]
            if crop.size == 0:
                continue
            out_name = f"{camera_id}_track{tid:03d}_{n}.jpg"
            cv2.imwrite(str(gallery_dir / out_name), crop, [cv2.IMWRITE_JPEG_QUALITY, 85])
            saved.append({
                "name": out_name, "track_id": tid, "frame": fi,
                "centroid": [int(cx), int(cy)],
                "stddev_xy": [meta["stddev_x"], meta["stddev_y"]],
                "track_frames": meta["frames"],
            })
    cap.release()

    return {
        "store_id": store_id, "camera_id": camera_id, "clip": str(clip_path),
        "total_tracks": len(samples), "candidate_tracks": len(candidates),
        "crops_saved": len(saved), "gallery_dir": str(gallery_dir),
        "saved": saved,
    }


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--layout", required=True, type=Path)
    p.add_argument("--store", default="STORE_001")
    p.add_argument("--camera", default="CAM_05")
    p.add_argument("--data-dir", type=Path, default=Path("/data"))
    p.add_argument("--gallery-dir", type=Path, default=Path("/data/staff_gallery"))
    p.add_argument("--min-frames-frac", type=float, default=0.5,
                   help="Track must persist for this fraction of total frames.")
    p.add_argument("--max-stddev-px", type=float, default=50.0,
                   help="Centroid stddev (px) ceiling for 'stationary'.")
    p.add_argument("--crops-per-track", type=int, default=3)
    args = p.parse_args()

    result = run(args.layout, args.store, args.camera, args.data_dir,
                 args.gallery_dir, args.min_frames_frac, args.max_stddev_px,
                 args.crops_per_track)
    import json
    print(json.dumps(result, indent=2))
    return 0 if "error" not in result else 1


if __name__ == "__main__":
    sys.exit(main())
