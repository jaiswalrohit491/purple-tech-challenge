"""Cluster all ByteTrack tracks across in-store cameras into N physical persons,
then label each as staff or customer using a dual gallery (staff_gallery +
customer_gallery).

The pipeline:
  1. For each `CAM_XX_trackYYY.jpg` crop in the extracted track-crops dirs,
     compute a ResNet50 ImageNet feature (the same backbone used by
     pipeline/staff_reid.py — no external API, no fine-tuning).
  2. Compare each track's embedding to the staff_gallery centroid and the
     customer_gallery centroid. The closer gallery wins → preliminary
     staff/customer label.
  3. Within the customer-labeled tracks, K-means with K = num_customers
     splits them into distinct customer identities (one cluster = one
     physical customer). Same for staff with K = num_staff.
  4. Output a single merged JSONL where visitor_id is the canonical
     person ID (CUSTOMER_01, STAFF_03, …) and is_staff reflects the label.

This is the "merge tracks within and across cameras" pass that the brief asks
for. It replaces tracker.py's per-camera re-entry rewrite for the in-store
cameras because once tracks are merged by appearance, REENTRY collapses into
"same visitor_id appears more than once" naturally.

Usage:
    python -m pipeline.cluster_and_label \\
        --events-dir /events \\
        --cameras CAM_01 CAM_02 CAM_05 \\
        --crops-dirs /events/track_crops/CAM_01 /events/track_crops/CAM_02 /events/track_crops/CAM_05 \\
        --staff-gallery /data/staff_gallery \\
        --customer-gallery /data/customer_gallery \\
        --num-staff 5 --num-customers 2 \\
        --out /events/STORE_001_merged.jsonl
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np

TRACK_RE = re.compile(r"(CAM_\d+)_track(\d+)")
VISITOR_RE = re.compile(r"(CAM_\d+)#(\d+)")


def load_resnet():
    import torch
    from torchvision import models, transforms

    weights = models.ResNet50_Weights.IMAGENET1K_V2
    m = models.resnet50(weights=weights)
    m.fc = torch.nn.Identity()
    m.eval()
    t = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])
    return m, t, torch


def embed(path: Path, model, transform, torch) -> np.ndarray:
    from PIL import Image
    with Image.open(path) as img:
        x = transform(img.convert("RGB")).unsqueeze(0)
        with torch.no_grad():
            v = model(x).numpy()[0]
    n = np.linalg.norm(v) or 1.0
    return v / n


def gallery_centroid(gallery_dir: Path, model, transform, torch) -> np.ndarray | None:
    if not gallery_dir.is_dir():
        return None
    embs = []
    for f in sorted(gallery_dir.iterdir()):
        if f.suffix.lower() in (".jpg", ".jpeg", ".png"):
            embs.append(embed(f, model, transform, torch))
    if not embs:
        return None
    c = np.mean(embs, axis=0)
    return c / (np.linalg.norm(c) or 1.0)


def _dbscan_cos(X: np.ndarray, eps: float, min_samples: int) -> np.ndarray:
    """DBSCAN on unit-norm vectors using cosine distance. Returns cluster labels."""
    n = X.shape[0]
    D = 1.0 - X @ X.T
    np.fill_diagonal(D, 0.0)
    labels = np.full(n, -1, dtype=int)
    visited = np.zeros(n, dtype=bool)
    cid = 0
    def neigh(i): return np.where(D[i] <= eps)[0]
    for i in range(n):
        if visited[i]:
            continue
        visited[i] = True
        N = neigh(i)
        if len(N) < min_samples:
            continue
        labels[i] = cid
        seeds = list(N)
        while seeds:
            j = seeds.pop()
            if not visited[j]:
                visited[j] = True
                Nj = neigh(j)
                if len(Nj) >= min_samples:
                    seeds.extend(int(k) for k in Nj if not visited[k])
            if labels[j] == -1:
                labels[j] = cid
        cid += 1
    return labels


def kmeans_cos(X: np.ndarray, k: int, iters: int = 100, seed: int = 42):
    """K-means with cosine distance on unit-norm vectors. Returns assignments + centroids."""
    rng = np.random.default_rng(seed)
    n = X.shape[0]
    if n <= k:
        # Each point is its own cluster.
        return np.arange(n), X.copy()

    # K-means++ initialisation.
    centers = [X[rng.integers(0, n)]]
    for _ in range(k - 1):
        sims = X @ np.array(centers).T
        d = 1.0 - sims.max(axis=1)
        d = np.maximum(d, 1e-9)
        probs = d / d.sum()
        centers.append(X[rng.choice(n, p=probs)])
    C = np.array(centers)

    for _ in range(iters):
        assign = (X @ C.T).argmax(axis=1)
        new_C = np.zeros_like(C)
        for i in range(k):
            mask = assign == i
            if mask.any():
                m = X[mask].mean(axis=0)
                new_C[i] = m / (np.linalg.norm(m) or 1.0)
            else:
                new_C[i] = C[i]
        if np.allclose(C, new_C, atol=1e-6):
            break
        C = new_C
    return assign, C


def parse_crops(crops_dirs: list[Path]) -> list[tuple[str, int, Path]]:
    out = []
    for d in crops_dirs:
        if not d.is_dir():
            continue
        for f in sorted(d.glob("*.jpg")):
            m = TRACK_RE.match(f.name)
            if m:
                out.append((m.group(1), int(m.group(2)), f))
    return out


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--events-dir", type=Path, default=Path("/events"))
    p.add_argument("--cameras", nargs="+", default=["CAM_01", "CAM_02", "CAM_05"])
    p.add_argument("--crops-dirs", nargs="+", type=Path, required=True)
    p.add_argument("--staff-gallery", type=Path, default=Path("/data/staff_gallery"))
    p.add_argument("--customer-gallery", type=Path, default=Path("/data/customer_gallery"))
    p.add_argument("--staff-threshold", type=float, default=0.45,
                   help="Cosine distance threshold against the staff gallery. "
                        "Used only when --customer-gallery is missing.")
    p.add_argument("--num-staff", type=int, default=0,
                   help="If 0, infer the number of staff identities via DBSCAN "
                        "on the staff-classified tracks (auto-K).")
    p.add_argument("--num-customers", type=int, default=0,
                   help="If 0, infer customer identity count via DBSCAN (auto-K).")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--store-id", default="STORE_001")
    args = p.parse_args()

    print("loading ResNet50…", file=sys.stderr)
    model, transform, torch = load_resnet()

    tracks = parse_crops(args.crops_dirs)
    if not tracks:
        print("no crops found", file=sys.stderr)
        return 1
    print(f"embedding {len(tracks)} tracks…", file=sys.stderr)
    X = np.vstack([embed(t[2], model, transform, torch) for t in tracks])

    staff_c = gallery_centroid(args.staff_gallery, model, transform, torch)
    cust_c = gallery_centroid(args.customer_gallery, model, transform, torch)
    if staff_c is None:
        print("staff gallery empty — cannot label", file=sys.stderr)
        return 1
    has_customer_gallery = cust_c is not None

    # Distances to gallery (1 - cosine).
    d_staff = 1.0 - X @ staff_c
    if has_customer_gallery:
        d_cust = 1.0 - X @ cust_c
        is_customer_track = d_cust < d_staff
    else:
        # No customer gallery — pure-data, uniform-only classification:
        # tracks within `staff_threshold` of the auto-built staff gallery are
        # staff; everything else is a customer. The threshold defaults to 0.45,
        # picked from the bimodal distance histogram in the operator's data.
        is_customer_track = d_staff > args.staff_threshold

    customer_idxs = np.where(is_customer_track)[0]
    staff_idxs = np.where(~is_customer_track)[0]
    print(f"preliminary: {len(staff_idxs)} staff tracks, {len(customer_idxs)} customer tracks",
          file=sys.stderr)

    # Sub-cluster each role into physical persons. K is either operator-provided
    # (--num-staff/--num-customers > 0) or inferred via DBSCAN (auto-K mode).
    canonical_per_track: dict[int, tuple[str, bool]] = {}

    def _subcluster(idxs: np.ndarray, target_k: int, role: str, is_staff: bool) -> int:
        if not len(idxs):
            return 0
        Xx = X[idxs]
        if target_k > 0:
            assign, _ = kmeans_cos(Xx, target_k)
            k_used = target_k
        else:
            # Auto-K via DBSCAN on cosine distance. Tighter eps than the
            # staff-gallery threshold because within a role we're separating
            # identities, not classes.
            assign = _dbscan_cos(Xx, eps=0.20, min_samples=2)
            # Re-number to start from 0 contiguously (DBSCAN may emit -1=noise).
            unique = sorted(set(int(a) for a in assign if a >= 0))
            mapping = {old: new for new, old in enumerate(unique)}
            next_id = len(unique)
            for k, a in enumerate(assign):
                a = int(a)
                if a < 0:
                    assign[k] = next_id  # each noise point = its own identity
                    next_id += 1
                else:
                    assign[k] = mapping[a]
            k_used = next_id
        for i, c in zip(idxs, assign):
            canonical_per_track[int(i)] = (f"{role}_{int(c) + 1:02d}", is_staff)
        return k_used

    n_staff_clusters = _subcluster(staff_idxs, args.num_staff, "STAFF", True)
    n_customer_clusters = _subcluster(customer_idxs, args.num_customers, "CUSTOMER", False)

    # Map (camera, track_id) -> canonical.
    track_map: dict[tuple[str, int], tuple[str, bool]] = {}
    for idx, (cam, tid, _) in enumerate(tracks):
        track_map[(cam, tid)] = canonical_per_track[idx]

    # Rewrite events.
    out_events: list[dict] = []
    unmatched = 0
    for cam in args.cameras:
        ef = args.events_dir / f"{args.store_id}_{cam}.jsonl"
        if not ef.exists():
            continue
        with ef.open() as f:
            for line in f:
                if not line.strip():
                    continue
                e = json.loads(line)
                m = VISITOR_RE.match(e["visitor_id"])
                if not m:
                    out_events.append(e); continue
                key = (m.group(1), int(m.group(2)))
                if key not in track_map:
                    unmatched += 1
                    continue
                canon, is_staff = track_map[key]
                e["visitor_id"] = canon
                e["is_staff"] = is_staff
                # Stamp the source track into metadata for auditing.
                meta = dict(e.get("metadata") or {})
                meta["source_track"] = f"{key[0]}#{key[1]}"
                e["metadata"] = meta
                out_events.append(e)

    # Synthesise one ENTRY event per canonical person at their earliest
    # appearance, so /metrics.unique_visitors (which filters event_type in
    # ENTRY/REENTRY) returns the right customer count without a SQL change.
    earliest: dict[str, dict] = {}
    for e in out_events:
        canon = e["visitor_id"]
        if canon not in earliest or e["timestamp"] < earliest[canon]["timestamp"]:
            earliest[canon] = e
    import uuid as _uuid
    entry_events = []
    for canon, e in earliest.items():
        if e.get("is_staff"):  # entries are only synthesised for customers
            continue
        entry_events.append({
            "event_id": str(_uuid.uuid4()),
            "store_id": e["store_id"],
            "camera_id": e["camera_id"],
            "visitor_id": canon,
            "event_type": "ENTRY",
            "timestamp": e["timestamp"],
            "zone_id": None,
            "dwell_ms": 0,
            "is_staff": False,
            "confidence": 0.9,
            "metadata": {
                "queue_depth": None, "sku_zone": None, "session_seq": 0,
                "synthetic": True, "reason": "first appearance after track merging",
            },
        })
    out_events.extend(entry_events)
    out_events.sort(key=lambda e: e["timestamp"])

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w") as f:
        for e in out_events:
            f.write(json.dumps(e, separators=(",", ":")) + "\n")

    # Summary.
    canon_role = {}
    for e in out_events:
        canon_role[e["visitor_id"]] = "STAFF" if e["is_staff"] else "CUSTOMER"
    customers = sorted([c for c, r in canon_role.items() if r == "CUSTOMER"])
    staff = sorted([c for c, r in canon_role.items() if r == "STAFF"])

    print(json.dumps({
        "tracks_in": len(tracks),
        "unmatched_events": unmatched,
        "events_out": len(out_events),
        "synthetic_entries": len(entry_events),
        "customers": customers,
        "staff": staff,
        "n_customers": len(customers),
        "n_staff": len(staff),
        "out": str(args.out),
    }, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
