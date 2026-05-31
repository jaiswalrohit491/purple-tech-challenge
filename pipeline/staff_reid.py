"""Pure-data staff classifier.

Architecture:
  1. A small gallery of confirmed-staff crops lives in `data/staff_gallery/*.jpg`.
     The gallery is built once per store by `pipeline/build_staff_gallery.py`
     using a stationary-track heuristic on the billing camera, then visually
     reviewed.
  2. At inference time, each visitor's first usable crop is embedded with a
     pre-trained torchvision ResNet50 (final classification head removed, so
     the output is the 2048-dim avg-pool feature vector — generic visual
     similarity, no person-Re-ID fine-tuning needed for a demo).
  3. Cosine distance to the nearest gallery entry. Below threshold → staff.

No external APIs, no labeled training set beyond the handful of staff crops
the operator approves. Replaces the previous VLM/HSV combo in
`pipeline/staff_vlm.py` for environments where outbound API calls are not
allowed (the brief's case).
"""
from __future__ import annotations

import io
from dataclasses import dataclass
from pathlib import Path
from threading import Lock

import numpy as np


@dataclass(frozen=True)
class StaffVerdict:
    is_staff: bool
    confidence: float
    reason: str
    source: str  # "gallery" | "no-gallery" | "no-crop" | "error"


class GalleryStaffClassifier:
    """Cosine-similarity classifier against a small store-specific gallery.

    Threshold: cosine distance < 0.30 ⇒ staff. Tunable per store by passing
    `distance_threshold`. The default is intentionally conservative so a
    customer wearing similar dark clothing isn't false-positived — the cost
    of missing a staff member (slight inflation of customer count) is much
    lower than the cost of a customer being silently dropped from metrics.
    """

    def __init__(self, gallery_dir: str | Path, distance_threshold: float = 0.30):
        self.gallery_dir = Path(gallery_dir)
        self.distance_threshold = distance_threshold
        self._lock = Lock()
        self._cache: dict[str, StaffVerdict] = {}
        self._model = None
        self._embedder = None
        self._gallery_embeddings: np.ndarray | None = None
        self._gallery_paths: list[Path] = []
        self._loaded = False

    def _ensure_loaded(self) -> None:
        if self._loaded:
            return
        self._loaded = True  # idempotent guard even on failure

        # Use the shared person embedder (OSNet re-ID, ResNet50 fallback) so
        # detection-time staff tagging uses the SAME vectors as the downstream
        # cross-camera merge in cluster_and_label.py.
        from .reid import get_embedder

        self._embedder = get_embedder()
        if not self._embedder.available:
            return
        self._model = object()  # sentinel: backend is live

        if not self.gallery_dir.is_dir():
            return
        paths = sorted([p for p in self.gallery_dir.iterdir()
                        if p.suffix.lower() in (".jpg", ".jpeg", ".png")])
        if not paths:
            return

        embeddings = []
        for p in paths:
            try:
                emb = self._embedder.embed_path(p)
                if emb is not None:
                    embeddings.append(emb)
                    self._gallery_paths.append(p)
            except Exception as e:
                print(f"staff_reid: skipping gallery image {p.name}: {e}")
        if embeddings:
            self._gallery_embeddings = np.vstack(embeddings)

    def _embed_pil(self, pil_img) -> np.ndarray:
        return self._embedder.embed_pil(pil_img)

    def classify(self, visitor_id: str, crop_jpeg: bytes | None) -> StaffVerdict:
        with self._lock:
            cached = self._cache.get(visitor_id)
        if cached is not None:
            return StaffVerdict(cached.is_staff, cached.confidence,
                                cached.reason, "gallery-cached")

        self._ensure_loaded()
        verdict = self._classify_uncached(crop_jpeg)
        with self._lock:
            self._cache[visitor_id] = verdict
        return verdict

    def _classify_uncached(self, crop_jpeg: bytes | None) -> StaffVerdict:
        if crop_jpeg is None:
            return StaffVerdict(False, 0.0, "no-crop", "no-crop")
        if self._model is None:
            return StaffVerdict(False, 0.0, "model unavailable", "error")
        if self._gallery_embeddings is None or len(self._gallery_embeddings) == 0:
            return StaffVerdict(False, 0.0, "empty gallery", "no-gallery")

        try:
            from PIL import Image
            img = Image.open(io.BytesIO(crop_jpeg)).convert("RGB")
            query = self._embed_pil(img)
        except Exception as e:
            return StaffVerdict(False, 0.0, f"embed-error:{type(e).__name__}", "error")

        # Cosine distance = 1 - cosine_similarity. Both vectors are unit-norm.
        sims = self._gallery_embeddings @ query
        max_sim = float(sims.max())
        min_dist = 1.0 - max_sim
        nearest_idx = int(sims.argmax())
        nearest = self._gallery_paths[nearest_idx].name

        is_staff = min_dist < self.distance_threshold
        return StaffVerdict(
            is_staff=is_staff,
            confidence=max(0.0, min(1.0, max_sim)),
            reason=f"d={min_dist:.3f} thr={self.distance_threshold} nearest={nearest}",
            source="gallery",
        )

    @property
    def gallery_size(self) -> int:
        return len(self._gallery_paths)
