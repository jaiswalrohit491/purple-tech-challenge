"""Person embedding backend — OSNet re-ID with a ResNet50 fallback.

The whole pipeline (staff classification, cross-camera identity merge) rests on
turning a person crop into a vector where the *same* person across cameras is
close and *different* people are far. ImageNet ResNet50 features were the first
cut, but they aren't trained for person re-identification, so same-person
fragments drift apart and identity counts over-fragment.

This module centralises embedding behind one `PersonEmbedder` so every stage
uses the same vectors:

  - **Primary: OSNet (x1.0) trained on MSMT17** — a real person-re-ID network.
    Weights are baked into the image at build time (see Dockerfile) from the
    HuggingFace mirror `kaiyangzhou/osnet`, so there is **no runtime download**
    and nothing to reach over the network. Input is 256x128 (the re-ID
    convention), ImageNet-normalised; the 512-dim global feature is returned.

  - **Fallback: torchvision ResNet50 (ImageNet)** — used automatically if the
    OSNet checkpoint is missing or fails to load. This keeps `docker compose up`
    and the acceptance gate working even if the weights were never baked
    (e.g. a build-time download failure). 2048-dim, 224x224 centre-crop.

Both backends return L2-normalised vectors so cosine distance = 1 - dot.
`PersonEmbedder.backend` reports which one is live, and the pipeline logs it so a
reviewer can see whether re-ID or the fallback produced a given run.
"""
from __future__ import annotations

import io
import os
from pathlib import Path

import numpy as np

DEFAULT_WEIGHTS = os.getenv("REID_WEIGHTS", "/opt/models/osnet_x1_0_msmt17.pth")

# Which appearance signature to use:
#   "color" (default) — torso HSV histogram. The discriminative cue on this
#       footage: separates differently-dressed people (grey shirt vs tan bag)
#       and merges each one's fragments. Needs no torch.
#   "fused"    — colour CONCATENATED with a deep body embedding (ResNet50),
#       weighted (REID_COLOR_WEIGHT, default 0.6). In principle the deep part
#       separates two *similarly*-dressed shoppers — but on this masked,
#       top-down footage the deep features aren't discriminative (they read both
#       customers as a generic "person") and DILUTE the colour signal, merging
#       the two customers. Kept for diverse/real data or a fine-tuned re-ID model.
#   "resnet50"/"osnet"/"auto" — deep features only.
# IMPORTANT: same-/different-person separation for *identically*-dressed people
# is delivered by the spatiotemporal CANNOT-LINK in pipeline/identity.py
# (concurrent tracks on one camera = different people), not by appearance — that
# is the robust signal here. Re-entry is handled in cluster_and_label.py (REENTRY).
REID_BACKEND = os.getenv("REID_BACKEND", "color").lower()
REID_COLOR_WEIGHT = float(os.getenv("REID_COLOR_WEIGHT", "0.6"))

# Imagenet normalisation, shared by both backends.
_MEAN = [0.485, 0.456, 0.406]
_STD = [0.229, 0.224, 0.225]


class PersonEmbedder:
    """Lazy-loaded person embedder. Prefers OSNet re-ID, falls back to ResNet50."""

    def __init__(self, weights_path: str | Path | None = None, device: str = "cpu"):
        self.weights_path = Path(weights_path or DEFAULT_WEIGHTS)
        self.device = device
        self._model = None
        self._transform = None
        self._torch = None
        self.backend: str | None = None  # "osnet-reid" | "resnet50" | None
        self.dim: int | None = None
        self._loaded = False

    # ------------------------------------------------------------------ load
    def _ensure_loaded(self) -> None:
        if self._loaded:
            return
        self._loaded = True  # idempotent even on failure

        # Colour-only backend needs no torch — it's a torso HSV histogram.
        if REID_BACKEND in ("color", "colour"):
            self.backend = "color-hsv"
            self.dim = 24
            self._model = object()  # sentinel: backend live, no torch model
            return

        try:
            import torch
            from torchvision import transforms
        except ImportError:
            # No torch: fused/deep backends degrade to colour-only.
            self.backend = "color-hsv"
            self.dim = 24
            self._model = object()
            return
        self._torch = torch

        # Fused = colour histogram + deep body embedding (the robust default).
        # The deep complement is ResNet50: its texture/body features are
        # *discriminative* (high variance), which is what we want to SEPARATE
        # two similarly-dressed shoppers. OSNet is identity-invariant and
        # collapses everyone on this footage, so it's a poor fusion partner here
        # (use REID_BACKEND=osnet explicitly for diverse stores).
        if REID_BACKEND in ("fused", "fuse"):
            self._load_resnet(transforms)
            if self._model is None:
                self.backend = "color-hsv"; self.dim = 24; self._model = object()
                return
            self._deep_backend = self.backend       # "resnet50"
            self.backend = f"fused(color+{self._deep_backend})"
            self.dim = 24 + self.dim                 # self.dim was the deep dim
            return

        # Single deep backend.
        if REID_BACKEND in ("osnet", "auto"):
            if self._try_load_osnet(transforms):
                return
            if REID_BACKEND == "osnet":
                print("reid: REID_BACKEND=osnet but weights unavailable; "
                      "falling back to ResNet50")
        self._load_resnet(transforms)

    def _color_signature(self, pil_img) -> np.ndarray:
        """L2-normalised torso HSV histogram (12 hue + 6 sat + 6 value bins).
        Cosine distance on this separates dark-uniform / grey / tan-khaki, etc."""
        import cv2
        rgb = np.asarray(pil_img.convert("RGB"))
        bgr = rgb[:, :, ::-1]
        h, w = bgr.shape[:2]
        # Torso band: central 30-70% horizontally, 20-55% vertically — clothing,
        # avoiding head, legs, and the colourful shelf background at the edges.
        roi = bgr[int(0.20 * h):int(0.55 * h), int(0.30 * w):int(0.70 * w)]
        if roi.size == 0:
            roi = bgr
        hsv = cv2.cvtColor(np.ascontiguousarray(roi), cv2.COLOR_BGR2HSV)
        H = hsv[:, :, 0].ravel(); S = hsv[:, :, 1].ravel(); V = hsv[:, :, 2].ravel()
        hh, _ = np.histogram(H, bins=12, range=(0, 180))
        sh, _ = np.histogram(S, bins=6, range=(0, 256))
        vh, _ = np.histogram(V, bins=6, range=(0, 256))
        vec = np.concatenate([hh, sh, vh]).astype(np.float64)
        return vec / (np.linalg.norm(vec) or 1.0)

    def _try_load_osnet(self, transforms) -> bool:
        if not self.weights_path.is_file():
            return False
        try:
            from .osnet import osnet_x1_0

            model = osnet_x1_0(num_classes=1000)
            sd = self._torch.load(self.weights_path, map_location=self.device)
            state = sd.get("state_dict", sd) if isinstance(sd, dict) else sd
            # Strip DataParallel prefix and the dataset-specific classifier head
            # (MSMT17 has 4101 identities; we only want the feature trunk).
            state = {
                k.replace("module.", ""): v
                for k, v in state.items()
                if not k.replace("module.", "").startswith("classifier.")
            }
            model.load_state_dict(state, strict=False)
            model.eval()
            self._model = model
            # OSNet re-ID convention: 256x128 (HxW).
            self._transform = transforms.Compose([
                transforms.Resize((256, 128)),
                transforms.ToTensor(),
                transforms.Normalize(mean=_MEAN, std=_STD),
            ])
            self.backend = "osnet-reid"
            self.dim = 512
            return True
        except Exception as exc:  # noqa: BLE001 — any failure → fall back
            print(f"reid: OSNet load failed ({type(exc).__name__}: {exc}); "
                  f"falling back to ResNet50")
            return False

    def _load_resnet(self, transforms) -> None:
        try:
            from torchvision import models
        except ImportError:
            return
        weights = models.ResNet50_Weights.IMAGENET1K_V2
        backbone = models.resnet50(weights=weights)
        backbone.fc = self._torch.nn.Identity()
        backbone.eval()
        self._model = backbone
        self._transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=_MEAN, std=_STD),
        ])
        self.backend = "resnet50"
        self.dim = 2048

    @property
    def available(self) -> bool:
        self._ensure_loaded()
        return self._model is not None

    # ------------------------------------------------------------- embedding
    def _deep_embed(self, pil_img) -> np.ndarray:
        x = self._transform(pil_img.convert("RGB")).unsqueeze(0)
        with self._torch.no_grad():
            v = self._model(x).cpu().numpy()[0]
        return v / (np.linalg.norm(v) or 1.0)

    def _embed_pil(self, pil_img) -> np.ndarray:
        if self.backend == "color-hsv":
            return self._color_signature(pil_img)
        if self.backend and self.backend.startswith("fused"):
            # Weighted concat of unit colour + unit deep vectors, so that
            # cosine(a,b) = w_color·cos_color + w_deep·cos_deep. Colour groups by
            # clothing; deep separates same-colour different bodies.
            wc = max(0.0, min(1.0, REID_COLOR_WEIGHT))
            color = self._color_signature(pil_img) * (wc ** 0.5)
            deep = self._deep_embed(pil_img) * ((1.0 - wc) ** 0.5)
            return np.concatenate([color, deep])
        return self._deep_embed(pil_img)

    def embed_pil(self, pil_img) -> np.ndarray | None:
        self._ensure_loaded()
        if self._model is None:
            return None
        return self._embed_pil(pil_img)

    def embed_path(self, path: str | Path) -> np.ndarray | None:
        self._ensure_loaded()
        if self._model is None:
            return None
        from PIL import Image
        with Image.open(path) as img:
            return self._embed_pil(img)

    def embed_bytes(self, data: bytes | None) -> np.ndarray | None:
        self._ensure_loaded()
        if self._model is None or data is None:
            return None
        from PIL import Image
        with Image.open(io.BytesIO(data)) as img:
            return self._embed_pil(img)


_SHARED: PersonEmbedder | None = None


def get_embedder(weights_path: str | Path | None = None) -> PersonEmbedder:
    """Process-wide singleton so the model is loaded once across pipeline stages."""
    global _SHARED
    if _SHARED is None:
        _SHARED = PersonEmbedder(weights_path)
    return _SHARED
