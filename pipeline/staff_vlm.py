"""Staff classifier using Claude vision, with cache + rule-based fallback.

Cost discipline: at most ONE VLM call per visitor (per process lifetime). The
result is cached by `visitor_id` and reused for every subsequent event from
that visitor. If the Anthropic SDK isn't available, the API key isn't set, or
the call fails, we fall back to a colour-histogram heuristic that flags
uniform-coloured torso patches as staff.

Tunables live in this file because they're domain-specific (uniform colour,
prompt text). Promoting them to env vars is unwarranted for a single-tenant
deployment.
"""
from __future__ import annotations

import base64
import io
import os
import re
from dataclasses import dataclass
from threading import Lock

UNIFORM_PROMPT = (
    "You are classifying a single person visible in retail CCTV footage. "
    "Apex Retail staff wear a navy-blue polo shirt with a small white logo "
    "on the left chest and a dark name badge. Customers wear varied casual "
    "clothing. Return STRICT JSON only, no prose: "
    '{"is_staff": <bool>, "confidence": <0-1 float>, "reason": "<short>"}. '
    "If the image is blurry, occluded, the face is blurred such that the "
    "uniform is not clearly visible, or you cannot see the torso, return "
    'is_staff=false with confidence <= 0.4.'
)


@dataclass(frozen=True)
class StaffVerdict:
    is_staff: bool
    confidence: float
    reason: str
    source: str  # "vlm" | "fallback" | "cached"


class StaffClassifier:
    """One instance per pipeline run. Thread-safe cache (the post-pass tracker
    may classify on a worker thread)."""

    def __init__(self, model: str = "claude-sonnet-4-6"):
        self.model = model
        self._cache: dict[str, StaffVerdict] = {}
        self._lock = Lock()
        self._client = None
        try:
            import anthropic  # noqa: F401  # heavy import, only when actually used
            self._available = bool(os.getenv("ANTHROPIC_API_KEY"))
        except ImportError:
            self._available = False

    def _client_lazy(self):
        if self._client is None:
            import anthropic
            self._client = anthropic.Anthropic()
        return self._client

    def classify(self, visitor_id: str, crop_jpeg: bytes | None) -> StaffVerdict:
        with self._lock:
            cached = self._cache.get(visitor_id)
        if cached is not None:
            return StaffVerdict(cached.is_staff, cached.confidence, cached.reason, "cached")

        verdict = self._classify_uncached(crop_jpeg)
        with self._lock:
            self._cache[visitor_id] = verdict
        return verdict

    def _classify_uncached(self, crop_jpeg: bytes | None) -> StaffVerdict:
        if crop_jpeg is None or not self._available:
            return self._fallback(crop_jpeg)

        try:
            b64 = base64.standard_b64encode(crop_jpeg).decode("ascii")
            client = self._client_lazy()
            msg = client.messages.create(
                model=self.model,
                max_tokens=120,
                messages=[{
                    "role": "user",
                    "content": [
                        {"type": "image", "source": {
                            "type": "base64", "media_type": "image/jpeg", "data": b64,
                        }},
                        {"type": "text", "text": UNIFORM_PROMPT},
                    ],
                }],
            )
            text = "".join(b.text for b in msg.content if getattr(b, "type", "") == "text")
            data = _parse_json_or_none(text)
            if not data:
                return self._fallback(crop_jpeg)
            return StaffVerdict(
                is_staff=bool(data.get("is_staff", False)),
                confidence=float(data.get("confidence", 0.5)),
                reason=str(data.get("reason", ""))[:120],
                source="vlm",
            )
        except Exception as exc:
            return StaffVerdict(False, 0.3, f"vlm-error:{type(exc).__name__}", "fallback")

    def _fallback(self, crop_jpeg: bytes | None) -> StaffVerdict:
        """Cheap heuristic: average colour of the upper-torso strip vs uniform target.
        Returns low-confidence is_staff=true when the patch is close to navy blue.
        Used when no API key is configured or the call fails."""
        if crop_jpeg is None:
            return StaffVerdict(False, 0.2, "no-crop", "fallback")
        try:
            from PIL import Image
        except ImportError:
            return StaffVerdict(False, 0.2, "no-pillow", "fallback")
        try:
            img = Image.open(io.BytesIO(crop_jpeg)).convert("RGB")
            w, h = img.size
            # Torso strip: middle 60% width, top 25-55% height.
            torso = img.crop((int(w * 0.2), int(h * 0.25), int(w * 0.8), int(h * 0.55)))
            r, g, b = _avg_rgb(torso)
            # Navy blue target (R<60, G<70, B>80) — very loose.
            is_navy = r < 80 and g < 90 and b > 70 and b > r
            return StaffVerdict(
                is_staff=is_navy,
                confidence=0.55 if is_navy else 0.35,
                reason=f"rgb=({r},{g},{b}) navy={is_navy}",
                source="fallback",
            )
        except Exception as exc:
            return StaffVerdict(False, 0.2, f"fallback-error:{type(exc).__name__}", "fallback")


def _avg_rgb(img) -> tuple[int, int, int]:
    px = list(img.getdata())
    n = len(px) or 1
    r = sum(p[0] for p in px) // n
    g = sum(p[1] for p in px) // n
    b = sum(p[2] for p in px) // n
    return r, g, b


def _parse_json_or_none(text: str) -> dict | None:
    import json
    text = text.strip()
    # Tolerate fenced JSON blocks.
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except Exception:
        return None
