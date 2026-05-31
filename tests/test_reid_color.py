# PROMPT:
#   "Unit-test the torso colour-histogram appearance signature in
#    pipeline/reid.py (the default backend). It must: be L2-normalised, be
#    deterministic, give ~0 distance for identical crops, and clearly separate
#    different clothing colours (dark uniform vs grey vs a saturated colour) —
#    the property that lets the two customers resolve without a prior."
#
# CHANGES MADE:
#   - Build solid-colour PIL images so the expected colour separation is
#     unambiguous, and assert on cosine DISTANCE ordering (same << different)
#     rather than absolute values, which are histogram-bin dependent.

from __future__ import annotations

import numpy as np
from PIL import Image

from pipeline.reid import PersonEmbedder


def _img(rgb):
    return Image.new("RGB", (120, 360), rgb)


def _embedder():
    e = PersonEmbedder()
    # Force the colour backend regardless of process default, so the test is
    # hermetic and needs no torch.
    e._loaded = True
    e.backend = "color-hsv"
    e.dim = 24
    e._model = object()
    return e


def _cos_dist(a, b):
    return 1.0 - float(np.dot(a, b))


def test_color_signature_is_unit_norm_and_right_dim():
    e = _embedder()
    v = e.embed_pil(_img((20, 20, 20)))
    assert v.shape == (24,)
    assert abs(np.linalg.norm(v) - 1.0) < 1e-6


def test_identical_crops_zero_distance():
    e = _embedder()
    a = e.embed_pil(_img((130, 130, 130)))
    b = e.embed_pil(_img((130, 130, 130)))
    assert _cos_dist(a, b) < 1e-6


def test_colours_separate_more_than_shades():
    """Dark-uniform vs grey vs saturated red: different colours are far apart;
    a slightly different grey shade stays close. This ordering is what makes
    the two customers (grey shirt / tan bag) resolve away from dark staff."""
    e = _embedder()
    dark = e.embed_pil(_img((20, 20, 20)))
    grey = e.embed_pil(_img((130, 130, 130)))
    grey2 = e.embed_pil(_img((140, 140, 140)))     # near-identical shade
    red = e.embed_pil(_img((200, 40, 40)))

    assert _cos_dist(dark, grey) > 0.3            # uniform vs grey shirt: far
    assert _cos_dist(grey, red) > 0.3             # grey vs saturated colour: far
    assert _cos_dist(grey, grey2) < _cos_dist(grey, dark)   # shade << colour change


def test_deterministic():
    e = _embedder()
    img = _img((90, 120, 60))
    assert np.allclose(e.embed_pil(img), e.embed_pil(img))
