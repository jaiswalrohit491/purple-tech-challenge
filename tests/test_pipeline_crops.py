# PROMPT:
#   "Write a unit test for the crop-quality gate in pipeline.cluster_and_label.
#    parse_crops must drop tracks whose representative crop is occluded/partial
#    (near-square or landscape box) or too small, and keep well-framed portrait
#    person crops. Build tiny synthetic JPEGs with PIL."
#
# CHANGES MADE:
#   - Asserted on both the kept list and the dropped-names list.
#   - Added the exact billing-counter failure case (a ~square 226x234 crop,
#     the real CAM_05#14 shape) to pin the regression that motivated the gate.

from __future__ import annotations

from pathlib import Path

from PIL import Image

from pipeline.cluster_and_label import parse_crops


def _img(path: Path, w: int, h: int) -> None:
    Image.new("RGB", (w, h), (20, 20, 20)).save(path)


def test_crop_quality_gate(tmp_path):
    d = tmp_path / "CAM_05"
    d.mkdir()
    # Well-framed full-body person: tall portrait → KEEP.
    _img(d / "CAM_05_track010.jpg", 292, 832)        # h/w = 2.85
    # Occluded top-down counter blob (the real CAM_05#14 shape) → DROP.
    _img(d / "CAM_05_track014.jpg", 226, 234)        # h/w = 1.04
    # Landscape box (person bending / cut off) → DROP.
    _img(d / "CAM_05_track070.jpg", 384, 241)        # h/w = 0.63
    # Tiny crop, short side below floor → DROP.
    _img(d / "CAM_05_track099.jpg", 40, 120)         # short side 40 < 60

    kept, dropped = parse_crops([d], min_aspect=1.3, min_short_side=60)

    kept_ids = {tid for _, tid, _ in kept}
    assert kept_ids == {10}, f"only the portrait crop should survive, got {kept_ids}"
    assert set(dropped) == {
        "CAM_05_track014.jpg",
        "CAM_05_track070.jpg",
        "CAM_05_track099.jpg",
    }


def test_crop_quality_gate_disabled(tmp_path):
    """min_aspect=0 disables the aspect filter (short-side floor still applies)."""
    d = tmp_path / "CAM_01"
    d.mkdir()
    _img(d / "CAM_01_track001.jpg", 226, 234)        # square, but big enough
    kept, dropped = parse_crops([d], min_aspect=0.0, min_short_side=60)
    assert {tid for _, tid, _ in kept} == {1}
    assert dropped == []
