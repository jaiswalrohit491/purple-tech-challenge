"""Load store_layout.json into typed dataclasses.

The challenge brief says store_layout.json contains: zone names, camera
coverage, and open hours. The exact shape isn't published, so this module
accepts a few plausible variants and normalises them. If the actual file ships
with a wildly different shape, only this file needs to change.

Expected (and produced internally) shape:

    {
      "stores": [
        {
          "store_id": "STORE_BLR_002",
          "city": "Bengaluru",
          "open_hours": {"start": "10:00", "end": "21:00", "tz": "Asia/Kolkata"},
          "cameras": [
            {
              "camera_id": "CAM_ENTRY_01",
              "view": "ENTRY",
              "entry_line": [[100, 540], [1820, 540]],
              "entry_normal_inward": [0, 1],
              "zones": []
            },
            {
              "camera_id": "CAM_FLOOR_01",
              "view": "FLOOR",
              "zones": [
                {"zone_id": "SKINCARE", "polygon": [[200,200],[800,200],[800,600],[200,600]]}
              ]
            },
            {
              "camera_id": "CAM_BILL_01",
              "view": "BILLING",
              "zones": [
                {"zone_id": "BILLING_QUEUE", "polygon": [...]}
              ]
            }
          ]
        }
      ]
    }
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class Zone:
    zone_id: str
    polygon: list[tuple[float, float]]
    brand_name: str | None = None  # populated when the zone maps to a POS-CSV brand


@dataclass(frozen=True)
class Camera:
    camera_id: str
    view: str  # "ENTRY" | "FLOOR" | "BILLING" | "BACK_OFFICE"
    entry_line: tuple[tuple[float, float], tuple[float, float]] | None
    entry_normal_inward: tuple[float, float] | None
    zones: list[Zone] = field(default_factory=list)
    # Path to the video clip for this camera (relative to repo root). When set,
    # `run.sh` drives detection straight off the layout file — no per-camera CLI
    # wiring needed.
    clip_path: str | None = None
    # When true, every event emitted from this camera is force-stamped
    # is_staff=true. Used for cameras that only see staff (e.g. back office).
    force_staff: bool = False


@dataclass(frozen=True)
class Store:
    store_id: str
    city: str
    open_hours: dict
    cameras: list[Camera]

    def camera(self, camera_id: str) -> Camera | None:
        return next((c for c in self.cameras if c.camera_id == camera_id), None)


def _to_tuple_list(points) -> list[tuple[float, float]]:
    return [(float(x), float(y)) for x, y in points]


def load_layout(path: str | Path) -> dict[str, Store]:
    """Return a dict keyed by store_id."""
    raw = json.loads(Path(path).read_text())
    stores_raw = raw.get("stores") or raw  # tolerate either {"stores": [...]} or bare list
    out: dict[str, Store] = {}
    for s in stores_raw:
        cams: list[Camera] = []
        for c in s.get("cameras", []):
            entry_line = c.get("entry_line")
            if entry_line:
                entry_line = (tuple(map(float, entry_line[0])), tuple(map(float, entry_line[1])))
            normal = c.get("entry_normal_inward")
            if normal:
                normal = (float(normal[0]), float(normal[1]))
            zones = [
                Zone(
                    zone_id=z["zone_id"],
                    polygon=_to_tuple_list(z["polygon"]),
                    brand_name=z.get("brand_name"),
                )
                for z in c.get("zones", [])
            ]
            cams.append(Camera(
                camera_id=c["camera_id"],
                view=c.get("view", "FLOOR"),
                entry_line=entry_line,
                entry_normal_inward=normal,
                zones=zones,
                clip_path=c.get("clip_path"),
                force_staff=bool(c.get("force_staff", False)),
            ))
        out[s["store_id"]] = Store(
            store_id=s["store_id"],
            city=s.get("city", "Unknown"),
            open_hours=s.get("open_hours", {}),
            cameras=cams,
        )
    return out
