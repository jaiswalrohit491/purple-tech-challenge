"""Billing-queue depth counter.

The billing camera's queue zone is a polygon. At each frame we count the
number of person bboxes whose bottom-center is inside it, then smooth with a
median filter over ~1s (15 frames at 15fps). The smoothed value is what we
stamp onto `BILLING_QUEUE_JOIN.metadata.queue_depth`.
"""
from __future__ import annotations

from collections import deque
from statistics import median

from shapely.geometry import Polygon


class QueueDepthCounter:
    def __init__(self, queue_polygon: Polygon, smoothing_frames: int = 15):
        self.poly = queue_polygon
        self._window: deque[int] = deque(maxlen=smoothing_frames)

    def step(self, bottom_centers: list[tuple[float, float]]) -> int:
        """Call once per billing-camera frame. Returns the current smoothed depth."""
        from shapely.geometry import Point
        count = sum(1 for xy in bottom_centers if self.poly.contains(Point(xy)))
        self._window.append(count)
        return int(median(self._window))
