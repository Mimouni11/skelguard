"""Person tracking: associate detections across frames so a skeleton has an identity.

Greedy IoU association, no re-identification: a person who is lost and comes back is a
new id. That is deliberate for the probes - a broken track is reported, not hidden - and
it is good enough for gait, where a clip holds one walking pass and the subject never
leaves the frame. WP1 replaces this with Hungarian matching plus a Kalman prediction,
which is what keeps ids stable through a crossing.
"""

from __future__ import annotations

import numpy as np

from skelguard.probe import iou


class Tracker:
    """Greedy IoU association. No re-id: a lost person comes back as a new id."""

    def __init__(self, thr: float = 0.3, max_miss: int = 10):
        self.thr, self.max_miss = thr, max_miss
        self.tracks: dict[int, dict] = {}
        self.next_id = 0

    def update(self, boxes: list[np.ndarray], frame_i: int) -> list[tuple[int, np.ndarray]]:
        live = [(tid, t) for tid, t in self.tracks.items()
                if frame_i - t["last"] <= self.max_miss]
        pairs = sorted(
            ((iou(t["box"], b), tid, bi)
             for tid, t in live for bi, b in enumerate(boxes)),
            key=lambda p: -p[0],
        )
        used_t, used_b, assigned = set(), set(), []
        for score, tid, bi in pairs:
            if score < self.thr or tid in used_t or bi in used_b:
                continue
            used_t.add(tid)
            used_b.add(bi)
            self.tracks[tid].update(box=boxes[bi], last=frame_i)
            assigned.append((tid, boxes[bi]))
        for bi, b in enumerate(boxes):
            if bi in used_b:
                continue
            tid = self.next_id
            self.next_id += 1
            self.tracks[tid] = {"box": b, "last": frame_i}
            assigned.append((tid, b))
        return assigned
