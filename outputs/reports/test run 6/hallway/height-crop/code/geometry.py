"""Scene geometry: floor homography, the expected-height model, and metric kinematics.

Two independent ways to reason about where a person is:

  Homography      4 measured floor points -> a matrix mapping image pixels to metres on
                  the floor. Needed for real speed in m/s (WP3) and for zones in metres.
                  Requires calibration; see scripts/calibrate_scene.py.

  Height model    For a camera with little roll, a fixed-height object's pixel height is
                  linear in the image y of its feet:  h_px = k * (y_feet - y_horizon).
                  Both constants fit from detections alone - no measurement needed. Gives
                  a geometric plausibility gate on boxes, and a crop scale, for free.

The height model was fitted on test run 3a: three detectors independently recovered the
same horizon and slope on the same clip (within 2%), so it recovers real camera geometry
rather than overfitting. It fails on wide-angle overhead views, which break the pinhole
assumption - undistort first, or fall back to the homography.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import numpy as np

from skelguard import paths

# A standing adult, used to turn apparent height into distance.
PERSON_HEIGHT_M = 1.70
# Plausible walking speed, for sanity-checking a calibration.
WALK_MS = (0.8, 2.0)


# --------------------------------------------------------------------- height model
@dataclass
class HeightModel:
    """h_px = k * (y_feet - y_horizon), fitted from detections."""

    k: float
    y_horizon: float
    n: int = 0
    inlier_rate: float = float("nan")
    median_rel_err: float = float("nan")

    def expected_px(self, y_feet: np.ndarray | float) -> np.ndarray | float:
        """How tall a person standing with their feet at this image y should appear."""
        return self.k * (np.asarray(y_feet, dtype=float) - self.y_horizon)

    def plausible(self, y_feet, box_h, tol: float = 0.35) -> np.ndarray:
        """True where the observed height is within `tol` of the expected height.

        This is the geometric gate: a box whose size is impossible for a person standing
        at that spot is rejected regardless of the detector's confidence.
        """
        exp = np.asarray(self.expected_px(y_feet), dtype=float)
        obs = np.asarray(box_h, dtype=float)
        with np.errstate(divide="ignore", invalid="ignore"):
            rel = np.abs(obs - exp) / np.maximum(exp, 1e-6)
        return (exp > 0) & (rel <= tol)

    def crop_scale(self, y_feet, target_px: int = 320) -> float:
        """Upscale factor needed to bring a person at this position to `target_px`."""
        exp = float(self.expected_px(y_feet))
        return float(np.clip(target_px / max(exp, 1.0), 1.0, 8.0))

    def distance_proxy(self, y_feet) -> np.ndarray | float:
        """Relative depth, up to an unknown focal length: Z ~ 1 / h_px.

        Useful for ordering and for a tiering threshold. One real measurement, or a
        homography, turns this into metres.
        """
        return PERSON_HEIGHT_M / np.maximum(self.expected_px(y_feet), 1e-6)

    def to_json(self) -> dict:
        return asdict(self)


def fit_height_model(y_feet: np.ndarray, box_h: np.ndarray, *, iters: int = 2000,
                     tol: float = 0.15, seed: int = 0) -> HeightModel | None:
    """RANSAC fit of h_px = k*(y_feet - y_horizon), then least squares on the inliers.

    RANSAC rather than plain least squares because the input contains false positives
    (a chair scoring 0.17) that would drag an ordinary fit.
    """
    x = np.asarray(y_feet, dtype=float)
    y = np.asarray(box_h, dtype=float)
    n = len(x)
    if n < 10:
        return None

    rng = np.random.default_rng(seed)
    best_inliers, best = -1, None
    for _ in range(iters):
        i, j = rng.choice(n, 2, replace=False)
        if abs(x[i] - x[j]) < 1e-6:
            continue
        k = (y[i] - y[j]) / (x[i] - x[j])
        if k <= 0:                      # further away (smaller y) must mean smaller
            continue
        y0 = x[i] - y[i] / k
        pred = k * (x - y0)
        ok = (pred > 0) & (np.abs(pred - y) / np.maximum(y, 1.0) < tol)
        if ok.sum() > best_inliers:
            best_inliers, best = int(ok.sum()), (k, y0)

    if best is None:
        return None

    k, y0 = best
    pred = k * (x - y0)
    ok = (pred > 0) & (np.abs(pred - y) / np.maximum(y, 1.0) < tol)
    if ok.sum() >= 2:                   # refit on inliers only
        A = np.vstack([x[ok], np.ones(int(ok.sum()))]).T
        m, c = np.linalg.lstsq(A, y[ok], rcond=None)[0]
        if m > 0:
            k, y0 = float(m), float(-c / m)

    pred = k * (x - y0)
    rel = np.abs(pred - y) / np.maximum(y, 1.0)
    return HeightModel(k=float(k), y_horizon=float(y0), n=n,
                       inlier_rate=float(ok.mean()),
                       median_rel_err=float(np.median(rel[ok])) if ok.any() else float("nan"))


# --------------------------------------------------------------------- homography
@dataclass
class Scene:
    """A calibrated camera position: pixels <-> metres on the floor."""

    name: str
    image_points: list[list[float]]     # 4 clicked floor points, in pixels
    world_points: list[list[float]]     # the same 4, in metres
    frame_w: int
    frame_h: int
    H: list[list[float]] | None = None          # image -> world
    height_model: dict | None = None
    note: str = ""

    # ---- construction ----
    @staticmethod
    def from_rectangle(name: str, image_points, width_m: float, length_m: float,
                       frame_w: int, frame_h: int, note: str = "") -> "Scene":
        """4 points clicked clockwise from the near-left corner of a floor rectangle."""
        world = [[0.0, 0.0], [width_m, 0.0], [width_m, length_m], [0.0, length_m]]
        src = np.asarray(image_points, dtype=np.float32)
        dst = np.asarray(world, dtype=np.float32)
        H = cv2.getPerspectiveTransform(src, dst)
        return Scene(name=name, image_points=np.asarray(image_points).tolist(),
                     world_points=world, frame_w=frame_w, frame_h=frame_h,
                     H=H.tolist(), note=note)

    # ---- transforms ----
    @property
    def H_arr(self) -> np.ndarray:
        if self.H is None:
            raise ValueError(f"scene {self.name!r} has no homography")
        return np.asarray(self.H, dtype=np.float64)

    def to_floor(self, pts_px: np.ndarray) -> np.ndarray:
        """(N,2) image pixels -> (N,2) metres on the floor plane."""
        p = np.asarray(pts_px, dtype=np.float32).reshape(-1, 1, 2)
        return cv2.perspectiveTransform(p, self.H_arr.astype(np.float32)).reshape(-1, 2)

    def to_image(self, pts_m: np.ndarray) -> np.ndarray:
        """(N,2) floor metres -> (N,2) image pixels."""
        p = np.asarray(pts_m, dtype=np.float32).reshape(-1, 1, 2)
        inv = np.linalg.inv(self.H_arr).astype(np.float32)
        return cv2.perspectiveTransform(p, inv).reshape(-1, 2)

    @property
    def model(self) -> HeightModel | None:
        return HeightModel(**self.height_model) if self.height_model else None

    # ---- persistence ----
    def save(self, path: Path | None = None) -> Path:
        path = path or (paths.CALIB / f"{self.name}.json")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(asdict(self), indent=2), encoding="utf-8")
        return path

    @staticmethod
    def load(name_or_path: str | Path) -> "Scene":
        p = Path(name_or_path)
        if not p.exists():
            p = paths.CALIB / f"{name_or_path}.json"
        if not p.exists():
            raise SystemExit(f"[FAIL] no calibration at {p}. Run scripts/calibrate_scene.py")
        return Scene(**json.loads(p.read_text(encoding="utf-8")))


# --------------------------------------------------------------------- kinematics
def foot_point(box: np.ndarray) -> np.ndarray:
    """Where a person box meets the floor: the bottom-centre of the box."""
    return np.array([(box[0] + box[2]) / 2.0, box[3]], dtype=np.float32)


def ankle_midpoint(pts_px: np.ndarray) -> np.ndarray:
    """Midpoint of the two ankles (BlazePose 27, 28) - the brief's intrusion signal."""
    return (pts_px[27] + pts_px[28]) / 2.0


def speeds_ms(positions_m: np.ndarray, times_s: np.ndarray) -> np.ndarray:
    """Instantaneous speed in m/s from a floor track. Same length as the input."""
    p = np.asarray(positions_m, dtype=float)
    t = np.asarray(times_s, dtype=float)
    if len(p) < 2:
        return np.zeros(len(p))
    d = np.linalg.norm(np.diff(p, axis=0), axis=1)
    dt = np.diff(t)
    v = np.divide(d, dt, out=np.zeros_like(d), where=dt > 0)
    return np.concatenate([[v[0]], v])


def sanity_report(speeds: np.ndarray) -> dict:
    """Is this calibration believable? Walking is roughly 0.8-2.0 m/s."""
    v = np.asarray(speeds, dtype=float)
    v = v[np.isfinite(v)]
    if not len(v):
        return {}
    lo, hi = WALK_MS
    return {
        "median_ms": float(np.median(v)),
        "p90_ms": float(np.quantile(v, 0.90)),
        "max_ms": float(v.max()),
        "frac_in_walking_range": float(((v >= lo) & (v <= hi)).mean()),
        "frac_impossible": float((v > 10.0).mean()),   # nobody sprints at 36 km/h
    }
