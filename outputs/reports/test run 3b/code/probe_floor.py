"""Test run 3b - floor tracking: pixels -> metres, with the geometric gate and Kalman.

Three questions, all unanswerable in pixel space:

  1. Does the geometric gate kill false positives that confidence alone lets through?
     (the 0.17-scoring chair). We measure the rejection rate at each confidence.
  2. Is the speed we compute on the floor plane physically plausible? Walking is
     0.8-2.0 m/s. A calibration that says 7 m/s is wrong, and nothing else tells us.
  3. Does Kalman belong in metres or in pixels? A person walking at constant speed
     has constant velocity in metres and a wildly non-constant one in pixels, because
     perspective compresses distance with depth. We run both and compare.

Usage:

    uv run python scripts/probe_floor.py "data/raw/clip.mp4" --scene hallway

The tracker here is deliberately dumb (greedy IoU, no re-identification): this run is
about geometry, not about tracking. A broken track is reported, not hidden.
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
import time
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from skelguard import paths  # noqa: E402
from skelguard.geometry import (  # noqa: E402
    WALK_MS,
    Scene,
    foot_point,
    sanity_report,
    speeds_ms,
)
from skelguard.probe import draw_box, iou, set_deterministic  # noqa: E402


def load_yolo(name: str):
    from ultralytics import YOLO

    return YOLO(str(paths.MODELS / name))


def person_boxes(model, conf: float, frame_bgr) -> list[np.ndarray]:
    """[x1,y1,x2,y2,score] per person. We keep the score to report what the gate does
    to each confidence band."""
    res = model.predict(frame_bgr, classes=[0], conf=conf, verbose=False)[0]
    if res.boxes is None or len(res.boxes) == 0:
        return []
    xyxy = res.boxes.xyxy.cpu().numpy()
    sc = res.boxes.conf.cpu().numpy()
    return [np.append(xyxy[i], sc[i]).astype(np.float32) for i in range(len(sc))]


def conf_band(score: float) -> str:
    return "<0.25" if score < 0.25 else ("0.25-0.5" if score < 0.5 else ">=0.5")

OK_COLOUR = (0, 200, 0)
GATED_COLOUR = (0, 0, 255)
GRID_COLOUR = (0, 160, 0)
SNAPSHOT = ["scripts/probe_floor.py", "src/skelguard/geometry.py"]


# ------------------------------------------------------------------ Kalman
class KF2D:
    """Constant-velocity Kalman filter on a 2D point. Units are whatever you feed it."""

    def __init__(self, xy, q: float, r: float):
        self.x = np.array([xy[0], xy[1], 0.0, 0.0], dtype=float)
        self.P = np.diag([r, r, 1.0, 1.0])
        self.q, self.r = q, r

    def step(self, xy, dt: float) -> np.ndarray:
        F = np.eye(4)
        F[0, 2] = F[1, 3] = dt
        Q = np.eye(4) * (self.q * max(dt, 1e-3))
        self.x = F @ self.x
        self.P = F @ self.P @ F.T + Q
        H = np.zeros((2, 4))
        H[0, 0] = H[1, 1] = 1.0
        R = np.eye(2) * self.r
        y = np.asarray(xy, dtype=float) - H @ self.x
        S = H @ self.P @ H.T + R
        K = self.P @ H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        self.P = (np.eye(4) - K @ H) @ self.P
        return self.x[:2].copy()


def smooth(points: np.ndarray, times: np.ndarray, q: float, r: float) -> np.ndarray:
    """Kalman-smooth a track of 2D points sampled at `times`."""
    pts = np.asarray(points, dtype=float)
    if len(pts) < 2:
        return pts.copy()
    kf = KF2D(pts[0], q, r)
    out = [pts[0].copy()]
    for i in range(1, len(pts)):
        out.append(kf.step(pts[i], float(times[i] - times[i - 1])))
    return np.asarray(out)


# ------------------------------------------------------------------ tracking
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


# ------------------------------------------------------------------ drawing
def draw_grid(frame, scene: Scene, step: float = 1.0) -> None:
    w_m = max(p[0] for p in scene.world_points)
    l_m = max(p[1] for p in scene.world_points)

    def line(a, b):
        p = scene.to_image(np.array([a, b], dtype=np.float32))
        if np.all(np.isfinite(p)):
            cv2.line(frame, tuple(np.int32(p[0])), tuple(np.int32(p[1])),
                     GRID_COLOUR, 1, cv2.LINE_AA)

    for x in np.arange(0, w_m + 1e-6, step):
        line([x, 0], [x, l_m])
    for y in np.arange(0, l_m + 1e-6, step):
        line([0, y], [w_m, y])


# ------------------------------------------------------------------ main probe
def run(args) -> int:
    set_deterministic(args.seed)
    scene = Scene.load(args.scene)
    if scene.H is None:
        raise SystemExit(f"[FAIL] scene {scene.name!r} has no homography. "
                         "Re-run calibrate_scene.py with --width/--length.")
    model = scene.model
    if model is None:
        print("  [warn] this scene has no height model: the geometric gate is disabled.")

    cap = cv2.VideoCapture(str(args.video))
    if not cap.isOpened():
        raise SystemExit(f"[FAIL] could not open {args.video}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    rep_dir = paths.OUT_REPORTS / args.run
    vid_dir = paths.OUT_VIDEOS / args.run
    for d in (rep_dir, vid_dir):
        d.mkdir(parents=True, exist_ok=True)

    writer = None
    if not args.no_video:
        out_path = vid_dir / f"{args.video.stem}__floor.mp4"
        writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"),
                                 fps / max(1, args.stride), (w, h))

    yolo = load_yolo(args.model)
    tracker = Tracker(thr=args.iou, max_miss=args.max_miss)
    rows: list[dict] = []
    n_boxes = n_gated = 0
    gate_by_conf: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    t0 = time.perf_counter()
    frame_i = n_frames = 0

    while True:
        ok, frame = cap.read()
        if not ok or (args.max_frames and frame_i >= args.max_frames):
            break
        if frame_i % args.stride:
            frame_i += 1
            continue
        n_frames += 1
        t_s = frame_i / fps

        boxes = person_boxes(yolo, args.conf, frame)
        keep: list[np.ndarray] = []
        rejected: list[np.ndarray] = []
        for b in boxes:
            n_boxes += 1
            foot = foot_point(b)
            good = True
            if model is not None:
                good = bool(model.plausible(foot[1], b[3] - b[1], tol=args.gate_tol))
            gate_by_conf[conf_band(float(b[4]))][0] += 1
            if good:
                keep.append(b)
            else:
                gate_by_conf[conf_band(float(b[4]))][1] += 1
                n_gated += 1
                rejected.append(b)

        for tid, box in tracker.update(keep, frame_i):
            foot = foot_point(box)
            fl = scene.to_floor(foot.reshape(1, 2))[0]
            rows.append({"frame": frame_i, "t_s": round(t_s, 4), "track": tid,
                         "x_px": float(foot[0]), "y_px": float(foot[1]),
                         "X_m": float(fl[0]), "Y_m": float(fl[1]),
                         "box_h": float(box[3] - box[1]),
                         "score": float(box[4]) if len(box) > 4 else float("nan")})
            if writer is not None:
                draw_box(frame, box[:4], OK_COLOUR,
                         f"#{tid} ({fl[0]:.1f},{fl[1]:.1f})m")

        if writer is not None:
            draw_grid(frame, scene, args.grid)
            for b in rejected:
                draw_box(frame, b[:4], GATED_COLOUR, f"GATED {b[4]:.2f}")
            cv2.rectangle(frame, (0, 0), (w, 24), (0, 0, 0), -1)
            cv2.putText(frame, f"{args.scene}  conf={args.conf}  gate_tol={args.gate_tol}"
                               f"  green grid = {args.grid:g} m",
                        (8, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1,
                        cv2.LINE_AA)
            writer.write(frame)
        frame_i += 1

    cap.release()
    if writer is not None:
        writer.release()
    elapsed = time.perf_counter() - t0

    # ------------------------------------------------------------ per-track speeds
    by_track: dict[int, list[dict]] = defaultdict(list)
    for r in rows:
        by_track[r["track"]].append(r)

    track_stats = []
    all_raw, all_m, all_px = [], [], []
    for tid, rs in sorted(by_track.items()):
        if len(rs) < args.min_track:
            continue
        t = np.array([r["t_s"] for r in rs])
        pm = np.array([[r["X_m"], r["Y_m"]] for r in rs])
        pp = np.array([[r["x_px"], r["y_px"]] for r in rs])

        v_raw = speeds_ms(pm, t)
        # Kalman in METRES: smooth where motion is actually linear.
        v_m = speeds_ms(smooth(pm, t, args.q_m, args.r_m), t)
        # Kalman in PIXELS: smooth first, then map. This is the thing we claim is wrong.
        v_px = speeds_ms(scene.to_floor(smooth(pp, t, args.q_px, args.r_px)), t)

        all_raw.append(v_raw)
        all_m.append(v_m)
        all_px.append(v_px)
        track_stats.append({
            "track": tid, "n": len(rs), "dur_s": round(float(t[-1] - t[0]), 2),
            "path_m": round(float(np.linalg.norm(np.diff(pm, axis=0), axis=1).sum()), 2),
            "v_raw": round(float(np.median(v_raw)), 2),
            "v_kf_m": round(float(np.median(v_m)), 2),
            "v_kf_px": round(float(np.median(v_px)), 2),
        })

    # ------------------------------------------------------------ report
    def cat(xs):
        return np.concatenate(xs) if xs else np.array([])

    lines: list[str] = []
    p = lines.append
    p("=" * 92)
    p(f"TEST RUN 3b - FLOOR TRACKING   {args.video.name}")
    p(f"scene={scene.name}  {w}x{h} @{fps:.1f}fps  frames={n_frames}  "
      f"conf={args.conf}  stride={args.stride}")
    p("=" * 92)

    p("\n[1] GEOMETRIC GATE  (does floor geometry reject what confidence accepts?)")
    if model is None:
        p("    disabled: this scene has no height model")
    else:
        p(f"    height model: h_px = {model.k:.3f} * (y_feet - {model.y_horizon:.1f})"
          f"   tol={args.gate_tol:.0%}")
        p(f"    {'conf band':<12}{'boxes':>8}{'gated':>8}{'rejected':>11}")
        for band in ("<0.25", "0.25-0.5", ">=0.5", "n/a"):
            tot, bad = gate_by_conf.get(band, [0, 0])
            if tot:
                p(f"    {band:<12}{tot:>8}{bad:>8}{bad / tot:>10.1%}")
        p(f"    {'TOTAL':<12}{n_boxes:>8}{n_gated:>8}"
          f"{(n_gated / n_boxes if n_boxes else 0):>10.1%}")
        p("    A high rejection rate in the low band and a near-zero one in the high band")
        p("    is the result we want: the gate removes junk without touching real people.")

    p(f"\n[2] TRACKS  (>= {args.min_track} frames)")
    if not track_stats:
        p("    none. Either nothing was detected, or every track is shorter than the")
        p("    minimum - which is itself the finding.")
    else:
        p(f"    {'id':>4}{'n':>6}{'dur_s':>8}{'path_m':>9}{'v_raw':>8}{'v_kf_m':>9}"
          f"{'v_kf_px':>9}   (median m/s)")
        for s in track_stats:
            p(f"    {s['track']:>4}{s['n']:>6}{s['dur_s']:>8}{s['path_m']:>9}"
              f"{s['v_raw']:>8}{s['v_kf_m']:>9}{s['v_kf_px']:>9}")

    p(f"\n[3] IS THE CALIBRATION BELIEVABLE?  (walking is {WALK_MS[0]}-{WALK_MS[1]} m/s)")
    for name, v in (("raw", cat(all_raw)), ("kalman in METRES", cat(all_m)),
                    ("kalman in PIXELS", cat(all_px))):
        s = sanity_report(v)
        if not s:
            p(f"    {name:<18} no samples")
            continue
        p(f"    {name:<18} median={s['median_ms']:5.2f}  p90={s['p90_ms']:5.2f}  "
          f"max={s['max_ms']:6.2f}  in walking range={s['frac_in_walking_range']:6.1%}  "
          f"impossible={s['frac_impossible']:.1%}")
    p("    `impossible` (>10 m/s) should be ~0. If it is not, the homography is wrong,")
    p("    or the tracker is swapping identities between people.")

    p("\n[4] WHERE DOES KALMAN BELONG?")
    vm, vp = cat(all_m), cat(all_px)
    if len(vm) and len(vp):
        jm, jp = float(np.std(np.diff(vm))), float(np.std(np.diff(vp)))
        sm = sanity_report(vm)["frac_in_walking_range"]
        sp = sanity_report(vp)["frac_in_walking_range"]
        p(f"    speed jitter (std of frame-to-frame change):  metres={jm:.3f}  "
          f"pixels={jp:.3f}")
        p(f"    fraction inside walking range:                metres={sm:.1%}  "
          f"pixels={sp:.1%}")
        p("    Smoothing in pixels assumes constant pixel velocity, which perspective")
        p("    makes false: the same stride covers fewer pixels further away, so the")
        p("    filter fights the geometry and drags the estimate near the horizon.")
        p(f"    VERDICT: {'metres' if (sm >= sp and jm <= jp) else 'inconclusive on this clip'}")
    else:
        p("    not enough track data to compare")

    p(f"\n[5] SPEED   {n_frames} frames in {elapsed:.1f}s  "
      f"= {n_frames / max(elapsed, 1e-6):.1f} fps end-to-end (detector + homography)")

    text = "\n".join(lines)
    print("\n" + text)

    (rep_dir / f"{args.video.stem}__floor_report.txt").write_text(text, encoding="utf-8")
    with (rep_dir / f"{args.video.stem}__floor_tracks.csv").open(
            "w", newline="", encoding="utf-8") as f:
        wri = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else
                             ["frame", "t_s", "track", "x_px", "y_px", "X_m", "Y_m",
                              "box_h", "score"])
        wri.writeheader()
        wri.writerows(rows)

    # frozen snapshot, so the numbers above can always be traced to the code
    code = rep_dir / "code"
    code.mkdir(exist_ok=True)
    for rel in SNAPSHOT:
        src = ROOT / rel
        if src.exists():
            shutil.copy2(src, code / Path(rel).name)
    (rep_dir / "run_meta.json").write_text(json.dumps({
        "video": str(args.video), "scene": scene.name, "conf": args.conf,
        "gate_tol": args.gate_tol, "iou": args.iou, "stride": args.stride,
        "seed": args.seed, "frames": n_frames, "fps_src": fps,
        "height_model": scene.height_model,
    }, indent=2), encoding="utf-8")
    print(f"\nwrote {rep_dir}")
    if writer is not None:
        print(f"wrote {vid_dir}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("video", type=Path)
    ap.add_argument("--scene", required=True, help="calib/<scene>.json")
    ap.add_argument("--model", default="yolov8n.pt", help="yolov8n.pt | yolov8s.pt | yolov8m.pt")
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--gate-tol", type=float, default=0.35,
                    help="allowed relative error vs expected height")
    ap.add_argument("--iou", type=float, default=0.30)
    ap.add_argument("--max-miss", type=int, default=10)
    ap.add_argument("--min-track", type=int, default=15)
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--max-frames", type=int, default=0)
    ap.add_argument("--grid", type=float, default=1.0)
    ap.add_argument("--q-m", type=float, default=0.5, help="process noise, metres")
    ap.add_argument("--r-m", type=float, default=0.10, help="measurement noise, metres")
    ap.add_argument("--q-px", type=float, default=200.0, help="process noise, pixels")
    ap.add_argument("--r-px", type=float, default=25.0, help="measurement noise, pixels")
    ap.add_argument("--no-video", action="store_true")
    ap.add_argument("--run", default="test run 3b")
    ap.add_argument("--seed", type=int, default=0)
    return run(ap.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
