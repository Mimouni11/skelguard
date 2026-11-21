"""Test run 4 - floor tracking and per-person MediaPipe skeletons.

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
from skelguard.probe import (  # noqa: E402
    POSE_CONNECTIONS, ProbeConfig, draw_box, make_landmarker,
    pose_on_crop, set_deterministic,
)
from skelguard.tracking import Tracker  # noqa: E402


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
SNAPSHOT = ["scripts/probe_floor.py", "src/skelguard/geometry.py", "src/skelguard/probe.py"]


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


def windowed_speeds(pos: np.ndarray, t: np.ndarray, win_s: float) -> np.ndarray:
    """Speed from net displacement over a time window, not between adjacent frames.

    Frame-to-frame differencing of a box's foot point is dominated by detector jitter:
    a few pixels of wobble at 25 fps is metres per second of noise. A window of a few
    tenths of a second is still well inside a stride, so it does not hide real motion.
    """
    p = np.asarray(pos, dtype=float)
    tt = np.asarray(t, dtype=float)
    if len(p) < 2 or win_s <= 0:
        return speeds_ms(p, tt)
    half = win_s / 2.0
    out = np.zeros(len(p))
    for i in range(len(p)):
        j = int(np.searchsorted(tt, tt[i] - half, side="left"))
        k = int(np.searchsorted(tt, tt[i] + half, side="right")) - 1
        j, k = max(0, min(j, len(p) - 1)), max(0, min(k, len(p) - 1))
        if k <= j:
            j, k = max(0, i - 1), min(len(p) - 1, i + 1)
        dt = tt[k] - tt[j]
        out[i] = float(np.linalg.norm(p[k] - p[j]) / dt) if dt > 0 else 0.0
    return out


def max_window_displacement(pos: np.ndarray, t: np.ndarray, win_s: float) -> float:
    """Largest net floor displacement this track achieves over any window of `win_s`.

    Net, not path length: a jittering box accumulates path without going anywhere, which
    is exactly how a mannequin looks. Measured in metres, so one threshold is valid
    everywhere in the frame - the same test in pixels would mean different things at
    different depths.
    """
    p = np.asarray(pos, dtype=float)
    tt = np.asarray(t, dtype=float)
    if len(p) < 2:
        return 0.0
    best = 0.0
    j = 0
    for i in range(len(p)):
        while tt[i] - tt[j] > win_s:
            j += 1
        if i > j:
            best = max(best, float(np.linalg.norm(p[i] - p[j])))
    return best


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
    rep_dir.mkdir(parents=True, exist_ok=True)

    writer = None
    if not args.no_video:
        vid_dir.mkdir(parents=True, exist_ok=True)
        out_path = vid_dir / f"{args.video.stem}__floor.mp4"
        writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"),
                                 fps / max(1, args.stride), (w, h))

    yolo = load_yolo(args.model)
    pose_cfg = ProbeConfig(
        model=args.pose_model, crop=True, crop_margin=args.pose_margin,
        crop_target_h=args.pose_target_h, crop_max_scale=args.pose_max_scale,
        det_conf=args.pose_conf, track_conf=args.pose_conf,
    )
    pose_context = make_landmarker(pose_cfg, image_mode=True, num_poses=1)
    pose_lm = pose_context.__enter__()
    tracker = Tracker(thr=args.iou, max_miss=args.max_miss)
    rows: list[dict] = []
    skeleton_rows: list[dict] = []
    n_pose = n_pose_missed = 0
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
        # Draw on a copy. Annotating `frame` in place would feed the next track's crop
        # an image with the previous track's skeleton painted across it, which silently
        # lowered pose recall whenever video was being written.
        canvas = frame.copy() if writer is not None else None

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
            # A/B: size the upscale from the calibrated expected height at this foot
            # position, or from the detected box height. The box can be truncated by
            # occlusion or the frame edge; the expected height cannot.
            expected_h = None
            if args.height_crop and model is not None:
                e = float(model.expected_px(foot[1]))
                expected_h = e if e > 0 else None
            obs = pose_on_crop(
                pose_lm, frame, box, pose_cfg, w, h, person_h_px=expected_h
            )
            pts, visibility, crop_scale = obs.pts_px, obs.mean_visibility, obs.scale
            pose_found = obs.found
            n_pose += int(pose_found)
            n_pose_missed += int(not pose_found)
            rows.append({"frame": frame_i, "t_s": round(t_s, 4), "track": tid,
                         "x_px": float(foot[0]), "y_px": float(foot[1]),
                         "X_m": float(fl[0]), "Y_m": float(fl[1]),
                         "box_h": float(box[3] - box[1]),
                         "score": float(box[4]) if len(box) > 4 else float("nan"),
                         "pose_found": pose_found,
                         "pose_visibility": visibility if pose_found else float("nan"),
                         "crop_scale": crop_scale})
            if pose_found:
                for landmark_i, (x_px, y_px) in enumerate(pts):
                    srow = {"frame": frame_i, "t_s": round(t_s, 4),
                            "track": tid, "landmark": landmark_i,
                            "x_px": float(x_px), "y_px": float(y_px),
                            "visibility": float(obs.visibility[landmark_i]),
                            "presence": float(obs.presence[landmark_i]),
                            "mean_visibility": visibility,
                            "crop_scale": crop_scale}
                    if obs.world is not None:
                        wx, wy, wz = obs.world[landmark_i]
                        srow |= {"world_x": float(wx), "world_y": float(wy),
                                 "world_z": float(wz)}
                    skeleton_rows.append(srow)
                if canvas is not None:
                    for a, b in POSE_CONNECTIONS:
                        cv2.line(canvas, tuple(np.int32(pts[a])), tuple(np.int32(pts[b])),
                                 (0, 220, 255), 2, cv2.LINE_AA)
                    for x_px, y_px in pts:
                        cv2.circle(canvas, (int(x_px), int(y_px)), 2, (40, 40, 255), -1)
            if canvas is not None:
                draw_box(canvas, box[:4], OK_COLOUR,
                         f"#{tid} ({fl[0]:.1f},{fl[1]:.1f})m"
                         f" {'pose' if pose_found else 'no pose'}")

        if writer is not None and canvas is not None:
            draw_grid(canvas, scene, args.grid)
            for b in rejected:
                draw_box(canvas, b[:4], GATED_COLOUR, f"GATED {b[4]:.2f}")
            cv2.rectangle(canvas, (0, 0), (w, 24), (0, 0, 0), -1)
            cv2.putText(canvas, f"{args.scene}  conf={args.conf}  gate_tol={args.gate_tol}"
                                f"  green grid = {args.grid:g} m",
                        (8, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1,
                        cv2.LINE_AA)
            writer.write(canvas)
        frame_i += 1

    cap.release()
    pose_context.__exit__(None, None, None)
    if writer is not None:
        writer.release()
    elapsed = time.perf_counter() - t0

    # ------------------------------------------------------------ per-track speeds
    by_track: dict[int, list[dict]] = defaultdict(list)
    for r in rows:
        by_track[r["track"]].append(r)

    track_stats = []
    all_raw, all_m, all_px = [], [], []
    moving_m: list = []
    static_tracks: set = set()
    for tid, rs in sorted(by_track.items()):
        if len(rs) < args.min_track:
            continue
        t = np.array([r["t_s"] for r in rs])
        pm = np.array([[r["X_m"], r["Y_m"]] for r in rs])
        pp = np.array([[r["x_px"], r["y_px"]] for r in rs])

        win = args.speed_window
        v_raw = windowed_speeds(pm, t, win)
        # Kalman in METRES: smooth where motion is actually linear.
        v_m = windowed_speeds(smooth(pm, t, args.q_m, args.r_m), t, win)
        # Kalman in PIXELS: smooth first, then map. This is the thing we claim is wrong.
        v_px = windowed_speeds(scene.to_floor(smooth(pp, t, args.q_px, args.r_px)), t, win)

        all_raw.append(v_raw)
        all_m.append(v_m)
        all_px.append(v_px)

        # Static-object filter: scenery that looks like a person does not go anywhere.
        disp = max_window_displacement(pm, t, args.static_window)
        moving = disp >= args.static_move_m
        if moving:
            moving_m.append(v_m)
        else:
            static_tracks.add(tid)
        track_stats.append({
            "track": tid, "n": len(rs), "dur_s": round(float(t[-1] - t[0]), 2),
            "path_m": round(float(np.linalg.norm(np.diff(pm, axis=0), axis=1).sum()), 2),
            "net_m": round(disp, 2), "moving": moving,
            "pose": round(float(np.mean([r["pose_found"] for r in rs])), 3),
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
    p(f"FLOOR TRACKING + MEDIAPIPE POSE   [{args.run}]   {args.video.name}")
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
        if n_gated < 20:
            p(f"    Only {n_gated} boxes were rejected, too few to show the gate removes")
            p("    junk. On a clip with no confirmed false positives this is expected,")
            p("    and the gate's usefulness stays unproven rather than disproven.")
        else:
            p("    A high rejection rate in the low confidence band and a near-zero one in")
            p("    the high band is the result we want: junk removed, real people kept.")

    p(f"\n[2] TRACKS  (>= {args.min_track} frames)")
    if not track_stats:
        p("    none. Either nothing was detected, or every track is shorter than the")
        p("    minimum - which is itself the finding.")
    else:
        p(f"    {'id':>4}{'n':>6}{'dur_s':>8}{'path_m':>9}{'net_m':>8}{'pose':>7}"
          f"{'v_raw':>8}{'v_kf_m':>9}{'v_kf_px':>9}  kind")
        for s in track_stats:
            p(f"    {s['track']:>4}{s['n']:>6}{s['dur_s']:>8}{s['path_m']:>9}"
              f"{s['net_m']:>8}{s['pose']:>7.0%}{s['v_raw']:>8}{s['v_kf_m']:>9}"
              f"{s['v_kf_px']:>9}  {'person' if s['moving'] else 'SCENERY'}")
        p("    path_m is distance travelled including jitter; net_m is how far it")
        p("    actually got. Scenery accumulates path without displacement.")

    p(f"\n[3] IS THE CALIBRATION BELIEVABLE?  (walking is {WALK_MS[0]}-{WALK_MS[1]} m/s,"
      f" speed window = {args.speed_window:.2f}s)")
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

    p(f"\n[4b] STATIC-OBJECT FILTER  (net displacement < {args.static_move_m} m over any "
      f"{args.static_window:.0f}s window)")
    if not track_stats:
        p("    no tracks to classify")
        n_static = n_static_frames = 0
    else:
        stat = [s for s in track_stats if not s["moving"]]
        mov = [s for s in track_stats if s["moving"]]
        n_static, n_static_frames = len(stat), sum(s["n"] for s in stat)
        n_tot_frames = sum(s["n"] for s in track_stats)
        p(f"    scenery: {n_static} tracks, {n_static_frames} frames "
          f"({n_static_frames / max(1, n_tot_frames):.1%} of tracked frames)")
        p(f"    people : {len(mov)} tracks, {n_tot_frames - n_static_frames} frames")
        if stat:
            p(f"    pose recall  scenery={np.mean([s['pose'] for s in stat]):.1%}   "
              f"people={np.mean([s['pose'] for s in mov]) if mov else float('nan'):.1%}")
        p("    A human-sized object standing on the floor is geometrically PERFECT, so")
        p("    the height gate cannot reject it. Only motion separates the two. In a")
        p("    restricted zone a static false positive would alarm forever, so this")
        p("    filter is what makes the intrusion signal usable, not the height gate.")
        vmov = cat(moving_m)
        if len(vmov) and len(vm):
            s_all, s_mov = sanity_report(vm), sanity_report(vmov)
            p(f"    speed median  all tracks={s_all['median_ms']:.2f} m/s  ->  "
              f"people only={s_mov['median_ms']:.2f} m/s")
            p(f"    in walking range  all={s_all['frac_in_walking_range']:.1%}  ->  "
              f"people only={s_mov['frac_in_walking_range']:.1%}")

    p(f"\n[5] MEDIAPIPE POSE   found={n_pose}  missed={n_pose_missed}  "
      f"recall against kept YOLO boxes={n_pose / max(1, n_pose + n_pose_missed):.1%}")
    p(f"    model={args.pose_model}  target crop height={args.pose_target_h}px  "
      f"confidence={args.pose_conf:.2f}")
    p("    Pose recall is relative to kept YOLO detections, not ground truth.")
    p(f"\n[6] SPEED   {n_frames} frames in {elapsed:.1f}s  "
      f"= {n_frames / max(elapsed, 1e-6):.1f} fps end-to-end (YOLO + pose + homography)")

    text = "\n".join(lines)
    print("\n" + text)

    (rep_dir / f"{args.video.stem}__floor_report.txt").write_text(text, encoding="utf-8")
    with (rep_dir / f"{args.video.stem}__floor_tracks.csv").open(
            "w", newline="", encoding="utf-8") as f:
        wri = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else
                             ["frame", "t_s", "track", "x_px", "y_px", "X_m", "Y_m",
                              "box_h", "score", "pose_found", "pose_visibility", "crop_scale"])
        wri.writeheader()
        wri.writerows(rows)
    with (rep_dir / f"{args.video.stem}__skeletons.csv").open(
            "w", newline="", encoding="utf-8") as f:
        fields = ["frame", "t_s", "track", "landmark", "x_px", "y_px",
                  "visibility", "presence", "world_x", "world_y", "world_z",
                  "mean_visibility", "crop_scale"]
        wri = csv.DictWriter(f, fieldnames=fields, restval="")
        wri.writeheader()
        wri.writerows(skeleton_rows)

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
        "pose": {"model": args.pose_model, "conf": args.pose_conf,
                 "target_h": args.pose_target_h, "margin": args.pose_margin,
                 "max_scale": args.pose_max_scale,
                 "detections": n_pose + n_pose_missed, "found": n_pose,
                 "missed": n_pose_missed},
    }, indent=2), encoding="utf-8")
    print(f"\nwrote {rep_dir}")
    if writer is not None:
        print(f"wrote {vid_dir}")

    s_m = sanity_report(cat(all_m))
    return {
        "clip": args.video.stem, "scene": scene.name,
        "frame": f"{w}x{h}", "frames": n_frames,
        "height_model": "yes" if model is not None else "none",
        "height_crop": bool(args.height_crop and model is not None),
        "boxes": n_boxes, "gated": n_gated,
        "gate_rate": round(n_gated / n_boxes, 4) if n_boxes else float("nan"),
        "n_tracks": len(track_stats),
        "n_scenery": n_static, "scenery_frames": n_static_frames,
        "median_ms_people": round(sanity_report(cat(moving_m)).get("median_ms", float("nan")), 3),
        "in_walk_people": round(
            sanity_report(cat(moving_m)).get("frac_in_walking_range", float("nan")), 4),
        "longest_track": max((s["n"] for s in track_stats), default=0),
        "pose_found": n_pose, "pose_missed": n_pose_missed,
        "pose_recall": round(n_pose / max(1, n_pose + n_pose_missed), 4),
        "median_ms": round(s_m.get("median_ms", float("nan")), 3),
        "p90_ms": round(s_m.get("p90_ms", float("nan")), 3),
        "max_ms": round(s_m.get("max_ms", float("nan")), 3),
        "in_walk_range": round(s_m.get("frac_in_walking_range", float("nan")), 4),
        "impossible": round(s_m.get("frac_impossible", float("nan")), 4),
        "fps_end_to_end": round(n_frames / max(elapsed, 1e-6), 2),
    }


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
    ap.add_argument("--speed-window", type=float, default=0.40,
                    help="seconds of displacement to measure speed over (0 = per-frame)")
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--max-frames", type=int, default=0)
    ap.add_argument("--grid", type=float, default=1.0)
    ap.add_argument("--q-m", type=float, default=0.5, help="process noise, metres")
    ap.add_argument("--r-m", type=float, default=0.10, help="measurement noise, metres")
    ap.add_argument("--q-px", type=float, default=200.0, help="process noise, pixels")
    ap.add_argument("--r-px", type=float, default=25.0, help="measurement noise, pixels")
    ap.add_argument("--no-video", action="store_true")
    ap.add_argument("--pose-model", choices=["lite", "full", "heavy"], default="full")
    ap.add_argument("--pose-conf", type=float, default=0.5)
    ap.add_argument("--pose-target-h", type=int, default=320)
    ap.add_argument("--pose-margin", type=float, default=0.20)
    ap.add_argument("--pose-max-scale", type=float, default=8.0)
    ap.add_argument("--static-window", type=float, default=3.0,
                    help="seconds over which a real person must have gone somewhere")
    ap.add_argument("--static-move-m", type=float, default=0.5,
                    help="net metres below which a track is scenery, not a person")
    ap.add_argument("--height-crop", action="store_true",
                    help="size the pose crop from the calibrated expected height "
                         "instead of the detected box height")
    ap.add_argument("--run", default="test run 4")
    ap.add_argument("--seed", type=int, default=0)
    run(ap.parse_args())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
