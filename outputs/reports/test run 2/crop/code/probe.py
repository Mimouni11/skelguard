"""Core of the WP0 multi-person pose probe, shared by the single-clip and batch runners.

Two modes:

  full   MediaPipe PoseLandmarker on the whole frame, up to `num_poses` people.
         What the plan calls "MediaPipe first".

  crop   YOLOv8n finds person boxes; each box is expanded, cropped, upscaled so the
         person is ~`crop_target_h` px tall, and MediaPipe runs single-pose on that
         crop. Landmarks are mapped back to full-frame coordinates.
         What the plan calls the fallback. Measured here before committing to it.

There is no ground truth on found footage, so YOLOv8n person detection is the reference:
a YOLO person box with no matching pose counts as a MediaPipe miss. That is a comparison
against another detector, not against truth - say so in the report.
"""

from __future__ import annotations

import json
import platform
import shutil
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from skelguard import paths

# BlazePose 33-landmark skeleton, hard-coded so we do not depend on mp.solutions.
POSE_CONNECTIONS: tuple[tuple[int, int], ...] = (
    (0, 1), (1, 2), (2, 3), (3, 7), (0, 4), (4, 5), (5, 6), (6, 8), (9, 10),
    (11, 12), (11, 13), (13, 15), (15, 17), (15, 19), (15, 21), (17, 19),
    (12, 14), (14, 16), (16, 18), (16, 20), (16, 22), (18, 20),
    (11, 23), (12, 24), (23, 24),
    (23, 25), (25, 27), (27, 29), (27, 31), (29, 31),
    (24, 26), (26, 28), (28, 30), (28, 32), (30, 32),
)

POSE_COLOURS = [(0, 220, 255), (0, 255, 120), (255, 140, 0), (255, 0, 200), (120, 120, 255)]
YOLO_COLOUR = (90, 90, 90)
MISS_COLOUR = (0, 0, 255)
NOT_SCORED_COLOUR = (60, 60, 60)   # a real person deliberately left out of the scoring

MODEL_FILES = {"lite": paths.POSE_LITE, "full": paths.POSE_FULL, "heavy": paths.POSE_HEAVY}
VIDEO_EXTS = (".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v", ".mpg", ".mpeg", ".wmv")

FPS_TARGET = 15.0

# Measured in test run 1 and 2: full-frame MediaPipe detects ~0% of people below 150 px
# of height and is only reliable above ~250 px. The crop pass exists to clear that bar.
RELIABLE_PERSON_PX = 250


@dataclass
class ProbeConfig:
    model: str = "full"
    num_poses: int = 5
    width: int = 0          # 0 = native. We standardise on native 720p.
    stride: int = 1
    max_frames: int = 0
    det_conf: float = 0.5
    track_conf: float = 0.5
    yolo_conf: float = 0.35
    iou: float = 0.30
    use_yolo: bool = True
    write_video: bool = True
    run: str = "test run 1"   # outputs land in outputs/<kind>/<run>/
    # Score against the single largest YOLO person only. Needed whenever num_poses is
    # capped below the number of people present, or every extra person is a free "miss"
    # and the miss rate just measures the cap.
    primary_only: bool = False
    # --- crop mode ---
    crop: bool = False
    crop_margin: float = 0.20     # expand the person box by this fraction before cropping
    crop_target_h: int = 320      # upscale until the person is about this tall, in px
    crop_max_scale: float = 8.0   # never upscale more than this; past it there is no detail


@dataclass
class ProbeResult:
    clip: str
    frames: pd.DataFrame
    people: pd.DataFrame
    meta: dict = field(default_factory=dict)
    video_path: Path | None = None
    frames_csv: Path | None = None
    people_csv: Path | None = None


# --------------------------------------------------------------------------- snapshot
SNAPSHOT_FILES = (
    "src/skelguard/probe.py",
    "src/skelguard/paths.py",
    "scripts/probe_batch.py",
    "scripts/probe_pose.py",
)


def snapshot_run(run: str, cfg: ProbeConfig, clips: list[Path] | None = None) -> Path:
    """Freeze the code and settings that produced a run, next to its results.

    Three weeks from now the report will quote a number from a run. Without the exact
    script and parameters beside it, that number cannot be explained or repeated.
    """
    root = paths.ROOT
    out = paths.run_dir(paths.OUT_REPORTS, run)
    code = out / "code"
    code.mkdir(parents=True, exist_ok=True)

    copied = []
    for rel in SNAPSHOT_FILES:
        src = root / rel
        if src.exists():
            shutil.copy2(src, code / Path(rel).name)
            copied.append(rel)

    versions: dict[str, str] = {}
    for name, mod in (("mediapipe", "mediapipe"), ("ultralytics", "ultralytics"),
                      ("opencv", "cv2"), ("numpy", "numpy"), ("torch", "torch")):
        try:
            versions[name] = __import__(mod).__version__
        except Exception:  # noqa: BLE001
            pass

    try:
        commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=root,
                                capture_output=True, text=True, timeout=5).stdout.strip()
    except Exception:  # noqa: BLE001
        commit = ""

    meta = {
        "run": run,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "command": " ".join(sys.argv),
        "git_commit": commit or "(not a repo / unknown)",
        "config": asdict(cfg),
        "clips": [c.name for c in clips] if clips else [],
        "python": platform.python_version(),
        "platform": f"{platform.system()} {platform.release()}",
        "versions": versions,
        "code_snapshot": copied,
    }
    (out / "run_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return out


# --------------------------------------------------------------------------- geometry
def iou(a: np.ndarray, b: np.ndarray) -> float:
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    if inter <= 0:
        return 0.0
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    return float(inter / (area_a + area_b - inter))


def match_greedy(yolo: list[np.ndarray], poses: list[np.ndarray], thr: float) -> dict[int, int]:
    """Greedy IoU matching, YOLO box index -> pose index."""
    pairs = sorted(
        ((iou(y, p), yi, pi) for yi, y in enumerate(yolo) for pi, p in enumerate(poses)),
        reverse=True,
    )
    out: dict[int, int] = {}
    used: set[int] = set()
    for score, yi, pi in pairs:
        if score < thr or yi in out or pi in used:
            continue
        out[yi] = pi
        used.add(pi)
    return out


def landmarks_px(lms, w: int, h: int) -> np.ndarray:
    """MediaPipe normalised landmarks -> (33, 2) pixel coordinates."""
    return np.stack([
        np.clip([lm.x for lm in lms], 0, 1) * w,
        np.clip([lm.y for lm in lms], 0, 1) * h,
    ], axis=1).astype(np.float32)


def box_of(pts: np.ndarray) -> np.ndarray:
    return np.array([pts[:, 0].min(), pts[:, 1].min(),
                     pts[:, 0].max(), pts[:, 1].max()], dtype=np.float32)


def mean_visibility(lms) -> float:
    return float(np.mean([getattr(l, "visibility", 0.0) for l in lms]))


def any_overlap(boxes: list[np.ndarray], thr: float = 0.05) -> bool:
    """True if any two person boxes overlap - our proxy for 'people are crossing'."""
    for i in range(len(boxes)):
        for j in range(i + 1, len(boxes)):
            if iou(boxes[i], boxes[j]) > thr:
                return True
    return False


def expand_box(box: np.ndarray, margin: float, w: int, h: int) -> np.ndarray:
    """Grow a box by `margin` on each side, clamped to the frame."""
    bw, bh = box[2] - box[0], box[3] - box[1]
    dx, dy = bw * margin, bh * margin
    return np.array([max(0.0, box[0] - dx), max(0.0, box[1] - dy),
                     min(float(w), box[2] + dx), min(float(h), box[3] + dy)],
                    dtype=np.float32)


# --------------------------------------------------------------------------- drawing
def draw_pose_px(frame, pts: np.ndarray, colour) -> None:
    p = pts.astype(int)
    for a, b in POSE_CONNECTIONS:
        if a < len(p) and b < len(p):
            cv2.line(frame, tuple(p[a]), tuple(p[b]), colour, 2, cv2.LINE_AA)
    for x, y in p:
        cv2.circle(frame, (int(x), int(y)), 2, colour, -1, cv2.LINE_AA)


def draw_box(frame, box: np.ndarray, colour, label: str = "", thick: int = 2) -> None:
    x1, y1, x2, y2 = [int(v) for v in box]
    cv2.rectangle(frame, (x1, y1), (x2, y2), colour, thick)
    if label:
        cv2.putText(frame, label, (x1, max(14, y1 - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, colour, 1, cv2.LINE_AA)


# --------------------------------------------------------------------------- detectors
def make_landmarker(cfg: ProbeConfig, image_mode: bool = False, num_poses: int | None = None):
    """A PoseLandmarker. image_mode=True for independent crops (no temporal state)."""
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision

    path = MODEL_FILES[cfg.model]
    if not path.exists():
        raise SystemExit(
            f"[FAIL] {path.name} is missing.\n"
            f"       uv run python scripts/download_models.py"
            + ("" if cfg.model == "full" else " --all")
        )
    opts = vision.PoseLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=str(path)),
        running_mode=vision.RunningMode.IMAGE if image_mode else vision.RunningMode.VIDEO,
        num_poses=num_poses if num_poses is not None else cfg.num_poses,
        min_pose_detection_confidence=cfg.det_conf,
        min_pose_presence_confidence=cfg.det_conf,
        min_tracking_confidence=cfg.track_conf,
        output_segmentation_masks=False,
    )
    return vision.PoseLandmarker.create_from_options(opts)


def make_yolo():
    from ultralytics import YOLO

    return YOLO(str(paths.MODELS / "yolov8n.pt"))  # fetched here on first use (~6 MB)


def yolo_person_boxes(model, conf: float, frame_bgr) -> list[np.ndarray]:
    res = model.predict(frame_bgr, classes=[0], conf=conf, verbose=False)[0]
    if res.boxes is None or len(res.boxes) == 0:
        return []
    return [b.astype(np.float32) for b in res.boxes.xyxy.cpu().numpy()]


def pose_on_crop(landmarker, frame_bgr, box: np.ndarray, cfg: ProbeConfig,
                 w: int, h: int) -> tuple[np.ndarray | None, float, float]:
    """Crop one person, upscale, run single-pose. Returns (landmarks_px_full, vis, scale).

    Landmarks come back in FULL-FRAME pixel coordinates, so everything downstream is
    unchanged by the crop.
    """
    import mediapipe as mp

    ex = expand_box(box, cfg.crop_margin, w, h)
    x0, y0, x1, y1 = [int(round(v)) for v in ex]
    if x1 - x0 < 8 or y1 - y0 < 8:
        return None, 0.0, 1.0

    crop = frame_bgr[y0:y1, x0:x1]
    ch, cw = crop.shape[:2]

    person_h = max(1.0, float(box[3] - box[1]))
    scale = float(np.clip(cfg.crop_target_h / person_h, 1.0, cfg.crop_max_scale))
    if scale > 1.0:
        crop = cv2.resize(crop, (int(cw * scale), int(ch * scale)),
                          interpolation=cv2.INTER_CUBIC)

    rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
    res = landmarker.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb))
    if not res.pose_landmarks:
        return None, 0.0, scale

    lms = res.pose_landmarks[0]
    # normalised-in-crop -> pixels in the (unscaled) crop -> pixels in the full frame
    pts = landmarks_px(lms, cw, ch) + np.array([x0, y0], dtype=np.float32)
    return pts, mean_visibility(lms), scale


# --------------------------------------------------------------------------- the probe
def probe_clip(video: Path, cfg: ProbeConfig, yolo=None, verbose: bool = True) -> ProbeResult:
    """Run the probe over one clip. Returns tidy dataframes."""
    import mediapipe as mp

    video = Path(video)
    if not video.exists():
        raise SystemExit(f"[FAIL] no such video: {video}")
    if cfg.crop and not cfg.use_yolo:
        raise SystemExit("[FAIL] crop mode needs YOLO to find the person boxes")

    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise SystemExit(f"[FAIL] could not open: {video}")

    src_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    src_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    src_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    scale = cfg.width / src_w if cfg.width else 1.0
    w, h = int(src_w * scale), int(src_h * scale)
    mode = "crop" if cfg.crop else "full"

    if verbose:
        print(f"  source {src_w}x{src_h} @ {src_fps:.1f} fps, {n_frames} frames"
              f"  ->  infer at {w}x{h}  [{mode} mode]")

    paths.ensure_dirs()
    vid_dir = paths.run_dir(paths.OUT_VIDEOS, cfg.run)
    log_dir = paths.run_dir(paths.OUT_LOGS, cfg.run)
    stem = f"{video.stem}__{mode}_{cfg.model}_np{cfg.num_poses}_w{w}"
    out_video = vid_dir / f"probe_{stem}.mp4"
    frames_csv = log_dir / f"probe_{stem}_frames.csv"
    people_csv = log_dir / f"probe_{stem}_people.csv"

    writer = None
    if cfg.write_video:
        writer = cv2.VideoWriter(str(out_video), cv2.VideoWriter_fourcc(*"mp4v"),
                                 src_fps / cfg.stride, (w, h))
    if cfg.use_yolo and yolo is None:
        yolo = make_yolo()

    frame_rows: list[dict] = []
    people_rows: list[dict] = []

    # crop mode runs single-pose on independent crops -> IMAGE mode, no temporal state
    landmarker = make_landmarker(cfg, image_mode=cfg.crop, num_poses=1 if cfg.crop else None)

    with landmarker as lm:
        idx, kept = -1, 0
        t_wall = time.perf_counter()
        while True:
            ok, frame_bgr = cap.read()
            if not ok:
                break
            idx += 1
            if idx % cfg.stride:
                continue
            if cfg.max_frames and kept >= cfg.max_frames:
                break
            kept += 1

            if scale != 1.0:
                frame_bgr = cv2.resize(frame_bgr, (w, h), interpolation=cv2.INTER_AREA)
            t_sec = idx / src_fps

            # ---------------------------------------------------- person boxes (YOLO)
            yolo_boxes: list[np.ndarray] = []
            yolo_ms = 0.0
            if yolo is not None:
                t1 = time.perf_counter()
                yolo_boxes = yolo_person_boxes(yolo, cfg.yolo_conf, frame_bgr)
                yolo_ms = (time.perf_counter() - t1) * 1000

            crossing = any_overlap(yolo_boxes) if yolo_boxes else False

            # the set we actually score against, as indices into yolo_boxes
            eval_idx = list(range(len(yolo_boxes)))
            if cfg.primary_only and yolo_boxes:
                eval_idx = [max(eval_idx,
                                key=lambda i: ((yolo_boxes[i][2] - yolo_boxes[i][0])
                                               * (yolo_boxes[i][3] - yolo_boxes[i][1])))]
            eval_boxes = [yolo_boxes[i] for i in eval_idx]

            # ---------------------------------------------------- pose
            poses_px: list[np.ndarray] = []
            vis_list: list[float] = []
            scales: list[float] = []
            per_person: list[dict] = []   # crop mode only, aligned with eval_boxes

            t0 = time.perf_counter()
            if cfg.crop:
                for box in eval_boxes:
                    pts, vis, sc = pose_on_crop(lm, frame_bgr, box, cfg, w, h)
                    per_person.append({"pts": pts, "vis": vis, "scale": sc})
                    if pts is not None:
                        poses_px.append(pts)
                        vis_list.append(vis)
                        scales.append(sc)
            else:
                rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                res = lm.detect_for_video(
                    mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb),
                    int(idx / src_fps * 1000))
                for p in res.pose_landmarks:
                    poses_px.append(landmarks_px(p, w, h))
                    vis_list.append(mean_visibility(p))
            pose_ms = (time.perf_counter() - t0) * 1000

            pose_boxes = [box_of(p) for p in poses_px]

            # ---------------------------------------------------- match
            if cfg.crop:
                # one attempt per person box, so the pairing is known by construction
                matches = {ei: pi for pi, ei in
                           enumerate(i for i, d in enumerate(per_person)
                                     if d["pts"] is not None)}
                n_ghost = 0   # pose only ever runs inside a person box
            else:
                matches = match_greedy(eval_boxes, pose_boxes, cfg.iou) if eval_boxes else {}
                n_ghost = len(pose_boxes) - len(matches)

            frame_rows.append({
                "clip": video.stem, "frame": idx, "t": round(t_sec, 3), "mode": mode,
                "n_pose": len(pose_boxes),
                "n_yolo": len(yolo_boxes),
                "n_yolo_eval": len(eval_boxes),
                "n_matched": len(matches), "n_missed": len(eval_boxes) - len(matches),
                "n_ghost": n_ghost,
                "crossing": int(crossing),
                "pose_ms": round(pose_ms, 2), "yolo_ms": round(yolo_ms, 2),
                "mean_visibility": round(float(np.mean(vis_list)), 3) if vis_list else np.nan,
                "mean_crop_scale": round(float(np.mean(scales)), 2) if scales else np.nan,
            })

            for ei, box in enumerate(eval_boxes):
                row = {
                    "clip": video.stem, "frame": idx, "t": round(t_sec, 3), "mode": mode,
                    "box_h": round(float(box[3] - box[1]), 1),
                    "box_h_frac": round(float((box[3] - box[1]) / h), 4),
                    "detected": int(ei in matches), "crossing": int(crossing),
                }
                if cfg.crop:
                    d = per_person[ei]
                    row["crop_scale"] = round(d["scale"], 2)
                    row["vis"] = round(d["vis"], 3) if d["pts"] is not None else np.nan
                    row["upscaled_h"] = round(float(box[3] - box[1]) * d["scale"], 1)
                people_rows.append(row)

            # ---------------------------------------------------- draw
            if writer is not None:
                scored = {gi: ei for ei, gi in enumerate(eval_idx)}
                for gi, box in enumerate(yolo_boxes):
                    if gi not in scored:
                        draw_box(frame_bgr, box, NOT_SCORED_COLOUR, "not scored", 1)
                        continue
                    hit = scored[gi] in matches
                    draw_box(frame_bgr, box, YOLO_COLOUR if hit else MISS_COLOUR,
                             "" if hit else "MISSED", 1 if hit else 2)
                matched_poses = set(matches.values())
                for pi, pts in enumerate(poses_px):
                    ghost = (not cfg.crop) and yolo is not None and pi not in matched_poses
                    colour = MISS_COLOUR if ghost else POSE_COLOURS[pi % len(POSE_COLOURS)]
                    draw_pose_px(frame_bgr, pts, colour)
                    draw_box(frame_bgr, pose_boxes[pi], colour,
                             f"GHOST? {pi}" if ghost else f"pose {pi}", 1)
                hud = (f"t={t_sec:5.1f}s  {mode}  pose={len(pose_boxes)}"
                       f"  yolo={len(yolo_boxes)}  {1000 / max(pose_ms, 1e-6):5.1f} fps"
                       + ("  CROSSING" if crossing else ""))
                cv2.rectangle(frame_bgr, (0, 0), (w, 26), (0, 0, 0), -1)
                cv2.putText(frame_bgr, hud, (8, 18), cv2.FONT_HERSHEY_SIMPLEX,
                            0.55, (255, 255, 255), 1, cv2.LINE_AA)
                writer.write(frame_bgr)

            if verbose and kept % 100 == 0:
                print(f"    ... {kept} frames, t={t_sec:.1f}s", flush=True)

        wall = time.perf_counter() - t_wall

    cap.release()
    if writer is not None:
        writer.release()

    frames = pd.DataFrame(frame_rows)
    people = pd.DataFrame(people_rows)
    frames.to_csv(frames_csv, index=False)
    if not people.empty:
        people.to_csv(people_csv, index=False)

    return ProbeResult(
        clip=video.stem, frames=frames, people=people,
        meta={"src_w": src_w, "src_h": src_h, "src_fps": src_fps, "mode": mode,
              "infer_w": w, "infer_h": h, "wall_s": wall, "n_processed": len(frames)},
        video_path=out_video if writer is not None else None,
        frames_csv=frames_csv, people_csv=people_csv,
    )


# --------------------------------------------------------------------------- stats
def drop_events(frames: pd.DataFrame, min_frames: int = 2) -> list[tuple[float, float, int, int]]:
    """Runs of >= min_frames consecutive frames where MediaPipe found fewer people than YOLO.

    Returns (t_start, t_end, worst_miss_count, n_frames) per run.
    """
    out: list[tuple[float, float, int, int]] = []
    start: float | None = None
    last, worst, run_len = 0.0, 0, 0

    for _, r in frames.iterrows():
        if r["n_missed"] > 0:
            if start is None:
                start, worst, run_len = float(r["t"]), 0, 0
            worst = max(worst, int(r["n_missed"]))
            last = float(r["t"])
            run_len += 1
        elif start is not None:
            if run_len >= min_frames:
                out.append((start, last, worst, run_len))
            start = None

    if start is not None and run_len >= min_frames:
        out.append((start, last, worst, run_len))
    return out


# Fixed pixel bands, so clips of different resolutions stay comparable.
PX_BANDS = ((0, 60, "<60px"), (60, 100, "60-100"), (100, 150, "100-150"),
            (150, 250, "150-250"), (250, 10_000, ">250px"))


def stats_for(frames: pd.DataFrame, people: pd.DataFrame) -> dict:
    """Condense one clip (or a pool of clips) into the numbers the decision needs."""
    s: dict = {}
    s["n_frames"] = len(frames)
    s["duration_s"] = round(float(frames["t"].max() - frames["t"].min()), 1) if len(frames) else 0
    s["mode"] = frames["mode"].iloc[0] if len(frames) and "mode" in frames else "full"

    # Speed. In full mode MediaPipe short-circuits when it finds nobody, so a pooled
    # median flatters it badly on clips where it mostly fails.
    s["pose_ms_median"] = float(frames["pose_ms"].median())
    s["pose_ms_p95"] = float(frames["pose_ms"].quantile(0.95))
    hit = frames[frames["n_pose"] > 0]
    s["pose_ms_when_detecting"] = float(hit["pose_ms"].median()) if len(hit) else float("nan")
    s["fps_median"] = 1000.0 / s["pose_ms_median"] if s["pose_ms_median"] else float("nan")
    s["fps_when_detecting"] = (1000.0 / s["pose_ms_when_detecting"]
                               if s["pose_ms_when_detecting"] == s["pose_ms_when_detecting"]
                               else float("nan"))
    # Crop mode pays YOLO on every frame, so the honest end-to-end cost includes it.
    tot = float((frames["pose_ms"] + frames["yolo_ms"]).median())
    s["fps_end_to_end"] = 1000.0 / tot if tot else float("nan")

    if "n_ghost" in frames:
        s["n_ghost"] = int(frames["n_ghost"].sum())
        total_poses = int(frames["n_pose"].sum())
        s["ghost_rate"] = (s["n_ghost"] / total_poses) if total_poses else float("nan")
        mins = (frames["t"].max() - frames["t"].min()) / 60.0 if len(frames) else 0.0
        s["ghosts_per_min"] = (s["n_ghost"] / mins) if mins > 0 else float("nan")
        s["frames_with_ghost"] = float((frames["n_ghost"] > 0).mean())

    s["frames_with_no_pose"] = float((frames["n_pose"] == 0).mean())
    s["mean_poses"] = float(frames["n_pose"].mean())
    s["max_poses"] = int(frames["n_pose"].max()) if len(frames) else 0
    s["crossing_frame_rate"] = float(frames["crossing"].mean())
    s["count_agreement"] = float((frames["n_pose"] == frames["n_yolo_eval"]).mean()) \
        if "n_yolo_eval" in frames else float("nan")

    if people.empty:
        s["n_people_det"] = 0
        s["miss_rate"] = float("nan")
        return s

    s["n_people_det"] = len(people)
    s["recall"] = float(people["detected"].mean())
    s["miss_rate"] = 1.0 - s["recall"]
    if "vis" in people:
        got = people[people["detected"] == 1]
        s["mean_vis_detected"] = float(got["vis"].mean()) if len(got) else float("nan")
    if "crop_scale" in people:
        s["mean_crop_scale"] = float(people["crop_scale"].mean())

    # Recall by apparent size, in fixed pixel bands AND as a fraction of frame height.
    for lo, hi, label in PX_BANDS:
        g = people[(people["box_h"] >= lo) & (people["box_h"] < hi)]
        s[f"px_{label}_n"] = len(g)
        s[f"px_{label}"] = float(g["detected"].mean()) if len(g) else float("nan")

    bands = [(0.0, 0.15, "far"), (0.15, 0.30, "mid"), (0.30, 1.01, "near")]
    for lo, hi, key in bands:
        g = people[(people["box_h_frac"] >= lo) & (people["box_h_frac"] < hi)]
        s[f"recall_{key}"] = float(g["detected"].mean()) if len(g) else float("nan")
        s[f"n_{key}"] = len(g)

    cross = people[people["crossing"] == 1]
    s["recall_crossing"] = float(cross["detected"].mean()) if len(cross) else float("nan")
    s["n_crossing"] = len(cross)
    return s


def decide(miss_rate: float, fps: float, mode: str = "full") -> tuple[str, str]:
    """The rule, fixed before any numbers were seen. Returns (verdict, reasoning)."""
    if miss_rate != miss_rate:  # NaN
        return "NO DATA", "no YOLO reference was run, so there is no miss rate"
    if mode == "crop":
        if miss_rate < 0.10 and fps >= FPS_TARGET:
            v, why = ("CROP PIPELINE WORKS",
                      "the crop pass clears both the miss and the fps bar; adopt "
                      "YOLOv8 detect -> upscale -> MediaPipe single-pose")
        elif miss_rate < 0.10:
            v, why = ("CROP WORKS BUT IS SLOW",
                      f"misses are low but only {fps:.1f} fps end to end; try --model lite, "
                      "a smaller crop_target_h, or running pose every other frame")
        else:
            v, why = ("CROP IS NOT ENOUGH",
                      "cropping did not rescue recall, so the limit is image detail, not "
                      "framing; the honest conclusion is a resolution floor on the footage")
        return v, why
    if miss_rate < 0.10 and fps >= FPS_TARGET:
        v = "MEDIAPIPE FIRST"
        why = "misses under 10% and the fps target holds; keep YOLOv8+ByteTrack on the shelf"
    elif miss_rate < 0.25:
        v = "MEDIAPIPE + CROP PASS"
        why = ("misses between 10% and 25%; keep MediaPipe primary but add a per-person-crop "
               "pass for the small and occluded cases")
    else:
        v = "FALL BACK TO YOLOV8 + BYTETRACK"
        why = ("misses above 25%; use YOLOv8+ByteTrack for detection and tracking and run "
               "MediaPipe on each person crop. Everything downstream is unchanged")
    if fps < FPS_TARGET:
        why += f"; also under target at {fps:.1f} fps, try --model lite or a smaller --width"
    return v, why
