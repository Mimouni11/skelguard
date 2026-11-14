"""Core of the WP0 multi-person pose probe, shared by the single-clip and batch runners.

No ground truth exists on found footage, so YOLOv8n person detection is the reference
count: a YOLO person box that no MediaPipe pose matches is counted as a MediaPipe miss.
That is a comparison against another detector, not against truth - say so in the report.
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

    Three weeks from now the report will quote a number from 'test run 1'. Without the
    exact script and parameters beside it, that number cannot be explained or repeated.
    """
    root = paths.ROOT
    out = paths.run_dir(paths.OUT_REPORTS, run)
    code = out / "code"
    code.mkdir(parents=True, exist_ok=True)

    copied = []
    for rel in SNAPSHOT_FILES:
        src = root / rel
        if src.exists():
            dst = code / Path(rel).name
            shutil.copy2(src, dst)
            copied.append(rel)

    try:
        import mediapipe, ultralytics  # noqa: E401

        versions = {"mediapipe": mediapipe.__version__,
                    "ultralytics": ultralytics.__version__}
    except Exception:  # noqa: BLE001
        versions = {}
    try:
        import cv2 as _cv2
        import numpy as _np

        versions |= {"opencv": _cv2.__version__, "numpy": _np.__version__}
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


def landmarks_to_box(lms, w: int, h: int) -> np.ndarray:
    xs = np.clip([lm.x for lm in lms], 0, 1) * w
    ys = np.clip([lm.y for lm in lms], 0, 1) * h
    return np.array([xs.min(), ys.min(), xs.max(), ys.max()], dtype=np.float32)


def any_overlap(boxes: list[np.ndarray], thr: float = 0.05) -> bool:
    """True if any two person boxes overlap - our proxy for 'people are crossing'."""
    for i in range(len(boxes)):
        for j in range(i + 1, len(boxes)):
            if iou(boxes[i], boxes[j]) > thr:
                return True
    return False


# --------------------------------------------------------------------------- drawing
def draw_pose(frame, lms, w: int, h: int, colour) -> None:
    pts = [(int(np.clip(lm.x, 0, 1) * w), int(np.clip(lm.y, 0, 1) * h)) for lm in lms]
    for a, b in POSE_CONNECTIONS:
        if a < len(pts) and b < len(pts):
            cv2.line(frame, pts[a], pts[b], colour, 2, cv2.LINE_AA)
    for x, y in pts:
        cv2.circle(frame, (x, y), 2, colour, -1, cv2.LINE_AA)


def draw_box(frame, box: np.ndarray, colour, label: str = "", thick: int = 2) -> None:
    x1, y1, x2, y2 = [int(v) for v in box]
    cv2.rectangle(frame, (x1, y1), (x2, y2), colour, thick)
    if label:
        cv2.putText(frame, label, (x1, max(14, y1 - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, colour, 1, cv2.LINE_AA)


# --------------------------------------------------------------------------- detectors
def make_landmarker(cfg: ProbeConfig):
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
        running_mode=vision.RunningMode.VIDEO,  # timestamps -> temporal continuity
        num_poses=cfg.num_poses,
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


# --------------------------------------------------------------------------- the probe
def probe_clip(video: Path, cfg: ProbeConfig, yolo=None, verbose: bool = True) -> ProbeResult:
    """Run MediaPipe (and optionally YOLO) over one clip. Returns tidy dataframes."""
    import mediapipe as mp

    video = Path(video)
    if not video.exists():
        raise SystemExit(f"[FAIL] no such video: {video}")

    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise SystemExit(f"[FAIL] could not open: {video}")

    src_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    src_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    src_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    scale = cfg.width / src_w if cfg.width else 1.0
    w, h = int(src_w * scale), int(src_h * scale)

    if verbose:
        print(f"  source {src_w}x{src_h} @ {src_fps:.1f} fps, {n_frames} frames"
              f"  ->  infer at {w}x{h}")

    paths.ensure_dirs()
    vid_dir = paths.run_dir(paths.OUT_VIDEOS, cfg.run)
    log_dir = paths.run_dir(paths.OUT_LOGS, cfg.run)
    stem = f"{video.stem}__{cfg.model}_np{cfg.num_poses}_w{w}"
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

    with make_landmarker(cfg) as lm:
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
            rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

            t0 = time.perf_counter()
            res = lm.detect_for_video(mp_img, int(idx / src_fps * 1000))
            pose_ms = (time.perf_counter() - t0) * 1000

            pose_boxes = [landmarks_to_box(p, w, h) for p in res.pose_landmarks]
            vis = [float(np.mean([getattr(l, "visibility", 0.0) for l in p]))
                   for p in res.pose_landmarks]

            yolo_boxes: list[np.ndarray] = []
            yolo_ms = 0.0
            if yolo is not None:
                t1 = time.perf_counter()
                yolo_boxes = yolo_person_boxes(yolo, cfg.yolo_conf, frame_bgr)
                yolo_ms = (time.perf_counter() - t1) * 1000

            # Crossing is a property of the scene, so judge it on everyone present.
            crossing = any_overlap(yolo_boxes if yolo_boxes else pose_boxes)

            # The set we actually score against, as indices into yolo_boxes.
            eval_idx = list(range(len(yolo_boxes)))
            if cfg.primary_only and yolo_boxes:
                eval_idx = [max(eval_idx,
                                key=lambda i: ((yolo_boxes[i][2] - yolo_boxes[i][0])
                                               * (yolo_boxes[i][3] - yolo_boxes[i][1])))]
            eval_boxes = [yolo_boxes[i] for i in eval_idx]

            matches = match_greedy(eval_boxes, pose_boxes, cfg.iou) if eval_boxes else {}
            t_sec = idx / src_fps

            frame_rows.append({
                "clip": video.stem, "frame": idx, "t": round(t_sec, 3),
                "n_pose": len(pose_boxes),
                "n_yolo": len(yolo_boxes),          # everyone in the scene
                "n_yolo_eval": len(eval_boxes),     # who we score against
                "n_matched": len(matches), "n_missed": len(eval_boxes) - len(matches),
                # A pose matching no YOLO person: furniture, a coat stand, a reflection.
                # These become false intrusion alerts downstream, so they are counted.
                "n_ghost": len(pose_boxes) - len(matches),
                "crossing": int(crossing),
                "pose_ms": round(pose_ms, 2), "yolo_ms": round(yolo_ms, 2),
                "mean_visibility": round(float(np.mean(vis)), 3) if vis else np.nan,
            })

            for yi, box in enumerate(eval_boxes):
                people_rows.append({
                    "clip": video.stem, "frame": idx, "t": round(t_sec, 3),
                    "box_h": round(float(box[3] - box[1]), 1),
                    "box_h_frac": round(float((box[3] - box[1]) / h), 4),
                    "detected": int(yi in matches), "crossing": int(crossing),
                })

            if writer is not None:
                scored = {gi: ei for ei, gi in enumerate(eval_idx)}
                for gi, box in enumerate(yolo_boxes):
                    if gi not in scored:  # present, but outside the scored set
                        draw_box(frame_bgr, box, NOT_SCORED_COLOUR, "not scored", 1)
                        continue
                    hit = scored[gi] in matches
                    draw_box(frame_bgr, box, YOLO_COLOUR if hit else MISS_COLOUR,
                             "" if hit else "MISSED", 1 if hit else 2)
                matched_poses = set(matches.values())
                for pi, p in enumerate(res.pose_landmarks):
                    ghost = yolo is not None and pi not in matched_poses
                    colour = MISS_COLOUR if ghost else POSE_COLOURS[pi % len(POSE_COLOURS)]
                    draw_pose(frame_bgr, p, w, h, colour)
                    draw_box(frame_bgr, pose_boxes[pi], colour,
                             f"GHOST? {pi}" if ghost else f"pose {pi}", 1)
                hud = (f"t={t_sec:5.1f}s  pose={len(pose_boxes)}  yolo={len(yolo_boxes)}"
                       f"  {1000 / max(pose_ms, 1e-6):5.1f} fps"
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
        meta={"src_w": src_w, "src_h": src_h, "src_fps": src_fps,
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


def stats_for(frames: pd.DataFrame, people: pd.DataFrame) -> dict:
    """Condense one clip (or a pool of clips) into the numbers the decision needs."""
    s: dict = {}
    s["n_frames"] = len(frames)
    s["duration_s"] = round(float(frames["t"].max() - frames["t"].min()), 1) if len(frames) else 0

    # Speed. Split detecting / not: MediaPipe short-circuits when it finds nobody, so a
    # pooled median flatters it badly on clips where it mostly fails.
    s["pose_ms_median"] = float(frames["pose_ms"].median())
    s["pose_ms_p95"] = float(frames["pose_ms"].quantile(0.95))
    hit = frames[frames["n_pose"] > 0]
    s["pose_ms_when_detecting"] = float(hit["pose_ms"].median()) if len(hit) else float("nan")
    s["fps_median"] = 1000.0 / s["pose_ms_median"] if s["pose_ms_median"] else float("nan")
    s["fps_when_detecting"] = (1000.0 / s["pose_ms_when_detecting"]
                               if s["pose_ms_when_detecting"] == s["pose_ms_when_detecting"]
                               else float("nan"))

    # Ghost poses: skeletons on things that are not people. These become false alerts.
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
    s["count_agreement"] = float((frames["n_pose"] == frames["n_yolo"]).mean())

    if people.empty:
        s["n_people_det"] = 0
        s["miss_rate"] = float("nan")
        return s

    s["n_people_det"] = len(people)
    s["recall"] = float(people["detected"].mean())
    s["miss_rate"] = 1.0 - s["recall"]

    # Recall by apparent distance, using fixed box-height bands so clips are comparable.
    bands = [(0.0, 0.15, "far   <15%"), (0.15, 0.30, "mid 15-30%"), (0.30, 1.01, "near  >30%")]
    for lo, hi, label in bands:
        g = people[(people["box_h_frac"] >= lo) & (people["box_h_frac"] < hi)]
        s[f"recall_{label.split()[0]}"] = float(g["detected"].mean()) if len(g) else float("nan")
        s[f"n_{label.split()[0]}"] = len(g)

    cross = people[people["crossing"] == 1]
    s["recall_crossing"] = float(cross["detected"].mean()) if len(cross) else float("nan")
    s["n_crossing"] = len(cross)
    return s


def decide(miss_rate: float, fps: float) -> tuple[str, str]:
    """The rule, fixed before any numbers were seen. Returns (verdict, reasoning)."""
    if miss_rate != miss_rate:  # NaN
        return "NO DATA", "no YOLO reference was run, so there is no miss rate"
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
