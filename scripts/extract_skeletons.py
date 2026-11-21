"""Extract gait skeletons from clips into data/skeletons/, one parquet per clip.

This is the step the WP0 probes never did. `probe_batch.py` measures whether a pose was
found; it stores no landmarks. `probe_floor.py` stores landmarks but refuses to run
without a calibrated homography, which the gait room does not have. Gait identification
needs the landmarks themselves, so this script writes them and nothing else depends on
a calibration.

What it keeps, per frame, per track, per landmark:

    x_px, y_px              image pixels, for drawing and for anything view-dependent
    world_x, world_y, world_z   metres, origin at the midpoint of the hips

`world_*` is the one that matters. A person walking toward the camera changes apparent
size by more than 2x across a single pass - measured on walk05front, 301 px to 717 px -
so raw pixel landmarks encode distance from the camera far more strongly than they
encode who the person is. The world landmarks are hip-centred and metric, so that
variation is removed by construction instead of by dividing one noisy estimate by
another.

    visibility, presence    per landmark, not averaged

The gait cycle is segmented from the ankles (27, 28). A single mean over 33 points
cannot tell you whether *those* were seen: a frame with a crisp face and occluded legs
scores high and is useless, a frame with clear legs and a cropped head scores low and is
fine.

Faces are blurred in the annotated video. The project's premise is that every alert
comes from the skeleton and never from a face, and an output video that shows faces
would contradict that on the one artefact anybody actually looks at.

    uv run python scripts/extract_skeletons.py --folder data/raw --match front
    uv run python scripts/extract_skeletons.py "data/raw/walk5-another-person/walk05front.mp4"
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from skelguard import paths  # noqa: E402
from skelguard.probe import (  # noqa: E402
    POSE_CONNECTIONS, VIDEO_EXTS, ProbeConfig, make_landmarker, pose_on_crop,
    set_deterministic,
)
from skelguard.tracking import Tracker  # noqa: E402

# BlazePose 0-10 are nose, eyes, ears and mouth - the whole face.
FACE_LANDMARKS = list(range(11))

SKELETON_COLOUR = (0, 220, 255)
JOINT_COLOUR = (40, 40, 255)
BOX_COLOUR = (0, 200, 0)


# ------------------------------------------------------------------ labels
def load_subjects() -> dict[str, dict]:
    """session id -> {subject, condition}. Edit data/labels/subjects.csv to change."""
    path = paths.LABELS / "subjects.csv"
    if not path.exists():
        print(f"  [warn] no {path}; clips will be labelled subject '?'")
        return {}
    df = pd.read_csv(path)
    return {str(r.session): {"subject": str(r.subject), "condition": str(r.condition)}
            for r in df.itertuples()}


def parse_clip(clip: Path, subjects: dict[str, dict]) -> dict:
    """Pull session, view and subject out of the file name.

    walk01front-2.mp4 -> session walk01, view front, subject A.
    """
    stem = clip.stem.lower()
    m = re.match(r"^(walk\d+)", stem)
    session = m.group(1) if m else stem
    view = "side" if "side" in stem else ("front" if "front" in stem else "unknown")
    meta = subjects.get(session, {"subject": "?", "condition": "?"})
    return {"clip": clip.stem, "session": session, "view": view, **meta}


def find_clips(folder: Path, match: str = "") -> list[Path]:
    return sorted(p for p in folder.rglob("*")
                  if p.is_file() and p.suffix.lower() in VIDEO_EXTS
                  and (not match or match.lower() in p.name.lower()))


# ------------------------------------------------------------------ privacy
def pixelate(frame: np.ndarray, x1: int, y1: int, x2: int, y2: int, blocks: int = 6) -> None:
    """Destroy the detail in a region, in place.

    Pixelation rather than a Gaussian blur: a blur of a small face can be partly undone,
    and it is not obvious from the video whether it was strong enough. Resampling to a
    handful of blocks and back is unambiguous.
    """
    h, w = frame.shape[:2]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    if x2 - x1 < 2 or y2 - y1 < 2:
        return
    roi = frame[y1:y2, x1:x2]
    small = cv2.resize(roi, (blocks, blocks), interpolation=cv2.INTER_LINEAR)
    frame[y1:y2, x1:x2] = cv2.resize(small, (x2 - x1, y2 - y1),
                                     interpolation=cv2.INTER_NEAREST)


def blur_face(frame: np.ndarray, pts: np.ndarray | None, box: np.ndarray) -> None:
    """Hide the face, in place.

    With a pose, the face landmarks give a tight box that is padded generously - the
    landmarks sit on the features, the head is wider than they are. Without a pose we
    fall back to the top quarter of the person box, because a missing skeleton must
    never mean a visible face.

    The blur follows the detector: on a frame where YOLO finds nobody, nothing is
    blurred, because nothing is known to be there.
    """
    if pts is not None:
        fp = pts[FACE_LANDMARKS]
        x1, y1 = fp.min(axis=0)
        x2, y2 = fp.max(axis=0)
        pad_x, pad_y = (x2 - x1) * 0.7 + 6, (y2 - y1) * 0.9 + 6
        x1, x2 = x1 - pad_x, x2 + pad_x
        y1, y2 = y1 - pad_y * 1.2, y2 + pad_y
    else:
        x1, x2 = box[0], box[2]
        y1, y2 = box[1], box[1] + 0.25 * (box[3] - box[1])
    pixelate(frame, int(x1), int(y1), int(x2), int(y2))


# ------------------------------------------------------------------ extraction
def extract(clip: Path, meta: dict, yolo, args) -> tuple[pd.DataFrame, dict]:
    cap = cv2.VideoCapture(str(clip))
    if not cap.isOpened():
        raise SystemExit(f"[FAIL] could not open {clip}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    writer = None
    if not args.no_video:
        vid_dir = paths.run_dir(paths.OUT_VIDEOS, args.run)
        writer = cv2.VideoWriter(str(vid_dir / f"{clip.stem}__skeleton.mp4"),
                                 cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

    cfg = ProbeConfig(model=args.pose_model, crop=True, crop_margin=args.crop_margin,
                      crop_target_h=args.crop_target_h, crop_max_scale=args.crop_max_scale,
                      det_conf=args.pose_conf, track_conf=args.pose_conf)
    ctx = make_landmarker(cfg, image_mode=True, num_poses=1)
    lm = ctx.__enter__()
    tracker = Tracker(thr=args.iou, max_miss=args.max_miss)

    rows: list[dict] = []
    track_frames: dict[int, list] = {}
    frame_i, n_pose, n_miss = 0, 0, 0
    t0 = time.perf_counter()

    while True:
        ok, frame = cap.read()
        if not ok or (args.max_frames and frame_i >= args.max_frames):
            break
        t_s = frame_i / fps
        # Draw on a copy. Annotating the array the model reads silently lowered pose
        # recall the last time it was done in place - see wp0_pose_probe.md 8.5.
        canvas = frame.copy() if writer is not None else None

        res = yolo.predict(frame, classes=[0], conf=args.conf, verbose=False)[0]
        boxes = []
        if res.boxes is not None and len(res.boxes):
            boxes = [b.astype(np.float32) for b in res.boxes.xyxy.cpu().numpy()]

        for tid, box in tracker.update(boxes, frame_i):
            obs = pose_on_crop(lm, frame, box, cfg, w, h)
            n_pose += int(obs.found)
            n_miss += int(not obs.found)
            cx = float((box[0] + box[2]) / 2.0)
            track_frames.setdefault(tid, []).append((frame_i, cx, float(box[3])))

            if obs.found:
                for i in range(len(obs.pts_px)):
                    row = {
                        "frame": frame_i, "t_s": round(t_s, 4), "track": tid,
                        "landmark": i,
                        "x_px": float(obs.pts_px[i, 0]), "y_px": float(obs.pts_px[i, 1]),
                        "visibility": float(obs.visibility[i]),
                        "presence": float(obs.presence[i]),
                        "box_h": float(box[3] - box[1]),
                        "crop_scale": float(obs.scale),
                    }
                    if obs.world is not None:
                        row |= {"world_x": float(obs.world[i, 0]),
                                "world_y": float(obs.world[i, 1]),
                                "world_z": float(obs.world[i, 2])}
                    rows.append(row)

            if canvas is not None:
                blur_face(canvas, obs.pts_px, box)
                if obs.found:
                    p = obs.pts_px
                    for a, b in POSE_CONNECTIONS:
                        cv2.line(canvas, tuple(np.int32(p[a])), tuple(np.int32(p[b])),
                                 SKELETON_COLOUR, 2, cv2.LINE_AA)
                    for x_px, y_px in p:
                        cv2.circle(canvas, (int(x_px), int(y_px)), 3, JOINT_COLOUR, -1)
                x1, y1, x2, y2 = [int(v) for v in box[:4]]
                cv2.rectangle(canvas, (x1, y1), (x2, y2), BOX_COLOUR, 1)
                cv2.putText(canvas, f"#{tid}" + ("" if obs.found else "  no pose"),
                            (x1, max(14, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                            BOX_COLOUR, 2, cv2.LINE_AA)

        if writer is not None and canvas is not None:
            cv2.rectangle(canvas, (0, 0), (w, 26), (0, 0, 0), -1)
            cv2.putText(canvas, f"{clip.stem}   subject {meta['subject']}"
                                f" / {meta['condition']} / {meta['view']}"
                                f"   t={t_s:5.2f}s   faces pixelated",
                        (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1,
                        cv2.LINE_AA)
            writer.write(canvas)
        frame_i += 1

    cap.release()
    ctx.__exit__(None, None, None)
    if writer is not None:
        writer.release()
    elapsed = time.perf_counter() - t0

    # ---- which track is the subject? -------------------------------------------
    # Bystanders stand still; the subject walks across the frame. Rank by how far the
    # box centre actually travelled, and require a real track length so a two-frame
    # flicker cannot win.
    stats = {}
    for tid, seen in track_frames.items():
        xs = np.array([s[1] for s in seen])
        stats[tid] = {"n": len(seen), "span_px": float(xs.max() - xs.min())}
    primary = None
    if stats:
        moving = {t: s for t, s in stats.items()
                  if s["n"] >= args.min_track and s["span_px"] >= 0.10 * w}
        pool = moving or stats
        primary = max(pool, key=lambda t: (pool[t]["n"], pool[t]["span_px"]))

    df = pd.DataFrame(rows)
    if not df.empty:
        for k, v in meta.items():
            df[k] = v
        df["is_primary"] = df["track"] == primary
        df["fps"] = fps

    info = {
        **meta, "frames": frame_i, "fps": fps, "size": f"{w}x{h}",
        "poses": n_pose, "pose_missed": n_miss,
        "pose_rate": round(n_pose / max(1, n_pose + n_miss), 4),
        "tracks": {str(t): s for t, s in stats.items()},
        "primary_track": primary,
        "primary_frames": stats.get(primary, {}).get("n", 0) if primary is not None else 0,
        "landmark_rows": len(df),
        "has_world": bool(len(df)) and "world_x" in df.columns,
        "seconds": round(elapsed, 1),
        "fps_processing": round(frame_i / max(elapsed, 1e-6), 1),
    }
    return df, info


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("clips", nargs="*", type=Path, help="explicit clips, or use --folder")
    ap.add_argument("--folder", type=Path, default=paths.RAW)
    ap.add_argument("--match", default="", help="only clips whose name contains this")
    ap.add_argument("--conf", type=float, default=0.25, help="YOLO person confidence")
    ap.add_argument("--model", default="yolov8n.pt")
    ap.add_argument("--iou", type=float, default=0.30)
    ap.add_argument("--max-miss", type=int, default=10)
    ap.add_argument("--min-track", type=int, default=15)
    ap.add_argument("--pose-model", choices=["lite", "full", "heavy"], default="full")
    ap.add_argument("--pose-conf", type=float, default=0.5)
    ap.add_argument("--crop-target-h", type=int, default=320)
    ap.add_argument("--crop-margin", type=float, default=0.20)
    ap.add_argument("--crop-max-scale", type=float, default=8.0)
    ap.add_argument("--max-frames", type=int, default=0)
    ap.add_argument("--no-video", action="store_true")
    ap.add_argument("--run", default="skeletons")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    set_deterministic(args.seed)
    clips = [Path(c) for c in args.clips] or find_clips(args.folder, args.match)
    if not clips:
        raise SystemExit(f"[FAIL] no clips found in {args.folder} matching {args.match!r}")

    subjects = load_subjects()
    paths.SKELETONS.mkdir(parents=True, exist_ok=True)

    from ultralytics import YOLO
    yolo = YOLO(str(paths.MODELS / args.model))

    print(f"{len(clips)} clip(s) -> {paths.SKELETONS}")
    if not args.no_video:
        print(f"annotated video (faces pixelated) -> {paths.OUT_VIDEOS / args.run}\n")

    infos = []
    for i, clip in enumerate(clips, 1):
        meta = parse_clip(clip, subjects)
        print(f"[{i}/{len(clips)}] {clip.name}  subject={meta['subject']} "
              f"({meta['condition']}, {meta['view']})")
        df, info = extract(clip, meta, yolo, args)
        if df.empty:
            print("        no poses at all - skipped")
            infos.append(info)
            continue
        out = paths.SKELETONS / f"{clip.stem}.parquet"
        df.to_parquet(out, index=False)
        info["file"] = out.name
        infos.append(info)
        print(f"        pose {info['pose_rate']:.1%}  tracks={len(info['tracks'])}  "
              f"primary=#{info['primary_track']} ({info['primary_frames']}f)  "
              f"{info['landmark_rows']} rows  "
              f"world={'yes' if info['has_world'] else 'NO'}  "
              f"{info['fps_processing']} fps")

    rep = paths.run_dir(paths.OUT_REPORTS, args.run)
    (rep / "extract_meta.json").write_text(json.dumps(infos, indent=2), encoding="utf-8")

    print(f"\n{'clip':<26}{'subj':>5}{'view':>7}{'pose':>7}{'trk':>5}"
          f"{'primary':>9}{'rows':>8}")
    for info in infos:
        print(f"{info['clip'][:25]:<26}{info['subject']:>5}{info['view']:>7}"
              f"{info['pose_rate']:>7.1%}{len(info['tracks']):>5}"
              f"{str(info['primary_track']) + ' (' + str(info['primary_frames']) + 'f)':>9}"
              f"{info['landmark_rows']:>8}")
    print(f"\nwrote {rep / 'extract_meta.json'}")
    print("Check `primary` against the video before trusting any enrolment: if a")
    print("bystander was picked instead of the walker, the subject label is wrong.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
