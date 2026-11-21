"""WP0 / test run 3a - how far out can we hold a PERSON BOX?

Pose and tracking have different pixel budgets. Test run 2 showed MediaPipe needs ~150 px
of person height even with a crop pass. The question here is the detector's floor: if a
box survives at distances where pose does not, the pipeline can degrade gracefully -
keep tracking on the floor position, and attach a skeleton only when one is available.

Sweeps YOLO model size against confidence. Confidence is a post-filter on detection
scores, so inference runs ONCE per model per clip at a low floor and every threshold is
evaluated offline from the cached boxes. Re-running inference per threshold would take
forty minutes to produce identical numbers.

    uv run python scripts/probe_yolo_sweep.py
    uv run python scripts/probe_yolo_sweep.py --models yolov8n yolov8s yolov8m
    uv run python scripts/probe_yolo_sweep.py --confs 0.5 0.35 0.25 0.15 0.10 0.05

Writes outputs/reports/test run 3a/
    yolo_sweep_detections.csv   every box above the floor, with its score
    yolo_sweep_summary.csv      one row per (model, conf, clip)
"""

from __future__ import annotations

import argparse
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
    VIDEO_EXTS, WINDOW_FRAMES, binary_runs, set_deterministic,
)

# From test run 2: with a crop pass, MediaPipe is ~0% below this and reliable above it.
POSE_FLOOR_PX = 150


def find_clips(folder: Path) -> list[Path]:
    return sorted(p for p in folder.iterdir()
                  if p.is_file() and p.suffix.lower() in VIDEO_EXTS)


def detect_all(model_name: str, clips: list[Path], conf_floor: float,
               stride: int, max_frames: int) -> tuple[pd.DataFrame, dict]:
    """One inference pass per clip at the confidence floor. Returns every box + score."""
    from ultralytics import YOLO

    weights = paths.MODELS / f"{model_name}.pt"
    model = YOLO(str(weights))        # ultralytics fetches it here on first use

    rows: list[dict] = []
    timing: dict = {}
    for clip in clips:
        cap = cv2.VideoCapture(str(clip))
        if not cap.isOpened():
            print(f"    [skip] cannot open {clip.name}")
            continue
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        idx, kept, times = -1, 0, []
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            idx += 1
            if idx % stride:
                continue
            if max_frames and kept >= max_frames:
                break
            kept += 1
            t0 = time.perf_counter()
            res = model.predict(frame, classes=[0], conf=conf_floor, verbose=False,
                                augment=False)[0]
            times.append((time.perf_counter() - t0) * 1000)
            if res.boxes is None or len(res.boxes) == 0:
                continue
            xyxy = res.boxes.xyxy.cpu().numpy()
            scores = res.boxes.conf.cpu().numpy()
            for b, sc in zip(xyxy, scores):
                rows.append({
                    "model": model_name, "clip": clip.stem, "frame": idx,
                    "score": round(float(sc), 4),
                    "x1": round(float(b[0]), 1), "y1": round(float(b[1]), 1),
                    "x2": round(float(b[2]), 1), "y2": round(float(b[3]), 1),
                    "box_h": round(float(b[3] - b[1]), 1),
                    "box_w": round(float(b[2] - b[0]), 1),
                    "frame_h": h,
                })
        cap.release()
        timing[clip.stem] = {
            "n_frames": kept, "fps_src": fps,
            "ms_median": float(np.median(times)) if times else float("nan"),
        }
        print(f"    {clip.name[:38]:38s} {kept:5d} frames, "
              f"{float(np.median(times)) if times else 0:6.1f} ms/frame")
    return pd.DataFrame(rows), timing


def evaluate(det: pd.DataFrame, timing: dict, conf: float, n_frames: int,
             fps: float, clip: str, model: str) -> dict:
    """Stats for one (model, conf, clip), computed from the cached detections."""
    d = det[det["score"] >= conf]
    present = np.zeros(n_frames, dtype=int)
    if len(d):
        # frames are strided; index by rank so runs are consecutive in processed order
        idx = np.unique(d["frame_rank"].to_numpy())
        present[idx] = 1

    runs = binary_runs(present)
    out = {
        "model": model, "conf": conf, "clip": clip,
        "n_frames": n_frames,
        "n_boxes": len(d),
        "boxes_per_frame": round(len(d) / n_frames, 3) if n_frames else 0.0,
        "frames_with_box": round(float(present.mean()), 4),
        "longest_run": int(runs.max()),
        "longest_run_s": round(float(runs.max() / fps), 2) if fps else float("nan"),
        "usable_rate": round(float(runs[runs >= WINDOW_FRAMES].sum() / max(n_frames, 1)), 4),
        "n_windows": int(np.sum(np.maximum(runs - WINDOW_FRAMES + 1, 0))),
        "ms_median": round(timing.get(clip, {}).get("ms_median", float("nan")), 1),
    }
    if len(d):
        out["box_h_min"] = round(float(d["box_h"].min()), 1)
        out["box_h_p05"] = round(float(d["box_h"].quantile(0.05)), 1)
        out["box_h_med"] = round(float(d["box_h"].median()), 1)
        # boxes found BELOW the pose floor are exactly the far tier we are hunting
        below = d[d["box_h"] < POSE_FLOOR_PX]
        out["n_below_pose_floor"] = len(below)
        out["frac_below_pose_floor"] = round(len(below) / len(d), 3)
        # a standing person is taller than wide; wild ratios smell like false positives
        ratio = d["box_h"] / d["box_w"].clip(lower=1)
        out["implausible_ratio_frac"] = round(float(((ratio < 1.0) | (ratio > 6.0)).mean()), 3)
    else:
        out |= {"box_h_min": np.nan, "box_h_p05": np.nan, "box_h_med": np.nan,
                "n_below_pose_floor": 0, "frac_below_pose_floor": np.nan,
                "implausible_ratio_frac": np.nan}
    # single-frame blips: with no ground truth, flicker is our proxy for false positives
    out["singleton_runs"] = int((runs == 1).sum())
    return out


def render(det: pd.DataFrame, clips: list[Path], model: str, conf: float,
           run: str, stride: int, max_frames: int) -> None:
    """Draw one (model, conf) config onto the clips, straight from cached detections.

    No inference: the boxes were already found during the sweep, so this is just drawing.
    """
    vid_dir = paths.run_dir(paths.OUT_VIDEOS, run)
    d = det[(det["model"] == model) & (det["score"] >= conf)]
    print(f"\nrendering {model} @ conf {conf} -> {vid_dir}")

    for clip in clips:
        sub = d[d["clip"] == clip.stem]
        cap = cv2.VideoCapture(str(clip))
        if not cap.isOpened():
            continue
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        out = vid_dir / f"yolo_{clip.stem}__{model}_c{conf:.2f}.mp4"
        wr = cv2.VideoWriter(str(out), cv2.VideoWriter_fourcc(*"mp4v"),
                             fps / stride, (w, h))
        by_frame = {f: g for f, g in sub.groupby("frame")}

        idx, kept = -1, 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            idx += 1
            if idx % stride:
                continue
            if max_frames and kept >= max_frames:
                break
            kept += 1
            n_far = 0
            for _, b in by_frame.get(idx, pd.DataFrame()).iterrows():
                far = b["box_h"] < POSE_FLOOR_PX
                n_far += far
                # green: pose can work here. orange: far tier, box only.
                colour = (0, 165, 255) if far else (0, 220, 120)
                p1 = (int(b["x1"]), int(b["y1"]))
                p2 = (int(b["x2"]), int(b["y2"]))
                cv2.rectangle(frame, p1, p2, colour, 2)
                cv2.putText(frame, f"{b['score']:.2f} {b['box_h']:.0f}px",
                            (p1[0], max(12, p1[1] - 5)), cv2.FONT_HERSHEY_SIMPLEX,
                            0.45, colour, 1, cv2.LINE_AA)
            hud = (f"t={idx / fps:5.1f}s  {model} conf>={conf:.2f}  "
                   f"boxes={len(by_frame.get(idx, []))}  far(<{POSE_FLOOR_PX}px)={n_far}")
            cv2.rectangle(frame, (0, 0), (w, 24), (0, 0, 0), -1)
            cv2.putText(frame, hud, (8, 17), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (255, 255, 255), 1, cv2.LINE_AA)
            wr.write(frame)
        cap.release()
        wr.release()
        print(f"  {out.name}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--folder", type=Path, default=paths.RAW)
    ap.add_argument("--models", nargs="+", default=["yolov8n", "yolov8s", "yolov8m"])
    ap.add_argument("--confs", type=float, nargs="+",
                    default=[0.50, 0.35, 0.25, 0.15, 0.10])
    ap.add_argument("--conf-floor", type=float, default=0.05,
                    help="inference threshold; every --conf is filtered from this")
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--max-frames", type=int, default=0)
    ap.add_argument("--run", default="test run 3a")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--video", nargs="*", default=None, metavar="MODEL:CONF",
                    help="render annotated clips for these configs, e.g. "
                         "--video yolov8m:0.15 yolov8n:0.35 . Bare --video renders the "
                         "biggest model at the lowest confidence. Uses cached boxes, "
                         "so it costs no extra inference.")
    args = ap.parse_args()

    set_deterministic(args.seed)

    clips = find_clips(args.folder)
    if not clips:
        raise SystemExit(f"[FAIL] no videos in {args.folder}")
    if min(args.confs) < args.conf_floor:
        raise SystemExit(f"[FAIL] --conf-floor {args.conf_floor} must be <= every --conf")

    out_dir = paths.run_dir(paths.OUT_REPORTS, args.run)
    print(f"{len(clips)} clip(s), {len(args.models)} model(s), {len(args.confs)} threshold(s)")
    print(f"inference once per model at conf >= {args.conf_floor}, thresholds applied after")
    print(f"-> {out_dir}\n")

    all_det, all_rows = [], []
    for model in args.models:
        print(f"[{model}]")
        det, timing = detect_all(model, clips, args.conf_floor, args.stride, args.max_frames)
        if det.empty:
            print("    no detections at all\n")
            continue

        # map raw frame numbers to processed rank, so runs are consecutive under --stride
        for clip in det["clip"].unique():
            m = det["clip"] == clip
            frames = np.sort(det.loc[m, "frame"].unique())
            rank = {f: i for i, f in enumerate(range(0, int(frames.max()) + 1, args.stride))}
            det.loc[m, "frame_rank"] = det.loc[m, "frame"].map(rank)
        det["frame_rank"] = det["frame_rank"].fillna(0).astype(int)
        all_det.append(det)

        for clip_stem, t in timing.items():
            sub = det[det["clip"] == clip_stem]
            for conf in args.confs:
                all_rows.append(evaluate(sub, timing, conf, t["n_frames"],
                                         t["fps_src"] / args.stride, clip_stem, model))
        print()

    if not all_rows:
        raise SystemExit("[FAIL] nothing detected by any model")

    summary = pd.DataFrame(all_rows)
    det_all = pd.concat(all_det, ignore_index=True)
    det_all.to_csv(out_dir / "yolo_sweep_detections.csv", index=False)
    summary.to_csv(out_dir / "yolo_sweep_summary.csv", index=False)

    if args.video is not None:
        configs = args.video or [f"{args.models[-1]}:{min(args.confs):.2f}"]
        for spec in configs:
            try:
                m, c = spec.split(":")
                render(det_all, clips, m, float(c), args.run, args.stride, args.max_frames)
            except ValueError:
                print(f"  [skip] bad --video spec {spec!r}, expected MODEL:CONF")

    # ---------------------------------------------------------------- tables
    for clip in summary["clip"].unique():
        s = summary[summary["clip"] == clip]
        print("=" * 104)
        print(f"{clip}   ({int(s['n_frames'].iloc[0])} frames)")
        print("=" * 104)
        print(f"{'model':9s} {'conf':>5s} {'w/box':>7s} {'longest':>9s} {'usable':>7s} "
              f"{'windows':>8s} {'min px':>7s} {'p05 px':>7s} {'<150px':>7s} "
              f"{'blips':>6s} {'bad AR':>7s} {'ms':>6s}")
        print("-" * 104)
        for _, r in s.iterrows():
            print(f"{r['model']:9s} {r['conf']:5.2f} {r['frames_with_box']:7.1%} "
                  f"{r['longest_run']:7d}f {r['usable_rate']:7.1%} {r['n_windows']:8d} "
                  f"{r['box_h_min']:7.0f} {r['box_h_p05']:7.0f} "
                  f"{r['frac_below_pose_floor']:7.1%} {r['singleton_runs']:6d} "
                  f"{r['implausible_ratio_frac']:7.1%} {r['ms_median']:6.1f}")
        print()

    # ---------------------------------------------------------------- reading guide
    print("=" * 104)
    print("HOW TO READ THIS")
    print("=" * 104)
    print(f"  w/box     fraction of frames with at least one person box")
    print(f"  usable    fraction inside unbroken runs of >= {WINDOW_FRAMES} frames"
          f"  <- continuity, the metric that matters")
    print(f"  min/p05   smallest person heights still detected, in pixels")
    print(f"  <150px    share of boxes below MediaPipe's floor  <- THE FAR TIER."
          f" Big here means")
    print(f"            tracking survives where pose cannot, so graceful degradation works.")
    print(f"  blips     single-frame detections; with no ground truth this is our only")
    print(f"            proxy for false positives. Rising sharply as conf drops = noise.")
    print(f"  bad AR    boxes whose height/width is implausible for a standing person.")
    print()
    print("  Pick the lowest conf where 'usable' and '<150px' still climb while 'blips'")
    print("  and 'bad AR' stay flat. Then WATCH that config before trusting it: without")
    print("  ground truth, precision cannot be measured here, only smelled.")
    print(f"\nwrote:\n  {out_dir / 'yolo_sweep_summary.csv'}"
          f"\n  {out_dir / 'yolo_sweep_detections.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
