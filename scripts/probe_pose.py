"""WP0 / Step 3 - probe MediaPipe multi-person pose on ONE clip.

Answers, with numbers rather than impressions:
  1. Does MediaPipe find every person?
  2. Does it keep finding them far away, or when they cross?
  3. What fps do we get?

Runs at the clip's native resolution by default (we standardise on 720p).

    uv run python scripts/probe_pose.py data/raw/clip.mp4
    uv run python scripts/probe_pose.py data/raw/clip.mp4 --model lite --num-poses 8
    uv run python scripts/probe_pose.py data/raw/clip.mp4 --sweep    # fps only

For all clips at once and a pooled decision, use scripts/probe_batch.py instead.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from skelguard.probe import (  # noqa: E402
    MODEL_FILES, FPS_TARGET, ProbeConfig, decide, drop_events, make_landmarker,
    probe_clip, snapshot_run, stats_for,
)


def report(res, stats) -> None:
    print("\n" + "=" * 70)
    print(f"PROBE  -  {res.clip}")
    print(f"{stats['n_frames']} frames ({stats['duration_s']}s) in {res.meta['wall_s']:.1f}s"
          f"  at {res.meta['infer_w']}x{res.meta['infer_h']}")
    print("=" * 70)

    print("\n[3] SPEED (pose inference only)")
    print(f"    median        {stats['pose_ms_median']:6.1f} ms -> {stats['fps_median']:5.1f} fps")
    print(f"    p95           {stats['pose_ms_p95']:6.1f} ms -> "
          f"{1000 / stats['pose_ms_p95']:5.1f} fps")
    if stats["pose_ms_when_detecting"] == stats["pose_ms_when_detecting"]:
        print(f"    when it finds someone  {stats['pose_ms_when_detecting']:6.1f} ms -> "
              f"{stats['fps_when_detecting']:5.1f} fps   <- the honest number")
        print(f"    (MediaPipe short-circuits on empty frames, and {stats['frames_with_no_pose']:.0%}"
              f" of frames here had no pose at all)")
    fps = stats["fps_when_detecting"]
    fps = fps if fps == fps else stats["fps_median"]
    print(f"    target >= {FPS_TARGET:.0f} fps: {'PASS' if fps >= FPS_TARGET else 'FAIL'}")

    print("\n[1] COVERAGE")
    print(f"    poses per frame: {stats['mean_poses']:.2f} mean, max {stats['max_poses']}")
    print(f"    frames with NO pose at all: {stats['frames_with_no_pose']:.1%}")
    if not stats.get("n_people_det"):
        print("    (no YOLO reference - rerun without --no-yolo for a miss rate)")
        return
    print(f"    YOLO saw {stats['n_people_det']} person-detections; "
          f"MediaPipe matched {int(stats['recall'] * stats['n_people_det'])}")
    print(f"    >>> MISS RATE: {stats['miss_rate']:.1%}")
    print(f"    frames where both agree on the count: {stats['count_agreement']:.1%}")

    if "n_ghost" in stats:
        print("\n[1b] GHOST POSES (skeleton on something YOLO says is not a person)")
        print(f"    {stats['n_ghost']} ghost pose(s) = {stats['ghost_rate']:.1%} of all poses")
        print(f"    {stats['ghosts_per_min']:.1f} per minute, "
              f"in {stats['frames_with_ghost']:.1%} of frames")
        print("    These become FALSE INTRUSION ALERTS once a zone is drawn. Labelled")
        print("    'GHOST?' in red in the annotated video - check them before believing")
        print("    the number, since a ghost may also be a real person YOLO missed.")

    print("\n[2] WHERE IT FAILS")
    print("    by apparent size (person box height as a fraction of frame height):")
    for key, label in (("far", "far   <15%"), ("mid", "mid 15-30%"), ("near", "near  >30%")):
        n, r = stats.get(f"n_{key}", 0), stats.get(f"recall_{key}", float("nan"))
        print(f"      {label:12s} n={n:5d}  detected "
              + (f"{r:6.1%}" if r == r else "   n/a"))
    r = stats.get("recall_crossing", float("nan"))
    print(f"    during crossings (boxes overlapping)  n={stats.get('n_crossing', 0):5d}"
          f"  detected " + (f"{r:6.1%}" if r == r else "   n/a"))
    print(f"    crossing occurs in {stats['crossing_frame_rate']:.1%} of frames")

    events = drop_events(res.frames)
    if events:
        print(f"\n    {len(events)} drop event(s) - scrub to these in the annotated video:")
        for start, end, worst, nf in sorted(events, key=lambda e: -(e[1] - e[0]))[:10]:
            print(f"      {start:6.1f}s -> {end:6.1f}s  ({end - start:5.1f}s, {nf:4d} frames, "
                  f"up to {worst} person(s) missed)")

    verdict, why = decide(stats["miss_rate"], fps)
    print("\n" + "-" * 70)
    print(f"DECISION (rule fixed before the numbers): {verdict}")
    print(f"    {why}")
    print("-" * 70)

    print("\nwrote:")
    for p in (res.video_path, res.frames_csv, res.people_csv):
        if p:
            print(f"  {p}")
    print("\nOne clip is weak evidence. Run scripts/probe_batch.py once you have 4-5 clips.")


def sweep(args: argparse.Namespace) -> int:
    import mediapipe as mp

    cap = cv2.VideoCapture(str(args.video))
    frames = []
    while len(frames) < args.sweep_frames:
        ok, f = cap.read()
        if not ok:
            break
        frames.append(f)
    cap.release()
    if not frames:
        raise SystemExit("[FAIL] no frames read")

    src_h, src_w = frames[0].shape[:2]
    print(f"fps sweep on {len(frames)} frames of {Path(args.video).name} ({src_w}x{src_h})\n")
    print(f"{'model':8s} {'width':>6s} {'median ms':>10s} {'fps':>7s}   target>={FPS_TARGET:.0f}")
    print("-" * 48)

    for model in ("lite", "full", "heavy"):
        if not MODEL_FILES[model].exists():
            print(f"{model:8s} {'-':>6s}  not downloaded (run download_models.py --all)")
            continue
        for width in args.sweep_widths:
            if width > src_w:
                continue
            scale = width / src_w
            w, h = int(src_w * scale), int(src_h * scale)
            cfg = ProbeConfig(model=model, num_poses=args.num_poses,
                              det_conf=args.det_conf, track_conf=args.track_conf)
            times = []
            with make_landmarker(cfg) as lm:
                for i, f in enumerate(frames):
                    small = f if scale == 1.0 else cv2.resize(f, (w, h), cv2.INTER_AREA)
                    rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
                    img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                    t0 = time.perf_counter()
                    lm.detect_for_video(img, int(i / 25 * 1000))
                    times.append((time.perf_counter() - t0) * 1000)
            med = float(np.median(times[2:] if len(times) > 2 else times))  # drop warm-up
            print(f"{model:8s} {width:>6d} {med:>10.1f} {1000 / med:>7.1f}   "
                  f"{'PASS' if 1000 / med >= FPS_TARGET else 'fail'}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("video", type=Path, help="clip to probe, e.g. data/raw/clip.mp4")
    ap.add_argument("--model", choices=["lite", "full", "heavy"], default="full")
    ap.add_argument("--num-poses", type=int, default=5, help="max poses MediaPipe may return")
    ap.add_argument("--width", type=int, default=0, help="resize before inference (0 = native)")
    ap.add_argument("--stride", type=int, default=1, help="process 1 frame in N")
    ap.add_argument("--max-frames", type=int, default=0, help="stop after N processed frames")
    ap.add_argument("--det-conf", type=float, default=0.5)
    ap.add_argument("--track-conf", type=float, default=0.5)
    ap.add_argument("--yolo-conf", type=float, default=0.35)
    ap.add_argument("--iou", type=float, default=0.30, help="IoU to call a pose a match")
    ap.add_argument("--no-yolo", action="store_true", help="skip the reference (counts only)")
    ap.add_argument("--no-video", action="store_true", help="skip the annotated mp4")
    ap.add_argument("--run", default="test run 1",
                    help="name of this run; outputs go to outputs/<kind>/<run>/")
    ap.add_argument("--sweep", action="store_true", help="fps only, across models and widths")
    ap.add_argument("--sweep-frames", type=int, default=60)
    ap.add_argument("--sweep-widths", type=int, nargs="+", default=[1280, 960, 640])
    args = ap.parse_args()

    if args.sweep:
        return sweep(args)

    cfg = ProbeConfig(
        model=args.model, num_poses=args.num_poses, width=args.width, stride=args.stride,
        max_frames=args.max_frames, det_conf=args.det_conf, track_conf=args.track_conf,
        yolo_conf=args.yolo_conf, iou=args.iou,
        use_yolo=not args.no_yolo, write_video=not args.no_video, run=args.run,
    )
    snap = snapshot_run(args.run, cfg, [args.video])
    print(f"clip   {args.video}")
    print(f"run    '{args.run}'  (code frozen in {snap / 'code'})")
    res = probe_clip(args.video, cfg)
    report(res, stats_for(res.frames, res.people))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
