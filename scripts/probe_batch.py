"""WP0 / Step 3 - run the pose probe over EVERY clip in data/raw and decide once.

One clip is weak evidence; a clip where everyone happens to be close to the camera will
flatter MediaPipe, and one shot down a long corridor will damn it. This runs the same
probe over all of them, prints a per-clip table plus pooled numbers, and applies the
decision rule to the pool.

    uv run python scripts/probe_batch.py
    uv run python scripts/probe_batch.py --no-video      # faster, numbers only
    uv run python scripts/probe_batch.py --model lite

Writes outputs/reports/wp0_probe_summary.csv  (per clip, committed - it is a result)
       outputs/reports/wp0_probe_people.csv   (every person-detection, pooled)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from skelguard import paths  # noqa: E402
from skelguard.geometry import fit_height_model  # noqa: E402
from skelguard.probe import (  # noqa: E402
    FPS_TARGET, PX_BANDS, VIDEO_EXTS, WINDOW_FRAMES, ProbeConfig, decide, make_yolo,
    probe_clip, snapshot_run, stats_for,
)


def find_clips(folder: Path, match: str = "") -> list[Path]:
    """Every video under `folder`, recursively. `match` filters on the file name.

    Recursive because the gait footage is organised one folder per subject-session,
    and `match` because a session holds both camera angles and they are not
    interchangeable - the side view is framed too tight to pose.
    """
    return sorted(p for p in folder.rglob("*")
                  if p.is_file() and p.suffix.lower() in VIDEO_EXTS
                  and (not match or match.lower() in p.name.lower()))


def fit_clip_height_model(clip: Path, yolo, conf: float, stride: int,
                          max_frames: int):
    """Fit the perspective person-height model for this clip's camera view.

    Returns (model_to_use, fitted_model_for_reporting, status). A weak fit is retained
    for reporting but not used for crop scale; crop mode then falls back to the observed
    box height, which still upscales distant people to the target size.
    """
    cap = cv2.VideoCapture(str(clip))
    if not cap.isOpened():
        return None, None, "could not open clip"
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    feet_y: list[float] = []
    heights: list[float] = []
    frame_i = processed = 0
    while True:
        ok, frame = cap.read()
        if not ok or (max_frames and processed >= max_frames):
            break
        if frame_i % max(1, stride):
            frame_i += 1
            continue
        processed += 1
        result = yolo.predict(frame, classes=[0], conf=conf, verbose=False)[0]
        if result.boxes is not None and len(result.boxes):
            boxes = result.boxes.xyxy.cpu().numpy()
            feet_y.extend(float(b[3]) for b in boxes)
            heights.extend(float(b[3] - b[1]) for b in boxes)
        frame_i += 1
    cap.release()

    if len(feet_y) < 10:
        return None, None, f"not enough detections ({len(feet_y)})"
    if np.ptp(feet_y) < 0.15 * max(frame_h, 1):
        return None, None, "insufficient perspective span; using detected box height"
    fitted = fit_height_model(np.asarray(feet_y), np.asarray(heights))
    if fitted is None:
        return None, None, "fit failed; using detected box height"
    # Conservative checks reject the known overhead/wide-angle failure mode.
    horizon_ok = -0.25 * frame_h <= fitted.y_horizon <= 0.75 * frame_h
    if fitted.inlier_rate < 0.80 or fitted.median_rel_err > 0.10 or not horizon_ok:
        return None, fitted, "fit rejected; using detected box height"
    return fitted, fitted, "fit accepted; expected height drives crop scale"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--folder", type=Path, default=paths.RAW)
    ap.add_argument("--match", default="",
                    help="only clips whose file name contains this, e.g. 'front'")
    ap.add_argument("--model", choices=["lite", "full", "heavy"], default="full")
    ap.add_argument("--num-poses", type=int, default=5)
    ap.add_argument("--width", type=int, default=0, help="0 = native (we standardise on 720p)")
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--max-frames", type=int, default=0)
    ap.add_argument("--det-conf", type=float, default=0.5)
    ap.add_argument("--track-conf", type=float, default=0.5)
    ap.add_argument("--yolo-conf", type=float, default=0.35)
    ap.add_argument("--iou", type=float, default=0.30)
    ap.add_argument("--no-video", action="store_true", help="skip the annotated mp4s")
    ap.add_argument("--primary-only", action="store_true",
                    help="score against the largest person only; use with --num-poses 1")
    ap.add_argument("--crop", action="store_true",
                    help="YOLO box -> upscale -> MediaPipe single-pose on the crop")
    ap.add_argument("--crop-margin", type=float, default=0.20)
    ap.add_argument("--crop-target-h", type=int, default=320,
                    help="upscale each person to about this many px tall")
    ap.add_argument("--crop-max-scale", type=float, default=8.0)
    ap.add_argument("--run", default="test run 1",
                    help="name of this run; outputs go to outputs/<kind>/<run>/")
    args = ap.parse_args()

    clips = find_clips(args.folder, args.match)
    if not clips:
        raise SystemExit(f"[FAIL] no videos in {args.folder} "
                         f"(looked for {', '.join(VIDEO_EXTS)}"
                         + (f", name containing {args.match!r}" if args.match else "") + ")")

    cfg = ProbeConfig(
        model=args.model, num_poses=args.num_poses, width=args.width, stride=args.stride,
        max_frames=args.max_frames, det_conf=args.det_conf, track_conf=args.track_conf,
        yolo_conf=args.yolo_conf, iou=args.iou,
        use_yolo=True, write_video=not args.no_video, run=args.run,
        primary_only=args.primary_only,
        crop=args.crop, crop_margin=args.crop_margin,
        crop_target_h=args.crop_target_h, crop_max_scale=args.crop_max_scale,
    )
    if args.num_poses == 1 and not args.crop and not args.primary_only:
        print("  !! num_poses=1 without --primary-only: every extra person in frame is\n"
              "     counted as a miss, so the miss rate measures your cap, not MediaPipe.\n")

    snap = snapshot_run(args.run, cfg, clips)

    print(f"{len(clips)} clip(s) in {args.folder}")
    print(f"run    '{args.run}'  ->  outputs/{{videos,logs,reports}}/{args.run}/")
    print(f"code   frozen in {snap / 'code'} + run_meta.json")
    print(f"model={cfg.model}  num_poses={cfg.num_poses}  "
          f"width={'native' if not cfg.width else cfg.width}\n")

    yolo = make_yolo()  # load once, reuse across clips
    rows, all_people, all_frames = [], [], []

    for i, clip in enumerate(clips, 1):
        print(f"[{i}/{len(clips)}] {clip.name}")
        height_model, fitted_model, height_status = fit_clip_height_model(
            clip, yolo, cfg.yolo_conf, cfg.stride, cfg.max_frames
        )
        print(f"        height model: {height_status}")
        res = probe_clip(clip, cfg, yolo=yolo, height_model=height_model)
        s = stats_for(res.frames, res.people)
        fps = s["fps_when_detecting"]
        fps = fps if fps == fps else s["fps_median"]
        verdict, _ = decide(s["miss_rate"], fps, s.get("mode", "full"))
        rows.append({"clip": res.clip,
                     "src": f"{res.meta['src_w']}x{res.meta['src_h']}",
                     "src_h": res.meta["src_h"],
                     "height_model_status": height_status,
                     "height_model_k": fitted_model.k if fitted_model else np.nan,
                     "height_model_horizon": fitted_model.y_horizon if fitted_model else np.nan,
                     "height_model_inliers": fitted_model.inlier_rate
                     if fitted_model else np.nan,
                     "height_model_median_rel_err": fitted_model.median_rel_err
                     if fitted_model else np.nan,
                     **s, "fps_used": fps, "verdict": verdict})
        all_people.append(res.people)
        all_frames.append(res.frames)
        print(f"        miss {s['miss_rate']:.1%}  |  ghosts {s.get('n_ghost', 0)}"
              f"  |  {fps:.1f} fps  |  longest run "
              f"{s.get('run_max', 0)}f ({s.get('run_max_s', 0):.1f}s)  |  "
              f"usable {s.get('usable_window_rate', 0):.0%}\n")

    summary = pd.DataFrame(rows)
    people = pd.concat([p for p in all_people if not p.empty], ignore_index=True)
    frames = pd.concat(all_frames, ignore_index=True)

    rep_dir = paths.run_dir(paths.OUT_REPORTS, args.run)
    sum_csv = rep_dir / "wp0_probe_summary.csv"
    ppl_csv = rep_dir / "wp0_probe_people.csv"
    summary.to_csv(sum_csv, index=False)
    people.to_csv(ppl_csv, index=False)

    # ---------------------------------------------------------------- per clip
    print("=" * 86)
    print("PER CLIP")
    print("=" * 86)
    print(f"{'clip':26s} {'src':>9s} {'miss':>6s} {'ghost/min':>10s} {'no-pose':>8s} "
          f"{'fps':>6s} {'far':>7s} {'mid':>7s} {'near':>7s} {'cross':>7s}")
    print("-" * 100)
    for _, r in summary.iterrows():
        def pct(key):
            v = r.get(key, float("nan"))
            return f"{v:6.1%}" if v == v else "    n/a"
        print(f"{r['clip'][:26]:26s} {r['src']:>9s} {r['miss_rate']:5.1%} "
              f"{r.get('ghosts_per_min', float('nan')):10.1f} "
              f"{r['frames_with_no_pose']:7.0%} "
              f"{r['fps_used']:6.1f} {pct('recall_far')} {pct('recall_mid')} "
              f"{pct('recall_near')} {pct('recall_crossing')}")

    # Mixed resolutions make a pooled miss rate misleading: a 384x288 CAVIAR clip puts a
    # person in ~40 px, where MediaPipe cannot work, while our own footage will be 720p+.
    if summary["src_h"].nunique() > 1:
        print("\n  !! MIXED RESOLUTIONS: " + ", ".join(
            f"{r['src']}x{int((summary['src'] == r['src']).sum())}"
            for r in summary.drop_duplicates("src").to_dict("records")))
        print("     Pooling these weights low-res clips as if they were representative.")
        print("     Decide on the clips that match our own footage; treat the rest as a")
        print("     separate, harder condition and report them that way.")
        tall = summary[summary["src_h"] >= 700]
        if len(tall):
            print(f"\n     720p+ clips only: miss "
                  f"{(tall['miss_rate'] * tall['n_people_det']).sum() / max(tall['n_people_det'].sum(), 1):.1%}"
                  f"  over {int(tall['n_people_det'].sum())} person-detections")

    # ---------------------------------------------------------------- pooled
    pooled = stats_for(frames, people)
    fps_pool = pooled["fps_when_detecting"]
    fps_pool = fps_pool if fps_pool == fps_pool else pooled["fps_median"]

    print("\n" + "=" * 86)
    print(f"POOLED over {len(clips)} clip(s): {pooled['n_frames']} frames, "
          f"{pooled['n_people_det']} person-detections")
    print("=" * 86)
    print(f"  miss rate                  {pooled['miss_rate']:.1%}")
    if "n_ghost" in pooled:
        print(f"  ghost poses                {pooled['n_ghost']} "
              f"({pooled['ghost_rate']:.1%} of all poses, "
              f"{pooled['ghosts_per_min']:.1f}/min)  <- future false alerts")
    print(f"  frames with no pose at all {pooled['frames_with_no_pose']:.1%}")
    print(f"  count agreement with YOLO  {pooled['count_agreement']:.1%}")
    print(f"  fps when detecting         {fps_pool:.1f}  "
          f"(target >= {FPS_TARGET:.0f}: {'PASS' if fps_pool >= FPS_TARGET else 'FAIL'})")
    print("\n  recall by apparent size:")
    for key, label in (("far", "far   (<15% of frame height)"),
                       ("mid", "mid   (15-30%)"),
                       ("near", "near  (>30%)")):
        n, r = pooled.get(f"n_{key}", 0), pooled.get(f"recall_{key}", float("nan"))
        print(f"    {label:30s} n={n:6d}  " + (f"{r:6.1%}" if r == r else "   n/a"))
    r = pooled.get("recall_crossing", float("nan"))
    print(f"    {'during crossings':30s} n={pooled.get('n_crossing', 0):6d}  "
          + (f"{r:6.1%}" if r == r else "   n/a"))

    # The pixel-height curve: the single most transferable result from these probes,
    # because it is independent of clip resolution and framing.
    print("\n  recall by person height in PIXELS (comparable across clips):")
    for _, _, label in PX_BANDS:
        n = pooled.get(f"px_{label}_n", 0)
        r = pooled.get(f"px_{label}", float("nan"))
        bar = "#" * int(round((r if r == r else 0) * 40))
        print(f"    {label:>8s}  n={n:6d}  " + (f"{r:6.1%}" if r == r else "   n/a")
              + f"  {bar}")

    print(f"\n  CONTINUITY (recall says how often; this says whether it holds long enough)")
    print(f"    frames with a detection      {pooled.get('frames_detected_rate', 0):.1%}")
    print(f"    on/off flips                 {pooled.get('flips_per_s', 0):.1f} per second")
    print(f"    longest unbroken run         {pooled.get('run_max', 0)} frames "
          f"({pooled.get('run_max_s', 0):.1f}s),  median run "
          f"{pooled.get('run_median', 0):.0f}")
    print(f"    frames inside runs >= {WINDOW_FRAMES:<3d}    "
          f"{pooled.get('usable_window_rate', 0):.1%}"
          f"   -> {pooled.get('n_windows', 0)} training window(s) for WP5")
    if pooled.get("n_windows", 0) == 0:
        print("    !! no run reaches the window length: this footage yields NO training")
        print("       samples and no stable track, whatever the recall says.")

    if "mean_crop_scale" in pooled:
        print(f"\n  mean upscale factor applied: {pooled['mean_crop_scale']:.2f}x")
    if "mean_vis_detected" in pooled and pooled["mean_vis_detected"] == pooled["mean_vis_detected"]:
        print(f"  mean landmark visibility when detected: {pooled['mean_vis_detected']:.3f}"
              "   (low = a pose was returned but it is probably rubbish)")
    print(f"  end-to-end fps incl. YOLO:  {pooled['fps_end_to_end']:.1f}")

    verdict, why = decide(pooled["miss_rate"], fps_pool, pooled.get("mode", "full"))
    print("\n" + "-" * 86)
    print(f"DECISION (rule fixed before any numbers were seen): {verdict}")
    print(f"  {why}")
    print("-" * 86)

    if summary["verdict"].nunique() > 1:
        print("\n  NOTE: clips disagree -> " + ", ".join(
            f"{r['clip'][:20]}={r['verdict']}" for _, r in summary.iterrows()))
        print("  Weight the clips that look most like our own footage.")

    print(f"\nwrote:\n  {sum_csv}\n  {ppl_csv}")
    print("\nNext: watch the annotated videos at the worst clips, then write the verdict "
          "into docs/wp0_pose_probe.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
