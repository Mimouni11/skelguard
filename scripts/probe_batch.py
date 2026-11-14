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

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from skelguard import paths  # noqa: E402
from skelguard.probe import (  # noqa: E402
    FPS_TARGET, VIDEO_EXTS, ProbeConfig, decide, make_yolo, probe_clip, snapshot_run,
    stats_for,
)


def find_clips(folder: Path) -> list[Path]:
    return sorted(p for p in folder.iterdir()
                  if p.is_file() and p.suffix.lower() in VIDEO_EXTS)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--folder", type=Path, default=paths.RAW)
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
    ap.add_argument("--run", default="test run 1",
                    help="name of this run; outputs go to outputs/<kind>/<run>/")
    args = ap.parse_args()

    clips = find_clips(args.folder)
    if not clips:
        raise SystemExit(f"[FAIL] no videos in {args.folder} "
                         f"(looked for {', '.join(VIDEO_EXTS)})")

    cfg = ProbeConfig(
        model=args.model, num_poses=args.num_poses, width=args.width, stride=args.stride,
        max_frames=args.max_frames, det_conf=args.det_conf, track_conf=args.track_conf,
        yolo_conf=args.yolo_conf, iou=args.iou,
        use_yolo=True, write_video=not args.no_video, run=args.run,
    )

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
        res = probe_clip(clip, cfg, yolo=yolo)
        s = stats_for(res.frames, res.people)
        fps = s["fps_when_detecting"]
        fps = fps if fps == fps else s["fps_median"]
        verdict, _ = decide(s["miss_rate"], fps)
        rows.append({"clip": res.clip,
                     "src": f"{res.meta['src_w']}x{res.meta['src_h']}",
                     "src_h": res.meta["src_h"],
                     **s, "fps_used": fps, "verdict": verdict})
        all_people.append(res.people)
        all_frames.append(res.frames)
        print(f"        miss {s['miss_rate']:.1%}  |  ghosts {s.get('n_ghost', 0)}"
              f" ({s.get('ghosts_per_min', float('nan')):.1f}/min)  |  {fps:.1f} fps  |  "
              f"no-pose frames {s['frames_with_no_pose']:.0%}\n")

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

    verdict, why = decide(pooled["miss_rate"], fps_pool)
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
