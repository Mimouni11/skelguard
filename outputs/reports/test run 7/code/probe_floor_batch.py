"""Test run 5 - floor tracking across every calibrated clip, with a controlled A/B.

Run 4 left two questions open:

  1. Floor distance in metres was only ever computed for the hallway, because it was the
     only calibrated scene. Every other clip was measured in pixels, so "does the geometry
     preserve distance" was never actually tested anywhere else.
  2. Run 4 changed the crop-scale rule AND the YOLO confidence at the same time, so the
     recall it reported cannot be attributed to either. There was no control.

This script fixes both. It runs the floor probe over a list of clip=scene pairs at ONE
fixed confidence, and with --ab runs each calibrated clip twice - crop sized from the
detected box height, then from the calibrated expected height - changing nothing else.

    uv run python scripts/probe_floor_batch.py --ab --conf 0.25 --run "test run 5"

A scene with no height model is still tracked on the floor; only the gate and the
height-crop arm are skipped, and the report says so rather than silently omitting it.
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
from argparse import Namespace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import probe_floor  # noqa: E402

from skelguard import paths  # noqa: E402
from skelguard.geometry import Scene  # noqa: E402

# clip stem -> scene name. Edit here, or override with --pair.
DEFAULT_PAIRS = {
    "Workers walking in hallway of office": "hallway",
    "WalkByShop1front": "shopfront",
    "Walk1": "caviar_lobby",
    "Meet_Crowd": "caviar_lobby_2",
}

SNAPSHOT = ["scripts/probe_floor_batch.py", "scripts/probe_floor.py",
            "src/skelguard/geometry.py"]


def find_clip(stem: str) -> Path | None:
    for p in sorted(paths.RAW.iterdir()):
        if p.is_file() and p.stem == stem:
            return p
    return None


def make_args(video: Path, scene: str, run: str, a, height_crop: bool) -> Namespace:
    """Build the single-clip probe's arguments. Everything except `height_crop` is held
    identical between the two arms, which is the whole point of the control."""
    return Namespace(
        video=video, scene=scene, model=a.model, conf=a.conf, gate_tol=a.gate_tol,
        iou=a.iou, max_miss=a.max_miss, min_track=a.min_track,
        speed_window=a.speed_window, static_window=a.static_window,
        static_move_m=a.static_move_m, stride=a.stride, max_frames=a.max_frames,
        grid=a.grid, q_m=a.q_m, r_m=a.r_m, q_px=a.q_px, r_px=a.r_px,
        no_video=a.no_video, pose_model=a.pose_model, pose_conf=a.pose_conf,
        pose_target_h=a.pose_target_h, pose_margin=a.pose_margin,
        pose_max_scale=a.pose_max_scale, height_crop=height_crop,
        run=run, seed=a.seed,
    )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pair", action="append", default=[],
                    metavar="CLIP_STEM=SCENE",
                    help="override or extend the clip-to-scene mapping, repeatable")
    ap.add_argument("--ab", action="store_true",
                    help="run each calibrated clip twice: box-height vs expected-height crop")
    ap.add_argument("--model", default="yolov8n.pt")
    ap.add_argument("--conf", type=float, default=0.25,
                    help="ONE confidence for every clip and both arms - do not vary it")
    ap.add_argument("--gate-tol", type=float, default=0.35)
    ap.add_argument("--iou", type=float, default=0.30)
    ap.add_argument("--max-miss", type=int, default=10)
    ap.add_argument("--min-track", type=int, default=15)
    ap.add_argument("--speed-window", type=float, default=0.40)
    ap.add_argument("--static-window", type=float, default=3.0)
    ap.add_argument("--static-move-m", type=float, default=0.5)
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--max-frames", type=int, default=0)
    ap.add_argument("--grid", type=float, default=1.0)
    ap.add_argument("--q-m", type=float, default=0.5)
    ap.add_argument("--r-m", type=float, default=0.10)
    ap.add_argument("--q-px", type=float, default=200.0)
    ap.add_argument("--r-px", type=float, default=25.0)
    ap.add_argument("--no-video", action="store_true")
    ap.add_argument("--pose-model", choices=["lite", "full", "heavy"], default="full")
    ap.add_argument("--pose-conf", type=float, default=0.5)
    ap.add_argument("--pose-target-h", type=int, default=320)
    ap.add_argument("--pose-margin", type=float, default=0.20)
    ap.add_argument("--pose-max-scale", type=float, default=8.0)
    ap.add_argument("--run", default="test run 5")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    pairs = dict(DEFAULT_PAIRS)
    for spec in a.pair:
        if "=" not in spec:
            raise SystemExit(f"[FAIL] --pair needs CLIP_STEM=SCENE, got {spec!r}")
        stem, scene = spec.split("=", 1)
        pairs[stem.strip()] = scene.strip()

    jobs: list[tuple[Path, str, bool]] = []
    skipped: list[str] = []
    for stem, scene_name in pairs.items():
        clip = find_clip(stem)
        if clip is None:
            skipped.append(f"{stem}: no video in data/raw")
            continue
        try:
            scene = Scene.load(scene_name)
        except SystemExit:
            skipped.append(f"{stem}: no calibration {scene_name!r}")
            continue
        if scene.H is None:
            skipped.append(f"{stem}: scene {scene_name!r} has no homography")
            continue
        has_model = scene.height_model is not None
        jobs.append((clip, scene_name, False))
        if a.ab and has_model:
            jobs.append((clip, scene_name, True))
        elif a.ab:
            skipped.append(f"{stem}: no height model, A/B arm skipped")

    if not jobs:
        raise SystemExit("[FAIL] nothing to run. Calibrate a scene first.")

    rows = []
    for i, (clip, scene_name, height_crop) in enumerate(jobs, 1):
        arm = "expected-height crop" if height_crop else "box-height crop"
        sub = f"{a.run}/{scene_name}" + ("/height-crop" if height_crop else "/box-crop")
        print(f"\n[{i}/{len(jobs)}] {clip.name}  scene={scene_name}  {arm}")
        try:
            res = probe_floor.run(make_args(clip, scene_name, sub, a, height_crop))
        except SystemExit as exc:
            skipped.append(f"{clip.stem} ({arm}): {exc}")
            continue
        res["arm"] = arm
        rows.append(res)

    # ---------------------------------------------------------------- report
    rep = paths.OUT_REPORTS / a.run
    rep.mkdir(parents=True, exist_ok=True)

    lines: list[str] = []
    p = lines.append
    p("=" * 118)
    p(f"TEST RUN 5 - FLOOR TRACKING ON EVERY CALIBRATED CLIP   conf={a.conf} "
      f"(fixed)  pose={a.pose_model}  seed={a.seed}")
    p("=" * 118)

    p("\n[1] DOES THE GEOMETRY HOLD ON EACH CAMERA?")
    p(f"    {'clip':<30}{'scene':<16}{'arm':<22}{'tracks':>7}{'scenery':>8}"
      f"{'med m/s':>9}{'people':>8}{'in range':>10}{'impossible':>12}")
    for r in rows:
        p(f"    {r['clip'][:29]:<30}{r['scene'][:15]:<16}{r['arm']:<22}"
          f"{r['n_tracks']:>7}{r['n_scenery']:>8}{r['median_ms']:>9.2f}"
          f"{r['median_ms_people']:>8.2f}{r['in_walk_range']:>9.1%}"
          f"{r['impossible']:>11.1%}")
    p("    `med m/s` is over every track; `people` excludes tracks the static filter")
    p("    called scenery. A large gap between the two means the clip is dominated by")
    p("    objects the detector thinks are people, which is a finding about the clip,")
    p("    not about the calibration. For a fisheye view a wild median is expected.")

    p("\n[2] POSE AND THE GATE")
    p(f"    {'clip':<30}{'arm':<22}{'height model':<14}{'boxes':>7}{'gated':>7}"
      f"{'pose recall':>13}{'fps':>7}")
    for r in rows:
        p(f"    {r['clip'][:29]:<30}{r['arm']:<22}{r['height_model']:<14}"
          f"{r['boxes']:>7}{r['gated']:>7}{r['pose_recall']:>12.1%}"
          f"{r['fps_end_to_end']:>7.1f}")

    if a.ab:
        p("\n[3] A/B: DOES THE CALIBRATED HEIGHT IMPROVE THE POSE CROP?")
        p("    Same clip, same confidence, same seed. Only the crop-scale rule differs.")
        by_clip: dict[str, dict[bool, dict]] = {}
        for r in rows:
            by_clip.setdefault(r["clip"], {})[r["height_crop"]] = r
        any_pair = False
        for clip, arms in by_clip.items():
            if False not in arms or True not in arms:
                continue
            any_pair = True
            base, test = arms[False], arms[True]
            d = test["pose_recall"] - base["pose_recall"]
            p(f"    {clip[:40]:<42} box={base['pose_recall']:.1%}  "
              f"expected={test['pose_recall']:.1%}  delta={d:+.1%}")
        if not any_pair:
            p("    no clip had both arms: every calibrated scene lacks a height model.")
        else:
            p("    A positive delta means sizing the crop from calibrated geometry beats")
            p("    sizing it from a box that occlusion and frame edges can truncate.")

    if skipped:
        p("\n[!] SKIPPED")
        for s in skipped:
            p(f"    {s}")

    text = "\n".join(lines)
    print("\n" + text)
    (rep / "floor_batch_report.txt").write_text(text, encoding="utf-8")

    if rows:
        with (rep / "floor_batch_summary.csv").open("w", newline="",
                                                    encoding="utf-8") as f:
            wri = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            wri.writeheader()
            wri.writerows(rows)

    code = rep / "code"
    code.mkdir(exist_ok=True)
    for rel in SNAPSHOT:
        src = ROOT / rel
        if src.exists():
            shutil.copy2(src, code / Path(rel).name)
    (rep / "run_meta.json").write_text(json.dumps({
        "run": a.run, "conf": a.conf, "model": a.model, "ab": a.ab,
        "pairs": pairs, "seed": a.seed, "skipped": skipped,
    }, indent=2), encoding="utf-8")
    print(f"\nwrote {rep}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
