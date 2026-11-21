"""Enrol one person's walk, then answer: is this them, or not?

Verification, not identification. The question "is this Mimouni?" needs a template for
Mimouni and a threshold - it does not need a model of everyone else. That matters here,
because the only non-Mimouni footage available is a single 16-second clip of one person,
and a two-class classifier built on that was measuring which clip a window came from
rather than who was walking (41% balanced accuracy, below chance).

One-class enrolment avoids that trap. The threshold comes from how much Mimouni's own
walk varies between his own takes - 8 clips over 4 sessions, with a clothing change -
which is something this data genuinely can measure.

    uv run python scripts/verify_gait.py --enrol --subject A --name mimouni
    uv run python scripts/verify_gait.py --eval
    uv run python scripts/verify_gait.py --video "data/raw/test/test.mp4"

What the evaluation can and cannot tell you:

    FRR (false reject)  well measured. Leave-one-clip-out over 8 independent takes:
                        how often is Mimouni turned away on a take he was not enrolled
                        with.
    FAR (false accept)  one impostor identity, from one clip. That is a single data
                        point, not a rate, and the report says so rather than quoting
                        a percentage that would imply eight.
"""

from __future__ import annotations

import argparse
import json
import sys
from argparse import Namespace
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from skelguard import gait, paths  # noqa: E402

CYCLES = paths.INTERIM / "gait_cycles.parquet"


def template_path(name: str) -> Path:
    return paths.TRAINED / f"gait_template_{name}.json"


# ------------------------------------------------------------------ features
def stable_features(df: pd.DataFrame, cols: list[str],
                    max_between: float = 0.6) -> tuple[list[str], list[str]]:
    """Keep only features that stay put across the enrolled person's own takes.

    A feature whose value shifts more between one take and the next than it does within
    a take is describing the recording, not the walker. Those are precisely the features
    that made the two-class model score below chance: held-out clips of the same person
    landed nowhere near his other clips.

    Measured as (variance of the per-clip means) / (overall variance). Near 1 means the
    feature is almost entirely explained by which clip it came from.
    """
    keep, dropped = [], []
    for c in cols:
        v = df[c].to_numpy(float)
        total = float(np.var(v))
        if total < 1e-12:
            dropped.append(c)
            continue
        between = float(np.var(df.groupby("clip")[c].mean().to_numpy(float)))
        (keep if between / total <= max_between else dropped).append(c)
    return keep, dropped


def score(X: np.ndarray, centre: np.ndarray, scale: np.ndarray) -> np.ndarray:
    """Distance from the template, in units of the enrolled person's own spread.

    Normalised per feature by the enrolled spread, then averaged rather than summed, so
    the number does not change meaning when features are added or dropped. ~1.0 means
    "as far from the template as a typical enrolled sample".
    """
    z = (X - centre) / scale
    return np.sqrt((z ** 2).mean(axis=1))


# ------------------------------------------------------------------ enrolment
def enrol(df: pd.DataFrame, subject: str, name: str, target_frr: float,
          max_between: float) -> dict:
    mine = df[df["subject"] == subject]
    if mine.empty:
        raise SystemExit(f"[FAIL] no samples for subject {subject!r}")
    clips = sorted(mine["clip"].unique())
    if len(clips) < 3:
        print(f"  [warn] only {len(clips)} clip(s) enrolled - the threshold will be "
              "fitted on very little take-to-take variation")

    all_cols = gait.feature_columns(mine, "all")
    cols, dropped = stable_features(mine, all_cols, max_between)
    if len(cols) < 4:
        print(f"  [warn] only {len(cols)} stable feature(s); relaxing the filter")
        cols, dropped = all_cols, []

    X = mine[cols].to_numpy(float)
    centre = np.median(X, axis=0)
    scale = np.std(X, axis=0)
    scale[scale < 1e-9] = 1.0

    # Threshold from leave-one-clip-out: score each clip against a template built from
    # the OTHER clips, so the number reflects an unseen take rather than itself.
    genuine = []
    for c in clips:
        tr, te = mine[mine["clip"] != c], mine[mine["clip"] == c]
        if tr.empty:
            continue
        Xtr = tr[cols].to_numpy(float)
        ctr, str_ = np.median(Xtr, axis=0), np.std(Xtr, axis=0)
        str_[str_ < 1e-9] = 1.0
        genuine.append(score(te[cols].to_numpy(float), ctr, str_))
    g = np.concatenate(genuine) if genuine else score(X, centre, scale)
    threshold = float(np.quantile(g, 1.0 - target_frr))

    tpl = {
        "name": name, "subject": subject, "created": date.today().isoformat(),
        "features": cols, "dropped_features": dropped,
        "centre": centre.tolist(), "scale": scale.tolist(),
        "threshold": threshold, "target_frr": target_frr,
        "enrolled_clips": clips, "n_windows": int(len(mine)),
        "genuine_score_median": float(np.median(g)),
        "genuine_score_p90": float(np.quantile(g, 0.90)),
    }
    out = template_path(name)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(tpl, indent=2), encoding="utf-8")

    print(f"\nenrolled '{name}' (subject {subject})")
    print(f"    {len(mine)} windows from {len(clips)} clip(s): {', '.join(clips)}")
    print(f"    features kept {len(cols)} of {len(all_cols)}")
    if dropped:
        print(f"    dropped as take-dependent: {', '.join(dropped)}")
    print(f"    genuine score  median {np.median(g):.2f}   p90 {np.quantile(g, .90):.2f}")
    print(f"    threshold {threshold:.3f}  (accept below this; target FRR "
          f"{target_frr:.0%})")
    print(f"\nwrote {out}")
    return tpl


def load_template(name: str) -> dict:
    p = template_path(name)
    if not p.exists():
        raise SystemExit(f"[FAIL] {p} missing. Run with --enrol first.")
    return json.loads(p.read_text(encoding="utf-8"))


# ------------------------------------------------------------------ evaluation
def evaluate(df: pd.DataFrame, tpl: dict) -> None:
    cols = tpl["features"]
    name, subject = tpl["name"], tpl["subject"]
    thr = tpl["threshold"]
    mine = df[df["subject"] == subject]
    others = df[df["subject"] != subject]

    print("=" * 72)
    print(f"VERIFICATION - is this {name}?")
    print("=" * 72)
    print(f"\ntemplate: {len(cols)} features, {tpl['n_windows']} windows from "
          f"{len(tpl['enrolled_clips'])} clip(s), threshold {thr:.3f}")

    print(f"\n[1] FALSE REJECT - {name} on a take he was not enrolled with")
    print(f"    {'clip':<26}{'windows':>8}{'median':>9}{'accepted':>10}")
    frr_clip = []
    for c in sorted(mine["clip"].unique()):
        tr, te = mine[mine["clip"] != c], mine[mine["clip"] == c]
        Xtr = tr[cols].to_numpy(float)
        ctr, str_ = np.median(Xtr, axis=0), np.std(Xtr, axis=0)
        str_[str_ < 1e-9] = 1.0
        s = score(te[cols].to_numpy(float), ctr, str_)
        acc = float((s < thr).mean())
        frr_clip.append(float(np.median(s) < thr))
        print(f"    {c[:25]:<26}{len(s):>8}{np.median(s):>9.2f}{acc:>9.0%}")
    print(f"    clip-level: {int(sum(frr_clip))}/{len(frr_clip)} takes accepted "
          f"-> FRR {1 - np.mean(frr_clip):.0%}")

    print(f"\n[2] FALSE ACCEPT - people who are not {name}")
    centre = np.asarray(tpl["centre"], float)
    scale = np.asarray(tpl["scale"], float)
    if others.empty:
        print("    no impostor data at all - FAR is unmeasured, not zero")
        return
    n_ident = others["subject"].nunique()
    for subj in sorted(others["subject"].unique()):
        for c in sorted(others[others["subject"] == subj]["clip"].unique()):
            te = others[(others["subject"] == subj) & (others["clip"] == c)]
            s = score(te[cols].to_numpy(float), centre, scale)
            verdict = "ACCEPTED (false)" if np.median(s) < thr else "rejected"
            print(f"    {c[:25]:<26}{len(s):>8}{np.median(s):>9.2f}  subject {subj}"
                  f"  {verdict}")
    print(f"\n    {n_ident} impostor identity(ies) from "
          f"{others['clip'].nunique()} clip(s).")
    if n_ident < 3:
        print("    That is a handful of data points, not a false-accept RATE. Quoting")
        print("    a percentage here would imply a sample that does not exist. What it")
        print("    can show is whether the template rejects anyone at all.")

    print("\n[3] SEPARATION")
    sg = score(mine[cols].to_numpy(float), centre, scale)
    si = score(others[cols].to_numpy(float), centre, scale)
    print(f"    {name:<12} median {np.median(sg):.2f}  p90 {np.quantile(sg, .9):.2f}")
    print(f"    {'impostor':<12} median {np.median(si):.2f}  "
          f"p10 {np.quantile(si, .1):.2f}")
    gap = float(np.median(si) - np.median(sg))
    print(f"    gap {gap:+.2f}")
    if gap <= 0.1:
        print("    Impostors score about as close to the template as the enrolled")
        print("    person does. There is no usable decision here at any threshold.")


# ------------------------------------------------------------------ query
def verify_video(video: Path, tpl: dict, args) -> None:
    import extract_skeletons as ex
    from ultralytics import YOLO

    meta = {"clip": video.stem, "session": "query", "view": "unknown",
            "subject": "?", "condition": "query"}
    yolo = YOLO(str(paths.MODELS / "yolov8n.pt"))
    ex_args = Namespace(conf=0.25, iou=0.30, max_miss=10, min_track=15,
                        pose_model="full", pose_conf=0.5, crop_target_h=320,
                        crop_margin=0.20, crop_max_scale=8.0, max_frames=0,
                        no_video=args.no_video, run=args.run)
    print(f"extracting skeletons from {video.name} ...")
    skel, info = ex.extract(video, meta, yolo, ex_args)
    if skel.empty:
        raise SystemExit("[FAIL] no pose found in that video")
    print(f"  pose {info['pose_rate']:.1%}, primary track #{info['primary_track']} "
          f"({info['primary_frames']} frames)")

    q = gait.cycle_features(skel, mode=args.mode, win_frames=args.win,
                            stride=args.stride)
    if q.empty:
        raise SystemExit("[FAIL] no usable gait window - needs a few seconds of walking")

    cols = tpl["features"]
    missing = [c for c in cols if c not in q.columns]
    if missing:
        raise SystemExit(f"[FAIL] query is missing enrolled features: {missing}")
    s = score(q[cols].to_numpy(float), np.asarray(tpl["centre"], float),
              np.asarray(tpl["scale"], float))
    thr = tpl["threshold"]
    med = float(np.median(s))
    share = float((s < thr).mean())

    print("\n" + "=" * 60)
    print(f"QUERY: {video.name}   {len(s)} window(s)   template: {tpl['name']}")
    print("=" * 60)
    print(f"\n{'window':>7}{'t_s':>8}{'score':>8}  verdict")
    for i, v in enumerate(s):
        print(f"{i:>7}{q['t_start'].iloc[i]:>8.2f}{v:>8.2f}  "
              f"{'match' if v < thr else 'reject'}")
    print(f"\nmedian score {med:.2f}   threshold {thr:.2f}   "
          f"{share:.0%} of windows match")
    if med < thr:
        print(f"\n>>> VERDICT: {tpl['name'].upper()}   "
              f"(score {med:.2f} < {thr:.2f})")
    else:
        print(f"\n>>> VERDICT: NOT {tpl['name'].upper()}   "
              f"(score {med:.2f} >= {thr:.2f})")
    print(f"\nZONE '{args.zone}': "
          + ("AUTHORISED" if med < thr else "DENIED"))


# ------------------------------------------------------------------ main
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--enrol", action="store_true", help="build the template")
    ap.add_argument("--eval", action="store_true", help="FRR / FAR on stored samples")
    ap.add_argument("--video", type=Path, help="is the walker in this clip the enrolled person?")
    ap.add_argument("--name", default="mimouni", help="what to call the enrolled person")
    ap.add_argument("--subject", default="A", help="which subject label to enrol")
    ap.add_argument("--target-frr", type=float, default=0.10,
                    help="fraction of genuine windows the threshold may reject")
    ap.add_argument("--max-between", type=float, default=0.6,
                    help="drop features whose between-take variance exceeds this share")
    ap.add_argument("--mode", choices=["window", "cycle"], default="window")
    ap.add_argument("--win", type=int, default=45)
    ap.add_argument("--stride", type=int, default=10)
    ap.add_argument("--zone", default="lab")
    ap.add_argument("--no-video", action="store_true")
    ap.add_argument("--run", default="verify_query")
    args = ap.parse_args()

    if not (args.enrol or args.eval or args.video):
        ap.error("pick one of --enrol, --eval, --video")
    if not CYCLES.exists():
        raise SystemExit(f"[FAIL] {CYCLES} missing. "
                         "Run: uv run python scripts/gait_id.py --build")
    df = pd.read_parquet(CYCLES)

    tpl = (enrol(df, args.subject, args.name, args.target_frr, args.max_between)
           if args.enrol else load_template(args.name))
    if args.eval:
        evaluate(df, tpl)
    if args.video:
        verify_video(args.video, tpl, args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
