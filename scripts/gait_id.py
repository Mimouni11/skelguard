"""Identify a person by their walk, then decide whether they are allowed in the zone.

Three modes:

    uv run python scripts/gait_id.py --build          # skeletons -> gait cycles
    uv run python scripts/gait_id.py --eval           # how well does it actually work?
    uv run python scripts/gait_id.py --video clip.mp4 # who is this?

The classifier is a nearest-centroid match on z-scored gait features. Deliberately not
something larger: with a few dozen gait cycles from two subjects, an MLP would fit the
noise and report a number that means nothing. A centroid distance has one degree of
freedom per feature and its failure mode is visible in the margin.

Two things this prints that matter more than accuracy:

  margin      how much closer the winning subject is than the runner-up, in units of
              the enrolled spread. A forced choice between two people is right half the
              time by coin flip, so the decision is only worth anything when the margin
              is large. Below --min-margin the answer is UNCERTAIN, and an access
              control system that says "I don't know" is behaving correctly.

  by block    the same evaluation on body proportions alone and on movement alone.
              If anthropometry scores as well as everything together, the system is
              recognising body shape, not gait. That is a real finding and has to be
              reported as one rather than buried in a combined number.
"""

from __future__ import annotations

import argparse
import sys
from argparse import Namespace
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from skelguard import gait, paths  # noqa: E402

CYCLES = paths.INTERIM / "gait_cycles.parquet"


# ------------------------------------------------------------------ model
class NearestCentroid:
    """z-score the features on the training set, then match to the closest class mean."""

    def __init__(self) -> None:
        self.mu = self.sd = None
        self.classes: list[str] = []
        self.centroids: np.ndarray | None = None
        self.spread: float = 1.0

    def fit(self, X: np.ndarray, y: np.ndarray) -> "NearestCentroid":
        self.mu = X.mean(axis=0)
        self.sd = X.std(axis=0)
        self.sd[self.sd < 1e-9] = 1.0
        Z = (X - self.mu) / self.sd
        self.classes = sorted(set(y))
        self.centroids = np.stack([Z[y == c].mean(axis=0) for c in self.classes])
        # typical within-class distance, so the margin has a unit
        d = [np.linalg.norm(Z[y == c] - self.centroids[i], axis=1)
             for i, c in enumerate(self.classes)]
        self.spread = float(np.median(np.concatenate(d))) or 1.0
        return self

    def distances(self, X: np.ndarray) -> np.ndarray:
        Z = (X - self.mu) / self.sd
        return np.linalg.norm(Z[:, None, :] - self.centroids[None, :, :], axis=2)

    def predict(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """(labels, margin). Margin is the runner-up gap in units of class spread."""
        d = self.distances(X)
        order = np.argsort(d, axis=1)
        best = order[:, 0]
        if d.shape[1] < 2:
            margin = np.full(len(X), np.inf)
        else:
            margin = (d[np.arange(len(X)), order[:, 1]]
                      - d[np.arange(len(X)), best]) / self.spread
        return np.array([self.classes[i] for i in best]), margin


# ------------------------------------------------------------------ data
def build_cycles(mode: str, win: int, stride: int) -> pd.DataFrame:
    files = sorted(paths.SKELETONS.glob("*.parquet"))
    if not files:
        raise SystemExit(f"[FAIL] no skeletons in {paths.SKELETONS}. "
                         "Run scripts/extract_skeletons.py first.")
    out = []
    for f in files:
        df = pd.read_parquet(f)
        if "world_x" not in df.columns:
            print(f"  [skip] {f.name}: no world landmarks")
            continue
        if str(df["subject"].iloc[0]) in ("?", "nan"):
            print(f"  [skip] {f.name}: unlabelled subject")
            continue
        c = gait.cycle_features(df, mode=mode, win_frames=win, stride=stride)
        print(f"  {f.stem:<26} subject={df['subject'].iloc[0]}  samples={len(c)}")
        if len(c):
            out.append(c)
    if not out:
        raise SystemExit("[FAIL] no gait cycles found in any clip")
    cycles = pd.concat(out, ignore_index=True)
    CYCLES.parent.mkdir(parents=True, exist_ok=True)
    cycles.to_parquet(CYCLES, index=False)
    return cycles


def load_cycles() -> pd.DataFrame:
    if not CYCLES.exists():
        raise SystemExit(f"[FAIL] {CYCLES} missing. Run with --build first.")
    return pd.read_parquet(CYCLES)


def add_groups(df: pd.DataFrame, min_groups: int = 3) -> pd.DataFrame:
    """Evaluation groups: one per clip, except a subject filmed only once.

    Held-out data must come from a take the model never saw, or neighbouring cycles of
    the same walk end up on both sides of the split and the score is meaningless. A
    subject with a single clip cannot satisfy that, so its cycles are cut into
    contiguous chunks instead. That is weaker, and the report says so.
    """
    df = df.copy()
    df["group"] = df["clip"]
    df["group_is_split"] = False
    for subj, g in df.groupby("subject"):
        if g["clip"].nunique() > 1:
            continue
        idx = g.sort_values("cycle").index
        chunks = np.array_split(idx, min(min_groups, max(1, len(idx))))
        for k, ch in enumerate(chunks):
            df.loc[ch, "group"] = f"{subj}_part{k + 1}"
            df.loc[ch, "group_is_split"] = True
    return df


# ------------------------------------------------------------------ evaluation
def leave_one_group_out(df: pd.DataFrame, block: str) -> dict:
    cols = gait.feature_columns(df, block)
    X, y = df[cols].to_numpy(float), df["subject"].to_numpy(str)
    groups = df["group"].to_numpy(str)

    pred = np.empty(len(df), dtype=object)
    marg = np.zeros(len(df))
    for g in sorted(set(groups)):
        te = groups == g
        tr = ~te
        if len(set(y[tr])) < 2:
            pred[te], marg[te] = "(untrainable)", 0.0
            continue
        m = NearestCentroid().fit(X[tr], y[tr])
        pred[te], marg[te] = m.predict(X[te])

    ok = pred == y
    scored = pred != "(untrainable)"
    per_class = {c: float(ok[(y == c) & scored].mean())
                 for c in sorted(set(y)) if ((y == c) & scored).any()}
    return {
        "block": block, "n": int(scored.sum()), "n_features": len(cols),
        "accuracy": float(ok[scored].mean()) if scored.any() else float("nan"),
        "balanced": float(np.mean(list(per_class.values()))) if per_class else float("nan"),
        "per_class": per_class,
        "median_margin": float(np.median(marg[scored])) if scored.any() else float("nan"),
        "pred": pred, "truth": y, "scored": scored,
    }


def confusion(truth, pred, scored) -> str:
    labels = sorted(set(truth))
    lines = ["    " + "".join(f"{l:>8}" for l in labels) + "   <- predicted"]
    for t in labels:
        row = [int(((truth == t) & (pred == p) & scored).sum()) for p in labels]
        lines.append(f" {t:>2} " + "".join(f"{v:>8}" for v in row))
    return "\n".join(lines)


def report_eval(df: pd.DataFrame) -> None:
    print("=" * 74)
    print("GAIT IDENTIFICATION - leave-one-group-out")
    print("=" * 74)
    n_sub = df["subject"].nunique()
    print(f"\n{len(df)} gait cycles, {n_sub} subject(s), "
          f"{df['clip'].nunique()} clip(s), {df['group'].nunique()} group(s)")
    for s, g in df.groupby("subject"):
        print(f"    {s}: {len(g):>4} cycles from {g['clip'].nunique()} clip(s)"
              + ("   [single clip - groups are chunks of one walk, not real takes]"
                 if g["group_is_split"].any() else ""))
    print(f"\nChance level for {n_sub} classes: {1 / n_sub:.0%}")

    results = {}
    print(f"\n{'block':<12}{'feats':>7}{'cycles':>8}{'accuracy':>10}"
          f"{'balanced':>10}{'margin':>9}")
    for block in ("all", "anthro", "dynamics"):
        r = leave_one_group_out(df, block)
        results[block] = r
        print(f"{block:<12}{r['n_features']:>7}{r['n']:>8}{r['accuracy']:>9.1%}"
              f"{r['balanced']:>10.1%}{r['median_margin']:>9.2f}")

    # The baseline that matters. A classifier that cannot beat a constant answer has
    # learned nothing, and with classes this unbalanced raw accuracy hides that.
    maj = df["subject"].value_counts()
    print(f"{'always ' + str(maj.index[0]):<12}{'-':>7}{len(df):>8}"
          f"{maj.iloc[0] / len(df):>9.1%}{1 / n_sub:>10.1%}{'-':>9}")

    r = results["all"]
    base_acc, base_bal = maj.iloc[0] / len(df), 1 / n_sub
    print(f"\nvs the constant answer: accuracy {r['accuracy'] - base_acc:+.1%}, "
          f"balanced {r['balanced'] - base_bal:+.1%}")
    smallest = int(maj.iloc[-1])
    if smallest:
        print(f"    The smallest class has {smallest} sample(s), so one of them is "
              f"worth {0.5 / smallest:.1%} of balanced accuracy.")
        print("    If the gain above is a few times that, it is noise, not a result.")

    print("\nper subject (all features):")
    for c, v in r["per_class"].items():
        print(f"    {c}: {v:.1%}")
    print("\nconfusion (all features):")
    print(confusion(r["truth"], r["pred"], r["scored"]))

    a, d = results["anthro"]["balanced"], results["dynamics"]["balanced"]
    print("\nWHAT IS DOING THE WORK?")
    print(f"    body proportions alone : {a:.1%} balanced")
    print(f"    movement alone         : {d:.1%} balanced")
    if a >= d:
        print("    Proportions match or beat movement, so this is closer to body-shape")
        print("    recognition than to gait recognition. Report it that way.")
    else:
        print("    Movement beats proportions, which is the gait claim holding up.")
    print("\n    Both blocks come from the same skeletons, so neither can be cheating")
    print("    on clothing or appearance - but limb ratios are anatomy, not walking.")


# ------------------------------------------------------------------ predict
def predict_video(video: Path, df: pd.DataFrame, args) -> None:
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
        raise SystemExit("[FAIL] no pose found in that video at all")
    print(f"  pose {info['pose_rate']:.1%}, primary track #{info['primary_track']} "
          f"({info['primary_frames']} frames)")

    q = gait.cycle_features(skel, mode=args.mode, win_frames=args.win,
                            stride=args.stride)
    if q.empty:
        raise SystemExit("[FAIL] no complete gait cycle in that clip. "
                         "It needs a few seconds of uninterrupted walking.")

    cols = gait.feature_columns(df, args.block)
    model = NearestCentroid().fit(df[cols].to_numpy(float),
                                  df["subject"].to_numpy(str))
    labels, margins = model.predict(q[cols].to_numpy(float))

    print("\n" + "=" * 62)
    print(f"QUERY: {video.name}   {len(q)} gait cycle(s)   features={args.block}")
    print("=" * 62)
    print(f"\n{'cycle':>6}{'t_s':>8}{'frames':>8}{'verdict':>10}{'margin':>9}")
    for i, (lab, m) in enumerate(zip(labels, margins)):
        verdict = lab if m >= args.min_margin else "uncertain"
        print(f"{q['cycle'].iloc[i]:>6}{q['t_start'].iloc[i]:>8.2f}"
              f"{q['n_frames'].iloc[i]:>8}{verdict:>10}{m:>9.2f}")

    conf = [(c, int((labels == c).sum())) for c in model.classes]
    conf.sort(key=lambda p: -p[1])
    winner, votes = conf[0]
    share = votes / len(labels)
    med_margin = float(np.median(margins))

    print(f"\nvotes: " + ", ".join(f"{c}={n}" for c, n in conf))
    print(f"median margin: {med_margin:.2f}  (uncertain below {args.min_margin})")
    if med_margin < args.min_margin:
        print(f"\n>>> VERDICT: UNCERTAIN  (leaning {winner}, {share:.0%} of cycles)")
        print("    The two enrolled subjects are not separated well enough on this")
        print("    clip to make a claim. For access control that is a refusal.")
    else:
        print(f"\n>>> VERDICT: subject {winner}   {votes}/{len(labels)} cycles "
              f"({share:.0%}), margin {med_margin:.2f}")

    zones = {"A": args.zone_a.split(","), "B": args.zone_b.split(",")}
    allowed = zones.get(winner, [])
    ok = args.zone in allowed and med_margin >= args.min_margin
    print(f"\nZONE '{args.zone}': "
          + (f"AUTHORISED for {winner}" if ok else "DENIED"))
    if med_margin < args.min_margin:
        print("    denied because the identity is uncertain, not because of the rule")
    elif args.zone not in allowed:
        print(f"    {winner} is permitted in: {', '.join(allowed) or '(nothing)'}")


# ------------------------------------------------------------------ main
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--build", action="store_true", help="rebuild cycles from skeletons")
    ap.add_argument("--eval", action="store_true", help="leave-one-group-out evaluation")
    ap.add_argument("--video", type=Path, help="identify the walker in this clip")
    ap.add_argument("--block", choices=["all", "anthro", "dynamics"], default="all")
    ap.add_argument("--mode", choices=["window", "cycle"], default="window",
                    help="overlapping windows (more samples) or strict gait cycles")
    ap.add_argument("--win", type=int, default=45, help="window length in frames")
    ap.add_argument("--stride", type=int, default=10, help="window stride in frames")
    ap.add_argument("--min-margin", type=float, default=0.35,
                    help="below this the answer is UNCERTAIN")
    ap.add_argument("--min-groups", type=int, default=3)
    ap.add_argument("--zone", default="lab", help="which zone is being entered")
    ap.add_argument("--zone-a", default="lab,corridor", help="zones subject A may enter")
    ap.add_argument("--zone-b", default="corridor", help="zones subject B may enter")
    ap.add_argument("--no-video", action="store_true")
    ap.add_argument("--run", default="gait_query")
    args = ap.parse_args()

    if not (args.build or args.eval or args.video):
        ap.error("pick one of --build, --eval, --video")

    df = (build_cycles(args.mode, args.win, args.stride) if args.build
          else load_cycles())
    df = add_groups(df, args.min_groups)
    if args.build:
        print(f"\n{len(df)} samples ({args.mode} mode) -> {CYCLES}")

    if args.eval:
        report_eval(df)
    if args.video:
        predict_video(args.video, df, args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
