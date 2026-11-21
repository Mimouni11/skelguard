"""Gait features: turn a stored skeleton sequence into something that identifies a person.

The input is one track's landmarks over time, as written by scripts/extract_skeletons.py.
Everything here works on `world_*` - the hip-centred metric landmarks - and never on
pixels, because pixel coordinates encode distance from the camera far more strongly than
they encode identity. On walk05front the subject goes from 301 px to 717 px tall in a
single pass; the same walk in world coordinates barely changes scale at all.

Three normalisations, in order:

  1. re-centre on the hip midpoint   (removes where in the room they are)
  2. divide by torso length          (removes body size AND MediaPipe's absolute scale
                                      error, which is the least trustworthy thing it
                                      returns)
  3. rotate out body yaw             (removes which way they happened to be facing)

Step 2 deserves care. Dividing by torso length also throws away height, which is a
genuinely discriminative signal - so the anthropometric block below keeps *ratios*
between limbs, which survive the division, rather than absolute lengths, which do not.

The features come in two blocks that are deliberately kept separate:

  ANTHROPOMETRY   limb length ratios. These are body proportions, not gait. They will
                  probably carry most of the accuracy, and a result that does not say
                  so is dishonest - "we identified people by their walk" is a different
                  claim from "we identified people by the ratio of their shin to their
                  thigh".
  DYNAMICS        what actually moves: joint angle ranges over a gait cycle, cadence,
                  cycle duration. This is the gait claim.

Reporting them separately is the point. Either answer is a real finding.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# BlazePose indices
L_SHOULDER, R_SHOULDER = 11, 12
L_ELBOW, R_ELBOW = 13, 14
L_WRIST, R_WRIST = 15, 16
L_HIP, R_HIP = 23, 24
L_KNEE, R_KNEE = 25, 26
L_ANKLE, R_ANKLE = 27, 28
L_HEEL, R_HEEL = 29, 30
L_TOE, R_TOE = 31, 32

N_LANDMARKS = 33
# The joints the gait cycle is read from. If these are not visible the frame is useless
# however good the rest of the skeleton looks.
GAIT_JOINTS = (L_HIP, R_HIP, L_KNEE, R_KNEE, L_ANKLE, R_ANKLE)


# ------------------------------------------------------------------ loading
def track_arrays(df: pd.DataFrame, track: int | None = None
                 ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Long skeleton table -> (world (T,33,3), visibility (T,33), t_s (T,)).

    Picks the primary track by default, which is the one extract_skeletons decided was
    the walker rather than a bystander.
    """
    if track is None:
        d = df[df["is_primary"]] if "is_primary" in df.columns else df
    else:
        d = df[df["track"] == track]
    d = d.sort_values(["frame", "landmark"])
    frames = d["frame"].to_numpy()
    n = len(d) // N_LANDMARKS
    if n == 0 or len(d) % N_LANDMARKS:
        raise ValueError(f"expected whole poses of {N_LANDMARKS} landmarks, got {len(d)}")
    world = d[["world_x", "world_y", "world_z"]].to_numpy(float).reshape(n, N_LANDMARKS, 3)
    vis = d["visibility"].to_numpy(float).reshape(n, N_LANDMARKS)
    t = d["t_s"].to_numpy(float).reshape(n, N_LANDMARKS)[:, 0]
    _ = frames
    return world, vis, t


# ------------------------------------------------------------------ normalisation
def _rotate_y(w: np.ndarray, ang: np.ndarray) -> np.ndarray:
    """Rotate about the vertical axis by `ang` (T,) radians."""
    c, s = np.cos(ang)[:, None], np.sin(ang)[:, None]
    x, y, z = w[..., 0], w[..., 1], w[..., 2]
    return np.stack([c * x - s * z, y, s * x + c * z], axis=-1)


def _moving_average(v: np.ndarray, k: int) -> np.ndarray:
    if k < 2 or len(v) < k:
        return v
    pad = k // 2
    padded = np.pad(v, (pad, pad), mode="edge")
    return np.convolve(padded, np.ones(k) / k, mode="valid")[:len(v)]


def normalise(world: np.ndarray, yaw_smooth: int = 15) -> np.ndarray:
    """Hip-centred, torso-scaled, yaw-aligned skeletons."""
    w = world - world[:, [L_HIP, R_HIP], :].mean(axis=1, keepdims=True)

    shoulder = w[:, [L_SHOULDER, R_SHOULDER], :].mean(axis=1)
    hip = w[:, [L_HIP, R_HIP], :].mean(axis=1)
    torso = np.linalg.norm(shoulder - hip, axis=1)
    torso = torso[np.isfinite(torso) & (torso > 1e-6)]
    scale = float(np.median(torso)) if len(torso) else 1.0
    w = w / max(scale, 1e-6)

    # Yaw from the hip line, smoothed: the raw per-frame hip vector is noisy enough that
    # de-rotating on it would inject jitter into every joint angle downstream.
    hip_vec = w[:, R_HIP, :] - w[:, L_HIP, :]
    yaw = np.arctan2(hip_vec[:, 2], hip_vec[:, 0])
    yaw = np.unwrap(yaw)
    yaw = _moving_average(yaw, yaw_smooth)
    return _rotate_y(w, -yaw)


# ------------------------------------------------------------------ cycles
def joint_angle(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> np.ndarray:
    """Angle at b, in degrees, for (T,3) arrays."""
    v1, v2 = a - b, c - b
    n1 = np.linalg.norm(v1, axis=-1)
    n2 = np.linalg.norm(v2, axis=-1)
    cos = (v1 * v2).sum(-1) / np.maximum(n1 * n2, 1e-9)
    return np.degrees(np.arccos(np.clip(cos, -1.0, 1.0)))


def cycle_signal(w: np.ndarray) -> np.ndarray:
    """A one-period-per-gait-cycle waveform from the ankles.

    The vertical difference between the two ankles: one foot lifts while the other is
    planted, so this crosses zero once per step and completes one period per full gait
    cycle. Raw *distance* between the ankles would not work - it has two minima per
    cycle and cannot tell left-forward from right-forward.

    Vertical specifically, and not the principal axis of the 3D separation, which is
    what this did first. On a walk toward the camera that principal axis is dominated
    by z, and z is the axis MediaPipe estimates worst from a single view. It cost about
    half the detectable cycles on the frontal clips. The y axis is in-image and is the
    one the model actually gets right.
    """
    sep = w[:, L_ANKLE, 1] - w[:, R_ANKLE, 1]
    if len(sep) < 3:
        return np.zeros(len(sep))
    return _moving_average(sep - sep.mean(), 5)


def count_cycles(signal: np.ndarray) -> int:
    """Upward zero crossings - how many gait cycles a slice of signal contains."""
    return sum(1 for i in range(1, len(signal)) if signal[i - 1] <= 0 < signal[i])


def find_cycles(signal: np.ndarray, fps: float, min_s: float = 0.6,
                max_s: float = 2.0) -> list[tuple[int, int]]:
    """Gait cycles as upward zero crossings of `signal`, filtered to plausible lengths."""
    crossings = [i for i in range(1, len(signal))
                 if signal[i - 1] <= 0 < signal[i]]
    out = []
    for a, b in zip(crossings, crossings[1:]):
        dur = (b - a) / fps
        if min_s <= dur <= max_s:
            out.append((a, b))
    return out


def _resample(v: np.ndarray, n: int) -> np.ndarray:
    if len(v) < 2:
        return np.full(n, v[0] if len(v) else 0.0)
    return np.interp(np.linspace(0, len(v) - 1, n), np.arange(len(v)), v)


# ------------------------------------------------------------------ features
def anthropometry(w: np.ndarray) -> dict[str, float]:
    """Limb length ratios. Body proportions, not gait - reported as its own block."""
    def seg(i, j):
        return np.linalg.norm(w[:, i, :] - w[:, j, :], axis=-1)

    thigh = np.median(np.concatenate([seg(L_HIP, L_KNEE), seg(R_HIP, R_KNEE)]))
    shin = np.median(np.concatenate([seg(L_KNEE, L_ANKLE), seg(R_KNEE, R_ANKLE)]))
    upper = np.median(np.concatenate([seg(L_SHOULDER, L_ELBOW), seg(R_SHOULDER, R_ELBOW)]))
    fore = np.median(np.concatenate([seg(L_ELBOW, L_WRIST), seg(R_ELBOW, R_WRIST)]))
    sh_w = np.median(seg(L_SHOULDER, R_SHOULDER))
    hip_w = np.median(seg(L_HIP, R_HIP))
    foot = np.median(np.concatenate([seg(L_HEEL, L_TOE), seg(R_HEEL, R_TOE)]))
    leg = thigh + shin

    eps = 1e-6
    return {
        "a_shin_thigh": float(shin / max(thigh, eps)),
        "a_leg_torso": float(leg),              # torso is 1.0 after normalisation
        "a_arm_leg": float((upper + fore) / max(leg, eps)),
        "a_fore_upper": float(fore / max(upper, eps)),
        "a_shoulder_hip": float(sh_w / max(hip_w, eps)),
        "a_shoulder_w": float(sh_w),
        "a_hip_w": float(hip_w),
        "a_foot_shin": float(foot / max(shin, eps)),
    }


def dynamics(w: np.ndarray, sl: slice, fps: float,
             signal: np.ndarray | None = None) -> dict[str, float]:
    """What moves over one slice: joint angle ranges, swing, cadence.

    `signal` is what lets a fixed-length window report an honest cadence. A window is
    not one cycle, so its duration carries no rhythm information, but the number of
    crossings inside it does.
    """
    seg = w[sl]
    knee_l = joint_angle(seg[:, L_HIP], seg[:, L_KNEE], seg[:, L_ANKLE])
    knee_r = joint_angle(seg[:, R_HIP], seg[:, R_KNEE], seg[:, R_ANKLE])
    elb_l = joint_angle(seg[:, L_SHOULDER], seg[:, L_ELBOW], seg[:, L_WRIST])
    elb_r = joint_angle(seg[:, R_SHOULDER], seg[:, R_ELBOW], seg[:, R_WRIST])
    ank_sep = np.linalg.norm(seg[:, L_ANKLE] - seg[:, R_ANKLE], axis=-1)
    wri_sep = np.linalg.norm(seg[:, L_WRIST] - seg[:, R_WRIST], axis=-1)

    dur = len(seg) / fps
    n_cyc = max(count_cycles(signal[sl]), 1) if signal is not None else 1
    cycle_s = dur / n_cyc
    out = {
        "d_cycle_s": float(cycle_s),
        "d_cadence": float(2.0 / max(cycle_s, 1e-6)),   # steps per second
        "d_knee_range": float(np.mean([np.ptp(knee_l), np.ptp(knee_r)])),
        "d_knee_min": float(np.mean([knee_l.min(), knee_r.min()])),
        "d_knee_max": float(np.mean([knee_l.max(), knee_r.max()])),
        "d_knee_asym": float(abs(np.ptp(knee_l) - np.ptp(knee_r))),
        "d_elbow_range": float(np.mean([np.ptp(elb_l), np.ptp(elb_r)])),
        "d_elbow_mean": float(np.mean([elb_l.mean(), elb_r.mean()])),
        "d_ankle_sep_max": float(ank_sep.max()),
        "d_ankle_sep_mean": float(ank_sep.mean()),
        "d_wrist_sep_mean": float(wri_sep.mean()),
        # Shoulders, not hips: normalise() puts the hip midpoint at the origin, so hip
        # height is identically zero on every frame and would carry no information.
        "d_torso_bounce": float(
            np.ptp(seg[:, [L_SHOULDER, R_SHOULDER], 1].mean(axis=1))),
    }
    # resampled knee curve, coarse: the shape of the swing, not just its extremes
    for i, v in enumerate(_resample((knee_l + knee_r) / 2.0, 8)):
        out[f"d_knee_p{i}"] = float(v)
    return out


ANTHRO_PREFIX, DYN_PREFIX = "a_", "d_"


def cycle_features(df: pd.DataFrame, fps: float | None = None,
                   min_visibility: float = 0.3, mode: str = "window",
                   win_frames: int = 45, stride: int = 10) -> pd.DataFrame:
    """One row per sample, for one clip's primary track.

    mode="cycle"   one row per detected gait cycle. Cleanest unit, but it throws away
                   every frame that does not sit inside a complete cycle, and on this
                   footage that was over half of them - 29 samples from 60 seconds of
                   walking, which is too few to measure anything.
    mode="window"  overlapping fixed-length windows. Each still spans more than a full
                   cycle at 45 frames / 1.5 s, so the same features are meaningful, and
                   it yields roughly five times as many samples. The windows overlap,
                   so samples within one clip are correlated - which is exactly why the
                   evaluation holds out whole clips rather than random rows.
    """
    world, vis, t = track_arrays(df)
    if fps is None:
        fps = float(df["fps"].iloc[0]) if "fps" in df.columns else 30.0

    w = normalise(world)
    sig = cycle_signal(w)
    if mode == "cycle":
        spans = find_cycles(sig, fps)
    else:
        spans = [(a, a + win_frames)
                 for a in range(0, max(0, len(w) - win_frames + 1), stride)]
    if not spans:
        return pd.DataFrame()
    cycles = spans

    meta_cols = [c for c in ("clip", "session", "subject", "condition", "view")
                 if c in df.columns]
    meta = {c: df[c].iloc[0] for c in meta_cols}

    rows = []
    for ci, (a, b) in enumerate(cycles):
        sl = slice(a, b)
        # A cycle is only usable if the legs were actually seen through it.
        leg_vis = float(vis[sl][:, list(GAIT_JOINTS)].mean())
        if leg_vis < min_visibility:
            continue
        feats = anthropometry(w[sl]) | dynamics(w, sl, fps, signal=sig)
        rows.append({**meta, "cycle": ci, "t_start": float(t[a]),
                     "n_frames": b - a, "leg_visibility": round(leg_vis, 3), **feats})
    return pd.DataFrame(rows)


def feature_columns(df: pd.DataFrame, block: str = "all") -> list[str]:
    """Which columns are features, optionally restricted to one block."""
    pref = {"all": (ANTHRO_PREFIX, DYN_PREFIX),
            "anthro": (ANTHRO_PREFIX,),
            "dynamics": (DYN_PREFIX,)}[block]
    return [c for c in df.columns if c.startswith(pref)]
