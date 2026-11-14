# SkelGuard — Projet 2 : sécurisation passive de zones restreintes

Privacy-preserving surveillance of restricted areas (server room, lab, workshop, site).
Every alert is derived from **skeletons and kinematics only — faces are never used**.

**In:** a video file or a webcam stream.
**Out:** an annotated video (one box per person with ID, authorised/forbidden, movement class
and speed; the zone drawn; visible alerts), a CSV event log, a heatmap and a flow report.

## Events

| Event | Signal |
|---|---|
| Intrusion | ankle midpoint inside the zone polygon |
| Loitering | dwell time > T seconds |
| Running | speed > threshold (m/s, via homography) |
| Fall | hips drop fast + torso tilt + immobility |
| Entry / exit count | crossing a virtual line, per track ID |
| Object removed / carried | object boxes + wrist keypoints + track ID |
| Abandoned object *(if time)* | static object whose owner track has left |

## Pipeline

```
video -> MediaPipe PoseLandmarker (multi-pose) -> own tracker (Hungarian + Kalman)
      -> smoothing + homography to metres -> kinematics
      -> rules engine  AND  from-scratch MLP/LSTM  -> alerts + outputs
```

YOLOv8 + ByteTrack is the **fallback only**, used if MediaPipe misses too many people or
swaps IDs too often on our own footage. Everything downstream is unchanged either way.

## Setup

```powershell
uv sync --python 3.11                       # venv + all dependencies
uv run python scripts/download_models.py    # MediaPipe pose + object models
uv run python scripts/check_setup.py        # verify everything works
```

Python is pinned to **3.11** (MediaPipe does not support 3.13).
PyTorch is the **CPU** build by default — see the comment in `pyproject.toml` to switch to
CUDA 12.8 when the YOLOv8 fallback needs the GPU.

Run anything with `uv run python ...` — no `pip`, no manual venv activation.

## Layout

```
data/raw/          own filmed footage              (gitignored)
data/datasets/     URFD, HMDB51, CAVIAR/PETS       (gitignored)
data/skeletons/    extracted landmarks per track   (gitignored)
data/labels/       clip scripts + ground-truth event CSVs   (COMMITTED)
calib/             zones.json + homography matrix per scene (COMMITTED)
models/mediapipe/  .task / .tflite bundles         (gitignored, reproducible)
models/trained/    MLP / LSTM checkpoints          (gitignored)
outputs/           annotated videos, CSV logs, figures      (gitignored)
outputs/reports/   final metric tables + report    (COMMITTED)
src/skelguard/     the package
scripts/           entry points
```

## Work packages

| WP | Content | When |
|---|---|---|
| WP0 | Datasets, MediaPipe check, own footage with floor marks, label definitions | before the weekend |
| WP1 | Multi-pose + own tracker (Hungarian + Kalman); ID-switch measurement | Day 1 · Fri 20 Nov |
| WP2 | Zone click tool + homography, `calib/zones.json`, point-in-polygon on ankles | Day 1 |
| WP3 | Kinematics in metres: speed, acceleration, direction, dwell | Day 2 · Sat 21 Nov |
| WP4 | Per-track rule state machine | Day 2 |
| WP5 | 30-frame windows, MLP from scratch, then LSTM, vs the rules | Day 2 |
| WP8 | Objects: removed (level 1), carried (2), abandoned (3) | Day 3 · Sun 22 Nov |
| WP6 | Annotated video, CSV log, heatmap | Day 3 |
| — | Evaluation on own footage, report, demo video | 23–29 Nov, submit **30 Nov** |

**Drop order if time runs out:** abandoned object → carried object → LSTM → Kalman comparison
→ heatmap. Never cut: the brief minimum, multi-person tracking, homography, or the test on
own footage.

## Data principle

Train and validate on public datasets; **test on our own footage, which the models never see
during training.** All final numbers come from the own footage.

## Ethics

Only skeletons are kept, never faces or identities. Written consent from everyone filmed.
Out of scope: face recognition, identifying people, live deployment on a real site.
