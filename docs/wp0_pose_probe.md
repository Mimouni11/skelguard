# WP0 — Pose probe: what we measured, and what the architecture must therefore be

Decision record for test runs 1, 2, 3a, 3b and 4. Every number below comes from a run whose
code snapshot and CSVs are in `outputs/reports/<run>/`. Nothing here is from the
literature; it is all measured on our own clips.

**Headline:** MediaPipe cannot be the detector. It needs a person to be ~150 px tall to
produce anything and ~250 px to be reliable, and when it fails it invents skeletons on
furniture. The pipeline is therefore **YOLO detects → crop and upscale → MediaPipe poses
one person → homography puts them on the floor in metres.**

---

## 1. Test clips

| Clip | Source | Resolution | Person heights (p05 / median / p95) |
|---|---|---|---|
| `Workers walking in hallway of office.mp4` | found footage, corridor | 1280×720 @25 | 68 / 153 / 398 px |
| `Meet_Crowd.mpg` | CAVIAR | 384×288 @25 | 26 / 47 / 105 px |
| `Walk1.mpg` | CAVIAR | 384×288 @25 | 29 / 42 / 117 px |

All probes run at **native resolution**. An early run downscaled 1280→960 and shrank the
very quantity under test; that result was discarded.

---

## 2. Test run 1 — MediaPipe full-frame, multi-person

Scored against YOLOv8 person boxes as a reference detector, matched greedily by IoU.

| Clip | Miss rate |
|---|---|
| hallway | 73.9 % |
| Meet_Crowd | 100 % |
| Walk1 | 100 % |

**Recall by person pixel height (pooled across clips) — the core finding:**

| Person height | Recall |
|---|---|
| < 60 px | 0 % |
| 60–100 px | 0 % |
| 100–150 px | 0.6 % (1 of 163) |
| 150–250 px | 57.7 % |
| > 250 px | 100 % |

> **~150 px of person height is the floor for MediaPipe to work at all; ~250 px is the
> floor for it to work reliably.** This single table drives every other decision in the
> project, including how we film.

Speed was initially flattered — MediaPipe short-circuits on frames with no person, so a
pooled median read 71 fps when the real figure was ~31. We now report `fps_when_detecting`
and `fps_end_to_end` separately.

---

## 3. Test run 2 — two attempts to rescue MediaPipe

### 3a. Single-person mode (`num_poses=1`)

Recall did not improve. Ghost poses exploded: **262 ghosts in Meet_Crowd = 800/min**.

Capping MediaPipe at one pose makes it *force* a pose somewhere in the frame. This is the
table that got a skeleton. For a system whose output is an intrusion alert, a ghost is a
false alarm, so this mode is unusable regardless of recall.

### 3b. Detect → crop → upscale → pose (adopted)

YOLO box → expand 20 % → upscale so the person is ~320 px → single-pose on the crop →
landmarks mapped back to full-frame pixels.

| Clip | Miss, full-frame | Miss, crop | Upscale |
|---|---|---|---|
| hallway | 73.9 % | **8.0 %** | 2.5× |
| Walk1 | 100 % | 42.5 % | ~7× |
| Meet_Crowd | 100 % | 76.4 % | ~7× |

**Ghosts: zero.** Pose only ever runs inside a box YOLO called a person, so there is
nowhere for a ghost to appear. This is a structural property of the architecture, not a
tuned threshold.

### Continuity — the metric that actually matters

WP5 needs unbroken 30-frame windows to train on. Recall alone hides whether they exist.

| Clip | Longest run | Usable frames | 30-frame windows |
|---|---|---|---|
| hallway | 384 frames (15.4 s) | 82.2 % | many |
| Walk1 | 22 frames | — | **0** |
| Meet_Crowd | 3 frames | — | **0** |

> **CAVIAR yields no training data at all, whatever its recall says.** It is retained for
> tracking and crossing checks only. Training data must come from our own footage.

---

## 4. Test run 3a — detector sweep (deterministic, seeded)

Inference runs once per model per clip at a 0.05 confidence floor; every box is cached
with its score and all thresholds are then evaluated offline. Re-running changes nothing.

Hallway clip:

| Model | Conf | Frames with box | Longest run | Usable | ms/frame |
|---|---|---|---|---|---|
| yolov8n | 0.25 | 62 % | 404 | 57 % | 31 |
| yolov8m | 0.10 | 76 % | 409 | 70 % | 200 |

- `box_h_min` 32–57 px: **YOLO detects people roughly 3× smaller than MediaPipe can pose.**
  The detector is not the bottleneck; the pose model is.
- `frac_below_pose_floor` 55–65 % on the hallway: the majority of detections are of people
  too small to pose. Crop-and-upscale is what converts them into usable poses.
- 46.6 % of all detections score below 0.25.
- Speed on CPU: yolov8n 31–35 ms, yolov8s 77–92 ms, yolov8m 200–240 ms (≈5 fps — not live).

**Split the detector by purpose:**

| Path | Model | Conf | Why |
|---|---|---|---|
| Live | yolov8n | 0.25 | ~30 fps on CPU |
| Overnight batch | yolov8m | 0.15 | quality matters, latency does not |

The overnight batch extracts skeletons once into `data/skeletons/`, so WP3/4/5 iterate on
cached arrays instead of re-decoding video. That is what makes a 5 fps detector legitimate
there. If yolov8m is ever wanted live, that — not a vague wish — is the justification for
installing the CUDA build of torch.

---

## 5. Test run 3b — floor geometry

### The expected-height model, fitted with no measurement

For a camera with little roll, a fixed-height object's pixel height is linear in the image
y of its feet:

```
h_px = k · (y_feet − y_horizon)
```

Both constants RANSAC-fit from detections we had already cached, with no tape measure.
RANSAC rather than least squares because the inputs contain false positives (a chair
scoring 0.17) that would drag an ordinary fit.

| Detector | k | y_horizon | Inliers | Median rel. err |
|---|---|---|---|---|
| yolov8n | 0.569 | 36.6 px | 97.3 % | 3.2 % |
| yolov8s | 0.560 | 36.0 px | 99.6 % | 3.6 % |
| yolov8m | 0.562 | 36.8 px | 98.8 % | 4.5 % |

> Three independent detectors recover the same horizon and slope within 2 %. That is
> recovered camera geometry, not an overfit.

It **fails on CAVIAR** (46–61 % inliers, y_horizon −5107 and 169677): an overhead
wide-angle view breaks the pinhole assumption. Undistort first, or rely on the homography.

This also retired an earlier claim of mine that 0.25 was a hard confidence floor — that
advice had been driven by CAVIAR junk. On the hallway clip, 85–91 % of sub-0.25 boxes are
geometrically plausible, i.e. mostly real people.

### Homography

Four floor corners clicked once (`scripts/calibrate_scene.py`) → `calib/<scene>.json`.
A wrong homography raises no error, it just returns wrong metres, so the tool writes a
verification image with a 1 m floor grid and 1.7 m height sticks. **The grid must lie flat
and stay square as it recedes.** That check is mandatory before any speed is quoted.

### Results (hallway, conf 0.10, 948 frames)

| | median | p90 | max | in walking range | impossible (>10 m/s) |
|---|---|---|---|---|---|
| raw | 1.63 | 2.86 | 14.94 | 56.2 % | 0.6 % |
| Kalman in **metres** | 1.63 | 2.69 | 12.53 | **60.5 %** | 0.9 % |
| Kalman in pixels | 1.62 | 2.71 | 12.90 | 58.8 % | 0.8 % |

Median 1.63 m/s is normal walking pace — the calibration is believable. (60 % "in range"
is not a failure: 0.8–2.0 m/s is a narrow band, and a person slowing at a door is
legitimately below it.)

**Speed must be measured over a window, not between frames.** Frame-to-frame differencing
of a box's foot point is dominated by detector jitter — a 5 px wobble at 25 fps is metres
per second of noise. Moving to a 0.4 s window (still well inside one stride):

| | per-frame | 0.4 s window |
|---|---|---|
| speed jitter | 1.567 | **0.470** |
| in walking range | 41.5 % | **60.5 %** |
| max | 29.9 m/s | 12.5 m/s |
| impossible | 3.3 % | 0.9 % |

**Kalman belongs in metres** — 0.470 vs 0.518 jitter, 60.5 % vs 58.8 % in range.
Consistent in direction, ~10 % better. A person walking at constant speed has constant
velocity in metres and a wildly non-constant one in pixels, because perspective compresses
distance with depth. The margin is modest and should be reported as such, but filtering in
floor coordinates costs nothing since we compute them anyway.

### The geometric gate

Rejects a box whose size is impossible for a person standing at that spot, regardless of
the detector's confidence.

| Conf band | Boxes | Rejected |
|---|---|---|
| < 0.25 | 83 | 4.8 % |
| 0.25–0.5 | 98 | 1.0 % |
| ≥ 0.5 | 634 | 0.2 % |

The shape is right — it almost never touches a confident detection. But with only 6
rejections on a clip containing no false positives, **the gate is unproven, not
validated.** It is cheap insurance whose real test is footage that actually contains junk.

> **Superseded by 8.3.** That test has since been run. The gate is *disproven* for
> human-sized objects: it keeps a mannequin at 5.5 % geometric error. It rejects wrong
> *sizes*, not non-people.

### Extrapolation limit

One track read 37.4 m of path in 10.4 s = 3.6 m/s average. The cause: the calibration
rectangle is ~4 m long and that track ran past 7 m. Outside the clicked quad a one-pixel
foot error becomes a large metre error, because the pixel-to-metre ratio explodes near the
horizon.

> **Rule: click the calibration rectangle so it covers the whole zone of interest.**
> Never calibrate a 4 m patch and then ask for metres at 10 m.

Cost: ~30 fps end-to-end including the homography, which is free by comparison with
detection.

---

## 6. What this means for the build

1. **Pipeline:** YOLOv8n (live) / YOLOv8m (batch) → geometric gate → crop + upscale to
   ~320 px → MediaPipe single-pose → landmarks to full frame → homography to floor metres
   → Kalman in metres.
2. **Pose is optional, position is not.** Below ~150 px there is no skeleton, but there is
   still a floor position. Intrusion detection (is a person inside the zone?) survives
   distance; movement classification does not.
3. **Filming constraints — non-negotiable:**
   - people must stay ≥150 px tall, ideally ≥250 px → film at 1080p
   - calibration rectangle must span the whole monitored zone
   - put the tape marks down and measure them *before* filming; it cannot be done after
4. **Training data comes from our own footage.** CAVIAR gives zero usable 30-frame windows.
5. **Zones are defined in metres on the floor plane**, not in image pixels, so they stay
   valid if the camera is re-mounted — only the homography is re-clicked.

## 7. Test run 4 — pose wired into floor tracking

Test run 4 adds the adopted crop pose pass to the test run 3b floor tracker. Each kept
YOLO detection is expanded by 20%, upscaled toward 320 px person height (maximum 8×), and
sent to MediaPipe Full in single-pose image mode. The pose is attached to the track ID;
floor position still comes from the YOLO box foot point, so a pose miss does not erase the
person's location.

Hallway result (`conf=0.10`, 948 frames):

| Measure | Result |
|---|---:|
| Kept YOLO boxes / track observations | 809 |
| Poses found | 624 (77.1%) |
| Poses missed | 185 |
| Skeleton output | 20,592 landmarks (33 per pose) |
| End-to-end speed, YOLO + pose + homography | 15.4 fps |

Pose rate is against kept YOLO detections, not ground truth. This is a useful integration
result, not a final recall score. The full-model pipeline is only just above the project's
15 fps target on this CPU, and a first pass measured 13.6 fps; treat the real-time result as
borderline and try the Lite pose model if a stable margin is needed. The floor gate also
rejected only six boxes, with no confirmed false positives in this clip, so its ability to
remove junk remains unproven.

Artifacts are saved under `outputs/reports/test run 4/` and
`outputs/videos/test run 4/`. The report has per-observation pose status, the skeleton CSV
has one row per landmark, and the annotated video draws the skeleton over each track.

## 8. Test run 5 — every calibrated clip, with controls

Run 4 left two holes. Floor metres had only ever been computed for the hallway, because it
was the only calibrated scene, so "does the geometry preserve distance" was untested
anywhere else. And run 4 changed the crop-scale rule *and* the YOLO confidence together,
so its recall could not be attributed to either. Run 5 calibrates all four clips and holds
confidence fixed at 0.25 across every clip and both arms.

### 8.1 Geometry per camera

| Clip | Scene | Tracks | Median m/s | In walking range | Impossible |
|---|---|---:|---:|---:|---:|
| hallway | hallway | 3 | 1.59 | 62.5 % | 1.0 % |
| WalkByShop1front | shopfront | 17 | **0.05** | 12.7 % | 0.0 % |
| Walk1 | caviar_lobby | 2 | 2.52 | 28.4 % | 6.2 % |
| Meet_Crowd | caviar_lobby_2 | 2 | 2.09 | 32.9 % | 0.0 % |

The homography survives the fisheye views better than the height model did — Meet_Crowd
reports no impossible speeds at all. This is expected: a homography is a general
projective map of a plane and absorbs any planar perspective exactly, whereas the height
model additionally assumes a pinhole camera with no roll. Walk1's 6.2 % impossible comes
from two short tracks, not from the mapping.

Shopfront's 0.05 m/s is **not** a calibration failure. See 8.3.

### 8.2 A/B — does calibrated height improve the pose crop?

Same clip, same confidence, same seed; only the crop-scale rule differs.

| Arm | Pose recall | Max upscale |
|---|---:|---:|
| crop sized from detected box height | 87.3 % | 6.44× |
| crop sized from expected height | 87.4 % | 4.91× |

**Null result, and a real one:** the two arms produced different crops on 666 of 730
observations, and pose outcome flipped on 14 — 7 gained, 7 lost. On a clip where the
height model agrees with the detected box to 4 %, sizing the crop from geometry changes
nothing. Its one measurable effect is clamping the maximum upscale, which is worth keeping
as a guard against over-magnifying a truncated box, but it is not a recall improvement and
should not be presented as one.

### 8.3 The finding: geometry cannot tell a person from an object

WalkByShop1front produced 17 tracks. Twelve of them sit at **the same pixel, (286.6,
146.7) ± 0.1 px**, for 1467 frames combined — one for 28 seconds unbroken. They are a
mannequin and a poster in a shop window.

| | Tracks | Frames | Pose recall | Median score | Net displacement |
|---|---:|---:|---:|---:|---:|
| Static | 12 | 1467 | 5.3 % | 0.38 | 0.00–0.09 m |
| Moving | 5 | 812 | 30.7 % | 0.82 | 3.75–10.40 m |

Would the geometric gate have caught them? No:

| Object | Observed | Expected | Rel. error | Gate @35 % |
|---|---:|---:|---:|---|
| mannequin A | 103.0 px | 97.7 px | 5.5 % | keep |
| mannequin B | 37.1 px | 38.7 px | 4.1 % | keep |
| real walker | 118.4 px | 116.2 px | 1.9 % | keep |

> **The height gate is disproven for this failure mode, not merely unproven.** A
> human-sized object standing on the floor is geometrically *perfect*. The gate rejects
> things of the wrong size — the chair from run 3a, 77 px where a person should be ~250 px
> — and is structurally incapable of rejecting a mannequin. Section 5 previously called it
> "unproven"; that was too generous.

This matters more than a recall percentage: a static false positive inside a restricted
zone raises a **permanent** intrusion alarm.

### 8.4 The static-object filter

The only signal that separates them is motion. A track is scenery if its net floor
displacement stays below **0.5 m over any 3 s window**.

Net displacement, not path length — a jittering box accumulates path while going nowhere,
which is exactly what a mannequin looks like. And measured in metres, so one threshold is
valid everywhere in the frame; the same test in pixels would mean different things at
different depths.

Validation:

| Case | Net displacement | Classified |
|---|---:|---|
| 2 cm jitter in place | 0.110 m | scenery |
| walking at 1.4 m/s | 4.200 m | person |
| pacing ±30 cm | 0.554 m | person |

The filter takes the **maximum** over any window across the whole track, so a person who
walks in and then stands still remains a person. Only a track that has never moved since
it appeared is scenery. That distinction is essential here — a motionless intruder is
precisely who the system exists to catch.

### 8.5 A measurement bug worth recording

Runs 5 and 6 reported 83.8 % and 87.3 % pose recall on the hallway under identical
settings. The only difference was `--no-video`.

The cause: each skeleton was drawn onto the frame being analysed, so the second person in
a frame was cropped out of an image with the first person's skeleton painted across them.
**Writing the video was degrading the results the video displayed.** Drawing now happens
on a copy, and runs 6 and 7 agree to the digit (87.3 % / 87.4 %).

Two lessons that apply to the rest of the project:

- Annotation must never touch the array the model reads. Draw on a copy, always.
- Two runs that differ under "identical" settings mean a bug, not noise. The instinct to
  write it off as variance would have left a silent defect in every video-producing run,
  and would have made the A/B above unmeasurable.

### 8.6 Additions to the build rules

6. **Intrusion = person + displacement, never detection alone.** A detection that has
   never moved is scenery regardless of its confidence or its geometry.
7. **The height gate's job is size, not identity.** Keep it for wrong-sized detections and
   as an upscale clamp; do not rely on it to remove objects.
8. **Pose recall doubles for real people** (30.7 % vs 5.3 % on shopfront), so a persistently
   failing pose is itself weak evidence of a non-person — useful as a secondary signal.

## 9. Reproducing

```powershell
uv run python scripts/probe_batch.py --run "test run 1"
uv run python scripts/probe_pose.py "data/raw/<clip>" --crop --run "test run 2 crop"
uv run python scripts/probe_yolo_sweep.py --run "test run 3a"
uv run python scripts/calibrate_scene.py "data/raw/<clip>" --name hallway --width 1.8 --length 8.0 \
    --detections "outputs/reports/test run 3a/yolo_sweep_detections.csv"
uv run python scripts/probe_floor.py "data/raw/<clip>" --scene hallway --conf 0.10
uv run python scripts/probe_floor.py "data/raw/Workers walking in hallway of office.mp4" \
    --scene hallway --conf 0.10 --speed-window 0.40 --run "test run 4"
```

Each run writes its report, CSVs, `run_meta.json` and a **frozen copy of the code that
produced it** into `outputs/reports/<run>/`, so any number above can be traced back.
