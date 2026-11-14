# Ground truth for our own footage

Our own footage is the **test set**. Every final number in the report is computed by comparing
the pipeline's event CSV against the files in this folder, so these must be written **while
filming**, not reconstructed afterwards from memory.

Two files per clip, both committed to git (they are small text).

## 1. `<clip_id>.script.md` — what was staged

Written *before* the take, ticked off after. Free text, but it must state:

- scene / camera position id (must match the key in `calib/zones.json`)
- who is in frame (person A, person B — **never real names**)
- which scenario: authorised passage, intrusion, loitering, running, fall, object removed,
  object carried out, object left behind
- the objects present in the zone at the start

Example `lab_cam1_take03.script.md`:

```
scene: lab_cam1
people: A (dark hoodie), B (white t-shirt)
objects in zone at start: laptop on the desk, backpack on the floor
plan: A walks through the authorised corridor; B enters the zone at ~0:20,
      loiters ~8 s, takes the laptop at ~0:34 and leaves to the right.
```

## 2. `<clip_id>.events.csv` — the timed truth

One row per ground-truth event. Columns:

| column | meaning |
|---|---|
| `clip_id` | file stem of the video in `data/raw/`, no extension |
| `t_start` | seconds from the start of the clip (`12.4`) |
| `t_end` | seconds; equal to `t_start` for instantaneous events |
| `person` | staged label `A`, `B`, … (our label, not a track ID) |
| `event` | one of `intrusion, loitering, running, fall, entry, exit, object_removed, object_carried, object_abandoned` |
| `object` | COCO class for object events, else empty |
| `notes` | occlusions, anything that went wrong in the take |

Example:

```csv
clip_id,t_start,t_end,person,event,object,notes
lab_cam1_take03,20.1,20.1,B,intrusion,,
lab_cam1_take03,20.1,28.3,B,loitering,,
lab_cam1_take03,34.0,34.0,B,object_removed,laptop,hand briefly hidden by torso
```

`person` is a **staged** label. Mapping it to the pipeline's track IDs is part of the
evaluation; keeping them separate is what lets us count ID switches honestly.

## Conventions, decided once, now

- **Timestamps** are seconds from the first frame of the clip, one decimal is enough.
- **Loitering** gets one row spanning entry to exit; the threshold T is a *parameter we tune*,
  so the truth records the dwell, not whether it crossed a threshold.
- **Running** gets one row per continuous run segment.
- **Fall**: `t_start` is the first frame of the descent, `t_end` the frame where the body is
  at rest.
- **object_removed** vs **object_carried**: `removed` is the instant the object leaves its
  resting place; `carried` is the span it is held while moving. A theft produces both.
- Clip ids: `<scene>_<camera>_take<NN>`, e.g. `lab_cam1_take03`. The `<scene>_<camera>` part
  must exist in `calib/zones.json` or the homography will not be found.

## Consent

Keep the signed consent forms **outside this repo**. Record only that they exist:
`consent/ A: yes (2026-11-xx), B: yes (2026-11-xx)` in the clip script. Only skeletons are
kept from these videos; the raw footage is gitignored and does not leave the machine.

## Movement classes (for WP5)

Separate from events. The window labels are `walk`, `run`, `still`, `fall` — defined in
writing *before* labelling, 30-frame windows inheriting the clip/segment label, every video
resampled to the same fps, splits made **by video and by subject** so the same person never
appears in both train and test.
