"""Canonical project paths, so no module ever hard-codes a relative path."""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

DATA = ROOT / "data"
RAW = DATA / "raw"              # own filmed footage
DATASETS = DATA / "datasets"    # URFD, HMDB51, CAVIAR / PETS
SKELETONS = DATA / "skeletons"  # extracted landmarks, one file per video
INTERIM = DATA / "interim"      # windows, features, caches
LABELS = DATA / "labels"        # clip scripts + ground-truth event CSVs (committed)

CALIB = ROOT / "calib"          # zones.json + homography per scene (committed)

MODELS = ROOT / "models"
MP_MODELS = MODELS / "mediapipe"
TRAINED = MODELS / "trained"

POSE_FULL = MP_MODELS / "pose_landmarker_full.task"
POSE_LITE = MP_MODELS / "pose_landmarker_lite.task"
POSE_HEAVY = MP_MODELS / "pose_landmarker_heavy.task"
OBJECT_DETECTOR = MP_MODELS / "efficientdet_lite0.tflite"

OUTPUTS = ROOT / "outputs"
OUT_VIDEOS = OUTPUTS / "videos"
OUT_LOGS = OUTPUTS / "logs"
OUT_FIGURES = OUTPUTS / "figures"
OUT_REPORTS = OUTPUTS / "reports"


def ensure_dirs() -> None:
    """Create every output folder. Safe to call repeatedly."""
    for p in (RAW, DATASETS, SKELETONS, INTERIM, LABELS, CALIB,
              MP_MODELS, TRAINED, OUT_VIDEOS, OUT_LOGS, OUT_FIGURES, OUT_REPORTS):
        p.mkdir(parents=True, exist_ok=True)
