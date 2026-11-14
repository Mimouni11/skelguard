"""Download the MediaPipe task bundles into models/mediapipe/.

Usage:
    uv run python scripts/download_models.py          # pose (full) + object detector
    uv run python scripts/download_models.py --all    # also the heavy/light pose variants
    uv run python scripts/download_models.py --force  # re-download even if present

Models are gitignored on purpose: they are reproducible artefacts, not source.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import requests
from tqdm import tqdm

MODELS_DIR = Path(__file__).resolve().parents[1] / "models" / "mediapipe"
BASE = "https://storage.googleapis.com/mediapipe-models"

# name -> (url, why we want it)
MODELS: dict[str, tuple[str, str]] = {
    # --- pose (WP1) ---
    "pose_landmarker_full.task": (
        f"{BASE}/pose_landmarker/pose_landmarker_full/float16/latest/pose_landmarker_full.task",
        "DEFAULT. 33 landmarks, good accuracy/speed balance on a laptop CPU.",
    ),
    # --- objects (WP8) ---
    "efficientdet_lite0.tflite": (
        f"{BASE}/object_detector/efficientdet_lite0/float32/latest/efficientdet_lite0.tflite",
        "DEFAULT. COCO detector: backpack, handbag, suitcase, laptop, phone, bottle.",
    ),
}

EXTRA: dict[str, tuple[str, str]] = {
    "pose_landmarker_lite.task": (
        f"{BASE}/pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task",
        "Fallback if we cannot hold 15 fps.",
    ),
    "pose_landmarker_heavy.task": (
        f"{BASE}/pose_landmarker/pose_landmarker_heavy/float16/latest/pose_landmarker_heavy.task",
        "For the overnight batch extraction, where fps does not matter.",
    ),
    "efficientdet_lite2.tflite": (
        f"{BASE}/object_detector/efficientdet_lite2/float32/latest/efficientdet_lite2.tflite",
        "More accurate objects if lite0 misses the scripted removals.",
    ),
}


def download(name: str, url: str, dest_dir: Path, force: bool = False) -> bool:
    dest = dest_dir / name
    if dest.exists() and not force:
        print(f"  [skip]  {name}  ({dest.stat().st_size / 1e6:.1f} MB, already here)")
        return True
    tmp = dest.with_suffix(dest.suffix + ".part")
    try:
        with requests.get(url, stream=True, timeout=60) as r:
            r.raise_for_status()
            total = int(r.headers.get("content-length", 0))
            with open(tmp, "wb") as f, tqdm(
                total=total, unit="B", unit_scale=True, desc=f"  {name}", leave=True
            ) as bar:
                for chunk in r.iter_content(chunk_size=1 << 16):
                    f.write(chunk)
                    bar.update(len(chunk))
        tmp.replace(dest)
        return True
    except Exception as exc:  # noqa: BLE001 - we want the reason, then carry on
        print(f"  [FAIL]  {name}: {exc}", file=sys.stderr)
        tmp.unlink(missing_ok=True)
        return False


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--all", action="store_true", help="also fetch the lite/heavy variants")
    ap.add_argument("--force", action="store_true", help="re-download files that already exist")
    args = ap.parse_args()

    wanted = dict(MODELS)
    if args.all:
        wanted.update(EXTRA)

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    print(f"-> {MODELS_DIR}\n")

    ok = 0
    for name, (url, why) in wanted.items():
        print(f"{name}\n    {why}")
        ok += download(name, url, MODELS_DIR, force=args.force)
        print()

    failed = len(wanted) - ok
    print(f"{ok}/{len(wanted)} model(s) ready in models/mediapipe/")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
