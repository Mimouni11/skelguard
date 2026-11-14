"""Smoke test for the SkelGuard workspace.

Verifies that every library imports, that the MediaPipe task bundles load, and that a
multi-pose PoseLandmarker + an ObjectDetector can actually run on one frame.

    uv run python scripts/check_setup.py
    uv run python scripts/check_setup.py --image data/raw/some_frame.jpg
"""

from __future__ import annotations

import argparse
import platform
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = ROOT / "models" / "mediapipe"

OK, BAD, WARN = "[ ok ]", "[FAIL]", "[warn]"
_failures: list[str] = []


def report(ok: bool, label: str, detail: str = "") -> bool:
    print(f"{OK if ok else BAD} {label}" + (f"  {detail}" if detail else ""))
    if not ok:
        _failures.append(label)
    return ok


def check_imports() -> None:
    print("\n--- libraries ---")
    import cv2
    import matplotlib
    import numpy as np
    import pandas as pd
    import scipy
    import sklearn

    report(True, "numpy", np.__version__)
    report(True, "opencv", cv2.__version__)
    report(True, "scipy", scipy.__version__)
    report(True, "pandas", pd.__version__)
    report(True, "scikit-learn", sklearn.__version__)
    report(True, "matplotlib", matplotlib.__version__)

    import mediapipe as mp

    report(True, "mediapipe", mp.__version__)

    import torch

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    report(True, "torch", f"{torch.__version__}  (device: {dev})")
    if dev == "cpu":
        print(f"{WARN} torch is the CPU build - fine for the MLP/LSTM; see pyproject.toml "
              f"to switch to cu128 when the YOLOv8 fallback needs the GPU")

    try:
        import ultralytics

        report(True, "ultralytics (YOLOv8 fallback)", ultralytics.__version__)
    except Exception as exc:  # noqa: BLE001
        print(f"{WARN} ultralytics not usable yet: {exc}")


def _test_frame(image: Path | None):
    import cv2
    import numpy as np

    if image is not None:
        bgr = cv2.imread(str(image))
        if bgr is None:
            raise SystemExit(f"{BAD} could not read image: {image}")
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    # No footage yet: a grey frame is enough to prove the graph runs end to end.
    return np.full((480, 640, 3), 128, dtype=np.uint8)


def check_mediapipe(image: Path | None) -> None:
    print("\n--- mediapipe task bundles ---")
    import mediapipe as mp
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision

    rgb = _test_frame(image)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    src = f"frame from {image.name}" if image else "synthetic 640x480 grey frame"
    print(f"       input: {src}")

    pose_path = MODELS_DIR / "pose_landmarker_full.task"
    if not report(pose_path.exists(), "pose_landmarker_full.task present",
                  "" if pose_path.exists() else "run scripts/download_models.py"):
        return

    opts = vision.PoseLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=str(pose_path)),
        running_mode=vision.RunningMode.IMAGE,
        num_poses=4,  # the brief requires multi-person
    )
    with vision.PoseLandmarker.create_from_options(opts) as landmarker:
        res = landmarker.detect(mp_image)
    report(True, "PoseLandmarker runs (num_poses=4)", f"{len(res.pose_landmarks)} pose(s) found")

    det_path = MODELS_DIR / "efficientdet_lite0.tflite"
    if not report(det_path.exists(), "efficientdet_lite0.tflite present",
                  "" if det_path.exists() else "run scripts/download_models.py"):
        return

    d_opts = vision.ObjectDetectorOptions(
        base_options=mp_python.BaseOptions(model_asset_path=str(det_path)),
        running_mode=vision.RunningMode.IMAGE,
        score_threshold=0.4,
        category_allowlist=["backpack", "handbag", "suitcase", "laptop", "cell phone", "bottle"],
    )
    with vision.ObjectDetector.create_from_options(d_opts) as detector:
        d_res = detector.detect(mp_image)
    report(True, "ObjectDetector runs", f"{len(d_res.detections)} object(s) found")


def check_layout() -> None:
    print("\n--- folders ---")
    for rel in ["data/raw", "data/datasets", "data/skeletons", "data/labels",
                "calib", "models/mediapipe", "models/trained",
                "outputs/videos", "outputs/logs", "outputs/figures", "src/skelguard"]:
        report((ROOT / rel).is_dir(), rel)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--image", type=Path, default=None,
                    help="run the models on a real frame instead of a grey one")
    args = ap.parse_args()

    print(f"SkelGuard setup check")
    print(f"python {platform.python_version()}  |  {platform.system()} {platform.release()}")
    print(f"root   {ROOT}")

    check_layout()
    check_imports()
    check_mediapipe(args.image)

    print("\n" + "=" * 60)
    if _failures:
        print(f"{BAD} {len(_failures)} check(s) failed: {', '.join(_failures)}")
        return 1
    print(f"{OK} workspace is ready.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
