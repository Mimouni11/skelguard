"""WP2 / test run 3b - calibrate a camera position: pixels <-> metres on the floor.

Click the 4 corners of a floor rectangle whose real dimensions you measured, and this
saves the homography to calib/<scene>.json. It also writes a verification image with a
1 m grid projected back onto the frame: a wrong homography is silent otherwise, and the
grid is the only cheap way to catch it.

    uv run python scripts/calibrate_scene.py data/raw/clip.mp4 --name lab_cam1 \
        --width 2.0 --length 6.0

Click order, clockwise starting at the NEAR-LEFT corner:

        4 ___________ 3      far
         |           |
         |           |
        1|___________|2      near   <- start here, go clockwise

Width is the near edge (1->2), length is the near-to-far edge (1->4).

On our own footage the 4 points are tape marks measured once. On found footage they can
be estimated from floor tiles or a known corridor width - good enough to show that speed
is linear in floor coordinates, not good enough to quote an absolute error.

Also fits the expected-height model from a detections CSV, if given:

    --detections "outputs/reports/test run 3a/yolo_sweep_detections.csv"

That needs no measurement at all and gives the geometric plausibility gate.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from skelguard import paths  # noqa: E402
from skelguard.geometry import PERSON_HEIGHT_M, Scene, fit_height_model  # noqa: E402

CORNER_LABELS = ["1 near-left", "2 near-right", "3 far-right", "4 far-left"]


def grab_frame(video: Path, n: int) -> np.ndarray:
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise SystemExit(f"[FAIL] could not open {video}")
    cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, n))
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise SystemExit(f"[FAIL] could not read frame {n} of {video}")
    return frame


def click_points(frame: np.ndarray, title: str) -> list[list[float]]:
    pts: list[list[float]] = []
    disp = frame.copy()

    def redraw() -> None:
        nonlocal disp
        disp = frame.copy()
        for i, (x, y) in enumerate(pts):
            cv2.circle(disp, (int(x), int(y)), 5, (0, 220, 255), -1)
            cv2.putText(disp, CORNER_LABELS[i], (int(x) + 8, int(y) - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 220, 255), 1, cv2.LINE_AA)
        if len(pts) > 1:
            cv2.polylines(disp, [np.int32(pts)], len(pts) == 4, (0, 220, 255), 2)
        nxt = CORNER_LABELS[len(pts)] if len(pts) < 4 else "ENTER to accept"
        bar = f"click {nxt}   |   u = undo   |   r = reset   |   ESC = cancel"
        cv2.rectangle(disp, (0, 0), (disp.shape[1], 26), (0, 0, 0), -1)
        cv2.putText(disp, bar, (8, 18), cv2.FONT_HERSHEY_SIMPLEX,
                    0.55, (255, 255, 255), 1, cv2.LINE_AA)

    def on_mouse(event, x, y, flags, param) -> None:
        if event == cv2.EVENT_LBUTTONDOWN and len(pts) < 4:
            pts.append([float(x), float(y)])
            redraw()

    redraw()
    cv2.namedWindow(title, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(title, on_mouse)
    while True:
        cv2.imshow(title, disp)
        key = cv2.waitKey(20) & 0xFF
        if key == 27:
            cv2.destroyAllWindows()
            raise SystemExit("cancelled")
        if key in (ord("u"), 8) and pts:
            pts.pop()
            redraw()
        if key == ord("r"):
            pts.clear()
            redraw()
        if key in (13, 10) and len(pts) == 4:
            break
    cv2.destroyAllWindows()
    return pts


def verification_image(frame: np.ndarray, scene: Scene, step: float = 1.0) -> np.ndarray:
    """Project a 1 m floor grid back onto the frame. If it does not look like the floor,
    the calibration is wrong."""
    out = frame.copy()
    w_m = max(p[0] for p in scene.world_points)
    l_m = max(p[1] for p in scene.world_points)

    def line(a, b, colour, thick=1):
        try:
            p = scene.to_image(np.array([a, b], dtype=np.float32))
        except Exception:  # noqa: BLE001
            return
        if np.all(np.isfinite(p)):
            cv2.line(out, tuple(np.int32(p[0])), tuple(np.int32(p[1])), colour, thick,
                     cv2.LINE_AA)

    for x in np.arange(0, w_m + 1e-6, step):
        line([x, 0], [x, l_m], (0, 200, 0))
    for y in np.arange(0, l_m + 1e-6, step):
        line([0, y], [w_m, y], (0, 200, 0))

    # the calibration rectangle itself, and a 1.7 m stick at a few floor points
    cv2.polylines(out, [np.int32(scene.image_points)], True, (0, 220, 255), 2)
    model = scene.model
    if model:
        for y in np.arange(step, l_m + 1e-6, max(step, l_m / 4)):
            foot = scene.to_image(np.array([[w_m / 2, y]], dtype=np.float32))[0]
            if not np.all(np.isfinite(foot)):
                continue
            hpx = float(model.expected_px(foot[1]))
            if hpx <= 0:
                continue
            cv2.line(out, (int(foot[0]), int(foot[1])),
                     (int(foot[0]), int(foot[1] - hpx)), (255, 120, 0), 2, cv2.LINE_AA)
            cv2.putText(out, f"{PERSON_HEIGHT_M}m={hpx:.0f}px",
                        (int(foot[0]) + 6, int(foot[1] - hpx / 2)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 120, 0), 1, cv2.LINE_AA)

    cv2.rectangle(out, (0, 0), (out.shape[1], 26), (0, 0, 0), -1)
    cv2.putText(out, f"{scene.name}: green = {step:g} m floor grid, "
                     f"orange = expected person height",
                (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("video", type=Path)
    ap.add_argument("--name", required=True, help="scene id, e.g. lab_cam1")
    ap.add_argument("--width", type=float, help="near edge 1->2, in metres")
    ap.add_argument("--length", type=float, help="near-to-far edge 1->4, in metres")
    ap.add_argument("--frame", type=int, default=0, help="which frame to click on")
    ap.add_argument("--detections", type=Path, default=None,
                    help="a yolo_sweep_detections.csv to fit the height model from")
    ap.add_argument("--det-clip", default=None,
                    help="clip stem to filter the detections by (default: this video's)")
    ap.add_argument("--det-conf", type=float, default=0.5,
                    help="fit the height model on confident boxes only")
    ap.add_argument("--grid", type=float, default=1.0, help="verification grid step, m")
    ap.add_argument("--note", default="")
    args = ap.parse_args()

    frame = grab_frame(args.video, args.frame)
    h, w = frame.shape[:2]

    # ---------------------------------------------------- height model (no measuring)
    model = None
    if args.detections:
        import pandas as pd

        d = pd.read_csv(args.detections)
        stem = args.det_clip or args.video.stem
        d = d[d["clip"] == stem]
        if d.empty:
            print(f"  [warn] no detections for clip {stem!r} in {args.detections.name}")
        else:
            d = d[d["score"] >= args.det_conf]
            model = fit_height_model(d["y2"].to_numpy(float), d["box_h"].to_numpy(float))
            if model:
                print(f"height model: h_px = {model.k:.3f} * (y_feet - {model.y_horizon:.1f})")
                print(f"              n={model.n}  inliers={model.inlier_rate:.1%}  "
                      f"median rel err={model.median_rel_err:.1%}")
                if model.inlier_rate < 0.80:
                    print("  [warn] weak fit. Wide-angle or overhead cameras break the")
                    print("         pinhole assumption - undistort first, or rely on the")
                    print("         homography alone.")
            else:
                print("  [warn] height model did not fit")

    # ---------------------------------------------------- homography (needs measuring)
    scene: Scene
    if args.width and args.length:
        print(f"\nClick the 4 floor corners of your {args.width} x {args.length} m "
              f"rectangle, clockwise from near-left.")
        pts = click_points(frame, f"calibrate {args.name}")
        scene = Scene.from_rectangle(args.name, pts, args.width, args.length, w, h,
                                     note=args.note)
    else:
        print("\n[no --width/--length] saving the height model only, without a homography.")
        scene = Scene(name=args.name, image_points=[], world_points=[],
                      frame_w=w, frame_h=h, H=None, note=args.note)

    if model:
        scene.height_model = model.to_json()

    out = scene.save()
    print(f"\nwrote {out}")

    if scene.H is not None:
        vis = verification_image(frame, scene, step=args.grid)
        png = paths.OUT_FIGURES / f"calib_{args.name}.png"
        png.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(png), vis)
        print(f"wrote {png}")
        print("\nCHECK THE VERIFICATION IMAGE before trusting any speed in m/s:")
        print("  - the green grid must lie flat on the floor and stay square as it recedes")
        print("  - the orange sticks must look like the height of a person standing there")
        print("  A wrong homography produces no error, only wrong metres.")
        cv2.imshow(f"verify {args.name} - any key to close", vis)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
