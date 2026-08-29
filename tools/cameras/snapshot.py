"""Capture one frame per camera and save it, labelled by role.

Useful as a quick record of what each camera sees and how sharp it is. Roles come
from the mapping recorded by tools/cameras/identify.py.
"""

import sys
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config.cameras import FPS, HEIGHT, WIDTH, resolve  # noqa: E402

OUT_DIR = Path(__file__).resolve().parent.parent / "captures"
WARMUP_FRAMES = 10


def capture(path: str):
    """Open one camera and return a settled frame, or None on failure."""
    cap = cv2.VideoCapture(path, cv2.CAP_V4L2)
    if not cap.isOpened():
        return None, "could not open device"

    # MJPEG keeps three cameras within USB 2.0 bandwidth.
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)
    cap.set(cv2.CAP_PROP_FPS, FPS)

    try:
        frame = None
        # Early frames from UVC cameras are often black or half-exposed.
        for _ in range(WARMUP_FRAMES):
            ok, f = cap.read()
            if ok:
                frame = f
        if frame is None:
            return None, "opened but no frame returned"
        actual = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        return frame, f"{actual[0]}x{actual[1]}"
    finally:
        cap.release()


def label(frame, text: str):
    out = frame.copy()
    cv2.rectangle(out, (0, 0), (out.shape[1], 40), (0, 0, 0), -1)
    cv2.putText(out, text, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    return out


def sharpness(frame) -> float:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def main() -> int:
    OUT_DIR.mkdir(exist_ok=True)

    cameras = resolve(strict=False)
    if not cameras:
        print("no camera mapping available")
        print("run: python tools/cameras/identify.py")
        return 1

    tiles = []
    for role, device in cameras.items():
        print(f"=== {role}  ({device}) ===")
        if not Path(device).exists():
            print(f"  MISSING: {device}")
            continue

        frame, info = capture(device)
        if frame is None:
            print(f"  FAIL: {info}")
            continue

        dest = OUT_DIR / f"{role}.jpg"
        cv2.imwrite(str(dest), frame)
        mean, sharp = frame.mean(), sharpness(frame)
        print(f"  {info}  brightness={mean:.1f}  sharpness={sharp:.1f}  -> {dest}")
        if mean < 5:
            print("  WARNING: frame is almost black, check lens cap or lighting")
        if sharp < 50:
            print("  WARNING: very low sharpness, camera looks out of focus")
        tiles.append(label(frame, f"{role}  {device}  sharp={sharp:.0f}"))

    if tiles:
        sheet = cv2.hconcat([cv2.resize(t, (WIDTH, HEIGHT)) for t in tiles])
        sheet_path = OUT_DIR / "all_cameras.jpg"
        cv2.imwrite(str(sheet_path), sheet)
        print(f"\ncontact sheet ({len(tiles)} cameras) -> {sheet_path}")

    return 0 if tiles else 1


if __name__ == "__main__":
    sys.exit(main())
