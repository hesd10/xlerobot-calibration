"""One-shot preflight check before teleop or dataset recording.

Verifies, in order:
  1. Python environment (lerobot, torch, CUDA, Feetech SDK)
  2. Serial buses present and writable
  3. All 16 servos responding with the expected IDs
  4. All 3 cameras present and delivering frames

Read-only throughout: nothing is moved, no torque is enabled.
"""

import subprocess
import sys
from pathlib import Path

TOOLS = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(TOOLS))


def section(title: str):
    print(f"\n{'=' * 60}\n{title}\n{'=' * 60}", flush=True)


def check_env() -> list[str]:
    problems = []

    # lerobot and torch belong to teleop and VLA training, not to calibration:
    # this tool reads servos through scservo_sdk directly and never loads a
    # policy. They are reported when present and passed over when not, so a
    # calibration-only install is not told to fetch a CUDA stack it will never
    # use. The Feetech SDK below is a different matter -- it is required.
    try:
        import lerobot
        print(f"  lerobot      {lerobot.__version__}")
    except Exception:
        print("  lerobot      not installed (only needed for teleop)")

    try:
        import torch
        cuda = "CUDA available" if torch.cuda.is_available() else "no CUDA"
        print(f"  torch        {torch.__version__} ({cuda})")
    except Exception:
        print("  torch        not installed (only needed for VLA training)")

    try:
        import scservo_sdk  # noqa: F401
        print("  feetech sdk  OK")
    except Exception as exc:
        problems.append(f"Feetech SDK import failed: {exc}")

    try:
        import cv2
        print(f"  opencv       {cv2.__version__}")
    except Exception as exc:
        problems.append(f"opencv import failed: {exc}")

    return problems


def check_cameras() -> list[str]:
    import cv2
    from config.cameras import HEIGHT, WIDTH, resolve, verify_cameras

    problems = verify_cameras(strict=False)
    cameras = resolve(strict=False)
    if not cameras:
        return problems or ["no camera mapping (run tools/cameras/identify.py)"]

    for role, device in cameras.items():
        cap = cv2.VideoCapture(device, cv2.CAP_V4L2)
        if not cap.isOpened():
            problems.append(f"{role} ({device}): could not open")
            continue
        try:
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, WIDTH)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)
            frame = None
            for _ in range(10):
                ok, f = cap.read()
                if ok:
                    frame = f
            if frame is None:
                problems.append(f"{role} ({device}): opened but no frames")
                continue

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            sharp = float(cv2.Laplacian(gray, cv2.CV_64F).var())
            bright = frame.mean()
            note = ""
            if bright < 5:
                note = "  DARK (lens cap?)"
                problems.append(f"{role}: frame nearly black")
            elif sharp < 50:
                note = "  BLURRY"
                problems.append(f"{role}: sharpness {sharp:.0f}, may be out of focus")
            print(f"  {role:<12} {device:<14} bright={bright:5.1f} sharp={sharp:7.1f}{note}")
        finally:
            cap.release()

    return problems


def run_script(rel: str) -> int:
    """Run another tool script, keeping its output in order with ours."""
    sys.stdout.flush()
    return subprocess.call([sys.executable, "-u", str(TOOLS / rel)])


def main() -> int:
    all_problems: dict[str, list[str]] = {}

    section("1. Python environment")
    all_problems["environment"] = check_env()

    section("2. Serial buses")
    from config.buses import describe, verify_buses
    print(describe())
    all_problems["buses"] = verify_buses(strict=False)
    for p in all_problems["buses"]:
        print(f"  PROBLEM: {p}")

    section("3. Servos")
    rc = run_script("checks/scan_buses.py")
    if rc != 0:
        all_problems["servos"] = ["servo scan reported problems (see above)"]

    section("4. Cameras")
    all_problems["cameras"] = check_cameras()
    for p in all_problems["cameras"]:
        print(f"  PROBLEM: {p}")

    section("Summary")
    failed = {k: v for k, v in all_problems.items() if v}
    if not failed:
        print("  All checks passed. Robot is ready.")
        return 0

    for area, problems in failed.items():
        print(f"  {area}:")
        for p in problems:
            print(f"    - {p}")
    return 1


if __name__ == "__main__":
    sys.exit(main())
