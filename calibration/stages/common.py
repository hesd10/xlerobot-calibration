"""Shared helpers for stage scripts.

Keeps the interaction style identical across stages, so an operator learns the
conventions once. Also centralises the safety checks that every stage needs:
confirm hardware is present, confirm prerequisites were actually solved, and
refuse to overwrite a good result by accident.
"""

from __future__ import annotations

import sys
from pathlib import Path

CALIB = Path(__file__).resolve().parent.parent
TOOLS = CALIB.parent / "tools"
for _p in (str(CALIB), str(TOOLS)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from core import gates, storage  # noqa: E402

BAR = "-" * 68


class Aborted(Exception):
    """Operator ended the stage with Ctrl-C or Ctrl-D."""


def ask(prompt: str, default: str | None = None) -> str:
    suffix = f" [{default}]" if default else ""
    while True:
        try:
            answer = input(f"  {prompt}{suffix}: ").strip()
        except (EOFError, KeyboardInterrupt):
            raise Aborted from None
        if answer:
            return answer
        if default is not None:
            return default


def confirm(prompt: str, default: bool = False) -> bool:
    d = "y" if default else "n"
    while True:
        answer = ask(f"{prompt} (y/n)", d).lower()
        if answer in ("y", "yes"):
            return True
        if answer in ("n", "no"):
            return False
        print("    please answer y or n")


def ask_float(prompt: str, default: float | None = None,
              lo: float = -1e9, hi: float = 1e9) -> float:
    while True:
        raw = ask(prompt, None if default is None else f"{default:g}")
        try:
            value = float(raw)
        except ValueError:
            print("    not a number")
            continue
        if not (lo <= value <= hi):
            print(f"    must be between {lo:g} and {hi:g}")
            continue
        return value


def ask_int(prompt: str, default: int | None = None,
            lo: int = -10**9, hi: int = 10**9) -> int:
    while True:
        raw = ask(prompt, None if default is None else str(default))
        try:
            value = int(raw)
        except ValueError:
            print("    not an integer")
            continue
        if not (lo <= value <= hi):
            print(f"    must be between {lo} and {hi}")
            continue
        return value


def heading(text: str) -> None:
    print(f"\n{BAR}\n{text}\n{BAR}")


def require_results(*names: str) -> dict:
    """Load prerequisite results, or explain what is missing and stop."""
    loaded = {}
    missing = []
    for name in names:
        data = storage.load_result(name)
        if data is None:
            missing.append(name)
        else:
            loaded[name] = data
    if missing:
        print(f"\n  Cannot run: missing {', '.join(missing)}")
        print("  Run: python calibration/run.py --status")
        raise Aborted
    return loaded


def confirm_overwrite(*names: str) -> bool:
    """Ask before replacing results that already exist."""
    existing = [n for n in names if storage.load_result(n) is not None]
    if not existing:
        return True
    print(f"\n  These results already exist: {', '.join(existing)}")
    return confirm("  Overwrite them", default=False)


def report_gates(results: list[gates.GateResult]) -> bool:
    """Print gate outcomes and say plainly whether the stage is acceptable."""
    heading("Acceptance check")
    passed, text = gates.summarise(results)
    print(text)
    if passed:
        print("\n  Stage accepted.")
    else:
        print("\n  Stage NOT accepted. Later stages build on this, so an error")
        print("  here will be silently absorbed downstream rather than showing up")
        print("  as an obvious failure. Collect more or better data and retry.")
    return passed


def open_camera(role: str, width: int | None = None, height: int | None = None):
    """Open one robot camera by role, with the usual failure explanations.

    If the preview server is running it is asked to release this camera first,
    since a V4L2 device admits only one opener. Call release_camera() when done so
    the preview comes back.
    """
    import cv2
    from config.cameras import HEIGHT, WIDTH, resolve

    from core import preview_client

    width = width or WIDTH
    height = height or HEIGHT

    devices = resolve(strict=False)
    if role not in devices:
        raise RuntimeError(
            f"camera '{role}' not found. Run tools/cameras/identify.py to "
            f"record which physical camera is which.")

    borrowed = False
    if preview_client.is_running() and role in preview_client.roles():
        preview_client.pause(role)
        borrowed = True

    device = devices[role]
    cap = cv2.VideoCapture(device, cv2.CAP_V4L2)
    if not cap.isOpened():
        if borrowed:
            preview_client.resume(role)
        raise RuntimeError(
            f"could not open {role} ({device}).\n"
            f"    Something else holds this camera. Find it with:\n"
            f"      fuser -v {device}\n"
            f"    Common causes: tools/cameras/identify.py, or an earlier\n"
            f"    calibration run that was interrupted.")
    if borrowed:
        _BORROWED.add(role)
        _register_atexit()

    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    got = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
           int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    if got != (width, height):
        cap.release()
        release_camera(role)
        raise RuntimeError(
            f"{role} gave {got[0]}x{got[1]} when asked for {width}x{height}. "
            f"Intrinsics are resolution-specific, so this must match.")
    return cap, device


# Roles this process paused on the preview server, so they can be handed back.
_BORROWED: set[str] = set()
_ATEXIT_REGISTERED = False


def _register_atexit() -> None:
    """Hand cameras back even if the stage crashes or is interrupted.

    Without this a failed stage leaves the preview showing frozen images, which
    reads as a broken preview rather than a failed stage.
    """
    global _ATEXIT_REGISTERED
    if not _ATEXIT_REGISTERED:
        import atexit

        atexit.register(release_all_cameras)
        _ATEXIT_REGISTERED = True


def release_camera(role: str) -> None:
    """Return a camera to the preview server, if it was borrowed from it."""
    if role not in _BORROWED:
        return
    from core import preview_client

    preview_client.resume(role)
    _BORROWED.discard(role)


def release_all_cameras() -> None:
    for role in list(_BORROWED):
        release_camera(role)


def preview_hint() -> str:
    """Tell the operator how to get a live view, if they have not started one."""
    from core import preview_client

    if preview_client.is_running():
        return (f"  Live view of all three cameras: "
                f"{preview_client.BASE_URL}")
    return ("  For a live view of all three cameras during calibration, run in\n"
            "  another terminal:  python calibration/preview.py")


def load_board(name: str | None = None):
    """Load the measured board spec, or explain that stage -1 has not run."""
    from core import charuco, storage

    boards = charuco.load_boards()
    if not boards:
        print("\n  No board recorded. Run: python calibration/run.py --stage prep")
        raise Aborted
    if name and name in boards:
        spec = boards[name]
    elif name:
        print(f"\n  No board named '{name}'. Available: {', '.join(boards)}")
        raise Aborted
    else:
        recorded = storage.load_result("board")
        preferred = (recorded or {}).get("world_board")
        if preferred in boards:
            spec = boards[preferred]
        else:
            spec = boards.get("main", next(iter(boards.values())))

    warn_board_drift(spec)
    if not spec.measured:
        print(f"\n  NOTE: '{spec.name}' square size is not caliper-measured, so")
        print(f"  every distance solved from it carries the same scale error.")
    return spec


def warn_board_drift(spec) -> None:
    """Say so if board.json disagrees with what stage -1 recorded.

    The two can drift apart by editing board.json directly, and nothing else
    notices: stages read board.json while --results shows the stage record.
    """
    from core import storage

    recorded = storage.load_result("board")
    if not recorded:
        return
    entry = (recorded.get("boards") or {}).get(spec.name)
    if not entry:
        return

    interesting = ("squares_x", "squares_y", "square_mm", "marker_mm",
                   "dictionary", "legacy")
    drift = [(k, entry.get(k), getattr(spec, k)) for k in interesting
             if k in entry and entry[k] != getattr(spec, k)]
    if not drift:
        return

    print("\n  WARNING: board.json differs from what stage -1 recorded.")
    for key, was, now in drift:
        print(f"    {key}: recorded {was!r}, now {now!r}")
    print("  Using board.json. Rerun stage prep to bring the record up to date.")


def not_implemented(stage_number: str, title: str, plan: str) -> int:
    """Placeholder so the runner reports honestly instead of crashing."""
    heading(f"Stage {stage_number}: {title}")
    print("  Not implemented yet. The plan for this stage:\n")
    for line in plan.rstrip().splitlines():
        print(f"  {line}")
    print("\n  The skeleton, dependency checks and acceptance gates are in place,")
    print("  so this stage can be filled in without touching the others.")
    return 0
