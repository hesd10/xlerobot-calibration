"""Stage -1: physical preparation.

Walks the operator through the setup that has to be right before any data is
worth collecting, and records the board geometry. Nothing here is optional
paperwork: an unmeasured board puts a scale error on every result, and a lens
refocused later invalidates the intrinsics silently.

    python calibration/stages/stage_prep.py
"""

from __future__ import annotations

import sys
import threading
import time
from pathlib import Path

CALIB = Path(__file__).resolve().parent.parent
TOOLS = CALIB.parent / "tools"
for p in (str(CALIB), str(TOOLS)):
    if p not in sys.path:
        sys.path.insert(0, p)

from core import charuco, storage  # noqa: E402

BAR = "-" * 68


class Aborted(Exception):
    """Operator ended the session with Ctrl-C or Ctrl-D."""


def ask(prompt: str, default: str | None = None) -> str:
    suffix = f" [{default}]" if default else ""
    while True:
        try:
            answer = input(f"  {prompt}{suffix}: ").strip()
        except (EOFError, KeyboardInterrupt):
            # A traceback here would look like a crash rather than a choice.
            raise Aborted from None
        if answer:
            return answer
        if default is not None:
            return default


def ask_float(prompt: str, default: float | None = None,
              lo: float = 0.0, hi: float = 1e6) -> float:
    while True:
        raw = ask(prompt, None if default is None else f"{default:g}")
        try:
            value = float(raw)
        except ValueError:
            print("    not a number, try again")
            continue
        if not (lo < value < hi):
            print(f"    must be between {lo:g} and {hi:g}")
            continue
        return value


def ask_int(prompt: str, default: int | None = None, lo: int = 1, hi: int = 100) -> int:
    while True:
        raw = ask(prompt, None if default is None else str(default))
        try:
            value = int(raw)
        except ValueError:
            print("    not an integer, try again")
            continue
        if not (lo <= value <= hi):
            print(f"    must be between {lo} and {hi}")
            continue
        return value


def confirm(prompt: str, default: bool = False) -> bool:
    """Yes/no. Rejects anything that is not clearly y or n."""
    fallback = "y" if default else "n"
    while True:
        answer = ask(f"{prompt} (y/n)", fallback).lower()
        if answer in ("y", "yes"):
            return True
        if answer in ("n", "no"):
            return False
        print("    please answer y or n")


def checklist() -> bool:
    print(f"\n{BAR}\nPhysical checklist\n{BAR}")
    items = [
        ("The board is printed and mounted FLAT on something rigid",
         "A curled or bowed sheet is a shape error. No amount of data fixes it,\n"
         "     and the solver will quietly absorb it into the joint zeros."),
        ("All three lenses are focused and will not be touched again",
         "These are manual-focus modules. Intrinsics are only valid for the focus\n"
         "     they were solved at, so refocusing later silently invalidates them."),
        ("A touch point is chosen on each gripper",
         "A sharp, repeatable feature such as a jaw tip corner. Its position is\n"
         "     solved for, so it need not be measured, but it must never move."),
        ("The grippers can be held at a fixed closed position",
         "Stage 5 assumes the touch point is rigid relative to the last link.\n"
         "     A gripper that drifts open breaks that assumption."),
    ]
    for text, why in items:
        print(f"\n  {text}?")
        print(f"     why: {why}")
        if not confirm("  confirmed"):
            print("\n  Stopping here. Sort that out first; skipping it will cost more")
            print("  time later than doing it now.")
            return False
    return True


def measure_board(name: str, purpose: str,
                  detect_convention: bool = True) -> charuco.BoardSpec:
    print(f"\n{BAR}\nBoard: {name}  ({purpose})\n{BAR}")
    print("  Squares means chessboard squares across, not corners.")
    squares_x = ask_int("squares across (x)", 7, 3, 40)
    squares_y = ask_int("squares down (y)", 5, 3, 40)

    print("\n  Now the measurement that matters. Put calipers across several")
    print("  squares and divide, rather than measuring one square: the error")
    print("  averages down. Measure the PRINT, not the source file.")
    print("  A 1% printer scaling error becomes a 1% error on every distance")
    print("  this calibration reports, and nothing downstream can detect it.")
    square_mm = ask_float("measured square edge (mm)", None, 1.0, 500.0)

    print(f"\n  The marker is the black ArUco pattern inside a white square, so it")
    print(f"  must be smaller than {square_mm:g} mm.")
    while True:
        marker_mm = ask_float("measured marker edge (mm)",
                              round(square_mm * 0.75, 2), 0.5, 500.0)
        if marker_mm < square_mm:
            break
        print(f"    {marker_mm:g} mm is not smaller than the {square_mm:g} mm "
              f"square. Did you measure the square twice?")

    needed = (squares_x * squares_y) // 2
    print(f"\n  This board uses {needed} markers "
          f"({squares_x}x{squares_y} squares, half of them carry a marker).")
    print("  The number in a dictionary name is how many patterns it holds, so it")
    print("  must be at least that. Within a family the patterns are identical for")
    print("  shared ids, so any dictionary large enough will detect your board.")
    print("  Pick the one printed on the board if you know it.\n")

    families: dict[str, list[tuple[int, str]]] = {}
    for dict_name in charuco.canonical_names():
        size = charuco.dictionary_size(dict_name)
        families.setdefault(dict_name.rsplit("_", 1)[0], []).append((size, dict_name))

    # Plain N x N families first; they are what a printed board almost always
    # uses, and burying them under APRILTAG variants invites a wrong choice.
    def family_order(family: str) -> tuple[int, str]:
        plain = len(family) == len("DICT_4X4") and "X" in family
        return (0 if plain else 1, family)

    for family in sorted(families, key=family_order):
        entries = sorted(families[family])
        usable = [n for size, n in entries if size >= needed]
        too_small = [n for size, n in entries if size < needed]
        suffixes = ", ".join(n.rsplit("_", 1)[1] for n in usable)
        line = f"    {family + ':':<22} " + (suffixes or "none large enough")
        if too_small:
            line += "   (too small: " + ", ".join(
                n.rsplit("_", 1)[1] for n in too_small) + ")"
        print(line)

    while True:
        dictionary = ask("\n  aruco dictionary", "DICT_4X4_1000")
        if dictionary not in charuco.DICTIONARIES:
            print(f"    unknown dictionary. Names look like DICT_4X4_1000.")
            continue
        available = charuco.dictionary_size(dictionary)
        if available < needed:
            print(f"    {dictionary} holds only {available} markers but this "
                  f"board needs {needed}. Choose a larger one.")
            continue
        break

    spec = charuco.BoardSpec(
        squares_x=squares_x, squares_y=squares_y,
        square_mm=square_mm, marker_mm=marker_mm,
        dictionary=dictionary, name=name, measured=True)
    if detect_convention:
        spec = resolve_convention(spec)
    print(f"\n{spec.describe()}")
    return spec


def diagnose_failure(spec: charuco.BoardSpec) -> None:
    """Explain why no corners were found, by checking the stages separately.

    "Nothing detected" has several causes that look identical but need opposite
    responses. Markers detecting while corners do not means the spec is wrong;
    no markers at all means the dictionary is wrong or the board is not visible.
    """
    try:
        import cv2
        import preview
    except Exception:
        print("  (cannot investigate further without a camera)")
        return

    print("  Checking where it breaks down.")
    try:
        with preview.live_session(None) as (_url, feeds):
            import time
            time.sleep(1.5)
            frames = {r: f.latest()[0] for r, f in feeds.items()}
    except Exception as exc:
        print(f"  (could not grab frames: {exc})")
        return

    detector = cv2.aruco.ArucoDetector(spec.cv_dictionary(),
                                       cv2.aruco.DetectorParameters())
    marker_counts = {}
    for role, frame in frames.items():
        if frame is None:
            continue
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        _, ids, _ = detector.detectMarkers(gray)
        marker_counts[role] = 0 if ids is None else len(ids)

    print(f"\n  ArUco markers found with {spec.dictionary}:")
    for role, n in marker_counts.items():
        print(f"    {preview.camera_label(role):<22}{n:>4}")

    total = sum(marker_counts.values())
    if total == 0:
        print("\n  No markers at all, so this is not a geometry problem:")
        print("    - wrong dictionary (try another family: 5X5, 6X6, APRILTAG)")
        print("    - board not actually in view, too small in frame, or too dark")
        print("    - board badly out of focus")
        return

    print(f"\n  Markers detect fine but corners do not. That means the marker")
    print(f"  layout does not match a {spec.squares_x}x{spec.squares_y} board")
    print(f"  with legacy={spec.legacy}. Almost always one of:")
    print("    - squares_x and squares_y the other way round")
    print("    - the legacy flag (boards made before OpenCV 4.6)")
    print("\n  Rerun this stage and let the detection step settle it, or run:")
    print("    python calibration/run.py --stage prep")


def resolve_convention(spec: charuco.BoardSpec) -> charuco.BoardSpec:
    """Settle the two things about a board nobody can read off the printed sheet.

    Which side is "x" and whether the sheet uses OpenCV's pre-4.6 marker layout
    are both invisible to the eye, and both make ChArUco corner detection return
    nothing while ArUco marker detection keeps working perfectly. Rather than ask
    the operator to guess, measure it: hold the board up once and the marker
    layout identifies the geometry unambiguously.
    """
    print(f"\n{BAR}")
    print("  Two things about a printed board cannot be read off it: which side")
    print("  counts as x, and whether it uses OpenCV's older marker layout")
    print("  (boards generated before OpenCV 4.6). Getting either wrong finds")
    print("  markers normally but no chessboard corners, which is a confusing")
    print("  way to fail. Easier to just measure it.")
    print(f"{BAR}")

    if not confirm("\n  Hold the board up to a camera and detect this now",
                   default=True):
        print("  Skipped. If detection finds nothing later, come back to this.")
        return spec

    try:
        import preview
    except Exception as exc:
        print(f"  cannot open a camera: {exc}")
        return spec

    candidates = [(spec.squares_x, spec.squares_y)]
    try:
        with preview.live_session(None) as (url, feeds):
            print(f"\n  Open this in a browser:  {url}")
            print("  Hold the board flat and square-on, filling most of the")
            print("  frame of any one camera. Press Enter when it looks good.")
            try:
                input()
            except (EOFError, KeyboardInterrupt):
                raise Aborted from None

            scored = []
            for role, feed in feeds.items():
                frame, _ = feed.latest()
                if frame is None:
                    continue
                results = charuco.detect_geometry(
                    frame, spec.square_mm, spec.marker_mm, spec.dictionary,
                    candidates, name=spec.name)
                if results:
                    scored.append((role, results))
    except Aborted:
        raise
    except Exception as exc:
        print(f"  detection failed: {exc}")
        return spec

    if not scored:
        print("\n  No camera saw enough markers to decide. Usual cause is the")
        print("  wrong dictionary, since even a wrong size detects markers.")
        print("  Keeping what you entered.")
        return spec

    print(f"\n  {'camera':<22}{'best fit':<22}{'markers':>8}{'fit err':>10}")
    votes: dict[tuple[int, int, bool], int] = {}
    for role, results in scored:
        best = results[0]
        label = f"{best['squares_x']}x{best['squares_y']} legacy={best['legacy']}"
        print(f"  {preview.camera_label(role):<22}{label:<22}{best['matched']:>8}"
              f"{best['median_px']:>9.2f}px")
        if best["inlier_fraction"] > 0.9:
            key = (best["squares_x"], best["squares_y"], best["legacy"])
            votes[key] = votes.get(key, 0) + 1

    if not votes:
        print("\n  No candidate fitted cleanly. Keeping what you entered; check")
        print("  the dictionary and the square count on the printed sheet.")
        return spec

    (sx, sy, legacy), n_votes = max(votes.items(), key=lambda kv: kv[1])
    if len(votes) > 1:
        print(f"\n  Cameras disagreed, taking the majority ({n_votes} of "
              f"{len(scored)}).")

    changed = (sx, sy, legacy) != (spec.squares_x, spec.squares_y, spec.legacy)
    if not changed:
        print(f"\n  Confirmed: {sx}x{sy}, legacy={legacy}. What you entered "
              f"was right.")
        return spec

    print(f"\n  Detected {sx}x{sy} with legacy={legacy}, but you entered "
          f"{spec.squares_x}x{spec.squares_y} with legacy={spec.legacy}.")
    print("  The measurement is more reliable than the printed label here.")
    if not confirm("  Use the detected values", default=True):
        print("  Keeping what you entered.")
        return spec

    return charuco.BoardSpec(
        squares_x=sx, squares_y=sy, square_mm=spec.square_mm,
        marker_mm=spec.marker_mm, dictionary=spec.dictionary, legacy=legacy,
        name=spec.name, measured=spec.measured)


def verify_detectable(spec: charuco.BoardSpec) -> bool:
    """Confirm the board as specified is detectable, with a live view to watch.

    Opens a browser preview of all cameras with detected corners drawn on, so a
    wrong dictionary or swapped dimensions is obvious rather than showing up as a
    silent failure two stages later.
    """
    print(f"\n{BAR}\nLive check: can we see '{spec.name}'?\n{BAR}")
    print("  This opens a live view of all three cameras with detected corners")
    print("  marked in green, so you can confirm the spec you just entered is")
    print("  right before anything depends on it.")
    if not confirm("\n  Start the live check", default=True):
        print("  Skipped. If the spec is wrong, stage 1 will detect nothing.")
        return True

    try:
        import preview
    except Exception as exc:
        print(f"  cannot start the preview: {exc}")
        return confirm("  Continue anyway", default=True)

    try:
        with preview.live_session(spec) as (url, feeds):
            print(f"\n  Open this in a browser:  {url}\n")
            print("  Hold the board in front of each camera in turn. You want")
            print(f"  green dots on the corners, up to {spec.n_corners} of them.")
            print("  Fill most of the frame and keep the board flat and well lit.")
            print("\n  Watching. Press Enter when you have seen every camera "
                  "detect the board.")

            best = {role: 0 for role in feeds}
            stop = threading.Event()

            def watch():
                while not stop.is_set():
                    for role, feed in feeds.items():
                        best[role] = max(best[role], feed.n_corners)
                    time.sleep(0.1)

            watcher = threading.Thread(target=watch, daemon=True)
            watcher.start()
            try:
                input()
            except (EOFError, KeyboardInterrupt):
                pass
            stop.set()
            watcher.join(timeout=1.0)
    except Exception as exc:
        print(f"  preview failed: {exc}")
        return confirm("  Continue anyway", default=True)

    print(f"\n  Best detection per camera (of {spec.n_corners} corners):")
    for role, n in best.items():
        if n == 0:
            verdict = "nothing detected"
        elif n < spec.n_corners * 0.3:
            verdict = "detected, but only partially"
        else:
            verdict = "good"
        print(f"    {preview.camera_label(role):<22} {n:>4}   {verdict}")

    missed = [preview.camera_label(r) for r, n in best.items() if n == 0]
    if len(missed) == len(best):
        print("\n  No camera found any chessboard corners.")
        diagnose_failure(spec)
        return confirm("\n  Continue anyway", default=False)

    if missed:
        print(f"\n  These never saw the board: {', '.join(missed)}")
        print("  The spec is right, since other cameras detected it. Most likely")
        print("  you did not hold the board in front of these.")
        return confirm("  Continue anyway", default=True)

    print("\n  Every camera detected the board. Spec confirmed.")
    return True


def run() -> int:
    print(f"{BAR}\nStage -1: physical preparation\n{BAR}")
    print("  Records the board geometry and confirms the physical setup.")
    print("  Everything here is a prerequisite for trustworthy numbers later.")

    existing = charuco.load_boards()
    if existing:
        print(f"\n  Boards already recorded: {', '.join(existing)}")
        for spec in existing.values():
            print(spec.describe())
        if not confirm("\n  Replace them"):
            print("  Keeping the existing definition.")
            # Rewritten from board.json rather than left alone, so editing that
            # file or running identify_board.py cannot leave the stage result
            # describing a different board than the one every stage detects.
            storage.save_result("board", {
                "boards": {n: vars(s) for n, s in existing.items()},
                "world_board": "world" if "world" in existing else "main",
                "note": "kept existing definition",
            })
            return 0

    if not checklist():
        return 1

    boards: dict[str, charuco.BoardSpec] = {}
    main_board = measure_board("main", "intrinsics, and the world frame W")
    boards["main"] = main_board
    if not verify_detectable(main_board):
        print("\n  Main board was not verified. Nothing was saved.")
        return 1

    print(f"\n{BAR}")
    print("  A second, larger board can serve as the world frame if the main one")
    print("  is too small to stay visible across the head's pan sweep. Optional.")
    if confirm("  Add a separate world board"):
        world = measure_board("world", "the world frame W only")
        boards["world"] = world
        if not verify_detectable(world):
            print("\n  World board was not verified. Nothing was saved.")
            return 1

    charuco.save_boards(boards)
    storage.save_result("board", {
        "boards": {n: vars(s) for n, s in boards.items()},
        "world_board": "world" if "world" in boards else "main",
    })

    print(f"\n{BAR}\nStage -1 complete\n{BAR}")
    print(f"  Saved {len(boards)} board definition(s) to calibration/board.json")
    print("\n  From now on, do not:")
    print("    - refocus any lens")
    print("    - reprint or reflatten the board")
    print("    - move the touch point on either gripper")
    print("\n  Next: python calibration/run.py --stage 1")
    return 0


def main() -> int:
    try:
        return run()
    except Aborted:
        print("\n\n  Stopped. Nothing was saved; rerun when you are ready.")
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
