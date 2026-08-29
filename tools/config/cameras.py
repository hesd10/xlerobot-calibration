"""Camera identification for this XLeRobot unit.

The three robot cameras are indistinguishable at the USB level: same VID:PID
(05a3:9230), same serial string ("USB2.0_CAM1"). Neither /dev/videoN numbering nor
the USB port path is a dependable identity:

  - videoN is assigned by kernel enumeration order, which varies per boot even
    with nothing replugged (observed three different assignments in one day).
  - The USB port path is stable across reboots but breaks as soon as a camera is
    moved to a different socket.

So identity is established by a human looking at the picture, recorded once by
tools/cameras/identify.py, and stored in a mapping file. At runtime, resolve()
matches the saved entries against what is currently on the bus.
"""

import json
from pathlib import Path

MAPPING_FILE = Path(__file__).resolve().parent / "camera_mapping.json"

# The model's role names. Everything downstream -- intrinsics, arm fusion, verify,
# the baked XML -- names its cameras by these, so they must not change.
ROLES = ("left_wrist", "right_wrist", "head")

# What the identification page records instead. The operator can only pick a
# camera by the side they SEE it on. That physical side is the quantity that
# moves: back-to-front, the body turns 180 degrees while the working side stays
# put, so each flange turns 180 degrees to keep facing forward and the arm that
# was on the left is now on the right. The nominal role is the invariant, fixed
# by the camera's name and port. So the file is keyed physically, and resolve()
# is the one place that folds a physical side onto a model role for the declared
# mounting. Storing the model role directly would record an observation of the
# moving side as though it were the fixed one, so the flip would never be
# applied and every wrist assignment on a back-to-front robot would be mirrored
# -- the bug this avoids.
PHYSICAL_ROLES = ("left_wrist_physical", "right_wrist_physical", "head")
_PHYSICAL_SUFFIX = "_wrist_physical"


def _mounting_search_path() -> list[Path]:
    """Where workflow.json may sit, nearest declaration first.

    A stage does not run from the workspace root: the runner copies the tree
    into runtime/<stage>-<hash>/ and runs from there, placing workflow.json
    beside the results dir rather than in the cwd. Searching only the cwd found
    nothing and silently fell back to 'normal', which reads every saved wrist
    assignment as the opposite camera -- the failure this file exists to
    prevent. The calibration package resolves it from the results dir first and
    the cwd second, so this must look in the same places.
    """
    bases: list[Path] = []
    try:
        from core import storage  # available when running inside a stage
        bases.append(Path(storage.RESULTS_DIR).resolve().parent)
    except Exception:
        pass
    # Walking up from the cwd covers a standalone tool run started from anywhere
    # inside the workspace or a runtime tree. The cwd itself stays first so an
    # explicit workspace still wins.
    cwd = Path.cwd().resolve()
    bases.extend([cwd, *cwd.parents])
    return bases


def _declared_mounting() -> str:
    """Read the workspace's mounting declaration, defaulting to 'normal'.

    Kept deliberately dependency-free where it can be: this module sits under
    tools/ and must keep working when the calibration package is absent.
    """
    for base in _mounting_search_path():
        state = base / "workflow.json"
        if state.is_file():
            try:
                declared = json.loads(state.read_text()).get("mounting")
            except (OSError, json.JSONDecodeError):
                declared = None
            if declared in ("normal", "flipped"):
                return declared
    return "normal"


def _model_role(physical_role: str, mounting: str) -> str:
    """Fold a physical wrist role onto the model role for this mounting."""
    if not physical_role.endswith(_PHYSICAL_SUFFIX):
        return physical_role
    side = physical_role[: -len(_PHYSICAL_SUFFIX)]
    if mounting == "flipped":
        side = "right" if side == "left" else "left"
    return f"{side}_wrist"

# Robot cameras report this; the laptop's built-in webcam reports "FHD Webcam"
# or "IR Camera", which is enough to keep it out of the candidate list.
ROBOT_CAMERA_NAME = "USB2.0_CAM1"

WIDTH, HEIGHT, FPS = 640, 480, 30

SYSFS = Path("/sys/class/video4linux")


def _read(path: Path) -> str | None:
    try:
        return path.read_text().strip()
    except OSError:
        return None


def v4l_name(device: str) -> str | None:
    return _read(SYSFS / Path(device).name / "name")


def v4l_index(device: str) -> int | None:
    raw = _read(SYSFS / Path(device).name / "index")
    return int(raw) if raw is not None else None


def usb_attrs(device: str) -> dict[str, str | None]:
    """USB port path plus vendor/product/serial for a video node."""
    node = SYSFS / Path(device).name
    if not node.exists():
        return {}
    resolved = node.resolve()

    # .../usb3/3-1/3-1.1/3-1.1.1/3-1.1.1:1.0/video4linux/video2
    interface = None
    for part in reversed(resolved.parts):
        if ":" in part and "-" in part:
            interface = part
            break
    if interface is None:
        return {}

    port = interface.split(":")[0]
    # The USB device dir is the parent of the interface dir.
    dev_dir = None
    for parent in resolved.parents:
        if parent.name == interface:
            dev_dir = parent.parent
            break

    return {
        "port": port,
        "vid": _read(dev_dir / "idVendor") if dev_dir else None,
        "pid": _read(dev_dir / "idProduct") if dev_dir else None,
        "serial": _read(dev_dir / "serial") if dev_dir else None,
    }


def list_capture_devices(robot_only: bool = False) -> list[dict]:
    """Every /dev/videoN that can deliver frames, newest kernel info each call.

    Metadata nodes (index != 0) are skipped: they exist but never yield images.
    """
    out = []
    for node in sorted(SYSFS.glob("video*"), key=lambda p: int(p.name[5:])):
        device = f"/dev/{node.name}"
        if v4l_index(device) != 0:
            continue
        name = v4l_name(device) or ""
        is_robot = ROBOT_CAMERA_NAME in name
        if robot_only and not is_robot:
            continue
        out.append({
            "device": device,
            "name": name,
            "is_robot_camera": is_robot,
            **usb_attrs(device),
        })
    return out


def load_mapping() -> dict:
    """Read the saved role assignment, or {} when it has never been recorded."""
    if not MAPPING_FILE.is_file():
        return {}
    try:
        return json.loads(MAPPING_FILE.read_text())
    except (OSError, json.JSONDecodeError):
        return {}


def save_mapping(assignments: dict[str, dict], note: str = "") -> None:
    """Persist role -> camera identity, keyed by USB port with videoN as a hint."""
    from datetime import datetime

    payload = {
        "confirmed_at": datetime.now().isoformat(timespec="seconds"),
        "note": note,
        "roles": assignments,
    }
    MAPPING_FILE.write_text(json.dumps(payload, indent=2) + "\n")


class CameraMappingError(RuntimeError):
    pass


def _saved_by_model_role(saved: dict[str, dict]) -> dict[str, dict]:
    """Re-key a saved 'roles' block onto model roles for the declared mounting.

    New files are keyed physically (left_wrist_physical, ...) and fold onto the
    model role for the mounting. Old files were keyed by model role already;
    those are left as-is, which is exactly the normal-mounting reading they were
    recorded under.
    """
    physical = {r for r in saved if r.endswith(_PHYSICAL_SUFFIX)}
    if not physical:
        return dict(saved)
    mounting = _declared_mounting()
    out = {}
    for role, entry in saved.items():
        out[_model_role(role, mounting)] = entry
    return out


def resolve(strict: bool = True) -> dict[str, str]:
    """Return role -> current /dev/videoN, re-matched against the live bus.

    Matching prefers the USB port recorded at identification time. If a camera has
    moved to a different socket the port no longer matches, and there is no way to
    tell the three robot cameras apart, so identification must be redone.
    """
    mapping = load_mapping()
    if not mapping:
        if strict:
            raise CameraMappingError(
                "No camera mapping recorded.\n"
                "Run: python tools/cameras/identify.py"
            )
        return {}

    saved = _saved_by_model_role(mapping.get("roles", {}))
    live = {c["port"]: c for c in list_capture_devices(robot_only=True)}

    resolved: dict[str, str] = {}
    missing: list[str] = []
    for role in ROLES:
        entry = saved.get(role)
        if not entry:
            missing.append(f"{role}: not in mapping file")
            continue
        port = entry.get("port")
        if port in live:
            resolved[role] = live[port]["device"]
        else:
            missing.append(
                f"{role}: no robot camera on USB port {port} "
                f"(was {entry.get('device')} when identified)"
            )

    if missing and strict:
        detail = "\n  ".join(missing)
        raise CameraMappingError(
            f"Camera mapping no longer matches the hardware:\n  {detail}\n\n"
            f"Live robot cameras: {sorted(live)}\n"
            "Re-run: python tools/cameras/identify.py"
        )
    return resolved


def verify_cameras(strict: bool = True) -> list[str]:
    """Check the saved mapping still resolves to real robot cameras."""
    problems: list[str] = []
    try:
        resolved = resolve(strict=False)
    except CameraMappingError as exc:
        problems.append(str(exc))
        resolved = {}

    if not resolved:
        problems.append("no usable camera mapping (run tools/cameras/identify.py)")
    else:
        for role in ROLES:
            if role not in resolved:
                problems.append(f"{role}: could not be resolved to a device")
                continue
            device = resolved[role]
            name = v4l_name(device) or ""
            if ROBOT_CAMERA_NAME not in name:
                problems.append(f"{role}: {device} is '{name}', not a robot camera")

    if problems and strict:
        raise CameraMappingError("Camera verification failed:\n  " + "\n  ".join(problems))
    return problems


def describe() -> str:
    mapping = load_mapping()
    if not mapping:
        return "  no mapping recorded -- run: python tools/cameras/identify.py"

    lines = [f"  confirmed at: {mapping.get('confirmed_at', '?')}"]
    if mapping.get("note"):
        lines.append(f"  note: {mapping['note']}")
    lines.append("")

    resolved = resolve(strict=False)
    saved = _saved_by_model_role(mapping.get("roles", {}))
    lines.append(f"  {'ROLE':<12} {'DEVICE NOW':<14} {'PORT':<10} {'AT ID TIME':<12} STATUS")
    for role in ROLES:
        entry = saved.get(role, {})
        now = resolved.get(role, "-")
        was = entry.get("device", "-")
        port = entry.get("port", "-")
        status = "OK" if role in resolved else "NOT FOUND"
        if role in resolved and now != was:
            status = "OK (renumbered)"
        lines.append(f"  {role:<12} {now:<14} {port:<10} {was:<12} {status}")
    return "\n".join(lines)


if __name__ == "__main__":
    print("XLeRobot camera mapping:\n")
    print(describe())
    print("\nAll capture devices on this machine:\n")
    print(f"  {'DEVICE':<14} {'NAME':<28} {'PORT':<10} ROBOT?")
    for c in list_capture_devices():
        print(f"  {c['device']:<14} {c['name'][:27]:<28} {c.get('port', '-'):<10} "
              f"{'yes' if c['is_robot_camera'] else 'no'}")

    problems = verify_cameras(strict=False)
    print("\nOK: mapping resolves cleanly" if not problems
          else "\nPROBLEMS:\n  " + "\n  ".join(problems))
