"""The calibrated robot's geometry, gathered into one overview.

At the end of the run the operator wants a single answer to "where did the
calibration decide everything is?" -- the three cameras and the two arm roots,
with positions and pointing directions in one place.

Everything here is read out of the committed stage results. The camera pose
maths mirrors calibration/visualize_three_cameras.py and the arm-root maths
mirrors get_arm_root_position() in calibration/stages/stage5b_head_zero.py,
which were the CLI-era sources of the same numbers, so the overview cannot
drift from the geometry the rest of the pipeline uses.

All poses are expressed in the calibrated body frame, at the exact-zero
posture (head pan and tilt at zero, arms at their calibrated zeros). That is
the frame every downstream consumer of the calibration works in.
"""
from __future__ import annotations

import math
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

from . import mounting as mounting_mod
from .i18n import text as _text
from .workspace import Workspace

# Which files hold the final, normalized geometry. Stage 7 rewrites the body
# frame, so the *_zero variants are the ones to draw; the pre-normalize files
# are a different frame and would place the arms wrongly.
HEAD_RESULT = "head_zero"
TOUCH_RESULT = "touch_zero"

# The chassis-frame bodies each arm hangs from, matching model_map.ROOT_BODIES.
ROOT_BODIES = {"left_arm": "Rotation_Pitch", "right_arm": "Rotation_Pitch_2"}

MODEL_XML = (Path(__file__).resolve().parent.parent
             / "calibration" / "model" / "xlerobot_calib.xml")

def arm_label(arm: str, mounting: str = "normal") -> str:
    """Name an arm by the side the operator can point at.

    The key stays the model's name everywhere it is used as a key -- colours,
    lookups, the JSON the 3D page consumes -- and only the label turns, so a
    flipped mounting changes what is read and nothing that is computed.
    """
    side = mounting_mod.physical_side(arm, mounting)
    return _text(f"ov.arm.{side}_arm")


def camera_label(camera: str, mounting: str = "normal") -> str:
    """Name a camera by the side the operator can point at.

    The wrist cameras are bolted to the arms, so they turn with them: under a
    flipped mounting the role the model calls `left_wrist` is the camera on the
    operator's right. The head camera has no side and passes through.
    """
    if camera == "head":
        return _text("ov.camera.head")
    side = mounting_mod.physical_camera_side(camera, mounting)
    return _text(f"ov.camera.{side}_wrist")


# Kept for callers and tests that index them directly. Normal mounting, which
# is the identity mapping; anything user-facing passes the real mounting.
ARM_LABELS = {arm: arm_label(arm) for arm in ("left_arm", "right_arm")}
CAMERA_LABELS = {camera: camera_label(camera)
                 for camera in ("head", "left_wrist", "right_wrist")}


def _matmul(a: list[list[float]], b: list[list[float]]) -> list[list[float]]:
    return [[sum(a[i][k] * b[k][j] for k in range(4)) for j in range(4)]
            for i in range(4)]


def _rotation_from_axis_angle(axis: tuple[float, float, float],
                              angle: float) -> list[list[float]]:
    norm = math.sqrt(sum(component * component for component in axis))
    if norm == 0.0:
        raise ValueError("axis must be non-zero")
    x, y, z = (component / norm for component in axis)
    c, s = math.cos(angle), math.sin(angle)
    t = 1.0 - c
    return [
        [t * x * x + c, t * x * y - s * z, t * x * z + s * y],
        [t * x * y + s * z, t * y * y + c, t * y * z - s * x],
        [t * x * z - s * y, t * y * z + s * x, t * z * z + c],
    ]


def _joint_transform(axis: tuple[float, float, float], angle: float,
                     origin: list[float] | None) -> list[list[float]]:
    """A rotation about an axis through `origin`, as a 4x4 transform.

    This is the same construction head_model.pan_transform / tilt_transform
    use: rotate about the axis, but keep the axis line fixed in space.
    """
    rotation = _rotation_from_axis_angle(axis, angle)
    point = list(origin or (0.0, 0.0, 0.0))
    transform = [[0.0] * 4 for _ in range(4)]
    for row in range(3):
        for column in range(3):
            transform[row][column] = rotation[row][column]
        transform[row][3] = point[row] - sum(
            rotation[row][k] * point[k] for k in range(3))
    transform[3][3] = 1.0
    return transform


def head_pan_for_drawing(mounting: str = mounting_mod.NORMAL) -> float:
    """The pan angle to draw the head at, so it faces the operator's work area.

    Drawing is the one place that wants a posture rather than the zero. The
    stored head zero is the posture facing the calibration board, and which
    model angle that is depends only on the mounting: chassis-front (q = 0)
    normally, chassis-back (q = pi) back-to-front. Drawing both at q = 0 turned
    the back-to-front robot away from its own work area, so the picture showed
    the head looking one way and the arms reaching the other -- correct numbers
    arranged into a robot that does not exist.

    This is a property of the picture and stops here. It is not written to any
    result, never reaches the baked XML, and no stage reads it: the pipeline
    applies the same correspondence itself, at the single point where an encoder
    count becomes a model angle. Two independent applications of one fixed fact,
    not a value passed between them.
    """
    return math.pi if mounting == mounting_mod.FLIPPED else 0.0


def head_camera_pose(head: dict[str, Any],
                     pan: float = 0.0) -> list[list[float]] | None:
    """The head camera's pose in the body frame at the given pan, tilt = 0.

    Defaults to the exact-zero posture. At zero the pan and tilt rotations are
    identity regardless of the axis origin, so it reduces to T_tilt_cam; away
    from zero the pan rotation about the calibrated axis origin is what turns
    the camera, which is why the origin is honoured rather than assumed at the
    body origin.
    """
    tilt_cam = head.get("T_tilt_cam")
    if not tilt_cam:
        return None
    senses = head.get("senses") or (1.0, 1.0)
    origin = head.get("gauge", {}).get("pan_axis_origin_m")
    pan_sense = float(senses[0])
    tilt_sense = float(senses[1]) if len(senses) > 1 else 1.0
    pan_t = _joint_transform((0.0, 0.0, pan_sense), pan, origin)
    tilt = _joint_transform((0.0, tilt_sense, 0.0), 0.0, origin)
    return _matmul(_matmul(pan_t, tilt), [list(row) for row in tilt_cam])


def wrist_camera_position(arm: str, mount: list[list[float]],
                          wrist_cam: list[list[float]],
                          xml_path: Path | str | None = None) -> list[float] | None:
    """Where a wrist camera sits in the body frame, at the calibrated zero pose.

    T_wrist_cam is expressed in the gripper frame, so it only becomes a place in
    the body frame once the arm is posed. At the calibrated zero every joint is
    at q = 0 by construction, which is the posture the rest of this page draws,
    so the gripper pose there is a fixed property of the model and the arm's own
    mount correction.

    Returns None when the model cannot be loaded. The camera then keeps its
    gripper-frame offset in the table and is simply not placed in the scene,
    which is what this page did for every wrist camera before.
    """
    try:
        import numpy as np

        import sys
        calibration = str(Path(__file__).resolve().parent.parent / "calibration")
        if calibration not in sys.path:
            sys.path.insert(0, calibration)
        import model_map
    except Exception:
        return None
    try:
        sim = model_map.SimModel(xml_path or MODEL_XML)
        # Every arm joint at its calibrated zero: the posture this page draws.
        zeros = {motor: 0.0 for motor in model_map.REAL_TO_SIM
                 if motor.startswith(arm)}
        sim.set_joints(zeros)
        p, R = sim.body_pose_in_chassis(model_map.WRIST_BODIES[arm])
    except Exception:
        return None
    T_A_G = np.eye(4)
    T_A_G[:3, :3] = R
    T_A_G[:3, 3] = p
    # T_B_A corrects the arm's mount; it is applied about the arm root, exactly
    # as arm_root_position() does for the root itself.
    T = np.asarray(mount, float) @ T_A_G @ np.asarray(wrist_cam, float)
    return [float(v) * 1000.0 for v in T[:3, 3]]


def nominal_root_positions(xml_path: Path | str | None = None) -> dict[str, list[float]]:
    """Where the XML puts each arm root, in chassis coordinates, in metres.

    Read from the model rather than hardcoded, so the overview follows the XML
    if the mechanical layout is ever revised. Positions accumulate down the
    body chain, matching how MuJoCo composes nested body offsets. Every body
    from the chassis down to the arm roots has an identity-or-absent rotation
    on the translation path, so summing positions is exact here; a rotated
    intermediate body would need the full transform chain.
    """
    path = Path(xml_path or MODEL_XML)
    root = ET.parse(path).getroot()
    chassis = root.find("worldbody/body[@name='chassis']")
    if chassis is None:
        raise ValueError(f"{path}: no chassis body")

    found: dict[str, list[float]] = {}
    wanted = {name: arm for arm, name in ROOT_BODIES.items()}

    def walk(body: ET.Element, offset: tuple[float, float, float]) -> None:
        for child in body.findall("body"):
            text = (child.get("pos") or "0 0 0").split()
            here = tuple(offset[i] + float(text[i]) for i in range(3))
            name = child.get("name")
            if name in wanted:
                found[wanted[name]] = list(here)
            walk(child, here)

    # Positions are wanted relative to the chassis (matching MuJoCo's
    # body_pose_in_chassis), so accumulation starts at the chassis origin and
    # deliberately ignores the chassis's own offset from the world.
    walk(chassis, (0.0, 0.0, 0.0))
    missing = set(ROOT_BODIES) - set(found)
    if missing:
        raise ValueError(f"{path}: arm roots not found: {sorted(missing)}")
    return found


def arm_root_position(mount: list[list[float]],
                      nominal: list[float]) -> list[float]:
    """The arm root in the body frame, in millimetres.

    T_B_A is a correction applied on top of the XML's nominal root, not the
    root position itself, so it has to be composed with the nominal position.
    This mirrors get_arm_root_position() in stage5b_head_zero.py.
    """
    point = list(nominal) + [1.0]
    return [sum(mount[row][k] * point[k] for k in range(4)) * 1000.0
            for row in range(3)]


def _position_mm(transform: list[list[float]]) -> list[float]:
    return [transform[row][3] * 1000.0 for row in range(3)]


def _axis(transform: list[list[float]], column: int) -> list[float]:
    return [transform[row][column] for row in range(3)]


def _azimuth_elevation(direction: list[float]) -> tuple[float, float]:
    """Azimuth about +Z from +X, and elevation above the XY plane, in degrees."""
    x, y, z = direction
    horizontal = math.hypot(x, y)
    return (math.degrees(math.atan2(y, x)),
            math.degrees(math.atan2(z, horizontal)))


def collect(workspace: Workspace) -> dict[str, Any]:
    """Gather the calibrated camera and arm-root geometry.

    Returns a dict with `cameras`, `arms` and `frame`, plus `missing` naming
    any result file that was not available. A partial overview is better than
    none: a missing wrist result should not hide the head camera.
    """
    head = workspace.load_result(HEAD_RESULT) or {}
    touch = workspace.load_result(TOUCH_RESULT) or {}
    mounting = workspace.mounting
    missing: list[str] = []
    if not head:
        missing.append(f"{HEAD_RESULT}.json")
    if not touch:
        missing.append(f"{TOUCH_RESULT}.json")

    cameras: list[dict[str, Any]] = []
    pose = head_camera_pose(head, head_pan_for_drawing(mounting))
    if pose is not None:
        # The optical axis is the camera's +Z, the OpenCV convention used
        # throughout the calibration.
        direction = _axis(pose, 2)
        azimuth, elevation = _azimuth_elevation(direction)
        cameras.append({
            "key": "head",
            "label": camera_label("head", mounting),
            "position_mm": _position_mm(pose),
            "position_frame": "body",
            "optical_axis": direction,
            "axis_frame": "body",
            "azimuth_deg": azimuth,
            "elevation_deg": elevation,
        })

    arms: list[dict[str, Any]] = []
    try:
        nominal = nominal_root_positions()
    except (OSError, ET.ParseError, ValueError):
        # Without the model the mount transform is still reportable, just not
        # the absolute root position. Say so rather than inventing a number.
        nominal = {}
        missing.append("model/xlerobot_calib.xml")
    # Physically left arm first, so the tables and the 3D legend read in the
    # order the operator would walk the robot. The keys stay model-named.
    for arm in mounting_mod.arm_order(mounting):
        result = (touch.get("arms") or {}).get(arm) or {}
        mount = result.get("T_B_A")
        if mount:
            mount = [list(row) for row in mount]
            forward = _axis(mount, 0)
            azimuth, elevation = _azimuth_elevation(forward)
            entry = {
                "key": arm,
                "label": arm_label(arm, mounting),
                "x_axis": forward,
                "y_axis": _axis(mount, 1),
                "z_axis": _axis(mount, 2),
                "azimuth_deg": azimuth,
                "elevation_deg": elevation,
            }
            if arm in nominal:
                entry["position_mm"] = arm_root_position(mount, nominal[arm])
            arms.append(entry)
        # The wrist camera is calibrated in the gripper frame, so where it sits
        # depends on the arm's posture. At the calibrated zero -- the posture
        # this whole page describes -- that posture is known, so the camera can
        # be placed in the body frame properly rather than drawn at the arm
        # root as a stand-in. The gripper-frame offset is still reported, since
        # that is the calibrated quantity; the body position is derived.
        camera_key = "left_wrist" if arm == "left_arm" else "right_wrist"
        wrist = result.get("T_wrist_cam")
        if wrist:
            wrist = [list(row) for row in wrist]
            at_zero = result.get("optical_axis_at_zero")
            if at_zero:
                direction = [float(value) for value in at_zero]
                azimuth = result.get("optical_axis_azimuth_deg")
                elevation = result.get("optical_axis_elevation_deg")
                if azimuth is None or elevation is None:
                    azimuth, elevation = _azimuth_elevation(direction)
                axis_frame = "body"
            else:
                # No body-frame axis recorded: fall back to the gripper-frame
                # axis, and derive the angles from that same vector so the row
                # stays self-consistent.
                direction = _axis(wrist, 2)
                azimuth, elevation = _azimuth_elevation(direction)
                axis_frame = "gripper"
            entry = {
                "key": camera_key,
                "label": camera_label(camera_key, mounting),
                "position_mm": _position_mm(wrist),
                "position_frame": "gripper",
                "optical_axis": direction,
                "axis_frame": axis_frame,
                "azimuth_deg": azimuth,
                "elevation_deg": elevation,
            }
            body_at = (wrist_camera_position(arm, mount, wrist)
                       if mount else None)
            if body_at is not None:
                entry["body_position_mm"] = body_at
            cameras.append(entry)

    return {
        "cameras": cameras,
        "arms": arms,
        "frame": head.get("body_frame_id") or touch.get("body_frame_id"),
        # Which way the robot was standing. Every label above was turned by it,
        # and the drawn head posture follows it, so the picture has to say which
        # one it is rather than leaving the reader to infer it from the arms.
        "mounting": mounting,
        "missing": missing,
    }


def _round(values: list[float], digits: int = 2) -> str:
    return "  ".join(f"{value:+.{digits}f}" for value in values)


# Diagram geometry. The robot is roughly 800 mm tall and 300 mm wide across the
# arm roots, so a single millimetre-to-pixel scale keeps both views honest:
# equal distances look equal in both panels, which is the point of a scale
# drawing. Anything that would fall outside is clamped by the viewBox rather
# than silently rescaled.
_VIEW_W, _VIEW_H = 720, 380
_SCALE = 0.26
_ARROW = 150.0

_TOP_ORIGIN = (200.0, 190.0)
_SIDE_ORIGIN = (520.0, 300.0)

_COLOURS = {
    "head": "#e0a500",
    "left_wrist": "#d94f8a",
    "right_wrist": "#1aa3a3",
    "left_arm": "#1769aa",
    "right_arm": "#087443",
}


def _svg_escape(text: str) -> str:
    return (str(text).replace("&", "&amp;").replace("<", "&lt;")
            .replace(">", "&gt;").replace('"', "&quot;"))


def _top_xy(position: list[float]) -> tuple[float, float]:
    """Top view: robot +X (forward) draws up, robot +Y (left) draws left."""
    x, y = position[0], position[1]
    return (_TOP_ORIGIN[0] - y * _SCALE, _TOP_ORIGIN[1] - x * _SCALE)


def _side_xy(position: list[float]) -> tuple[float, float]:
    """Side view: robot +X (forward) draws right, robot +Z (up) draws up."""
    x, z = position[0], position[2]
    return (_SIDE_ORIGIN[0] + x * _SCALE, _SIDE_ORIGIN[1] - z * _SCALE)


def _marker(x: float, y: float, colour: str, label: str) -> str:
    return (f'<circle cx="{x:.1f}" cy="{y:.1f}" r="5" fill="{colour}"/>'
            f'<text x="{x + 8:.1f}" y="{y - 7:.1f}" font-size="11" '
            f'fill="{colour}">{_svg_escape(label)}</text>')


def _arrow(x: float, y: float, dx: float, dy: float, colour: str) -> str:
    length = math.hypot(dx, dy)
    if length < 1e-9:
        # A direction with no extent in this projection (an axis pointing
        # straight at the viewer) has no honest arrow to draw.
        return ""
    scale = _ARROW * _SCALE / length
    return (f'<line x1="{x:.1f}" y1="{y:.1f}" '
            f'x2="{x + dx * scale:.1f}" y2="{y + dy * scale:.1f}" '
            f'stroke="{colour}" stroke-width="2.2" '
            f'marker-end="url(#arrowhead)"/>')


def diagram(overview: dict[str, Any]) -> str:
    """A two-panel scale drawing of where the calibration put everything.

    Top view (looking down) and side view (looking from the robot's right).
    Both panels share one scale, so a distance in one is comparable to the
    other. Only body-frame items are drawn: a gripper-frame wrist position
    has no fixed place on a body-frame drawing, so those are shown by
    direction only, from the arm root they belong to.
    """
    parts: list[str] = []
    for origin, label, across, up in (
            (_TOP_ORIGIN, _text("ov.diagram.top"), "Y", "X"),
            (_SIDE_ORIGIN, _text("ov.diagram.side"), "X", "Z")):
        parts.append(
            f'<line x1="{origin[0] - 110:.0f}" y1="{origin[1]:.0f}" '
            f'x2="{origin[0] + 110:.0f}" y2="{origin[1]:.0f}" '
            f'stroke="#b9c2cc" stroke-dasharray="4 3"/>'
            f'<line x1="{origin[0]:.0f}" y1="{origin[1] - 150:.0f}" '
            f'x2="{origin[0]:.0f}" y2="{origin[1] + 110:.0f}" '
            f'stroke="#b9c2cc" stroke-dasharray="4 3"/>'
            f'<text x="{origin[0]:.0f}" y="{origin[1] + 128:.0f}" '
            f'font-size="12" fill="#5b6672" text-anchor="middle">'
            f'{_svg_escape(label)}</text>')

    arms_by_key = {arm["key"]: arm for arm in overview["arms"]}
    for arm in overview["arms"]:
        position = arm.get("position_mm")
        if not position:
            continue
        colour = _COLOURS[arm["key"]]
        for project in (_top_xy, _side_xy):
            x, y = project(position)
            parts.append(_marker(x, y, colour, arm["label"]))

    for camera in overview["cameras"]:
        colour = _COLOURS.get(camera["key"], "#5b6672")
        axis = camera["optical_axis"]
        if camera.get("position_frame") == "body":
            position = camera["position_mm"]
        else:
            # A gripper-frame camera: anchor its direction at the arm root so
            # the arrow still says which way the camera looks, without
            # claiming a body-frame position it does not have.
            arm_key = ("left_arm" if camera["key"].startswith("left")
                       else "right_arm")
            anchor = arms_by_key.get(arm_key, {}).get("position_mm")
            if not anchor:
                continue
            position = anchor
        if camera.get("axis_frame") != "body":
            # Only body-frame directions belong on a body-frame drawing.
            continue
        top = _top_xy(position)
        parts.append(_arrow(top[0], top[1], -axis[1], -axis[0], colour))
        parts.append(_marker(top[0], top[1], colour, camera["label"]))
        side = _side_xy(position)
        parts.append(_arrow(side[0], side[1], axis[0], -axis[2], colour))
        parts.append(_marker(side[0], side[1], colour, camera["label"]))

    body = "".join(part for part in parts if part)
    aria = _svg_escape(_text("ov.diagram.aria"))
    return (
        f'<svg viewBox="0 0 {_VIEW_W} {_VIEW_H}" width="100%" '
        f'role="img" aria-label="{aria}" '
        f'xmlns="http://www.w3.org/2000/svg">'
        f'<defs><marker id="arrowhead" markerWidth="8" markerHeight="6" '
        f'refX="7" refY="3" orient="auto">'
        f'<polygon points="0 0, 8 3, 0 6" fill="context-stroke"/>'
        f'</marker></defs>{body}</svg>')


def sections(workspace: Workspace) -> list[dict[str, Any]]:
    """The overview as summary sections, ready for the dashboard to render."""
    overview = collect(workspace)
    join = _text("ov.join")
    if overview["missing"] and not overview["cameras"] and not overview["arms"]:
        return [{
            "title": _text("ov.overview.title"),
            "columns": [_text("ov.notes.title")],
            "rows": [[_text("ov.missing", items=join.join(overview["missing"]))]],
        }]

    frames = {name: _text(f"ov.frame.{name}")
              for name in ("body", "gripper")}
    camera_rows = []
    for camera in overview["cameras"]:
        camera_rows.append([
            camera["label"],
            frames.get(camera.get("position_frame"), frames["body"]),
            _round(camera["position_mm"]),
            frames.get(camera.get("axis_frame"), frames["body"]),
            _round(camera["optical_axis"], 4),
            f"{camera['azimuth_deg']:+.2f}",
            f"{camera['elevation_deg']:+.2f}",
        ])

    arm_rows = []
    for arm in overview["arms"]:
        position = arm.get("position_mm")
        arm_rows.append([
            arm["label"],
            _round(position) if position else _text("ov.missingModel"),
            _round(arm["x_axis"], 4),
            f"{arm['azimuth_deg']:+.2f}",
            f"{arm['elevation_deg']:+.2f}",
        ])

    result = []
    if camera_rows or arm_rows:
        result.append({
            "title": _text("ov.diagram.title"),
            "columns": [],
            "rows": [],
            "svg": diagram(overview),
        })
    if camera_rows:
        result.append({
            "title": _text("ov.cameras.title"),
            "columns": [_text("sum.camera"),
                        _text("ov.cameras.posFrame"),
                        _text("ov.cameras.position"),
                        _text("ov.cameras.axisFrame"),
                        _text("ov.cameras.axis"),
                        _text("ov.azimuth"),
                        _text("ov.elevation")],
            "rows": camera_rows,
        })
    if arm_rows:
        result.append({
            "title": _text("ov.arms.title"),
            "columns": [_text("ov.armCol"),
                        _text("ov.cameras.position"),
                        _text("ov.arms.xAxis"),
                        _text("ov.azimuth"),
                        _text("ov.elevation")],
            "rows": arm_rows,
        })
    notes = [_text(key) for key in (
        "ov.note.frames", "ov.note.axis", "ov.note.wrist", "ov.note.roots")]
    if overview["frame"]:
        notes.append(_text("ov.note.frameId", value=overview["frame"]))
    if overview["missing"]:
        notes.append(_text("ov.missing", items=join.join(overview["missing"])))
    result.append({
        "title": _text("ov.notes.title"),
        "columns": [_text("ov.notes.title")],
        "rows": [[note] for note in notes],
    })
    return result
