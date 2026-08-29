"""Build concise, user-facing summaries from committed stage results."""
from __future__ import annotations

from typing import Any

from . import mounting as mounting_mod, robot_overview, robot_view3d
from .i18n import joint_label, text as _text
from .registry import BY_KEY
from .workspace import Workspace

# Mirrors calibration/core/gates.py; kept local so the summary layer does not
# depend on the calibration package being importable. test_result_summary.py
# checks the two agree, because these numbers are now shown to the operator and
# a silent drift would put a wrong limit on the screen.
TOUCH_RESIDUAL_GOOD_MM = 2.0
TOUCH_RESIDUAL_MAX_MM = 5.0


def _mm(value: float) -> str:
    """A limit as it should read in a sentence: 5 rather than 5.0."""
    return f"{value:g}"


def _arm_text(arm: str, mounting: str) -> str:
    """Name an arm by the side the operator sees it on.

    Takes the model's name, because that is what the result files store, and
    returns the operator's. Under a normal mounting the two agree and this is
    the identity.
    """
    side = mounting_mod.physical_side(arm, mounting)
    return _text("sum.leftArm" if side == "left" else "sum.rightArm")


def _by_arm(joints: dict[str, Any], mounting: str) -> list[tuple[str, Any]]:
    """Joint records ordered physically-left arm first, then the rest.

    The stage results are keyed by the model's joint names and their insertion
    order follows whatever order the stage happened to visit the arms in, which
    back-to-front is already the operator's order. Relying on that coincidence
    is what made these tables read right-before-left, so the order is stated
    here instead of inherited.
    """
    ordered: list[tuple[str, Any]] = []
    for arm in mounting_mod.arm_order(mounting):
        ordered.extend((joint, item) for joint, item in joints.items()
                       if joint.startswith(f"{arm}_"))
    ordered.extend((joint, item) for joint, item in joints.items()
                   if not any(joint.startswith(f"{arm}_")
                              for arm in ("left_arm", "right_arm")))
    return ordered


def _value(value: Any, digits: int = 3) -> Any:
    if isinstance(value, float):
        return round(value, digits)
    return value


def _section(title: str, columns: list[str], rows: list[list[Any]]) -> dict[str, Any]:
    return {"title": title, "columns": columns,
            "rows": [[_value(cell) for cell in row] for row in rows]}


def _load(workspace: Workspace, name: str) -> dict[str, Any]:
    return workspace.load_result(name) or {}


def _senses_summary(workspace: Workspace) -> list[dict[str, Any]]:
    result = _load(workspace, "senses")
    senses = result.get("senses", {})
    mounting = workspace.mounting
    rows = []
    for joint, item in _by_arm(senses, mounting):
        sign = item.get("sign")
        rows.append([
            joint_label(joint, mounting),
            f"{sign:+d}" if isinstance(sign, int) else _text("sum.unrecorded"),
            item.get("raw_before", "—"), item.get("raw_after", "—"),
            item.get("travel_counts", "—"),
        ])
    return [_section(_text("sum.senses.title"), [
        _text("sum.joint"), _text("sum.senses.sign"), _text("sum.senses.start"),
        _text("sum.senses.end"), _text("sum.senses.travel")], rows)]


def _intrinsics_summary(workspace: Workspace) -> list[dict[str, Any]]:
    mounting = workspace.mounting
    by_role = {}
    for name in BY_KEY["intrinsics"].outputs:
        item = _load(workspace, name)
        role = item.get("camera_role", name.removeprefix("intrinsics_"))
        by_role[role] = item
    rows = []
    # Head first, then the wrists physically left to right, named by the side
    # the operator sees rather than by the arm the model bolts them to.
    for role in mounting_mod.camera_order(mounting):
        item = by_role.get(role)
        if item is None:
            continue
        rows.append([
            robot_overview.camera_label(role, mounting),
            f"{item.get('width', '—')}x{item.get('height', '—')}",
            item.get("n_views_total", "—"),
            f"{item['coverage'] * 100:.1f}%" if isinstance(item.get("coverage"), (int, float)) else "—",
            f"{item['fit_rms_px']:.3f} px" if isinstance(item.get("fit_rms_px"), (int, float)) else "—",
            f"{item['holdout_rms_px']:.3f} px" if isinstance(item.get("holdout_rms_px"), (int, float)) else "—",
        ])
    return [_section(_text("sum.intr.title"), [
        _text("sum.camera"), _text("sum.intr.resolution"), _text("sum.intr.views"),
        _text("sum.intr.coverage"), _text("sum.intr.fitRms"),
        _text("sum.intr.holdoutRms")], rows)]


def _prepare_summary(workspace: Workspace) -> list[dict[str, Any]]:
    boards = _load(workspace, "board").get("boards", {})
    rows = []
    for name, board in boards.items():
        rows.append([name, f"{board.get('squares_x', '—')} x {board.get('squares_y', '—')}",
                     board.get("square_mm", "—"), board.get("marker_mm", "—"),
                     board.get("dictionary", "—")])
    return [_section(_text("sum.board.title"), [
        _text("sum.board.board"), _text("sum.board.size"), _text("sum.board.square"),
        _text("sum.board.marker"), _text("sum.board.dictionary")], rows)]


def _head_summary(workspace: Workspace) -> list[dict[str, Any]]:
    item = _load(workspace, "head")
    row = [item.get("n_views_total", "—"), item.get("pan_sweep_deg", "—"),
           item.get("tilt_sweep_deg", "—"), item.get("fit_rms_mm", "—"),
           item.get("holdout_rms_mm", "—"), item.get("holdout_rms_deg", "—")]
    return [_section(_text("sum.head.title"), [
        _text("sum.intr.views"), _text("sum.head.panSweep"), _text("sum.head.tiltSweep"),
        _text("sum.head.fitRms"), _text("sum.head.holdoutRms"),
        _text("sum.head.holdoutDeg")], [row])]


def _ranges_summary(workspace: Workspace) -> list[dict[str, Any]]:
    item = _load(workspace, "zeros")
    # zeros.json holds a serialised ZeroSet: {"zeros": {"joints": {...}}}, where
    # each joint maps to a record whose "raw" is the count. The travel scan sits
    # under {"ranges": {arm: {"travels": {joint: {...}}}}}.
    zeros = item.get("zeros", {}).get("joints", {})
    mounting = workspace.mounting
    ranges = item.get("ranges", {})
    rows = []
    # Physically-left arm first. The stored order already happened to be this,
    # because stage 4 walks the arms in working order, but that made the table
    # depend on a coincidence rather than a decision.
    for arm in mounting_mod.arm_order(mounting):
        arm_ranges = ranges.get(arm)
        if not isinstance(arm_ranges, dict):
            continue
        for joint, info in arm_ranges.get("travels", {}).items():
            if not isinstance(info, dict):
                continue
            record = zeros.get(joint)
            zero = record.get("raw", "—") if isinstance(record, dict) else "—"
            span_deg = info.get("span_deg")
            span = (f"{span_deg:.1f}°" if isinstance(span_deg, (int, float))
                    else info.get("span_counts", "—"))
            rows.append([joint_label(joint, mounting), zero, span])
    if not rows:
        rows = [[_text("sum.ranges.recorded"), len(zeros), "—"]]
    return [_section(_text("sum.ranges.title"), [
        _text("sum.joint"), _text("sum.ranges.zero"), _text("sum.ranges.span")], rows)]


def _arms_summary(workspace: Workspace) -> list[dict[str, Any]]:
    """Judge each arm against the stage's own residual gates.

    The stage writes no 'passed' flag, so a status must be derived; the previous
    default of True reported a pass even for a bad fit.
    """
    touch = _load(workspace, "touch")
    mounting = workspace.mounting
    arm_results = touch.get("arms", {})
    rows = []
    for arm in mounting_mod.arm_order(mounting):
        item = arm_results.get(arm)
        if not isinstance(item, dict):
            continue
        holdout = item.get("holdout_rms_mm")
        if not item.get("success", True):
            status = _text("sum.arms.failed")
        elif not isinstance(holdout, (int, float)):
            status = _text("sum.arms.noHoldout")
        elif holdout <= TOUCH_RESIDUAL_GOOD_MM:
            status = _text("sum.arms.good")
        elif holdout <= TOUCH_RESIDUAL_MAX_MM:
            status = _text("sum.arms.acceptable")
        else:
            status = _text("sum.arms.tooLarge").format(limit=_mm(TOUCH_RESIDUAL_MAX_MM))
        rows.append([_arm_text(arm, mounting), status, item.get("fit_rms_mm", "—"),
                     item.get("holdout_rms_mm", "—"), item.get("holdout_rms_deg", "—"),
                     f"{item.get('n_views_fit', '—')}/{item.get('n_views_holdout', '—')}"])
    sections = [_section(_text("sum.arms.title"), [
        _text("sum.arm"), _text("sum.status"), _text("sum.head.fitRms"),
        _text("sum.head.holdoutRms"), _text("sum.head.holdoutDeg"),
        _text("sum.arms.fitViews")], rows)]

    before = _load(workspace, "zeros").get("zeros", {}).get("joints", {})
    after = _load(workspace, "zeros_refined").get("zeros", {}).get("joints", {})
    zero_rows = []
    for joint, item in _by_arm(after, mounting):
        raw_after, raw_before = item.get("raw"), (before.get(joint) or {}).get("raw")
        if not isinstance(raw_after, int) or not isinstance(raw_before, int):
            continue
        delta = _unwrap_delta(raw_after - raw_before)
        solved = item.get("source") == "stage5_fusion"
        zero_rows.append([joint_label(joint, mounting), raw_before, raw_after,
                          f"{delta:+d}",
                          f"{_counts_to_deg(delta):+.2f}°" if solved
                          else _text("sum.arms.notSolved")])
    if zero_rows:
        sections.append(_section(_text("sum.arms.zeroTitle"), [
            _text("sum.joint"), _text("sum.arms.roughZero"), _text("sum.arms.refined"),
            _text("sum.arms.change"), _text("sum.arms.angle")], zero_rows))
    return sections


COUNTS_PER_TURN = 4096


def _unwrap_delta(delta: int, counts: int = COUNTS_PER_TURN) -> int:
    """Shortest signed count difference, matching core.servos.unwrap_delta.

    Duplicated rather than imported because the summary layer must stay usable
    without the calibration package on the path. A plain subtraction would
    report a 4-count move across the 0/4095 seam as 4092.
    """
    return (int(delta) + counts // 2) % counts - counts // 2


def _counts_to_deg(delta_counts: int) -> float:
    return delta_counts * 360.0 / COUNTS_PER_TURN


def _normalize_summary(workspace: Workspace) -> list[dict[str, Any]]:
    """Report what stage 7 actually moved, not merely that files exist.

    Stage 7 redefines the body frame and rewrites zeros, so the useful question
    is how far each zero shifted. Arm zeros are compared against stage 6
    (zeros_refined); head pan/tilt zeros live in a separate file and are
    compared against stage 4 (head), which is what this stage rewrites.
    """
    arm_before = _load(workspace, "zeros_refined").get("zeros", {}).get("joints", {})
    arm_after = _load(workspace, "zeros_zero").get("zeros", {}).get("joints", {})
    head_before = _load(workspace, "head").get("zeros", {}).get("joints", {})
    head = _load(workspace, "head_zero")
    head_after = head.get("zeros", {}).get("joints", {})
    mounting = workspace.mounting

    rows = []
    for before, after in ((head_before, head_after), (arm_before, arm_after)):
        for joint, item in _by_arm(after, mounting):
            raw_after = item.get("raw")
            raw_before = (before.get(joint) or {}).get("raw")
            label = joint_label(joint, mounting)
            if not isinstance(raw_after, int) or not isinstance(raw_before, int):
                rows.append([label, raw_before if raw_before is not None else "—",
                             raw_after if raw_after is not None else "—", "—", "—"])
                continue
            delta = _unwrap_delta(raw_after - raw_before)
            rows.append([label, raw_before, raw_after, f"{delta:+d}",
                         _text("sum.norm.unchanged") if delta == 0
                         else f"{_counts_to_deg(delta):+.2f}°"])

    unchanged = _text("sum.norm.unchanged")
    changed = sum(1 for row in rows if row[4] not in (unchanged, "—"))
    sections = [_section(_text("sum.norm.title", changed=changed),
                         [_text("sum.joint"), _text("sum.norm.before"),
                          _text("sum.norm.after"), _text("sum.arms.change"),
                          _text("sum.arms.angle")], rows)]

    frame_rows = []
    yaw = head.get("stage5b_yaw_correction_deg")
    tilt = head.get("stage5b_head_tilt_zero_correction_deg")
    if isinstance(yaw, (int, float)):
        frame_rows.append([_text("sum.norm.yaw"), f"{yaw:+.3f}°"])
    if isinstance(tilt, (int, float)):
        frame_rows.append([_text("sum.norm.tilt"), f"{tilt:+.3f}°"])
    # Both of these name arms, so both are read out physically-left first and
    # labelled by the side the operator can point at. The stored keys stay
    # model-named: `_L_mm` is the model's left_arm whichever way it is bolted.
    roots = {"left_arm": "stage5b_arm_root_L_mm",
             "right_arm": "stage5b_arm_root_R_mm"}
    for arm in mounting_mod.arm_order(mounting):
        point = head.get(roots[arm])
        if isinstance(point, list) and len(point) == 3:
            frame_rows.append([
                _text("sum.norm.root", arm=_arm_text(arm, mounting)),
                ", ".join(f"{v:.1f}" for v in point)])
    arm_results = _load(workspace, "touch_zero").get("arms", {})
    for arm in mounting_mod.arm_order(mounting):
        arm_result = arm_results.get(arm)
        if not isinstance(arm_result, dict):
            continue
        error = arm_result.get("forearm_heading_error_deg")
        if isinstance(error, (int, float)):
            frame_rows.append([
                _text("sum.norm.heading", arm=_arm_text(arm, mounting)),
                f"{error:+.3f}°"])
    frames = {item.get("body_frame_id")
              for item in (head, _load(workspace, "touch_zero"), _load(workspace, "zeros_zero"))
              if item.get("body_frame_id")}
    frame_rows.append([_text("sum.norm.sharedFrame"),
                       next(iter(frames), _text("sum.unrecorded"))
                       if len(frames) == 1
                       else _text("sum.norm.frameMismatch", count=len(frames))])
    if frame_rows:
        sections.append(_section(_text("sum.norm.frameTitle"),
                                 [_text("sum.item"), _text("sum.valueCol")], frame_rows))
    return sections


def _verify_summary(workspace: Workspace) -> list[dict[str, Any]]:
    item = _load(workspace, "validation")
    mounting = workspace.mounting
    labels = {role: robot_overview.camera_label(role, mounting)
              for role in ("head", "left_wrist", "right_wrist")}

    def number(value: Any, digits: int = 2, unit: str = "") -> str:
        return f"{value:.{digits}f}{unit}" if isinstance(value, (int, float)) else "—"

    rows, detail_rows, bias_rows = [], [], []
    camera_results = item.get("cameras", {})
    # Head first, then the wrists physically left to right.
    for camera in mounting_mod.camera_order(mounting):
        camera_result = camera_results.get(camera)
        if not isinstance(camera_result, dict):
            continue
        summary = camera_result.get("summary", {})
        label = labels.get(camera, camera)
        gates = camera_result.get("gates", {})
        # Name the gate that actually failed rather than just saying "failed".
        failed = [_text(name) for name, key in (
            ("sum.verify.gatePosition", "position_rms_mm"),
            ("sum.verify.gateRotation", "rotation_rms_deg"),
            ("sum.verify.gateSamples", "minimum_samples"))
            if gates.get(key) is False]
        # Same limits stage 8 itself applies.
        position_limit = 6.0 if camera == "head" else 8.0
        rows.append([
            label,
            _text("sum.verify.passed") if camera_result.get("passed")
            else _text("sum.verify.failed",
                     gates=_text("sum.verify.gateJoin").join(failed)),
            summary.get("count", "—"),
            number(summary.get("translation_rms_mm"), 2, f" / {position_limit:.0f} mm"),
            number(summary.get("rotation_rms_deg"), 3, " / 3 °"),
        ])
        detail_rows.append([
            label,
            number(summary.get("translation_rms_mm"), 2, " mm"),
            number(summary.get("translation_p95_mm"), 2, " mm"),
            number(summary.get("translation_max_mm"), 2, " mm"),
            number(summary.get("rotation_rms_deg"), 3, " °"),
            number(summary.get("rotation_p95_deg"), 3, " °"),
            number(summary.get("rotation_max_deg"), 3, " °"),
        ])
        # A large bias means a systematic offset left in the model; large RMS
        # with near-zero bias is just noise. Worth separating.
        translation_bias = summary.get("translation_bias_mm")
        rotation_bias = summary.get("rotation_bias_deg")
        bias_rows.append([
            label,
            ", ".join(number(v, 2) for v in translation_bias)
            if isinstance(translation_bias, list) else "—",
            ", ".join(number(v, 3) for v in rotation_bias)
            if isinstance(rotation_bias, list) else "—",
            number(summary.get("predicted_pixel_rms_px"), 2, " px"),
            number(summary.get("pnp_reprojection_rms_px"), 3, " px"),
        ])

    sections = [_section(_text("sum.verify.title"), [
        _text("sum.camera"), _text("sum.status"), _text("sum.samples"),
        _text("sum.verify.posRms"), _text("sum.verify.rotRms")], rows)]
    if detail_rows:
        sections.append(_section(_text("sum.verify.detailTitle"), [
            _text("sum.camera"), _text("sum.verify.posRmsPlain"),
            _text("sum.verify.posP95"), _text("sum.verify.posMax"),
            _text("sum.verify.rotRmsPlain"), _text("sum.verify.rotP95"),
            _text("sum.verify.rotMax")], detail_rows))
    if bias_rows:
        sections.append(_section(_text("sum.verify.biasTitle"), [
            _text("sum.camera"), _text("sum.verify.posBias"), _text("sum.verify.rotBias"),
            _text("sum.verify.pixelRms"), _text("sum.verify.pnpRms")], bias_rows))
    # This is the last stage, so the run ends with the whole calibrated
    # geometry in one place: where the three cameras and both arm roots are.
    sections.extend(robot_overview.sections(workspace))
    sections.extend(_view3d_section(workspace))
    return sections


def _view3d_section(workspace: Workspace) -> list[dict[str, Any]]:
    """Write the standalone 3D page and point the user at it.

    The overview tables and the flat diagram both lose information when two
    markers project to nearly the same spot, which mirrored arm roots do in a
    side view. A rotatable view has no such blind angle. It is written as a
    plain file rather than embedded so it survives outside the tool: it can be
    archived with the results or sent to someone who does not run the tool.
    """
    try:
        page = robot_view3d.build(workspace)
    except Exception as error:  # noqa: BLE001 - a diagram must never fail a run
        return [_section(_text("summary.view3dTitle"), [_text("sum.status")],
                         [[_text("summary.view3dFailed", error=error)]])]
    destination = workspace.results / "robot_overview_3d.html"
    destination.write_text(page, encoding="utf-8")
    # The path stays visible because the file outlives the tool, but the link
    # saves the operator from copying it into a browser by hand.
    return [_section(_text("summary.view3dTitle"),
                     [_text("summary.view3dFile"), _text("summary.view3dNote")],
                     [[{"link": "/results/robot_overview_3d.html",
                        "text": str(destination)},
                       _text("summary.view3dHint")]])]


def summarize_stage(workspace: Workspace, stage_key: str) -> dict[str, Any]:
    stage = BY_KEY[stage_key]
    builders = {
        "prepare": _prepare_summary,
        "intrinsics": _intrinsics_summary,
        "senses": _senses_summary,
        "head": _head_summary,
        "arm_ranges": _ranges_summary,
        "arms": _arms_summary,
        "normalize": _normalize_summary,
        "verify": _verify_summary,
    }
    sections = builders[stage_key](workspace)
    return {"stage_key": stage_key, "stage_number": stage.number,
            "title": stage.title,
            "sections": sections}
