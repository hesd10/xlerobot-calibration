"""Stage 5b: define head pan zero from arm mount symmetry.

Once both arms are calibrated, their physical symmetry defines a "forward"
direction for the robot. This stage:
  1. Loads both arm mount transforms from touch.json
  2. Queries the simulation for arm root body positions
  3. Applies T_B_A corrections to get true arm root positions in base frame
  4. Calculates the geometric forward direction (perpendicular to the line
     connecting the two arm roots)
  5. Updates T_W_B in head.json using closed-form pan zero shift
  6. Pins each arm's shoulder-pan zero with the sideways forearm link, which
     resolves the gauge freedom stage 5 leaves between pan zero and mount yaw

This is exact: rotations about one axis compose, so A(q) = A(q-d) A(d) and the
A(d) is absorbed into T_W^B. No recapture or re-solve needed.

Physical constraint: arm mount translations are fixed by hardware. If asymmetry
is detected, we warn but continue using the geometric midpoint direction.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from scipy.optimize import brentq
from core import (storage, head_model, se3, senses as senses_mod,
                  wrist_model, zeros as zeros_mod)
import frames
import model_map


ROLL_ZERO_LIMIT_DEG = 45.0
HEAD_TILT_ZERO_LIMIT_DEG = 45.0


def _head_optical_axis_at_zero(T_tilt_cam: np.ndarray,
                               axis_origin: np.ndarray,
                               senses: tuple[float, float]) -> np.ndarray:
    T_base_cam = head_model.T_base_cam(
        0.0, 0.0, T_tilt_cam, axis_origin, senses)
    optical = T_base_cam[:3, 2]
    return optical / np.linalg.norm(optical)


def resolve_head_tilt_zero(head: dict, senses: tuple[float, float],
                           axis_origin: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
    """Choose the nearest tilt zero with a horizontal, forward-facing optical axis.

    FRAME_RELATIVE (see frames.py): the operator sets the pan zero facing the
    board, which makes the axis forward by construction. This confirms the solve
    is self-consistent; it is not evidence that the frame itself is right.
    """
    T_tilt_cam = np.asarray(head["T_tilt_cam"], float)

    def transformed_mount(delta: float) -> np.ndarray:
        return head_model.shift_tilt_zero(
            T_tilt_cam, delta, axis_origin, sense=senses[1])

    def z_at(delta: float) -> float:
        return float(_head_optical_axis_at_zero(
            transformed_mount(delta), axis_origin, senses)[2])

    grid = np.linspace(-np.pi, np.pi, 1441)
    values = [z_at(value) for value in grid]
    roots = []
    for lo, hi, f_lo, f_hi in zip(grid[:-1], grid[1:], values[:-1], values[1:]):
        if f_lo == 0.0:
            roots.append(float(lo))
        elif f_lo * f_hi < 0.0:
            roots.append(float(brentq(z_at, lo, hi)))

    candidates = []
    for delta in roots:
        mount = transformed_mount(delta)
        optical = _head_optical_axis_at_zero(mount, axis_origin, senses)
        if frames.points_forward(optical):
            candidates.append((abs(delta), delta, mount, optical))
    if not candidates:
        raise ValueError("head: no horizontal optical-axis root facing forward")

    _, delta, mount, optical = min(candidates, key=lambda item: item[0])
    if abs(np.rad2deg(delta)) > HEAD_TILT_ZERO_LIMIT_DEG:
        raise ValueError(
            f"head: nearest horizontal tilt zero is {np.rad2deg(delta):+.1f} deg")
    return delta, mount, optical


def get_arm_root_position(sim, arm: str, T_B_A: np.ndarray) -> np.ndarray:
    """Get the arm root position in the base frame.
    
    The simulation's ROOT_BODIES give the nominal arm root position, and T_B_A
    is the correction transform applied on top. Returns position in meters.
    """
    root_body = model_map.ROOT_BODIES[arm]
    sim.set_joints({})
    p_nom, _ = sim.body_pose_in_chassis(root_body)
    # T_B_A and the nominal root position are both expressed in chassis frame.
    p_corrected = (T_B_A @ np.append(p_nom, 1.0))[:3]
    return p_corrected


def _pan_axis_half_turn(sim) -> np.ndarray:
    """A 180 degree turn of the base frame about the head's pan axis.

    Used when a back-to-front robot has produced a base frame a half turn from
    the model's. Taken about the pan axis rather than the origin so the chassis
    turns about the mast the head actually rotates on, which is what makes the
    arm mounts land back where they are bolted.

    This is a gauge change, not a re-solve: mount yaw and shoulder-pan zero are
    two halves of one freedom and only their sum is observable, so the turn must
    be paired with taking the same amount off each pan zero. The caller does
    that; see the shoulder-pan section below.
    """
    sim.set_joints({})
    origin, _ = sim.body_pose_in_chassis("head_pan_link")
    origin = np.asarray(origin, float)
    turn = np.eye(4)
    turn[:3, :3] = np.diag([-1.0, -1.0, 1.0])
    turn[:3, 3] = origin - turn[:3, :3] @ origin
    return turn


def calculate_symmetry_yaw(pos_L: np.ndarray, pos_R: np.ndarray) -> float:
    """Yaw correction, in radians, that squares the base frame to the arm mounts.

    The line joining the two arm roots is lateral, so its perpendicular is the
    sagittal direction. Two perpendiculars exist, 180 degrees apart; the one
    with the larger X is taken.

    Note that is +X, whereas the model's forward is -X (see frames.FORWARD), so
    the returned angle is measured from the robot's BACKWARD direction. That
    sounds like a bug and is not: the caller uses the value only as a relative
    correction, `zero_shift = symmetry_yaw`, and both perpendiculars give the
    same correction because a symmetric pair of mounts yields zero either way.
    The choice only has to be consistent between runs, which taking the larger X
    guarantees.

    It cannot repair a base frame that is a half turn out, because such a frame
    turns both mounts together and leaves them symmetric. Only the
    MODEL_ANCHORED check in frames.wrong_side_report catches that.
    """
    d = (pos_L - pos_R)[:2]  # xy projection
    perp1 = np.array([-d[1], d[0]])
    perp2 = np.array([d[1], -d[0]])
    sagittal = perp1 if perp1[0] > perp2[0] else perp2
    sagittal = sagittal / np.linalg.norm(sagittal)
    return float(np.arctan2(sagittal[1], sagittal[0]))


# Stage 5 leaves shoulder-pan zero and arm mount yaw indistinguishable: to
# contact data they are the same motion, so the solver pins mount yaw at 0 and
# lets the pan zero absorb whatever is left. That leaves the pan zero floating.
#
# The forearm links break the tie. In the model at q=0 both the Upper_Arm ->
# Lower_Arm and Lower_Arm -> Wrist links lie exactly in the sagittal plane
# (x component 0) pointing along -Y for the left arm and +Y for the right, and
# their heading in the XY plane turns degree for degree with shoulder pan while
# being completely unaffected by shoulder lift and elbow flex, which rotate
# about axes parallel to the link. So "this link points straight out to the
# side" is an observable that fixes the pan zero and nothing else.
#
# The Lower_Arm -> Wrist link is used: it is the longer of the two (135 mm vs
# 116 mm), so the same angular error moves its endpoint further and the heading
# is correspondingly less sensitive to model noise.
PAN_ZERO_LIMIT_DEG = 30.0

PAN_LINK_BODIES = {
    "left_arm": ("Lower_Arm", "Wrist_Pitch_Roll"),
    "right_arm": ("Lower_Arm_2", "Wrist_Pitch_Roll_2"),
}


def _pan_axis_rotation(sim, arm: str, angle: float) -> np.ndarray:
    """Rotation about this arm's pan axis, as a chassis-frame 4x4.

    The pan joint turns about chassis +Z through the arm root, so moving the pan
    zero and rotating the mount by the opposite amount cancel exactly.
    """
    sim.set_joints({})
    origin, _ = sim.body_pose_in_chassis(model_map.ROOT_BODIES[arm])
    origin = np.asarray(origin, float)
    cos, sin = np.cos(angle), np.sin(angle)
    rotation = np.array([[cos, -sin, 0.0], [sin, cos, 0.0], [0.0, 0.0, 1.0]])
    transform = np.eye(4)
    transform[:3, :3] = rotation
    transform[:3, 3] = origin - rotation @ origin
    return transform


def _forearm_heading(sim, arm: str, T_B_A: np.ndarray, pan: float) -> float:
    """Heading of the forearm link in the XY plane, in degrees.

    Measured with every other arm joint at its zero. Lift and elbow do not
    affect this heading, so leaving them at zero costs nothing.
    """
    angles = {name: 0.0 for name in model_map.ARM_JOINTS_NO_GRIPPER[arm]}
    angles[f"{arm}_shoulder_pan"] = float(pan)
    sim.set_joints(angles)
    near_body, far_body = PAN_LINK_BODIES[arm]
    p_near, _ = sim.body_pose_in_chassis(near_body)
    p_far, _ = sim.body_pose_in_chassis(far_body)
    near = (T_B_A @ np.append(p_near, 1.0))[:3]
    far = (T_B_A @ np.append(p_far, 1.0))[:3]
    link = far - near
    return float(np.rad2deg(np.arctan2(link[0], link[1])))


def resolve_shoulder_pan_zero(sim, arm: str, T_B_A: np.ndarray) -> tuple[float, float]:
    """Find the pan shift that points the forearm straight out to the side.

    Returns the model-space shift and the heading error it removes. The shift is
    the posture the servo is really in when it reads its current zero, so
    recording it moves the zero onto the sideways-pointing posture.
    """
    ideal = 180.0 if arm == "left_arm" else 0.0

    def error_at(pan: float) -> float:
        # Wrapped to (-180, 180] so the left arm's +-180 ideal has no seam.
        return (_forearm_heading(sim, arm, T_B_A, pan) - ideal + 180.0) % 360.0 - 180.0

    limit = np.deg2rad(PAN_ZERO_LIMIT_DEG)
    lo, hi = error_at(-limit), error_at(limit)
    # A sign change alone is not enough. At a heading error near 180 the wrap in
    # error_at() flips the sign at the seam, so lo and hi straddle it and the
    # bracket looks valid while containing no root at all: brentq then converges
    # on the seam and reports a near-zero shift for an arm a half turn out. That
    # is exactly what a robot used back-to-front produces. Requiring both ends to
    # be small keeps the bracket on the continuous stretch around the real root.
    if abs(lo) > 90.0 or abs(hi) > 90.0 or lo * hi > 0.0:
        raise ValueError(
            f"{arm}: no sideways forearm posture within +-{PAN_ZERO_LIMIT_DEG:.0f} deg "
            f"of the current pan zero (heading error {error_at(0.0):+.2f} deg)")
    delta = float(brentq(error_at, -limit, limit))
    if abs(np.rad2deg(delta)) > PAN_ZERO_LIMIT_DEG:
        raise ValueError(
            f"{arm}: nearest sideways pan zero is {np.rad2deg(delta):+.1f} deg")
    return delta, error_at(0.0)


def _roll_about_axis(delta: float) -> np.ndarray:
    screw = np.concatenate([
        wrist_model.ROLL_AXIS_IN_WRIST * float(delta), np.zeros(3)])
    return se3.exp_se3(screw)


def _optical_axis_at_zero(sim, arm: str, T_B_A: np.ndarray,
                          T_wrist_cam: np.ndarray) -> np.ndarray:
    angles = {name: 0.0 for name in model_map.ARM_JOINTS_NO_GRIPPER[arm]}
    sim.set_joints(angles)
    _, R_jaw = sim.body_pose_in_chassis(model_map.WRIST_BODIES[arm])
    optical = T_B_A[:3, :3] @ R_jaw @ T_wrist_cam[:3, 2]
    return optical / np.linalg.norm(optical)


def resolve_wrist_roll_zero(sim, arm: str, T_B_A: np.ndarray,
                            T_wrist_cam: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
    """Choose the nearest roll zero whose optical axis is horizontal.

    Two roots lie 180 degrees apart, so the tie is broken by which way the
    camera ends up facing. See frames.horizontal_quadrant_sign for the signs,
    which are not mirror-symmetric between the arms.
    """
    expected = frames.horizontal_quadrant_sign(arm)

    def transformed_mount(delta: float) -> np.ndarray:
        # The new zero is the old posture q=delta, so q_old=q_new+delta.
        return _roll_about_axis(delta) @ T_wrist_cam

    def z_at(delta: float) -> float:
        return float(_optical_axis_at_zero(
            sim, arm, T_B_A, transformed_mount(delta))[2])

    grid = np.linspace(-np.pi, np.pi, 1441)
    values = [z_at(value) for value in grid]
    roots = []
    for lo, hi, f_lo, f_hi in zip(grid[:-1], grid[1:], values[:-1], values[1:]):
        if f_lo == 0.0:
            roots.append(float(lo))
        elif f_lo * f_hi < 0.0:
            roots.append(float(brentq(z_at, lo, hi)))

    candidates = []
    for delta in roots:
        mount = transformed_mount(delta)
        optical = _optical_axis_at_zero(sim, arm, T_B_A, mount)
        got = np.sign([frames.forward_of(optical), frames.lateral(optical)])
        if np.array_equal(got, expected):
            candidates.append((abs(delta), delta, mount, optical))
    if not candidates:
        # Both roots sit in the opposite pair of quadrants when the base frame
        # is a half turn from the model's, which is what a back-to-front robot
        # produces. Say so, rather than leaving a bare geometric complaint.
        raise ValueError(
            f"{arm}: no horizontal optical-axis root pointing the expected way. "
            f"This stage defines the base frame, so it cannot be re-run on a "
            f"workspace whose frame has already been rewritten. Re-run it "
            f"against the stage 6 output instead, with the mounting switch set "
            f"to match how the robot was standing.")

    _, delta, mount, optical = min(candidates, key=lambda item: item[0])
    if abs(np.rad2deg(delta)) > ROLL_ZERO_LIMIT_DEG:
        raise ValueError(
            f"{arm}: nearest horizontal wrist-roll zero is {np.rad2deg(delta):+.1f} deg")
    return delta, mount, optical


def main() -> int:
    import common
    
    try:
        common.require_results("head", "touch", "zeros")
    except common.Aborted:
        return 1
    
    # Load results
    touch = storage.load_result("touch")
    head = storage.load_result("head")
    zeros_source = storage.load_result("zeros")
    
    if not touch or not touch.get("arms"):
        print("Error: No touch calibration found.")
        return 1
    
    arms = touch["arms"]
    if len(arms) < 2:
        print(f"Error: Only {len(arms)} arm(s) calibrated: {list(arms.keys())}")
        print("Both arms must be calibrated before running stage 5b.")
        return 1
    
    print("\n=== Stage 5b: Head pan zero from arm symmetry ===\n")
    
    # Load simulation
    sim = model_map.SimModel()
    
    # Extract arm root positions
    left = arms["left_arm"]
    right = arms["right_arm"]
    T_B_L = np.array(left["T_B_A"])
    T_B_R = np.array(right["T_B_A"])
    
    pos_L = get_arm_root_position(sim, "left_arm", T_B_L)
    pos_R = get_arm_root_position(sim, "right_arm", T_B_R)

    # Read the mounting before the first report, not just before the arithmetic
    # that needs it: every line below names an arm, and used back-to-front the
    # arm the model calls left_arm is the one on the operator's right. Printing
    # the stored side would have them checking the opposite arm.
    mounting = frames.declared_mounting()

    def report_roots() -> None:
        for arm, pos in (("left_arm", pos_L), ("right_arm", pos_R)):
            side = frames.physical_side(arm, mounting).capitalize()
            print(f"{side} arm root (mm): "
                  f"[{pos[0]*1000:.2f}, {pos[1]*1000:.2f}, {pos[2]*1000:.2f}]")

    report_roots()

    # Every other direction test in this stage is FRAME_RELATIVE in the sense
    # frames.py describes: phrased in the base frame, which is itself defined by
    # wherever the operator set the head pan zero. Those cannot detect a frame
    # that is a half turn out. This one is MODEL_ANCHORED, so it can.
    #
    # A robot used back-to-front lands here every time: the operator must set
    # the pan zero facing the board to capture anything, and on that robot
    # facing the board means facing chassis-back, so the solved frame comes out
    # a half turn from the model's.
    #
    # The measurement alone cannot say whether that half turn is intended. A
    # normal robot whose head pan zero was set half a turn out looks exactly the
    # same here, and quietly absorbing that would bake the mistake into every
    # later stage. So the declared mounting decides, and the measurement only
    # confirms it.
    turned = not frames.is_on_expected_side("left_arm", pos_L)
    print(f"\nDeclared mounting: {mounting}")
    print(f"Measured frame:    "
          f"{'half a turn from the model' if turned else 'as the model expects'}")

    # Both mountings must land here. Stage 5 converts its captured pan angles
    # into model angles using the mounting offset, so the frame that reaches
    # this stage is already the model's own however the robot is standing. A
    # turned frame is therefore a fault in either mounting rather than the
    # signature of one of them.
    if turned:
        raise ValueError(
            "the arm roots came out on the wrong sides of the base frame.\n"
            "  The usual cause is the head pan zero: it has to be set with the "
            "head facing the board.\n"
            "  Otherwise check the mounting switch really matches how the robot "
            "is standing, then redo the head stage and the arm stage.\n"
            f"  The arm on your {frames.physical_side('left_arm', mounting)} "
            f"sits {frames.lateral(pos_L) * 1000:+.1f} mm "
            "toward the robot's left; the model puts it on the other side.")

    # Nothing is absorbed here. A back-to-front robot has its captured pan
    # angles converted to model angles by stage 5, on the way in, so the frame
    # that reaches this stage is already the model's own however the robot is
    # standing. This stage only checks it.
    #
    # Absorbing the half turn here instead was tried and was wrong twice over:
    # it cancelled the very half turn the turned mount flanges physically have,
    # and it left this stage applying the correction it is supposed to be an
    # independent check on, so a genuinely bad stage 6 had nothing left to fail
    # against.
    report = frames.wrong_side_report({"left_arm": pos_L, "right_arm": pos_R})
    if report:
        print(f"\nError: {report}")
        return 1
    
    # Check symmetry and warn if needed
    y_asym = pos_L[1] + pos_R[1]
    x_diff = abs(pos_L[0] - pos_R[0])
    z_diff = abs(pos_L[2] - pos_R[2])
    
    WARN_Y = 0.005  # 5mm
    WARN_XZ = 0.005
    
    warnings = []
    if abs(y_asym) > WARN_Y:
        warnings.append(f"  Y asymmetry: {y_asym * 1000:.1f} mm")
    if x_diff > WARN_XZ:
        warnings.append(f"  X misalignment: {x_diff * 1000:.1f} mm")
    if z_diff > WARN_XZ:
        warnings.append(f"  Z misalignment: {z_diff * 1000:.1f} mm")
    
    if warnings:
        print(f"\n⚠ WARNING: Arm roots not perfectly symmetric:")
        for w in warnings:
            print(w)
        print("  (This is a fixed hardware constraint.)")
        print("  Continuing with geometric midpoint as forward direction.\n")
    else:
        print("\n✓ Arm roots are symmetric within tolerance.\n")
    
    # Calculate forward direction
    symmetry_yaw = calculate_symmetry_yaw(pos_L, pos_R)
    
    print(f"Calculated forward direction: {np.rad2deg(symmetry_yaw):+.4f}° yaw in base frame")
    print(f"  (perpendicular to line connecting arm roots)\n")
    
    # Load current T_W_B
    T_W_B_old = np.array(head["T_W_B"])
    head_zeros = zeros_mod.ZeroSet.from_dict(head.get("zeros"))
    if "head_motor_1" not in head_zeros.joints:
        print("Error: head.json does not contain its paired pan zero.")
        return 1
    got_senses = senses_mod.load()
    if got_senses is None:
        print("Error: no measured head pan sense found; rerun Stage 2.")
        return 1
    pan_sense = got_senses.sign("head_motor_1")
    tilt_sense = got_senses.sign("head_motor_2")

    # Apply corrections using the same measured encoder-to-model convention as Stage 3.
    axis_origin = np.array(head["gauge"]["pan_axis_origin_m"])
    tilt_shift, T_tilt_cam_new, head_optical = resolve_head_tilt_zero(
        head, (pan_sense, tilt_sense), axis_origin)
    head_zeros.record_shift(
        "head_motor_2", tilt_shift,
        reason="stage 5b head optical axis toward -X", sign=tilt_sense)
    print(
        f"Applying head tilt zero shift: {np.rad2deg(tilt_shift):+.4f}° "
        f"(optical axis [{head_optical[0]:+.5f}, {head_optical[1]:+.5f}, "
        f"{head_optical[2]:+.5f}])")

    # `symmetry_yaw` is the new body's forward direction expressed in the old
    # body frame, so it is directly the model-space zero shift. Negating it here
    # rotates the body frame away from arm-root symmetry by twice this angle.
    zero_shift = symmetry_yaw
    # record_shift() converts the model-space shift to raw encoder counts. The
    # head model consumes the unsigned encoder angle and applies pan_sense itself,
    # so the paired frame transform takes the corresponding encoder-space shift.
    encoder_zero_shift = pan_sense * zero_shift
    T_W_B_new = head_model.shift_pan_zero(
        T_W_B_old, encoder_zero_shift, axis_origin, sense=pan_sense)
    head_zeros.record_shift(
        "head_motor_1", zero_shift,
        reason="stage 5b arm-root symmetry", sign=pan_sense)

    print(f"Applying pan zero shift: {np.rad2deg(zero_shift):+.4f}° (closed-form, exact)\n")

    # A maps coordinates in the new body convention back into the old convention:
    # T_W_B_new = T_W_B_old @ A. Every old B-frame pose must therefore be
    # re-expressed as T_Bnew_X = inv(A) @ T_Bold_X.
    A_old_new = head_model.pan_transform(
        encoder_zero_shift, axis_origin, sense=pan_sense)
    A_new_old = np.linalg.inv(A_old_new)
    sources = {
        "head_fingerprint": storage.result_fingerprint(head),
        "touch_fingerprint": storage.result_fingerprint(touch),
        "zeros_fingerprint": storage.result_fingerprint(zeros_source),
        "head_saved_at": head.get("saved_at"),
        "touch_saved_at": touch.get("saved_at"),
        "zeros_saved_at": zeros_source.get("saved_at"),
    }
    frame_id = (
        f"stage5b:{sources['head_fingerprint'][:10]}:"
        f"{sources['touch_fingerprint'][:10]}:"
        f"{sources['zeros_fingerprint'][:10]}"
    )

    calibrated_zeros = zeros_mod.ZeroSet.from_dict(zeros_source.get("zeros"))
    touch_zero = dict(touch)
    touch_zero.pop("saved_at", None)
    touch_zero.pop("git_revision", None)
    touch_zero["arms"] = {}
    transformed_roots = {}
    # Physical-left first, so the two lines below appear in the order the
    # operator would walk the arms rather than in whatever order the JSON holds.
    for arm in frames.working_order(mounting):
        arm_result = arms[arm]
        spoken = frames.physical_side(arm, mounting) + " arm"
        transformed = dict(arm_result)
        T_B_A_new = A_new_old @ np.asarray(arm_result["T_B_A"], float)
        transformed["T_B_A"] = T_B_A_new.tolist()

        roll_motor = model_map.ARM_JOINTS_NO_GRIPPER[arm][-1]
        if roll_motor not in calibrated_zeros.joints:
            raise ValueError(f"{arm}: no wrist-roll zero in zeros.json")

        # Pin the gauge freedom left over from stage 5. Moving the pan zero
        # rotates the arm, so the mount absorbs the opposite rotation about the
        # same axis: the two cancel exactly, leaving every pose this arm can
        # reach unchanged while the zero lands on the sideways posture. The
        # compensation uses the delta actually realised after the zero is
        # rounded to whole counts, not the ideal one.
        pan_motor = f"{arm}_shoulder_pan"
        if pan_motor not in calibrated_zeros.joints:
            raise ValueError(f"{arm}: no shoulder-pan zero in zeros.json")
        pan_sign = got_senses.sign(pan_motor)
        pan_delta, pan_error = resolve_shoulder_pan_zero(sim, arm, T_B_A_new)
        pan_before = calibrated_zeros.joints[pan_motor].raw
        calibrated_zeros.record_shift(
            pan_motor, pan_delta,
            reason="stage 5b sideways forearm link fixes the pan gauge freedom",
            sign=pan_sign)
        pan_after = calibrated_zeros.joints[pan_motor].raw
        pan_realised = pan_sign * (pan_after - pan_before) * 2.0 * np.pi / model_map.COUNTS_PER_TURN
        T_B_A_new = T_B_A_new @ _pan_axis_rotation(sim, arm, -pan_realised)
        transformed["T_B_A"] = T_B_A_new.tolist()
        transformed["shoulder_pan_zero_correction_deg"] = float(np.rad2deg(pan_realised))
        transformed["forearm_heading_error_deg"] = float(pan_error)
        print(f"  {spoken} shoulder-pan zero: {np.rad2deg(pan_realised):+.4f} deg "
              f"(forearm heading was off by {pan_error:+.4f} deg)")

        delta, T_cam_new, optical = resolve_wrist_roll_zero(
            sim, arm, T_B_A_new, np.asarray(arm_result["T_wrist_cam"], float))
        calibrated_zeros.record_shift(
            roll_motor, delta, reason="stage 5b horizontal wrist-camera optical axis",
            sign=got_senses.sign(roll_motor))
        transformed["T_wrist_cam"] = T_cam_new.tolist()
        transformed["wrist_roll_zero_correction_deg"] = float(np.rad2deg(delta))
        transformed["optical_axis_at_zero"] = optical.tolist()
        transformed["optical_axis_azimuth_deg"] = float(
            np.rad2deg(np.arctan2(optical[1], optical[0])))
        transformed["optical_axis_elevation_deg"] = float(
            np.rad2deg(np.arctan2(optical[2], np.linalg.norm(optical[:2]))))
        touch_zero["arms"][arm] = transformed
        transformed_roots[arm] = get_arm_root_position(sim, arm, T_B_A_new)
        print(
            f"  {spoken} wrist-roll zero: {np.rad2deg(delta):+.4f} deg; "
            f"optical azimuth {transformed['optical_axis_azimuth_deg']:+.4f} deg")
    touch_zero["body_frame_id"] = frame_id
    touch_zero["stage5b_sources"] = sources
    touch_zero["T_Bnew_Bold"] = A_new_old.tolist()
    touch_zero["stage5b_applied"] = True

    # Update the head transform and its paired encoder zero as the other half of
    # the same body-frame contract. Head-local T_tilt_cam remains unchanged.
    result = dict(head)
    result.pop("saved_at", None)
    result.pop("git_revision", None)
    result["T_W_B"] = T_W_B_new.tolist()
    result["T_tilt_cam"] = T_tilt_cam_new.tolist()
    result["params"] = head_model.pack(
        T_W_B_new, T_tilt_cam_new).tolist()
    result["zeros"] = head_zeros.to_dict()
    result["stage5b_head_tilt_zero_correction_deg"] = float(
        np.rad2deg(tilt_shift))
    result["stage5b_head_optical_axis_at_zero"] = head_optical.tolist()
    result["body_frame_id"] = frame_id
    result["stage5b_sources"] = sources
    result["T_Bold_Bnew"] = A_old_new.tolist()
    result["stage5b_applied"] = True
    result["stage5b_yaw_correction_deg"] = float(np.rad2deg(symmetry_yaw))
    result["stage5b_arm_root_L_mm"] = (
        transformed_roots["left_arm"] * 1000).tolist()
    result["stage5b_arm_root_R_mm"] = (
        transformed_roots["right_arm"] * 1000).tolist()
    result["stage5b_note"] = (
        "Pan zero and body frame redefined from arm-root symmetry; tilt zero "
        "defined by the head optical axis toward -X; shoulder-pan zeros pinned "
        "by the sideways forearm link, with arm mounts rotated to compensate so "
        "arm poses are unchanged. Use this result only with touch_zero.json "
        "carrying the same body_frame_id."
    )

    zeros_zero = dict(zeros_source)
    zeros_zero.pop("saved_at", None)
    zeros_zero.pop("git_revision", None)
    zeros_zero["zeros"] = calibrated_zeros.to_dict()
    zeros_zero["body_frame_id"] = frame_id
    zeros_zero["stage5b_sources"] = sources
    zeros_zero["stage5b_applied"] = True

    storage.save_result("touch_zero", touch_zero)
    storage.save_result("head_zero", result)
    storage.save_result("zeros_zero", zeros_zero)

    print("✓ Head pan, body frame and wrist-roll zeros updated successfully.")
    print(f"  Head result:  {storage.RESULTS_DIR / 'head_zero.json'}")
    print(f"  Arm result:   {storage.RESULTS_DIR / 'touch_zero.json'}")
    print(f"  Joint zeros:  {storage.RESULTS_DIR / 'zeros_zero.json'}")
    print(f"  Paired body frame: {frame_id}")
    print(f"\nThe head's pan=0 now points {np.rad2deg(symmetry_yaw):+.4f}° from the previous forward.")
    print("This is the 'forward' direction defined by arm root symmetry.\n")
    
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
