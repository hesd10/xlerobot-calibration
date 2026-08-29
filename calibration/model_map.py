"""Joint naming and unit conversion between the real XLeRobot and the MuJoCo model.

The two use unrelated joint names; this module is the single place that reconciles
them.

Frame convention
----------------
Forward is -x, up is +z, so the robot's left is -y:

    left = up x forward = z_hat x (-x_hat) = -y_hat

Two independent pieces of geometry back the -x forward direction:

  - the head camera mesh mounts at xyz 0.025 0 0.03 on head_tilt_link, which
    rotates into world -x, i.e. the lens protrudes toward -x
  - the whole payload leans that way: both arm roots at x = -0.09, head at -0.10

The `_L` chain sits at y = -0.155 and the `_R` chain at y = +0.155, so the model's
_L/_R suffixes are correct: `_L` really is the robot's left arm.

Two mislabellings in the source assets, neither of which says anything about the
joint naming. Do not use either as evidence:

  - the URDF's wrist cameras are swapped: `fixed_Right_Arm_Camera` hangs off
    `Fixed_Jaw`, which is on the `_L` (left) chain
  - the wheel bodies are swapped: `left_wheel` is at y = +0.225, i.e. on the
    robot's right. See WHEELS below, which corrects for this.

Servo model
-----------
STS3215 encoders are 4096 counts/turn:

    q_sim = sign * (2*pi/4096) * (raw - raw_zero)
    raw   = sign * q_sim * 4096/(2*pi) + raw_zero

`sign` and `raw_zero` are calibration outputs, not known a priori. Until
calibration has run, ZEROS/SIGNS below hold nominal placeholders.
"""

from __future__ import annotations

import math
from pathlib import Path

MODEL_DIR = Path(__file__).resolve().parent / "model"
# The augmented model (cameras, sites, actuators) built by build_calib_model.py.
MODEL_XML = MODEL_DIR / "xlerobot_calib.xml"
VISUAL_XML = MODEL_DIR / "xlerobot_visual.xml"
URDF = MODEL_DIR / "xlerobot.urdf"

# Bodies within this distance of the sagittal plane count as centred, not sided.
CENTRE_TOL = 0.01

COUNTS_PER_TURN = 4096
RAD_PER_COUNT = 2.0 * math.pi / COUNTS_PER_TURN

# real motor name -> sim joint name. Suffixes line up directly.
REAL_TO_SIM: dict[str, str] = {
    # Robot's LEFT arm: bus1, servo IDs 1-6.
    "left_arm_shoulder_pan": "Rotation_L",
    "left_arm_shoulder_lift": "Pitch_L",
    "left_arm_elbow_flex": "Elbow_L",
    "left_arm_wrist_flex": "Wrist_Pitch_L",
    "left_arm_wrist_roll": "Wrist_Roll_L",
    "left_arm_gripper": "Jaw_L",
    # Robot's RIGHT arm: bus2, servo IDs 1-6.
    "right_arm_shoulder_pan": "Rotation_R",
    "right_arm_shoulder_lift": "Pitch_R",
    "right_arm_elbow_flex": "Elbow_R",
    "right_arm_wrist_flex": "Wrist_Pitch_R",
    "right_arm_wrist_roll": "Wrist_Roll_R",
    "right_arm_gripper": "Jaw_R",
    # Head, on bus1 as servo IDs 7 and 8.
    "head_motor_1": "head_pan_joint",
    "head_motor_2": "head_tilt_joint",
}

SIM_TO_REAL: dict[str, str] = {v: k for k, v in REAL_TO_SIM.items()}

# Sim bodies that terminate each kinematic chain, used as FK targets.
TIP_BODIES = {
    "left_arm": "Moving_Jaw",
    "right_arm": "Moving_Jaw_2",
    "head": "head_tilt_link",
}

# Last rigid link before the gripper opens, i.e. frame L_L / L_R in the plan.
WRIST_BODIES = {
    "left_arm": "Fixed_Jaw",
    "right_arm": "Fixed_Jaw_2",
}

# Arm root bodies, frames A_L / A_R in the plan.
ROOT_BODIES = {
    "left_arm": "Rotation_Pitch",
    "right_arm": "Rotation_Pitch_2",
}

# Body-frame convention. frames.py owns it; these names are kept because callers
# and docs refer to them. Previously they were declared here with no readers at
# all, so editing them changed nothing while the real convention sat in literals
# such as `p[1] < 0` scattered through the stages.
from frames import FORWARD as _FORWARD, LEFT as _LEFT
import frames

FORWARD_AXIS = tuple(float(value) for value in _FORWARD)
LEFT_AXIS = tuple(float(value) for value in _LEFT)

# URDF fixed_head_camera_link origin, in head_tilt_link coordinates. Rotated into
# world this comes out along -x, which is the evidence for FORWARD_AXIS.
HEAD_CAM_MOUNT_XYZ = (0.025, 0.0, 0.03)

# The model's wheel bodies are mislabelled: `left_wheel` sits at y = +0.225, which
# is the robot's right. Key on the physical side, not the body name. Wheels are not
# actuated in the calib model since the base is localised visually.
WHEELS = {
    "left": "right_wheel",   # y = -0.225, robot's left
    "right": "left_wheel",   # y = +0.225, robot's right
}

# Arm joints excluding the gripper: these five per arm carry TCP position, so they
# are the zero offsets solved during the contact-touch stage.
ARM_JOINTS_NO_GRIPPER = {
    "left_arm": [
        "left_arm_shoulder_pan",
        "left_arm_shoulder_lift",
        "left_arm_elbow_flex",
        "left_arm_wrist_flex",
        "left_arm_wrist_roll",
    ],
    "right_arm": [
        "right_arm_shoulder_pan",
        "right_arm_shoulder_lift",
        "right_arm_elbow_flex",
        "right_arm_wrist_flex",
        "right_arm_wrist_roll",
    ],
}

GRIPPERS = ("left_arm_gripper", "right_arm_gripper")
HEAD_JOINTS = ("head_motor_1", "head_motor_2")

# PLACEHOLDERS. raw_zero is the encoder reading that corresponds to the sim's q=0
# pose, and sign is +1 when increasing counts means increasing q_sim. Neither is
# known yet: the robot's current zeros are known to differ substantially from the
# model's, and reconciling them is what calibration stages 4 and 5 do. Using these
# values for anything other than smoke tests will give wrong answers.
NOMINAL_RAW_ZERO = 2048
ZEROS: dict[str, int] = {name: NOMINAL_RAW_ZERO for name in REAL_TO_SIM}
SIGNS: dict[str, int] = {name: 1 for name in REAL_TO_SIM}
CALIBRATED = False  # set once real zeros/signs are loaded from calibration output


def raw_to_rad(motor: str, raw: float,
               zeros: dict[str, int] | None = None,
               signs: dict[str, int] | None = None) -> float:
    """Encoder counts -> sim joint angle in radians."""
    z = (zeros or ZEROS)[motor]
    s = (signs or SIGNS)[motor]
    return s * RAD_PER_COUNT * (raw - z)


def rad_to_raw(motor: str, q: float,
               zeros: dict[str, int] | None = None,
               signs: dict[str, int] | None = None) -> float:
    """Sim joint angle in radians -> encoder counts."""
    z = (zeros or ZEROS)[motor]
    s = (signs or SIGNS)[motor]
    return s * q / RAD_PER_COUNT + z


class SimModel:
    """Thin wrapper over the MuJoCo model, addressed by real motor names."""

    def __init__(self, path: Path | str = MODEL_XML):
        import mujoco

        self.mj = mujoco
        self.model = mujoco.MjModel.from_xml_path(str(path))
        self.data = mujoco.MjData(self.model)
        self._qadr = {}
        for motor, sim_joint in REAL_TO_SIM.items():
            jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, sim_joint)
            if jid < 0:
                raise KeyError(f"sim joint '{sim_joint}' (for {motor}) not in model")
            self._qadr[motor] = self.model.jnt_qposadr[jid]

    def joint_range(self, motor: str) -> tuple[float, float]:
        jid = self.mj.mj_name2id(self.model, self.mj.mjtObj.mjOBJ_JOINT,
                                 REAL_TO_SIM[motor])
        lo, hi = self.model.jnt_range[jid]
        return float(lo), float(hi)

    def joint_axis(self, motor: str):
        """The joint's rotation axis in WORLD coordinates, at the current pose.

        Reads what MuJoCo computed rather than the XML's `axis` attribute. The
        two differ whenever the joint's body carries a rotation, and taking the
        raw attribute cost this project a badly mirrored head calibration.
        """
        import numpy as np

        jid = self.mj.mj_name2id(self.model, self.mj.mjtObj.mjOBJ_JOINT,
                                 REAL_TO_SIM[motor])
        return np.array(self.data.xaxis[jid], dtype=float)

    def joint_anchor(self, motor: str):
        """The point the joint rotates about, in WORLD coordinates."""
        import numpy as np

        jid = self.mj.mj_name2id(self.model, self.mj.mjtObj.mjOBJ_JOINT,
                                 REAL_TO_SIM[motor])
        return np.array(self.data.xanchor[jid], dtype=float)

    def set_joints(self, q: dict[str, float], reset: bool = True) -> None:
        """Set joint angles by real motor name, then run forward kinematics."""
        if reset:
            self.mj.mj_resetData(self.model, self.data)
        for motor, value in q.items():
            self.data.qpos[self._qadr[motor]] = value
        self.mj.mj_forward(self.model, self.data)

    def body_pose(self, body: str):
        """World pose of a body as (position, 3x3 rotation)."""
        import numpy as np

        bid = self.mj.mj_name2id(self.model, self.mj.mjtObj.mjOBJ_BODY, body)
        if bid < 0:
            raise KeyError(f"body '{body}' not in model")
        return self.data.xpos[bid].copy(), np.array(self.data.xmat[bid]).reshape(3, 3)

    def body_pose_in_chassis(self, body: str):
        """Pose of a body expressed in the chassis frame."""
        import numpy as np

        p_body, R_body = self.body_pose(body)
        p_chassis, R_chassis = self.body_pose("chassis")
        R_rel = R_chassis.T @ R_body
        p_rel = R_chassis.T @ (p_body - p_chassis)
        return p_rel, np.asarray(R_rel, dtype=float)

    def fk(self, q: dict[str, float], body: str):
        self.set_joints(q)
        return self.body_pose(body)

    def summary(self) -> str:
        m = self.model
        return (f"nq={m.nq} nv={m.nv} nbody={m.nbody} njnt={m.njnt} "
                f"ngeom={m.ngeom} nmesh={m.nmesh} nu={m.nu} ncam={m.ncam} nsite={m.nsite}")


def check_consistency() -> list[str]:
    """Confirm the mapping matches the model and the real robot's motor list."""
    problems = []
    try:
        sim = SimModel()
    except Exception as exc:
        return [f"could not load model: {exc}"]

    # Every mapped joint must exist and its limits must be finite.
    for motor in REAL_TO_SIM:
        lo, hi = sim.joint_range(motor)
        if lo == hi == 0.0:
            problems.append(f"{motor} -> {REAL_TO_SIM[motor]}: no joint range in model")

    # The model's own layout: the left arm is bolted toward frames.LEFT. This is
    # a property of the XML, not of how the robot is standing, so it holds for a
    # robot used back-to-front too.
    sim.set_joints({})
    for arm in frames.ARMS:
        tip, _ = sim.body_pose(TIP_BODIES[arm])
        if not frames.is_on_expected_side(arm, tip):
            want = "left" if frames.side_sign(arm) > 0 else "right"
            problems.append(
                f"{arm} tip ({TIP_BODIES[arm]}) sits "
                f"{frames.lateral(tip) * 1000:+.1f} mm toward the robot's left, "
                f"expected clearly to the {want}"
            )

    # Wrist and root bodies must land on the same side as their chain's tip.
    for arm in frames.ARMS:
        for label, table in (("wrist", WRIST_BODIES), ("root", ROOT_BODIES)):
            p, _ = sim.body_pose(table[arm])
            if not frames.is_on_expected_side(arm, p):
                problems.append(
                    f"{arm} {label} ({table[arm]}) is at "
                    f"y={p[1]:+.4f}, wrong side for {arm}"
                )

    # WHEELS corrects the model's swapped wheel names; verify the correction holds.
    for side, body in WHEELS.items():
        p, _ = sim.body_pose(body)
        if frames.side_of(p) != side:
            problems.append(
                f"WHEELS['{side}'] -> {body} is at y={p[1]:+.4f}, wrong side"
            )

    # The head camera mount must protrude forward, the basis for FORWARD_AXIS.
    import numpy as np

    _, r_tilt = sim.body_pose("head_tilt_link")
    if not frames.points_forward(r_tilt @ np.array(HEAD_CAM_MOUNT_XYZ)):
        problems.append(
            "head camera mount no longer points forward; the forward-axis "
            "assumption behind the left/right mapping needs rechecking"
        )

    # Elements build_calib_model.py adds; absent if the visual model is loaded.
    for cam in ("head_camera", "left_wrist_camera", "right_wrist_camera"):
        if sim.mj.mj_name2id(sim.model, sim.mj.mjtObj.mjOBJ_CAMERA, cam) < 0:
            problems.append(f"camera '{cam}' missing; run build_calib_model.py")
    for site in ("left_touch_point", "right_touch_point", "body_frame_B",
                 "left_arm_root_A", "right_arm_root_A"):
        if sim.mj.mj_name2id(sim.model, sim.mj.mjtObj.mjOBJ_SITE, site) < 0:
            problems.append(f"site '{site}' missing; run build_calib_model.py")

    # Wrist cameras must hang off the matching arm, despite the URDF's swapped names.
    for side in ("left", "right"):
        cid = sim.mj.mj_name2id(sim.model, sim.mj.mjtObj.mjOBJ_CAMERA,
                                f"{side}_wrist_camera")
        if cid < 0:
            continue
        parent = sim.mj.mj_id2name(sim.model, sim.mj.mjtObj.mjOBJ_BODY,
                                   sim.model.cam_bodyid[cid])
        want = WRIST_BODIES[f"{side}_arm"]
        if parent != want:
            problems.append(
                f"{side}_wrist_camera hangs off '{parent}', expected '{want}'"
            )
    return problems


if __name__ == "__main__":
    sim = SimModel()
    print(f"model: {MODEL_XML}")
    print(f"  {sim.summary()}\n")

    print(f"  {'REAL MOTOR':<26} {'SIM JOINT':<18} {'RANGE (rad)':<22} DEG")
    for motor, sim_joint in REAL_TO_SIM.items():
        lo, hi = sim.joint_range(motor)
        print(f"  {motor:<26} {sim_joint:<18} [{lo:+.4f}, {hi:+.4f}]      "
              f"{math.degrees(hi - lo):7.1f}")

    sim.set_joints({})
    print("\n  chain tips at zero pose (robot frame: -x forward, -y left):")
    for role, body in TIP_BODIES.items():
        p, _ = sim.body_pose(body)
        print(f"    {role:<10} {body:<16} [{p[0]:+.4f} {p[1]:+.4f} {p[2]:+.4f}]  "
              f"{frames.side_of(p)}")

    problems = check_consistency()
    print("\nOK: mapping is consistent with the model" if not problems
          else "PROBLEMS:\n  " + "\n  ".join(problems))
