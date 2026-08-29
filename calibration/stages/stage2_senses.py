"""Stage 2: joint senses.

Asks the operator to move each joint in a named direction, reads raw counts
before and after, and records the sign. Nothing is driven; the arms stay
backdrivable and torque stays off throughout.

Why this is a stage of its own, ahead of everything solved
---------------------------------------------------------
A servo's positive direction is a wiring and assembly fact that the model file
cannot know, and no residual can recover it. On this robot both head pan senses
fit the capture to 3.56mm; the wrong one placed the world board 1515mm above the
floor against a measured 750mm and had the camera looking up at a board that was
plainly below it. Well conditioned, low residual, every gate green, and mirrored.

Fourteen joints is 2^14 combinations, so trying them is not an option either.
The only cure is to measure each one against a direction a person can name.

The direction words are generated from the model's own joint axes rather than
typed out, so they cannot drift away from the XML.
"""

from __future__ import annotations

import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

import common
import frames
from core import senses as senses_mod
from core import servos, storage

# The direction of each joint's POSITIVE rotation, in the words an operator can
# act on. Written out per joint rather than derived, because a derived phrase
# names whichever world axis the motion happens to be largest along, and that is
# not always the motion a person notices. Head tilt is the example: a positive
# tilt does move the camera forward, but what you see is it looking *down*, and
# "forward" sends people the wrong way.
#
# Every entry below was checked against the model, by rotating the joint 15
# degrees positive and watching where the far end went. `verify_directions()`
# re-runs that check, so these cannot silently drift from the XML.
POSITIVE = {
    "left_arm_shoulder_pan": "the whole arm swings CLOCKWISE, seen from above",
    "right_arm_shoulder_pan": "the whole arm swings CLOCKWISE, seen from above",
    "left_arm_shoulder_lift": "the arm swings UP",
    "right_arm_shoulder_lift": "the arm swings UP",
    "left_arm_elbow_flex": "the forearm swings DOWN",
    "right_arm_elbow_flex": "the forearm swings DOWN",
    "left_arm_wrist_flex": "the gripper swings DOWN",
    "right_arm_wrist_flex": "the gripper swings DOWN",
    "left_arm_wrist_roll":
        "the gripper rolls CLOCKWISE, looking along the arm towards its tip",
    "right_arm_wrist_roll":
        "the gripper rolls CLOCKWISE, looking along the arm towards its tip",
    "left_arm_gripper": "the jaws OPEN",
    "right_arm_gripper": "the jaws OPEN",
    "head_motor_1": "the head turns ANTICLOCKWISE, seen from above",
    "head_motor_2": "the head looks DOWN",
}

# How each stated direction is checked against the model: rotate the joint a
# little in its positive direction and assert something about the result. The
# predicate receives the joint's world axis and how the watched body moved, in
# millimetres, both in the robot frame (-x forward, -y left, +z up).
#
# Rotation senses are read off the axis, not the displacement: "clockwise from
# above" is a statement about the axis pointing down, which holds wherever the
# arm happens to be posed.
CHECKS = {
    "shoulder_pan": ("Moving_Jaw", lambda ax, dp: ax[2] < -0.9),
    "shoulder_lift": ("Moving_Jaw", lambda ax, dp: dp[2] > 5.0),
    "elbow_flex": ("Moving_Jaw", lambda ax, dp: dp[2] < -5.0),
    "wrist_flex": ("Moving_Jaw", lambda ax, dp: dp[2] < -5.0),
    "head_motor_1": ("head_tilt_link", lambda ax, dp: ax[2] > 0.9),
}

# Two joints cannot be judged from how their watched body's origin moves, because
# that origin sits on the joint's own hinge. They get their own probes.
#
#  - a gripper's jaws separating is a distance between two bodies, not a
#    displacement of one, and only a point out along the jaw blade shows it.
#  - the head camera is offset from the tilt joint, so "looks down" has to be
#    measured at the camera rather than at the link origin.
JAW_BLADE_M = np.array([0.0, -0.05, 0.0])
HEAD_CAM_M = np.array([0.025, 0.0, 0.03])


def gripper_opens(sim, motor: str, probe_deg: float) -> tuple[bool, str]:
    """Does a positive rotation separate the jaws?"""
    moving = "Moving_Jaw_2" if "right" in motor else "Moving_Jaw"
    fixed = "Fixed_Jaw_2" if "right" in motor else "Fixed_Jaw"
    gaps = []
    for angle in (0.0, probe_deg):
        sim.set_joints({motor: np.deg2rad(angle)})
        p, R = sim.body_pose(moving)
        f, _ = sim.body_pose(fixed)
        gaps.append(np.linalg.norm((p + R @ JAW_BLADE_M) - f) * 1000.0)
    return gaps[1] > gaps[0] + 0.5, f"jaw gap {gaps[0]:.1f} -> {gaps[1]:.1f} mm"


def roll_is_clockwise(sim, motor: str, probe_deg: float) -> tuple[bool, str]:
    """Is a positive roll clockwise, looking along the arm towards its tip?

    Clockwise from that viewpoint means the rotation axis points away from the
    viewer, i.e. along the arm's outward direction. Comparing the two directly
    is what makes this a real check: `abs(axis[1]) > 0.9` would accept either
    sense, since the two arms mirror and their roll axes point opposite ways.
    """
    side = "right" if "right" in motor else "left"
    root = "Rotation_Pitch_2" if side == "right" else "Rotation_Pitch"
    tip = "Moving_Jaw_2" if side == "right" else "Moving_Jaw"
    sim.set_joints({})
    axis = sim.joint_axis(motor)
    a, _ = sim.body_pose(root)
    b, _ = sim.body_pose(tip)
    outward = (b - a) / np.linalg.norm(b - a)
    dot = float(axis @ outward)
    return dot > 0.5, (f"axis . outward = {dot:+.3f}, so a positive roll is "
                       f"{'clockwise' if dot > 0 else 'anticlockwise'} "
                       f"towards the tip")


def head_looks_down(sim, motor: str, probe_deg: float) -> tuple[bool, str]:
    """Does a positive tilt lower the camera and drop its optical axis?"""
    heights, axes = [], []
    for angle in (0.0, probe_deg):
        sim.set_joints({motor: np.deg2rad(angle)})
        p, R = sim.body_pose("head_tilt_link")
        heights.append((p + R @ HEAD_CAM_M)[2] * 1000.0)
        axes.append((R @ np.array([0.0, 0.0, 1.0]))[2])
    ok = heights[1] < heights[0] - 0.5 and axes[1] < axes[0]
    return ok, (f"camera {heights[0]:.1f} -> {heights[1]:.1f} mm, "
                f"axis z {axes[0]:+.3f} -> {axes[1]:+.3f}")


SPECIAL = {
    "wrist_roll": roll_is_clockwise,
    "gripper": gripper_opens,
    "head_motor_2": head_looks_down,
}


def _watched_body(motor: str) -> str:
    for key, (body, _) in CHECKS.items():
        if key in motor:
            return "Moving_Jaw_2" if ("right" in motor
                                      and body == "Moving_Jaw") else body
    return "Moving_Jaw_2" if "right" in motor else "Moving_Jaw"


def check_direction(sim, motor: str,
                    probe_deg: float = 15.0) -> tuple[bool, str]:
    """Does the model agree with what POSITIVE claims for this joint?"""
    for key, probe in SPECIAL.items():
        if key in motor:
            return probe(sim, motor, probe_deg)

    for key, (body, predicate) in CHECKS.items():
        if key in motor:
            body = _watched_body(motor)
            sim.set_joints({})
            axis = sim.joint_axis(motor)
            before, _ = sim.body_pose(body)
            sim.set_joints({motor: np.deg2rad(probe_deg)})
            after, _ = sim.body_pose(body)
            moved = (after - before) * 1000.0
            return bool(predicate(axis, moved)), (
                f"axis {np.round(axis, 2).tolist()}, {body} moves "
                f"{np.round(moved, 1).tolist()} mm")

    raise KeyError(f"no direction check defined for {motor}")


def verify_directions(sim, probe_deg: float = 15.0) -> list[str]:
    """Confirm every stated direction still matches the model.

    The words in POSITIVE are what the operator acts on, so if the model changes
    under them the whole stage records mirrored signs while looking healthy. This
    turns that into a startup failure instead.
    """
    problems = []
    for motor in senses_mod.JOINTS:
        ok, evidence = check_direction(sim, motor, probe_deg)
        if not ok:
            problems.append(f"{motor}: POSITIVE says {POSITIVE[motor]!r}, "
                            f"but the model shows {evidence}")
    return problems


def build_prompts(sim) -> dict[str, dict]:
    """One instruction per joint, with the model evidence behind it.

    The words come from POSITIVE; the axis and displacement are recorded next to
    them so a doubtful prompt can be traced back to the model without re-deriving
    anything.
    """
    out: dict[str, dict] = {}
    for motor in senses_mod.JOINTS:
        sim.set_joints({})
        axis = sim.joint_axis(motor)
        _, evidence = check_direction(sim, motor)
        out[motor] = {
            "motor": motor,
            "direction": POSITIVE[motor],
            "axis": [round(float(v), 4) for v in axis],
            "model_evidence": evidence,
        }
    return out


def counts_to_deg(counts: int) -> float:
    return abs(counts) * 360.0 / servos.COUNTS_PER_TURN


class MotionTracker:
    """Continuously unwrap a demonstrated move while the prompt blocks on input."""

    def __init__(self, robot, motor: str, period: float = 0.02):
        self.robot = robot
        self.motor = motor
        self.period = period
        self.start_raw: int | None = None
        self.last_raw: int | None = None
        self.position = 0
        self.error: str | None = None
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def __enter__(self):
        raw = self.robot.read_raw(self.motor)
        if raw is None:
            self.error = f"cannot read {self.motor}"
            return self
        self.start_raw = self.last_raw = int(raw)
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        return self

    def _loop(self):
        while not self._stop.is_set():
            raw = self.robot.read_raw(self.motor)
            if raw is not None and self.last_raw is not None:
                raw = int(raw)
                self.position += servos.unwrap_delta(raw - self.last_raw)
                self.last_raw = raw
            time.sleep(self.period)

    def __exit__(self, *exc):
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)


def measure_joint(robot, motor: str, prompt: dict) -> senses_mod.Sense | None:
    """Read one joint's sense from a motion the operator performs.

    The operator marks a start and an end position explicitly and the sign comes
    from the difference. That matters when a joint is already against the stop it
    would have to move towards: rather than failing, they park it at the other
    extreme first, then travel the full range in the named direction.

    Returns None if the operator skips it, and refuses a move too small to be
    distinguished from encoder noise rather than recording a coin flip.
    """
    # The stored name is printed on its own line because the dashboard parses
    # this line to know which joint is being measured -- that is the only place
    # it learns it. The arm the operator can point at goes on the line below:
    # used back-to-front the flanges are turned, so the joint the model calls
    # left_arm_* is on their right, and the stored name alone would leave them
    # moving an arm that never responds.
    mounting = frames.declared_mounting()
    print(f"\n  {motor}")
    spoken = frames.spoken_joint(motor, mounting)
    if spoken != motor:
        print(f"    This is the {spoken}.")
    print(f"    Positive direction: {prompt['direction']}.")
    print(f"    If the joint is already at that end of its travel, move it to "
          f"the OTHER end first, then mark the start there.")

    while True:
        print("    Put the joint at the START position, then press Enter "
              "('s' to skip).")
        if common.ask("", "").strip().lower() == "s":
            return None
        servos.wait_until_still(robot, [motor])
        before = robot.read_raw(motor)
        if before is None:
            print("    Cannot read this servo. Check the cable.")
            if not common.confirm("retry", True):
                return None
            continue
        print(f"      start: {before} counts")

        print(f"    Now move it so that {prompt['direction']}, as far as it "
              f"comfortably goes. Press Enter when there.")
        with MotionTracker(robot, motor) as tracker:
            if tracker.error:
                print(f"    {tracker.error}. Starting this joint again.")
                continue
            if common.ask("", "").strip().lower() == "s":
                return None
        servos.wait_until_still(robot, [motor])
        after = robot.read_raw(motor)
        if after is None:
            print("    Lost the servo mid-move. Starting this joint again.")
            continue
        print(f"      end:   {after} counts")

        # Use accumulated motion, not endpoint shortest-arc differencing. Wrist
        # roll can legitimately move beyond 180 degrees during this demonstration.
        travel = int(tracker.position)
        if abs(travel) < senses_mod.MIN_TRAVEL_COUNTS:
            print(f"    Only {abs(travel)} counts ({counts_to_deg(travel):.1f} "
                  f"deg) between start and end. That is too close to encoder "
                  f"noise to tell the direction apart.")
            print(f"    Move at least {senses_mod.MIN_TRAVEL_COUNTS} counts "
                  f"({counts_to_deg(senses_mod.MIN_TRAVEL_COUNTS):.0f} deg). If "
                  f"the joint cannot travel that far in this direction, it was "
                  f"already at the stop: start from the other end.")
            continue

        sign = 1 if travel > 0 else -1
        print(f"    moved {travel:+d} counts ({counts_to_deg(travel):.1f} deg) "
              f"-> sense {sign:+d}")
        return senses_mod.Sense(
            name=motor, sign=sign, raw_before=before, raw_after=after,
            travel_counts=travel, prompt=prompt["direction"])


def report(got: senses_mod.SenseSet, prompts: dict) -> None:
    common.heading("Recorded senses")
    print(f"  {'joint':<26} {'sense':>6} {'travel':>8}  positive direction")
    for motor in senses_mod.JOINTS:
        s = got.senses.get(motor)
        if s is None:
            print(f"  {motor:<26} {'--':>6} {'':>8}  not recorded")
            continue
        travel = f"{s.travel_counts:+d}" if s.travel_counts is not None else "?"
        print(f"  {motor:<26} {s.sign:+6d} {travel:>8}  {s.prompt}")

    flipped = [n for n, s in got.senses.items() if s.sign < 0]
    print(f"\n  {len(flipped)} of {len(got.senses)} joints count against the "
          f"model's axis.")
    if flipped:
        print("  " + ", ".join(flipped))
    print("\n  A mix is normal; it reflects how the servos were wired and")
    print("  assembled, not a mistake.")


def check(got: senses_mod.SenseSet) -> list:
    """Acceptance: every joint measured, and measured convincingly."""
    from core import gates as gates_mod

    out = [gates_mod.GateResult(
        name="joints recorded", passed=got.complete,
        value=float(len(got.senses)), threshold=float(len(senses_mod.JOINTS)),
        direction="min", unit="joints",
        detail="" if got.complete else "missing: " + ", ".join(got.missing))]

    weak = got.weak
    out.append(gates_mod.GateResult(
        name="travel per joint", passed=not weak,
        value=float(len(weak)), threshold=0.0, direction="max",
        unit="joints below the minimum",
        detail="" if not weak else
        "re-do these with a bigger move: " + ", ".join(weak)))
    return out


def main() -> int:
    import model_map

    common.heading("Stage 2: joint senses")
    print("  Which way does each servo turn its joint, relative to the model's")
    print("  own axis? This cannot be inferred later: a wrong sense fits the")
    print("  data just as well as the right one, and mirrors the result.")
    print("\n  Nothing is driven. Torque stays off; you move each joint by hand.")
    if frames.declared_mounting() == frames.FLIPPED:
        print("\n  This robot is declared back-to-front, so the flanges are")
        print("  turned: each joint below is named by the arm you can see it on,")
        print("  with the name the saved results use in brackets.")

    try:
        if not common.confirm_overwrite("senses"):
            return 1
    except common.Aborted:
        return 1

    sim = model_map.SimModel()
    wrong = verify_directions(sim)
    if wrong:
        print("\n  The stated directions no longer match the model:")
        for w in wrong:
            print(f"    - {w}")
        print("\n  Stopping. These words are what you act on, so if they are")
        print("  stale every sense recorded here would be mirrored. Fix")
        print("  POSITIVE in this file to match the model first.")
        return 1
    prompts = build_prompts(sim)

    try:
        robot = servos.RawRobot()
    except Exception as exc:
        print(f"\n  Cannot reach the servos: {exc}")
        print("  Check the USB adapters and that no other process holds them.")
        return 1

    with robot:
        problems = robot.verify()
        if problems:
            print("\n  Servo bus problems:")
            for p in problems:
                print(f"    - {p}")
            if not common.confirm("continue anyway", False):
                return 1

        got = senses_mod.SenseSet()
        mounting = frames.declared_mounting()
        print("\n  Fourteen joints. Press 's' to skip one and come back to it.")
        # Left arm first, by the operator's left. Back-to-front that is the arm
        # the model calls right_arm, so the order is taken from the mounting
        # rather than from the stored order.
        for motor in frames.in_working_order(senses_mod.JOINTS, mounting):
            sense = measure_joint(robot, motor, prompts[motor])
            if sense is not None:
                got.record(sense)

        while got.missing:
            missing = frames.in_working_order(got.missing, mounting)
            print("\n  Still unrecorded: "
                  + ", ".join(frames.spoken_joint(m, mounting)
                              for m in missing))
            if not common.confirm("go through them again", True):
                break
            for motor in missing:
                sense = measure_joint(robot, motor, prompts[motor])
                if sense is not None:
                    got.record(sense)

    report(got, prompts)
    checks = check(got)
    if not common.report_gates(checks):
        print("\n  Not saved. Re-run this stage once the gaps are filled.")
        return 1

    payload = got.to_dict()
    payload["prompts"] = prompts
    storage.save_result("senses", payload)
    print("\n  Saved. Next: python calibration/run.py --stage 3")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
