"""Stage 4: rough arm zeros and measured joint travel.

Two jobs, and it is worth being clear that only one of them is about accuracy.

The zeros are deliberately rough
--------------------------------
Stage 5 solves the real zeros from contact data. Measured on synthetic contacts
with 1mm of touch placement error and 0.1 degrees of joint noise, an initial zero
guess wrong by 90 degrees converges to within 0.23 degrees of truth -- the same
answer an exact guess gives. So there is no point being careful here, and the
prompt says so, because an operator who believes this pose must be precise will
spend twenty minutes achieving nothing.

The ranges are not rough
------------------------
What the zero pose is actually for is the encoder wrap. These are single-turn
absolute encoders and several of these joints travel more than half a turn, so a
range measured as two endpoint readings can come out as the short way round: wrist
roll's 320 degrees reported as 40. Posing near zero first puts every extreme within
half a turn of the start, and `core/ranges.py` accumulates the steps in between, so
the travel is recovered even past a full turn.

A range that fills the whole 0-4095 span is a legitimate result for a flexible
joint, not a fault, and the report says as much rather than flagging it.

One arm at a time, by hand, torque off throughout. Holding both arms at a pose
simultaneously is a two-hand job with no hands left for the keyboard. Within an
arm, though, every joint is tracked at once: each has its own accumulator, and a
live table shows which ones still fall short, so there is no reason to make the
operator work through them in a fixed order.
"""

from __future__ import annotations

import json
import sys
import threading
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import common  # noqa: E402
import frames  # noqa: E402
from core import arm_model, gates, ranges, senses as senses_mod  # noqa: E402
from core import servos, storage, zeros as zeros_mod  # noqa: E402

ARMS = ("left_arm", "right_arm")

# Every joint of an arm, gripper included. For the gripper, Stage 4's manually
# closed posture is the zero; the standard encoder scale maps counts to angle.
def arm_joints(arm: str) -> list[str]:
    from model_map import ARM_JOINTS_NO_GRIPPER
    return list(ARM_JOINTS_NO_GRIPPER[arm]) + [f"{arm}_gripper"]


def name_column(motors, mounting: str) -> int:
    """Width of the joint-name column for the tables the operator reads.

    Back-to-front the spoken names name the arm the operator can see and run
    wider than the plain ones, so a fixed width would push every later column
    out of line. The plain width is the floor so normal runs are unchanged.
    """
    return max([28] + [len(frames.spoken_joint(m, mounting)) for m in motors])


# How the model's zero pose looks, in words. The operator cannot read a quaternion
# and should not have to; these describe the pose the model's q=0 puts the arm in.
#
# Checked against model/xlerobot_calib.xml at q=0: every link of the arm has a
# chassis X component of 0, so the whole arm lies in the sideways plane. It does
# NOT point forward. Left arm link directions at zero, in chassis XYZ:
#   Rotation_Pitch -> Upper_Arm        [ 0, -0.286,  0.958]  107 mm, rising
#   Upper_Arm      -> Lower_Arm        [ 0, -0.970, -0.241]  116 mm, outward
#   Lower_Arm      -> Wrist_Pitch_Roll [ 0, -0.999,  0.039]  135 mm, outward
#   Wrist_Pitch_Roll -> Fixed_Jaw      [ 0, -1.000,  0.000]   60 mm, outward
# The jaw hinge (Jaw_L / Jaw_R) runs along chassis X at zero, so the jaws swing
# open in the vertical plane that contains the forearm.
ZERO_POSE = (
    "shoulder pan   : arm reaching straight out to the side, level with the "
    "shoulder, not swung forward or back",
    "shoulder lift  : upper arm angled slightly down from the shoulder",
    "elbow flex     : forearm straight out to the side, very nearly horizontal",
    "wrist flex     : wrist straight, gripper in line with the forearm",
    "wrist roll     : the hinge pin of the jaws pointing forward, so the jaws "
    "open one up and one down",
    "gripper        : jaws closed, just touching",
)

# Polling interval while the operator sweeps a joint. Fast enough that no joint can
# cover half a turn between samples by hand, which is what keeps the unwrap honest.
POLL_SECONDS = 0.04

# A joint must cover at least this share of the range the model allows before its
# measured travel is trusted as a bound.
RANGE_FRACTION = 0.5

# Where the live sweep snapshot is published for a supervising UI to read. The
# operator sweeping six joints at once cannot also read a terminal, so the
# numbers that decide whether a sweep is finished have to be visible while it is
# still happening. Written under data/ because it is transient progress, not a
# stage result.
LIVE_FILE = "arm_range_live.json"


def live_path() -> Path:
    path = Path(storage.DATA_DIR) / LIVE_FILE
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def publish_live(payload: dict | None) -> None:
    """Replace the live snapshot atomically, or clear it when nothing is live.

    A reader polling this file must never see a half-written table, so the
    write goes to a temporary file and is renamed into place.
    """
    path = live_path()
    try:
        if payload is None:
            path.unlink(missing_ok=True)
            return
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(payload, default=storage.json_default))
        tmp.replace(path)
    except OSError:
        # The snapshot is a convenience for the UI. Losing it must not
        # interrupt a sweep the operator is in the middle of.
        pass


class Tracker:
    """Polls a group of servos in the background while the operator moves them.

    `input()` blocks, so the sampling that makes wrap-aware tracking work cannot
    live on the main thread. A daemon thread reads continuously and folds each
    reading into the travel accumulator.
    """

    def __init__(self, robot, names: list[str], arm: str = "",
                 allowed_deg: dict[str, float] | None = None,
                 publish: bool = False):
        self.robot = robot
        self.names = list(names)
        self.arm = arm
        self.allowed_deg = dict(allowed_deg or {})
        self.publish = publish
        self.set = ranges.RangeSet()
        self.live: dict[str, int] = {}
        self.error: str | None = None
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

        # The sweep is thousands of encoder readings, and the result keeps
        # about a dozen numbers per joint: the extremes, the span, and whether
        # the joint crossed the 0/4095 seam. Everything downstream rests on
        # that last flag -- RangeAngleTracker uses it to pick which branch of
        # the encoder a reading belongs to -- and it is decided by continuity
        # between consecutive samples. If the sweep is jerky, or a read is
        # dropped, or the operator moves faster than the poll, a genuine wrap
        # can be missed or a false one invented, and the summary looks
        # identical either way. Keeping the raw sequence is what makes that
        # answerable without another hour of sweeping. It costs a few hundred
        # KB; there are no images in this stage.
        self._log = None
        self._log_lock = threading.Lock()
        self._log_t0 = 0.0
        self._session = None

    def _seed(self) -> bool:
        for name in self.names:
            raw = self.robot.read_raw(name)
            if raw is None:
                self.error = f"cannot read {name}"
                return False
            self.set.begin(name, raw)
            self.live[name] = raw
        return True

    def snapshot(self) -> dict:
        """The current travel of every tracked joint, for display.

        Every position here is a continuous count accumulated by `Travel`, not a
        difference of raw readings, so a joint sitting across the 0/4095 seam
        reports the travel it actually made rather than the short way round.
        """
        joints = []
        for name in self.names:
            travel = self.set.travels.get(name)
            if travel is None:
                continue
            allowed = self.allowed_deg.get(name)
            joints.append({
                "joint": name,
                "raw": self.live.get(name),
                "start_raw": int(travel.start_raw),
                "position_deg": counts_to_deg(travel.position),
                "min_deg": counts_to_deg(travel.lowest),
                "max_deg": counts_to_deg(travel.highest),
                "raw_min": travel.raw_at(travel.lowest),
                "raw_max": travel.raw_at(travel.highest),
                "span_deg": travel.span_deg,
                "span_counts": travel.span_counts,
                "allowed_deg": allowed,
                "enough": (allowed is None
                           or travel.span_deg >= RANGE_FRACTION * allowed),
                "over_one_turn": travel.span_counts >= servos.COUNTS_PER_TURN,
                "wrapped": bool(travel.wrapped),
                "samples": int(travel.steps),
            })
        return {"arm": self.arm, "joints": joints, "updated_at": time.time()}

    def _open_log(self) -> None:
        """Start the raw sample log for this sweep, if a workspace exists."""
        if not self.arm:
            return
        try:
            path = storage.session_path("stage4_sweep", self.arm)
            storage.archive_session(path)
            # A session.json is written even though this stage stores no
            # frames, because archive_session() keys off it: without one, a
            # second sweep silently overwrites the first rather than being
            # kept beside it.
            session = storage.CaptureSession(path, storage.SessionMeta(
                stage="4", purpose="continuous joint travel sweep",
                notes={"arm": self.arm, "joints": self.names,
                       "poll_seconds": POLL_SECONDS,
                       "counts_per_turn": servos.COUNTS_PER_TURN,
                       "seed_raw": dict(self.live),
                       "samples": "samples.jsonl"}))
            self._log = (path / "samples.jsonl").open("w")
            self._log.write(json.dumps({
                "record": "header",
                "arm": self.arm,
                "joints": self.names,
                "poll_seconds": POLL_SECONDS,
                "counts_per_turn": servos.COUNTS_PER_TURN,
                "started_at": datetime.now().isoformat(timespec="seconds"),
                "seed_raw": dict(self.live),
                "note": "One line per poll: t is seconds since the sweep "
                        "started, raw is the encoder count per joint as read. "
                        "The travel summary in zeros.json is derived from "
                        "exactly this sequence.",
            }) + "\n")
            self._log_t0 = time.monotonic()
            self._session = session
        except Exception:
            # A sweep that cannot write its log is still a valid sweep. The
            # operator is at the robot; failing here would cost them the run
            # for the sake of a diagnostic.
            self._log = None

    def _record(self, raw: dict[str, int]) -> None:
        if self._log is None or not raw:
            return
        try:
            with self._log_lock:
                self._log.write(json.dumps({
                    "t": round(time.monotonic() - self._log_t0, 3),
                    "raw": raw,
                }) + "\n")
        except Exception:
            self._log = None

    def _close_log(self) -> None:
        if self._log is None:
            return
        try:
            with self._log_lock:
                self._log.write(json.dumps({
                    "record": "summary",
                    "ended_at": datetime.now().isoformat(timespec="seconds"),
                    "travels": {n: t.to_dict()
                                for n, t in self.set.travels.items()},
                }) + "\n")
                self._log.close()
        except Exception:
            pass
        finally:
            self._log = None
        if self._session is not None:
            try:
                self._session.finish(
                    swept=True,
                    samples={n: int(t.steps)
                             for n, t in self.set.travels.items()})
            except Exception:
                pass

    def _loop(self) -> None:
        while not self._stop.is_set():
            sample = {}
            for name in self.names:
                raw = self.robot.read_raw(name)
                if raw is not None:
                    self.set.update({name: raw})
                    self.live[name] = raw
                    sample[name] = int(raw)
            self._record(sample)
            if self.publish:
                publish_live(self.snapshot())
            time.sleep(POLL_SECONDS)

    def __enter__(self) -> "Tracker":
        if self._seed():
            self._open_log()
            if self.publish:
                publish_live(self.snapshot())
            self._thread = threading.Thread(target=self._loop, daemon=True)
            self._thread.start()
        return self

    def __exit__(self, *exc) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        self._close_log()
        if self.publish:
            publish_live(None)

    def travel(self, name: str) -> ranges.Travel | None:
        return self.set.travels.get(name)


def counts_to_deg(counts: float) -> float:
    return counts * 360.0 / servos.COUNTS_PER_TURN


def model_span_deg(sim, motor: str) -> float:
    import numpy as np

    lo, hi = sim.joint_range(motor)
    if hi <= lo:
        return 360.0
    return float(np.rad2deg(hi - lo))


def pose_zero(robot, arm: str, got_senses) -> dict[str, int] | None:
    """Have the operator pose one arm near the model's zero, and read it.

    Returns raw counts per joint, or None if the operator gave up on this arm.
    """
    # Name the arm the operator can point at. Used back-to-front the flanges are
    # turned, so the arm the model calls left_arm is the one on their right, and
    # posing by the model's name would put every zero half a turn out.
    mounting = frames.declared_mounting()
    side = frames.physical_side(arm, mounting)
    common.heading(f"{side} arm: rough zero pose")
    if mounting == frames.FLIPPED:
        print(f"  This is the arm on your {side} as you face the robot's")
        print(f"  working side. The saved results call it {arm}, because the")
        print("  flange is turned; that is expected, not a mix-up.")
    print("  Pose this arm by hand to roughly this shape:\n")
    for line in ZERO_POSE:
        print(f"    {line}")
    print("\n  Match this pose carefully, preferably within about 10 degrees per joint.")
    print("  Stage 5 Fusion refines these zeros within a guarded ±30 degree window.")
    print("  A link pointing in the opposite direction is not the same zero pose,")
    print("  even if the adjoining links still look collinear.")
    print("\n  Torque is off; the arm stays where you put it only if it is")
    print("  balanced, so support it while you read the counts.")
    print("  Leave the other arm alone for now.")

    while True:
        if not common.confirm(f"is the {side} arm posed", True):
            if common.confirm("skip this arm entirely", False):
                return None
            continue
        names = arm_joints(arm)
        servos.wait_until_still(robot, names)
        readings = {n: robot.read_raw(n) for n in names}
        missing = [n for n, v in readings.items() if v is None]
        if missing:
            print(f"\n  No reading from: {', '.join(missing)}")
            print("  Check the bus cable and the servo IDs, then try again.")
            if not common.confirm("retry", True):
                return None
            continue

        # The stored name stays in the first column: the dashboard parses these
        # rows to mirror the table live, and it renames them for the operator
        # itself. The heading above already says which arm this is.
        print(f"\n  {'joint':<28} {'raw':>6}")
        for n in names:
            print(f"  {n:<28} {readings[n]:>6}")
        if common.confirm("accept these as the rough zero", True):
            return {n: int(v) for n, v in readings.items()}


def sweep_arm(robot, sim, arm: str, zero_raw: dict[str, int]) -> ranges.RangeSet:
    """Measure the travel of every joint on one arm, in a single live sweep.

    All joints are tracked at once rather than one at a time. Each joint has its
    own independent accumulator, so moving several together costs nothing, and
    the live table shows which joints still fall short -- which is the only thing
    a per-joint prompt was ever providing, minus the need to memorise an order.

    Sweeping continues on the *same* tracker when a joint falls short. Restarting
    would discard the travel already accumulated and, worse, move the start away
    from the pose whose counts were recorded as the zero.
    """
    names = arm_joints(arm)
    allowed = {motor: model_span_deg(sim, motor) for motor in names}
    out = ranges.RangeSet()
    mounting = frames.declared_mounting()
    side = frames.physical_side(arm, mounting)
    common.heading(f"{side} arm: joint travel")
    print("  Sweep every joint on this arm, in any order, as many at a time as")
    print("  you like. For each one: move it to one extreme, then the other,")
    print("  then back near where it started.")
    print("\n  The table on screen updates as you move, and shows each joint's")
    print("  current position, the minimum and maximum it has reached, and")
    print("  whether it has covered enough. Travel is accumulated continuously,")
    print("  so crossing the 0/4095 encoder seam is measured correctly.")
    print("\n  Move steadily. Stop at the mechanical stop, or at whatever limit")
    print("  the wiring or a self-collision imposes. Do not force anything.")

    with Tracker(robot, names, arm=arm, allowed_deg=allowed,
                 publish=True) as tracker:
        if tracker.error:
            print(f"    {tracker.error}. Check the cable.")
            return out
        while True:
            print("\n    Sweep the joints now. Press Enter when every joint is "
                  "done and back near its start ('s' to stop here).")
            answer = common.ask("", "").strip().lower()
            short = [n for n in names
                     if (tracker.travel(n) is None
                         or not ranges.sense_agrees(tracker.travel(n),
                                                    allowed[n], RANGE_FRACTION))]
            if answer == "s" or not short:
                break
            print("\n    These joints have not covered half the range the model "
                  "allows:")
            sw = name_column(short, mounting)
            for name in short:
                travel = tracker.travel(name)
                span = 0.0 if travel is None else travel.span_deg
                print(f"      {frames.spoken_joint(name, mounting):<{sw}} "
                      f"{span:>6.0f} of {allowed[name]:>4.0f} deg")
            print("    Either they are obstructed, or the sweep stopped short.")
            if not common.confirm("keep sweeping this arm", True):
                break

        for motor in names:
            travel = tracker.travel(motor)
            if travel is None:
                continue
            out.travels[motor] = travel
            out.zero_raw[motor] = int(zero_raw.get(motor, travel.start_raw))

    w = name_column(names, mounting)
    print(f"\n  {'joint':<{w}} {'travel':>9}  raw span")
    for motor in names:
        spoken = frames.spoken_joint(motor, mounting)
        travel = out.travels.get(motor)
        if travel is None:
            print(f"  {spoken:<{w}} {'--':>9}  not measured")
            continue
        print(f"  {spoken:<{w}} {travel.span_deg:>8.1f}d  "
              f"{travel.raw_at(travel.lowest)}..{travel.raw_at(travel.highest)}"
              f"{', crossed the 0/4095 seam' if travel.wrapped else ''}")
        if travel.span_counts >= servos.COUNTS_PER_TURN:
            print("    This sweep covers a full encoder turn, so a later raw "
                  "reading cannot be mapped to a unique joint angle. Reduce "
                  "the sweep to the joint's actual mechanical limits.")

        # The sweep should have begun at the pose whose counts were recorded as
        # the zero. If it did not, the joint drifted between the two steps and
        # the zero no longer describes the arm the ranges were measured on.
        drift = abs(servos.unwrap_delta(travel.start_raw - out.zero_raw[motor]))
        if drift > zeros_mod.PAIRING_TOLERANCE_COUNTS:
            print(f"    Note: swept from {travel.start_raw}, but the zero was "
                  f"recorded at {out.zero_raw[motor]} ({drift} counts apart). "
                  f"The joint moved in between; harmless for a rough zero.")
    return out


def report(rough: dict[str, dict[str, int]], measured: dict[str, ranges.RangeSet],
           sim) -> None:
    common.heading("Rough zeros and measured travel")
    mounting = frames.declared_mounting()
    shown = [m for arm in ARMS if arm in rough for m in arm_joints(arm)]
    w = name_column(shown, mounting)
    print(f"  {'joint':<{w}} {'zero':>6} {'travel':>10} {'model':>8}  raw span")
    for arm in frames.working_order(mounting):
        if arm not in rough:
            continue
        for motor in arm_joints(arm):
            name = frames.spoken_joint(motor, mounting)
            zero = rough[arm].get(motor)
            travel = measured.get(arm, ranges.RangeSet()).travels.get(motor)
            allowed = model_span_deg(sim, motor)
            if travel is None:
                print(f"  {name:<{w}} {zero if zero is not None else '--':>6} "
                      f"{'--':>10} {allowed:>7.0f}d  not measured")
                continue
            seam = " (wraps)" if travel.wrapped else ""
            print(f"  {name:<{w}} {zero:>6} {travel.span_deg:>9.0f}d "
                  f"{allowed:>7.0f}d  "
                  f"{travel.raw_at(travel.lowest)}..{travel.raw_at(travel.highest)}"
                  f"{seam}")

    print("\n  The zeros above are starting guesses, nothing more. Stage 5 replaces")
    print("  them with solved values; it needs them only to start in the right")
    print("  basin, and tolerates being 90 degrees out.")
    print("\n  Gauge convention recorded with these zeros: the arm mount's yaw is")
    print("  held at zero, because rotating shoulder pan and yawing the whole arm")
    print("  are the same motion as far as contact data can tell. Shoulder pan's")
    print("  solved zero absorbs the mount yaw.")


def check(rough: dict[str, dict[str, int]], measured: dict[str, ranges.RangeSet],
          sim) -> list:
    """Acceptance: both arms posed, every joint swept, every sweep convincing."""
    posed = [a for a in ARMS if a in rough]
    out = [gates.GateResult(
        name="arms posed", passed=len(posed) == len(ARMS),
        value=float(len(posed)), threshold=float(len(ARMS)),
        direction="min", unit="arms",
        detail="" if len(posed) == len(ARMS)
        else "missing: " + ", ".join(a for a in ARMS if a not in rough))]

    wanted = [m for a in posed for m in arm_joints(a)]
    have = [m for a in posed for m in measured.get(a, ranges.RangeSet()).travels]
    out.append(gates.GateResult(
        name="joints swept", passed=len(have) == len(wanted),
        value=float(len(have)), threshold=float(len(wanted)),
        direction="min", unit="joints",
        detail="" if len(have) == len(wanted)
        else "missing: " + ", ".join(m for m in wanted if m not in have)))

    short = []
    for arm in posed:
        for motor, travel in measured.get(arm, ranges.RangeSet()).travels.items():
            if not ranges.sense_agrees(travel, model_span_deg(sim, motor),
                                       RANGE_FRACTION):
                short.append(f"{motor} ({travel.span_deg:.0f}d)")
    out.append(gates.GateResult(
        name="travel per joint", passed=not short,
        value=float(len(short)), threshold=0.0, direction="max",
        unit="joints below half the model's range",
        detail="" if not short else "; ".join(short)))

    full_turn = [
        motor for arm in posed
        for motor, travel in measured.get(arm, ranges.RangeSet()).travels.items()
        if travel.span_counts >= servos.COUNTS_PER_TURN
    ]
    out.append(gates.GateResult(
        name="absolute angle ranges", passed=not full_turn,
        value=float(len(full_turn)), threshold=0.0, direction="max",
        unit="joints spanning a full turn",
        detail="" if not full_turn else "; ".join(full_turn)))
    return out


def build_payload(rough: dict[str, dict[str, int]],
                  measured: dict[str, ranges.RangeSet]) -> dict:
    zset = zeros_mod.ZeroSet()
    for arm, readings in rough.items():
        for motor, raw in readings.items():
            zset.add(motor, raw, source="rough_pose",
                     note="stage 4 hand pose; a starting guess, not a result")
    return {
        "zeros": zset.to_dict(),
        "ranges": {arm: rs.to_dict() for arm, rs in measured.items()},
        # Written down because stage 5 must fix it and a later reader will wonder
        # why one parameter of thirteen is held.
        "gauge": {"arm_mount_yaw": 0.0,
                  "se3_index": arm_model.MOUNT_YAW_INDEX,
                  "absorbed_by": "shoulder_pan zero",
                  "why": "shoulder pan rotation and arm mount yaw are the same "
                         "motion to contact data, so one must be fixed"},
        "solved_joints": list(arm_model.SOLVED_JOINTS),
    }


def main() -> int:
    import model_map

    common.heading("Stage 4: rough arm zeros and joint travel")
    print("  Two things, with different standards of care.")
    print("\n  The zeros are rough on purpose. Stage 5 solves the real ones from")
    print("  contact data and reaches the same answer from a guess 90 degrees out,")
    print("  so precision here buys nothing.")
    print("\n  The travel measurement is the part that matters. Several of these")
    print("  joints turn more than half a turn, and a single-turn encoder cannot")
    print("  tell a 320 degree range from a 40 degree one unless the sweep starts")
    print("  near the zero and is sampled the whole way. That is why the pose")
    print("  comes first.")
    print("\n  Nothing is driven. Torque stays off; you move the arms by hand.")

    try:
        common.require_results("senses")
        if not common.confirm_overwrite("zeros"):
            return 1
    except common.Aborted:
        return 1

    got_senses = senses_mod.require([m for a in ARMS for m in arm_joints(a)])
    sim = model_map.SimModel()

    try:
        robot = servos.RawRobot()
    except Exception as exc:
        print(f"\n  Cannot reach the servos: {exc}")
        print("  Check the USB adapters and that no other process holds them.")
        return 1

    rough: dict[str, dict[str, int]] = {}
    measured: dict[str, ranges.RangeSet] = {}
    with robot:
        problems = robot.verify()
        if problems:
            print("\n  Servo bus problems:")
            for p in problems:
                print(f"    - {p}")
            if not common.confirm("continue anyway", False):
                return 1

        # Left arm first, by the operator's left. Back-to-front the flanges are
        # turned, so that is the arm the model calls right_arm.
        mounting = frames.declared_mounting()
        for arm in frames.working_order(mounting):
            readings = pose_zero(robot, arm, got_senses)
            if readings is None:
                continue
            rough[arm] = readings
            measured[arm] = sweep_arm(robot, sim, arm, readings)
            print(f"\n  {frames.physical_side(arm, mounting)}"
                  " arm done. You can let it rest now.")

    report(rough, measured, sim)
    if not common.report_gates(check(rough, measured, sim)):
        print("\n  Not saved. Re-run this stage to fill the gaps.")
        return 1

    storage.save_result("zeros", build_payload(rough, measured))
    print("\n  Saved. Next: python calibration/run.py --stage 3")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
