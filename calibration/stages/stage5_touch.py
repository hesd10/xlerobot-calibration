"""Stage 5: arm contact calibration.

The operator drives a fixed point on the gripper jaw onto a named ChArUco corner
and presses a key. The corner's world position is known from the board geometry,
stage 3 solved where the board sits relative to the robot base, so each touch gives
one known base-frame point that a known joint configuration must reach.

Per arm, thirteen parameters: arm mounting (6), four joint zeros, touch point (3).
Twelve are observable and the mount's yaw is held; see `core/arm_model.py`.

What the guidance is for
------------------------
Touch count is the least useful thing to chase. Thirty touches made from one arm
posture determine almost nothing, because a joint zero is only pinned down by
touches that moved that joint. So this tracks the spread of each joint across the
touches taken so far and asks for the joint that has moved least, rather than just
counting.

Measured on synthetic contacts with realistic noise, twelve well-spread touches
recover the zeros to about 1.2 degrees and twenty-four to about 0.8. The floor is
set by servo angle noise, not by touch count -- accuracy is almost exactly
proportional to it -- so past about twenty-four touches there is nothing more to
win here.

Wrist roll is excluded on purpose: rotating it and sliding the touch point along
its axis are the same motion. Stage 6 solves it from the wrist camera.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import common  # noqa: E402
import model_map  # noqa: E402
from core import arm_model, arm_solve, ranges, senses as senses_mod  # noqa: E402
from core import servos, storage, zeros as zeros_mod  # noqa: E402

ARMS = ("left_arm", "right_arm")

# Aim for this many touches per arm. Chosen from the synthetic sweep above: the
# accuracy curve is flat past here, so asking for more only tires the operator.
TARGET_TOUCHES = 24

# Warn when a joint has barely moved across the touches so far.
WANT_SPREAD_DEG = 70.0


def corner_world(spec, index: int) -> np.ndarray:
    """One interior corner's position in the world frame, which is the board frame."""
    return np.asarray(spec.corner_positions()[index], dtype=float)


def corner_label(spec, index: int) -> str:
    """Name a corner the way the operator can find it on the printed board.

    Row and column of interior corners, counted from the board's origin corner,
    which is the one detect_geometry fixed as the frame origin.
    """
    per_row = spec.squares_x - 1
    return f"corner {index} (row {index // per_row + 1}, col {index % per_row + 1})"


def suggest_corners(spec, n: int = 12) -> list[int]:
    """A spread of corners across the board rather than a huddle in one area.

    Touches clustered on adjacent corners vary the arm posture hardly at all, which
    is the failure this stage's gates exist to catch. Picking spread-out corners up
    front makes the good behaviour the easy one.
    """
    per_row = spec.squares_x - 1
    rows = spec.squares_y - 1

    # Spread the requested count over the board's full extent rather than stepping
    # by a fixed number of squares. Stepping by a couple of squares puts consecutive
    # suggestions 40 mm apart, which the arm can reach without changing posture
    # meaningfully -- the very thing this stage's gates reject.
    n_rows = max(2, int(round(np.sqrt(n * rows / max(1, per_row)))))
    n_cols = max(2, int(np.ceil(n / n_rows)))

    def pick(count: int, extent: int) -> list[int]:
        if count == 1:
            return [extent // 2]
        return [int(round(i * (extent - 1) / (count - 1))) for i in range(count)]

    row_idx = pick(n_rows, rows)
    col_idx = pick(n_cols, per_row)

    out = []
    for i, r in enumerate(row_idx):
        cols = col_idx[::-1] if i % 2 else col_idx
        for c in cols:
            out.append(r * per_row + c)
    return out[:n]


def stage4_rough_zero_set(stage4: dict) -> zeros_mod.ZeroSet:
    """Stage 5's baseline zeros, even after zeros.json has been overwritten.

    Stage 4 records the rough hand pose in ranges[*].zero_raw. That survives a
    later Stage 5 save, so reruns must rebuild the baseline from there instead of
    reusing the latest contents of zeros.json.
    """
    out = zeros_mod.ZeroSet.from_dict(stage4.get("zeros"))
    ranges = stage4.get("ranges") or {}
    for arm in ARMS:
        zero_raw = (ranges.get(arm) or {}).get("zero_raw") or {}
        for name in model_map.ARM_JOINTS_NO_GRIPPER[arm]:
            if name in zero_raw:
                out.add(name, int(zero_raw[name]), source="rough_pose",
                        note="stage 4 hand pose; a starting guess, not a result")
    return out


def reset_arm_to_rough(recorded: zeros_mod.ZeroSet, rough_zeros: zeros_mod.ZeroSet,
                       arm: str) -> None:
    """Overwrite one arm's solved joints with the stage 4 baseline."""
    for name in model_map.ARM_JOINTS_NO_GRIPPER[arm]:
        recorded.add(
            name,
            rough_zeros.joints[name].raw,
            source="rough_pose",
            note="stage 4 hand pose; a starting guess, not a result",
        )


def spreads(taken: list[dict], arm: str) -> dict[str, float]:
    """Degrees of travel each solved joint has covered across the touches so far."""
    out = {}
    for name in arm_model.joint_names(arm):
        vals = [np.rad2deg(t["angles"][name]) for t in taken
                if name in t.get("angles", {})]
        out[name] = float(max(vals) - min(vals)) if len(vals) > 1 else 0.0
    return out


def next_advice(taken: list[dict], arm: str) -> str:
    """What to vary for the next touch: the joint that has moved least so far."""
    if len(taken) < 2:
        return "vary the whole arm posture between touches, not just the corner"
    sp = spreads(taken, arm)
    worst = min(sp, key=sp.get)
    if sp[worst] >= WANT_SPREAD_DEG:
        return "spread is good on every joint; keep varying the posture"
    short = worst.replace(f"{arm}_", "").replace("_", " ")
    return (f"{short} has only covered {sp[worst]:.0f} deg -- reach this corner "
            f"with that joint somewhere different")


def describe_touch_point(arm: str) -> None:
    print("  The touch point is a fixed spot on the STATIC jaw, not the moving one:")
    print("  a point on the moving jaw shifts whenever the gripper opens, and the")
    print("  fit has no way to know it did.")
    print("\n  Pick one identifiable spot -- a corner of the fixed jaw tip works --")
    print("  and use the SAME spot for every touch on this arm. Its exact location")
    print("  is solved, so it does not need measuring; it only needs to be the same")
    print("  point each time.")


def take_touches(robot, sim, spec, arm: str, zero_raw: dict[str, int],
                 signs: dict[str, int], measured_ranges: ranges.RangeSet,
                 T_W_B: np.ndarray) -> list[dict]:
    """Collect contacts for one arm, guiding the operator toward useful postures."""
    common.heading(f"{arm.replace('_', ' ')}: touches")
    describe_touch_point(arm)

    suggestions = suggest_corners(spec, TARGET_TOUCHES)
    print(f"\n  Aim for {TARGET_TOUCHES} touches. Suggested corners, spread across")
    print("  the board so the arm posture has to change between them:")
    print("    " + ", ".join(str(i) for i in suggestions[:12]))
    print("\n  For each touch: pose the arm so your chosen jaw point sits exactly on")
    print("  the corner, hold it there, then press Enter. Torque is off, so support")
    print("  the arm. Enter 'd' when done, 'u' to undo the last touch.")

    solved = arm_model.joint_names(arm)
    all_names = list(zero_raw)
    taken: list[dict] = []
    while True:
        print(f"\n  [{len(taken)}/{TARGET_TOUCHES}] {next_advice(taken, arm)}")
        answer = common.ask("    corner index (or d/u)", "").strip().lower()
        if answer == "d":
            if len(taken) < arm_solve.MIN_TOUCHES:
                print(f"    Only {len(taken)} touches; the fit needs at least "
                      f"{arm_solve.MIN_TOUCHES}.")
                if not common.confirm("stop anyway", False):
                    continue
            break
        if answer == "u":
            if taken:
                dropped = taken.pop()
                print(f"    dropped the touch at corner {dropped['corner']}")
            continue
        try:
            index = int(answer)
        except ValueError:
            print("    Give a corner index, or 'd' to finish.")
            continue
        if not 0 <= index < spec.n_corners:
            print(f"    This board has corners 0..{spec.n_corners - 1}.")
            continue

        print(f"    {corner_label(spec, index)}. Hold the jaw point on it, then "
              f"press Enter.")
        common.ask("", "")
        if not servos.wait_until_still(robot, solved, tolerance=3):
            print("    The arm is still moving. Steady it and try this touch again.")
            continue
        raw = {n: robot.read_raw(n) for n in all_names}
        missing = [n for n, v in raw.items() if v is None]
        if missing:
            print(f"    Lost {', '.join(missing)}. Touch not recorded.")
            continue

        angles = ranges.angles_from_ranges(
            {n: int(v) for n, v in raw.items()}, zero_raw, signs,
            measured_ranges)
        target_world = corner_world(spec, index)
        target_base = (np.linalg.inv(T_W_B) @ np.append(target_world, 1.0))[:3]
        taken.append({"corner": index, "raw": raw, "angles": angles,
                      "target_base": target_base.tolist(),
                      "target_world": target_world.tolist()})
        print(f"    recorded ({len(taken)} touches)")
    return taken


def report(arm: str, result: dict | None) -> None:
    common.heading(f"{arm.replace('_', ' ')}: contact fit")
    if result is None:
        print(f"  Too few touches to solve (need {arm_solve.MIN_TOUCHES}).")
        return

    print(f"  {result['n_touches_total']} touches "
          f"({result['n_touches_fit']} fitted, "
          f"{result['n_touches_holdout']} held back)")
    print(f"  holdout {result['holdout_rms_mm']:.2f} mm rms, "
          f"{result['holdout_max_mm']:.2f} mm worst")
    print(f"  fit     {result['fit_rms_mm']:.2f} mm rms")
    print(f"  condition number {result['condition_number']:.1e}")

    print("\n  Solved zero corrections to stage 4's rough pose:")
    for name, deg in sorted(result["zeros_deg"].items()):
        print(f"    {name:<28} {deg:+8.2f} deg")

    print(f"\n  Touch point on the jaw: "
          f"{', '.join(f'{v:+.1f}' for v in result['touch_point_mm'])} mm")
    print(f"  Arm mount offset:       "
          f"{', '.join(f'{v:+.1f}' for v in result['mount_translation_mm'])} mm, "
          f"{', '.join(f'{v:+.2f}' for v in result['mount_rotation_deg'])} deg")

    print("\n  Joint spread across the touches:")
    for name in arm_model.joint_names(arm):
        span = result.get(f"spread_{name}_deg", 0.0)
        flag = "" if span >= WANT_SPREAD_DEG else "   <-- narrow"
        print(f"    {name:<28} {span:7.0f} deg{flag}")

    print("\n  The shoulder pan zero also carries the arm mount's yaw, which is held")
    print("  at zero by convention. Read it as a calibration constant, not as a")
    print("  physical property of that joint alone.")


def main() -> int:
    common.heading("Stage 5: arm contact calibration")
    print("  Touch a fixed point on each gripper's static jaw to known board")
    print("  corners. Each touch is one known point the arm must reach, which is")
    print("  what determines the joint zeros and how the arm is mounted.")
    print("\n  What matters is the variety of postures, not the number of touches.")
    print("  A zero is only pinned down by touches that moved its joint, so this")
    print("  tool asks you to vary the joint that has moved least.")
    print("\n  Nothing is driven. Torque stays off; you pose the arms by hand.")

    try:
        results = common.require_results("senses", "zeros", "head")
        if not common.confirm_overwrite("touch"):
            return 1
    except common.Aborted:
        return 1

    spec = common.load_board()
    common.warn_board_drift(spec)

    head = results["head"]
    T_W_B = np.asarray(head["T_W_B"], dtype=float)
    stage4 = results["zeros"]
    rough_zeros = stage4_rough_zero_set(stage4)
    recorded = zeros_mod.ZeroSet.from_dict(stage4.get("zeros"))
    measured_ranges = {
        arm: ranges.RangeSet.from_dict(data)
        for arm, data in (stage4.get("ranges") or {}).items()
    }
    got_senses = senses_mod.load()
    if got_senses is None:
        print("\n  No joint senses recorded. Run stage 2 first.")
        return 1

    print("\n  The board must not have moved since stage 3. Its position relative")
    print("  to the robot is what makes each corner a known point; if either was")
    print("  nudged, every touch here is measured against the wrong place.")
    if not common.confirm("board and robot base both untouched since stage 3", False):
        print("\n  Re-run stage 3 first to re-measure where the board sits.")
        return 1

    sim = model_map.SimModel()
    try:
        robot = servos.RawRobot()
    except Exception as exc:
        print(f"\n  Cannot reach the servos: {exc}")
        return 1

    out: dict[str, dict] = {}
    captures: dict[str, list] = {}
    with robot:
        for arm in ARMS:
            names = [m for m in model_map.ARM_JOINTS_NO_GRIPPER[arm]]
            zero_raw = {n: rough_zeros.joints[n].raw for n in names
                        if n in rough_zeros.joints}
            if len(zero_raw) != len(names):
                print(f"\n  Stage 4 has no rough zero for "
                      f"{', '.join(n for n in names if n not in zero_raw)}.")
                print("  Re-run stage 4 for this arm.")
                continue
            signs = {n: got_senses.sign(n) for n in names}

            # No pairing check against stage 4's pose here. Every touch is taken
            # from a different posture by design, so the arm is always far from
            # that pose and the check would fire on every run -- a warning that is
            # always true teaches the operator to ignore warnings.

            arm_ranges = measured_ranges.get(arm)
            if arm_ranges is None or any(n not in arm_ranges.travels for n in names):
                print(f"\n  Stage 4 has incomplete ranges for {arm}; skipping.")
                continue
            taken = take_touches(
                robot, sim, spec, arm, zero_raw, signs, arm_ranges, T_W_B)
            if len(taken) < arm_solve.MIN_TOUCHES:
                print(f"\n  {arm}: not enough touches to solve; skipped.")
                continue
            captures[arm] = taken
            # No zero guess, and that is not an omission. The recorded angles already
            # have stage 4's rough zeros applied, so what this fit solves is the
            # correction to them, whose best starting estimate is zero.
            result = arm_solve.fit(
                sim, arm,
                [t["angles"] for t in taken],
                [np.asarray(t["target_base"], float) for t in taken],
                zeros_guess=None)
            report(arm, result)
            if result is not None:
                out[arm] = result
                reset_arm_to_rough(recorded, rough_zeros, arm)
                for joint_name, zero_deg in result["zeros_deg"].items():
                    rough_raw = rough_zeros.joints[joint_name].raw
                    sense = got_senses.sign(joint_name)
                    new_raw = servos.zero_with_angle_correction(
                        rough_raw, np.deg2rad(zero_deg), sense)
                    recorded.add(joint_name, new_raw, source="contact",
                                 note=f"stage 5: {zero_deg:+.2f} deg from rough pose")

    if not out:
        print("\n  Nothing solved. Not saved.")
        return 1

    checks = []
    for arm, result in out.items():
        for gate in arm_solve.grade(result):
            gate.name = f"{arm}: {gate.name}"
            checks.append(gate)
    gates_passed = common.report_gates(checks)
    if not gates_passed:
        print("\n  Some gates failed. Later stages may produce inaccurate results.")
        if not common.confirm("Save anyway", default=False):
            print("  Not saved. Re-run with more varied postures.")
            return 1
        print("  Saving despite failed gates...")

    storage.save_result("touch", {
        "arms": out,
        "captures": {a: [{k: v for k, v in t.items() if k != "angles"}
                         for t in taken] for a, taken in captures.items()},
        "zeros_used": rough_zeros.to_dict(),
        "senses_used": {n: got_senses.sign(n)
                        for a in ARMS
                        for n in model_map.ARM_JOINTS_NO_GRIPPER[a]},
    })
    storage.save_result("zeros", {
        "zeros": recorded.to_dict(),
        "ranges": stage4.get("ranges", {}),
        "gauge": stage4.get("gauge", {}),
        "solved_joints": stage4.get("solved_joints", list(arm_model.SOLVED_JOINTS)),
    })
    print("\n  Saved. Next: python calibration/run.py --stage 6")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
