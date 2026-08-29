"""User-facing workflow registry for the guided calibration application."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class StageGuide:
    key: str
    number: int
    title: str
    purpose: str
    action: str
    completion: str
    requires: tuple[str, ...]
    outputs: tuple[str, ...]
    hardware: bool = True
    legacy_aliases: tuple[str, ...] = ()
    notice: str = ""


# From the head stage onward the board defines the world frame: stage 6 reuses
# the T_W_B solved here, and stage 8 measures fresh views against it. Moving the
# base or the board invalidates all of that silently -- nothing errors, the
# numbers are just wrong -- so the stage that opens the window says so first.
FIXED_NOTICE = (
    "From this stage until calibration is finished, the robot base and the "
    "board must not move relative to each other. Every later stage is measured "
    "against the frame solved here, so nudging either one means starting again "
    "from this stage."
)

STILL_FIXED_NOTICE = (
    "The robot base and the board must still be exactly where they were during "
    "the head stage. If either has moved, redo the head stage first."
)


STAGES = (
    StageGuide(
        "prepare", 1, "Setup and calibration board",
        "Establish trustworthy scale, lenses and a physical reference.",
        "Measure the board squares, lock the focus on all three lenses, and "
        "check that the robot base and the board can be held still.",
        "Board parameters saved, all three views sharp, and camera roles "
        "confirmed.",
        (), ("board",), hardware=True, legacy_aliases=("prep", "-1")),
    StageGuide(
        "intrinsics", 2, "Camera intrinsics",
        "Measure focal length, principal point and lens distortion for all "
        "three cameras.",
        "Take the head, left wrist and right wrist cameras in turn, collecting "
        "views from near and far, tilted, and at the edges of the frame.",
        "All three cameras meet the coverage requirement and pass the holdout "
        "reprojection limit.",
        ("board",),
        ("intrinsics_head", "intrinsics_left_wrist", "intrinsics_right_wrist"),
        legacy_aliases=("1",)),
    StageGuide(
        "senses", 3, "Joint directions",
        "Map real encoder directions onto the model's joint axes.",
        "Move each joint by hand a short way in the direction the interface "
        "asks for.",
        "Every head and arm joint direction has been measured and saved.",
        (), ("senses",), legacy_aliases=("2",)),
    StageGuide(
        "arm_ranges", 4, "Arm rough zeros and travel",
        "Record each arm joint's starting zero and measure its usable travel "
        "across the encoder wrap point.",
        "Pose each arm near its zero to record the zero, then sweep each joint "
        "slowly through its full usable range.",
        "Every target joint on both arms has a rough zero and a valid travel "
        "range.",
        ("senses",), ("zeros",), legacy_aliases=("4",)),
    StageGuide(
        "head", 5, "Head, world frame and head camera",
        "Use a fixed board to solve the world frame, the head kinematics and "
        "the head camera extrinsics together.",
        "Hold the robot and the board still, move pan and tilt by hand, and "
        "collect sharp views with good coverage.",
        "Head fit and holdout errors pass, and the paired head result is saved.",
        ("intrinsics_head", "senses"), ("head",), legacy_aliases=("3", "2-3"),
        notice=FIXED_NOTICE),
    StageGuide(
        "arms", 6, "Arm calibration",
        "Solve arm mounting, joint zeros and wrist camera extrinsics together.",
        "Point each wrist camera at the fixed board in turn, moving shoulder, "
        "elbow and wrist to get wide coverage.",
        "Both arms pass the fit and holdout limits, and the result is saved as "
        "one piece.",
        ("senses", "zeros", "head", "intrinsics_left_wrist", "intrinsics_right_wrist"),
        ("touch", "zeros_refined"), legacy_aliases=("5", "5f", "fusion"),
        notice=STILL_FIXED_NOTICE),
    StageGuide(
        "normalize", 7, "Body frame and zero conventions",
        "Define the body's forward direction from arm symmetry and fix the "
        "head and wrist-roll zero conventions.",
        "Check the inputs, then run the calculation; this stage collects "
        "nothing new.",
        "head_zero, touch_zero and zeros_zero share one body frame and input "
        "fingerprint.",
        ("head", "touch", "zeros_refined"), ("head_zero", "touch_zero", "zeros_zero"),
        hardware=False, legacy_aliases=("5b",)),
    StageGuide(
        "verify", 8, "Independent verification and export",
        "Check the finished model against fresh poses, without adjusting any "
        "calibration parameter.",
        "Move the head and both arms by hand, collecting at least ten fresh "
        "valid views per camera.",
        "All three cameras pass, and the report, robot.yaml and MuJoCo XML are "
        "written.",
        ("touch_zero", "zeros_zero", "head_zero", "senses", "intrinsics_head",
         "intrinsics_left_wrist", "intrinsics_right_wrist"),
        ("validation", "robot_yaml"), legacy_aliases=("8",),
        notice=STILL_FIXED_NOTICE),
)

BY_KEY = {stage.key: stage for stage in STAGES}


def validate_registry() -> None:
    numbers = [stage.number for stage in STAGES]
    if numbers != list(range(1, len(STAGES) + 1)):
        raise ValueError(f"stage numbers must be continuous: {numbers}")
    keys = [stage.key for stage in STAGES]
    if len(keys) != len(set(keys)):
        raise ValueError("stage keys must be unique")
    producers: dict[str, str] = {}
    for stage in STAGES:
        for output in stage.outputs:
            if output in producers:
                raise ValueError(f"duplicate output {output}: {producers[output]} and {stage.key}")
            producers[output] = stage.key
    known = set(producers)
    for stage in STAGES:
        missing = set(stage.requires) - known
        if missing:
            raise ValueError(f"{stage.key} requires unknown results: {sorted(missing)}")


validate_registry()
