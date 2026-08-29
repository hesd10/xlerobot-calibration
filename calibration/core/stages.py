"""Stage registry: dependencies, gates and progress tracking.

This is what makes the procedure guided rather than a pile of scripts. Each stage
declares what it needs, what it produces, and how to tell whether its output is
good enough. The runner then always knows which stage is next, refuses to run one
whose inputs are missing, and refuses to advance past one whose output failed its
acceptance test.

The ordering is not arbitrary. Intrinsics must come before any pose is solved,
because PnP without a correct K produces a confidently wrong answer. Arm zeros
must come before wrist camera extrinsics, because the extrinsic solve assumes
forward kinematics is already right.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from . import storage


@dataclass
class Stage:
    key: str
    number: str
    title: str
    purpose: str
    # Result files that must exist before this stage can run.
    requires: list[str] = field(default_factory=list)
    # Result files this stage writes.
    produces: list[str] = field(default_factory=list)
    # Physical setup the operator must not disturb once this stage has run.
    locks: list[str] = field(default_factory=list)
    manual: str = ""
    script: str | None = None
    optional: bool = False
    # Other numbers that reach this stage. Stage 2 and 3 of the document are one
    # stage here, so both numbers must still find it.
    aliases: list[str] = field(default_factory=list)

    def matches(self, token: str) -> bool:
        return token in (self.key, self.number) or token in self.aliases


STAGES: list[Stage] = [
    Stage(
        key="prep",
        number="-1",
        title="Physical preparation",
        purpose="Lock lens focus, measure the printed board, choose the touch point.",
        produces=["board"],
        locks=["lens focus", "board geometry"],
        manual=(
            "1. Print the ChArUco board(s). Mount flat on something rigid; a curled\n"
            "   sheet is a scale and shape error that nothing downstream can undo.\n"
            "2. MEASURE the printed square size with calipers and record it. Do not\n"
            "   trust the nominal size: printer scaling is routinely off by 1-2%,\n"
            "   and that error propagates into every distance the calibration\n"
            "   reports.\n"
            "3. Focus each of the three lenses by hand for the distance it will\n"
            "   actually work at, then lock them. These are manual-focus modules,\n"
            "   so focus cannot be restored in software once intrinsics are solved.\n"
            "4. Pick the touch point on each gripper: a sharp, repeatably\n"
            "   identifiable feature such as a jaw tip corner. Its exact position is\n"
            "   solved for, so it need not be measured, but it must never move.\n"
        ),
        script="stages/stage_prep.py",
    ),
    Stage(
        key="intrinsics",
        number="1",
        title="Camera intrinsics",
        purpose="Solve K and distortion for head, left wrist and right wrist.",
        requires=["board"],
        produces=["intrinsics_head", "intrinsics_left_wrist", "intrinsics_right_wrist"],
        locks=["lens focus", "capture resolution"],
        manual=(
            "Hold the board in front of one camera at a time. Cover the whole frame\n"
            "including the corners, tilt up to about 45 degrees, and vary distance.\n"
            "Corner coverage is what determines the distortion terms; a stack of\n"
            "centred frontal views gives a low error and a useless model.\n"
            "\n"
            "This opens a page at http://127.0.0.1:8422. Pick a camera there and\n"
            "only that one streams, which also keeps three 500mA modules from\n"
            "fighting over one USB bus.\n"
        ),
        script="stages/stage1_web.py",
    ),
    Stage(
        key="senses",
        number="2",
        aliases=["directions", "signs"],
        title="Joint senses",
        purpose="Record which way each servo turns its joint, relative to the "
                "model's axis.",
        produces=["senses"],
        manual=(
            "For each joint the page names a direction in plain words and asks you\n"
            "to move the joint that way by hand. It reads raw counts before and\n"
            "after and derives the sign. Nothing is driven; torque stays off.\n"
            "\n"
            "This exists because a servo's positive direction is a wiring and\n"
            "assembly fact, and the model file cannot know it. Guessing it wrong is\n"
            "not caught by any residual: on this unit the head pan sense fitted the\n"
            "data to 3.6mm either way, while placing the board 750mm from where it\n"
            "actually was and aiming the camera at the ceiling.\n"
            "\n"
            "Do this before anything is solved, and never afterwards: every stage\n"
            "downstream is expressed in these signs.\n"
        ),
        script="stages/stage2_senses.py",
    ),
    Stage(
        key="zeros",
        number="4",
        title="Rough arm zeros and joint travel",
        purpose="Record a starting zero per arm joint and measure each joint's travel.",
        requires=["senses"],
        produces=["zeros"],
        manual=(
            "One arm at a time, by hand, torque off. Pose it near the model's zero\n"
            "pose, record the counts, then sweep each joint to both ends.\n"
            "\n"
            "The zero is deliberately rough. Stage 5 solves the real zeros from\n"
            "contact data and reaches the same answer from a guess ninety degrees\n"
            "out, so care spent on precision here buys nothing.\n"
            "\n"
            "What the zero pose is actually for is the encoder. These are\n"
            "single-turn absolute encoders and several joints travel more than half\n"
            "a turn -- wrist roll covers 320 degrees -- so a range measured as two\n"
            "endpoint readings can come out as the short way round. Starting near\n"
            "the zero puts every extreme within half a turn, and the sweep is\n"
            "sampled continuously so the travel accumulates correctly.\n"
            "\n"
            "A joint whose range fills the whole 0-4095 span is a normal result for\n"
            "a flexible joint, not a fault.\n"
        ),
        script="stages/stage4_zeros.py",
    ),
    # Runs after zeros despite the lower number: the numbers are the tokens
    # operators have always typed and are kept stable, while the list order is
    # what the runner follows. Zeros moved ahead of head because it means
    # swinging both arms about by hand, and from here to stage 5 the base and
    # board must not move at all.
    Stage(
        key="head",
        number="3",
        aliases=["2-3", "world"],
        title="World frame, head mechanism and head camera extrinsics",
        purpose="Clamp everything, define W = the board frame, then solve T_W^B, "
                "the pan axis and the head camera mount.",
        requires=["intrinsics_head", "senses"],
        produces=["head"],
        locks=["robot base position", "world board position",
               "head zero convention"],
        manual=(
            "Clamp the robot base and the large board so neither can shift. Every\n"
            "later result is expressed relative to this board, so if the base moves\n"
            "afterwards the calibration is void. The clamp has to hold from here\n"
            "through stage 5, which reuses this board pose; the arm handling that\n"
            "would most likely disturb it is already done, in stage 4.\n"
            "\n"
            "The board must stay visible throughout, so pan sweeps only as far as\n"
            "the field of view allows. A sweep of about +/-25 to 30 degrees pins the\n"
            "vertical axis to roughly a millimetre; below +/-15 degrees the axis\n"
            "estimate degrades quickly and is not worth collecting.\n"
            "The tilt zero is fixed by convention, not solved: it is nearly\n"
            "inseparable from the camera mount rotation.\n"
        ),
        script="stages/stage23_head.py",
    ),
    Stage(
        key="touch",
        number="5",
        aliases=["5f", "fusion"],
        title="Arm and wrist-camera fusion calibration",
        purpose="Solve arm mounting, four joint zeros and wrist camera mounts from wrist views.",
        requires=["senses", "zeros", "head", "intrinsics_left_wrist",
                  "intrinsics_right_wrist"],
        # One result holding both arms, not two. Both are solved in a single pass
        # against the same board placement, so splitting them into separate outputs
        # would let the runner believe one arm's result is usable while the board has
        # since moved out from under the other.
        produces=["touch"],
        manual=(
            "Use one wrist camera at a time to observe the fixed ChArUco board. Move\n"
            "the arm by hand through diverse postures while keeping the board sharp\n"
            "and well inside the image. The page switches cameras when you switch\n"
            "arms, so only one USB video stream is open.\n"
            "\n"
            "Vary shoulder pan, lift, elbow and especially wrist flex over broad\n"
            "ranges, and change camera height and orientation. View count alone is\n"
            "not enough: repeated nearby poses leave joint zeros undetermined.\n"
            "\n"
            "The board and robot base must not have moved since stage 3. Encoder\n"
            "counts are continuously unwrapped across 4095/0 using the Stage 4 rough\n"
            "zeros and measured travel; do not cross a mechanical stop.\n"
            "\n"
            "Wrist roll participates in the camera motion but its zero remains a\n"
            "gauge with camera roll. Stage 6 resolves that physical convention.\n"
        ),
        script="stages/stage5_fusion.py",
    ),
    Stage(
        key="head_zero",
        number="5b",
        title="Body, head-pan and wrist-roll zero conventions",
        purpose="Align body forward from arm symmetry and make both wrist-camera optical axes horizontal.",
        requires=["head", "touch", "zeros"],
        produces=["head_zero", "touch_zero", "zeros_zero"],
        locks=["body frame convention"],
        manual=(
            "Nothing to do by hand and nothing to capture: this is arithmetic on\n"
            "results already in hand.\n"
            "\n"
            "The head zero cannot be measured against the world board, because a\n"
            "head rotation and a base rotation look identical to a camera watching\n"
            "one fixed board. The robot's own build symmetry supplies what the\n"
            "board cannot: the two arm roots are mirror images in the XML, so the\n"
            "head zero that makes the SOLVED arm roots symmetric is the one that\n"
            "has the head facing straight ahead.\n"
            "\n"
            "Roll and yaw come from that symmetry. Pitch does not: rotating about\n"
            "the sideways axis maps the mirror plane onto itself, so it leaves the\n"
            "symmetry untouched. Pitch is taken instead from the direction of the\n"
            "arm-root midpoint, matched to the model.\n"
            "\n"
            "The wrist-roll gauge is fixed at the same time: each camera's optical\n"
            "axis is made horizontal, with the left axis kept in quadrant III and\n"
            "the right in quadrant I. Matching camera-mount transforms preserve all\n"
            "Fusion predictions exactly. Re-running starts from Stage 3/5 sources,\n"
            "so none of these corrections accumulate.\n"
        ),
        script="stages/stage5b_head_zero.py",
    ),
    Stage(
        key="wrist_cams",
        number="6",
        title="Legacy independent wrist-camera solve",
        purpose="Optionally cross-check Fusion using a separate wrist-camera capture session.",
        requires=["touch_zero", "head_zero", "zeros_zero", "intrinsics_left_wrist",
                  "intrinsics_right_wrist"],
        produces=["wrist_left", "wrist_right"],
        manual=(
            "Pose each arm so its wrist camera sees the board, across many\n"
            "orientations. Sweep wrist roll widely: that motion is what separates\n"
            "the roll zero from the camera mount rotation.\n"
        ),
        script="stages/stage6_wrist_cams.py",
        optional=True,
    ),
    Stage(
        key="verify",
        number="8",
        title="Passive three-camera independent validation",
        purpose="Measure fixed-model accuracy from new manual poses without commanding motion.",
        requires=["touch_zero", "zeros_zero", "head_zero", "senses",
                  "intrinsics_head", "intrinsics_left_wrist", "intrinsics_right_wrist"],
        produces=["validation", "robot_yaml"],
        manual=(
            "Turn torque off and manually pose the head and both arms. Capture at\n"
            "least ten new ChArUco views per camera using the three Web UI tabs.\n"
            "Stage 8 never commands motion or changes calibration parameters. It\n"
            "always saves raw and shared-drift-corrected diagnostics, even when an\n"
            "acceptance gate fails.\n"
        ),
        script="stages/stage8_verify.py",
    ),
]

BY_KEY = {s.key: s for s in STAGES}


def result_exists(name: str) -> bool:
    data = storage.load_result(name)
    if data is None:
        return False
    if name == "touch" and data.get("method") == "wrist_camera_fusion":
        return bool(data.get("complete"))
    if name in ("head_zero", "touch_zero", "zeros_zero"):
        derived = [storage.load_result(n)
                   for n in ("head_zero", "touch_zero", "zeros_zero")]
        head, touch, zeros = [storage.load_result(n)
                              for n in ("head", "touch", "zeros")]
        if not all((*derived, head, touch, zeros)):
            return False
        frame_ids = {item.get("body_frame_id") for item in derived}
        if len(frame_ids) != 1 or None in frame_ids:
            return False
        source_sets = [item.get("stage5b_sources") for item in derived]
        if any(source != source_sets[0] for source in source_sets[1:]):
            return False
        sources = source_sets[0]
        return bool(
            sources
            and sources.get("head_fingerprint") == storage.result_fingerprint(head)
            and sources.get("touch_fingerprint") == storage.result_fingerprint(touch)
            and sources.get("zeros_fingerprint") == storage.result_fingerprint(zeros)
        )
    if name in ("wrist_left", "wrist_right"):
        head_zero = storage.load_result("head_zero")
        touch_zero = storage.load_result("touch_zero")
        zeros_zero = storage.load_result("zeros_zero")
        if not head_zero or not touch_zero or not zeros_zero:
            return False
        frame_id = data.get("body_frame_id")
        return bool(
            frame_id
            and frame_id == head_zero.get("body_frame_id")
            and frame_id == touch_zero.get("body_frame_id")
            and frame_id == zeros_zero.get("body_frame_id")
        )
    if name in ("validation", "robot_yaml"):
        validation = storage.load_result("validation")
        deploy = storage.load_result("robot_yaml")
        if not validation or not deploy:
            return False
        sources = validation.get("sources") or {}
        expected = {
            "head_zero": storage.result_fingerprint(storage.load_result("head_zero") or {}),
            "touch_zero": storage.result_fingerprint(storage.load_result("touch_zero") or {}),
            "zeros_zero": storage.result_fingerprint(storage.load_result("zeros_zero") or {}),
            "intrinsics_head": storage.result_fingerprint(storage.load_result("intrinsics_head") or {}),
            "intrinsics_left_wrist": storage.result_fingerprint(storage.load_result("intrinsics_left_wrist") or {}),
            "intrinsics_right_wrist": storage.result_fingerprint(storage.load_result("intrinsics_right_wrist") or {}),
        }
        return bool(
            all(sources.get(key) == value for key, value in expected.items())
            and sources.get("body_frame_id") == deploy.get("body_frame_id")
            and deploy.get("calibration_sources") == sources
            and isinstance(validation.get("passed"), bool)
        )
    return True


def stage_state(stage: Stage) -> str:
    """One of: done, ready, blocked."""
    if stage.produces and all(result_exists(n) for n in stage.produces):
        return "done"
    missing = [r for r in stage.requires if not result_exists(r)]
    return "blocked" if missing else "ready"


def missing_requirements(stage: Stage) -> list[str]:
    return [r for r in stage.requires if not result_exists(r)]


def partial_outputs(stage: Stage) -> list[str]:
    """Outputs present on disk even when their multi-part result is incomplete."""
    return [n for n in stage.produces if storage.load_result(n) is not None]


def next_stage() -> Stage | None:
    for stage in STAGES:
        if not stage.optional and stage_state(stage) != "done":
            return stage
    return None


def progress() -> list[tuple[Stage, str]]:
    return [(s, stage_state(s)) for s in STAGES]
