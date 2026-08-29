"""Translate legacy terminal prompts into explicit guided UI actions.

Prompts are emitted as stable keys plus their parameters rather than as
finished sentences, and the interface renders them. That keeps the prompt
identity separate from its wording, which matters because the interaction
token is derived from it: rewording a prompt must not look like the operator
has been moved to a different step.
"""
from __future__ import annotations

import re
from typing import Any

from .i18n import joint_label as _joint_label, text as _text
from .mounting import NORMAL


JOINT_PATTERN = (
    r"(?:left_arm|right_arm)_(?:shoulder_pan|shoulder_lift|elbow_flex|"
    r"wrist_flex|wrist_roll|gripper)|head_motor_[12]"
)


def _joint_labels() -> dict[str, str]:
    from .i18n import JOINTS
    return {joint: _joint_label(joint) for joint in JOINTS}


# Kept as a module-level mapping because other modules and tests index it
# directly.
JOINT_LABELS = _joint_labels()



def _latest_joint(text: str) -> str | None:
    matches = list(re.finditer(
        r"(?:left_arm|right_arm)_(?:shoulder_pan|shoulder_lift|elbow_flex|"
        r"wrist_flex|wrist_roll|gripper)|head_motor_[12]", text))
    return matches[-1].group(0) if matches else None


def _latest_arm(text: str) -> str:
    """Which arm the stage is talking about, as a translation key.

    Only the stage's spoken wording counts. The stored names printed in the
    tables below a prompt (``left_arm_shoulder_pan`` and friends) name the arm
    of the MODEL, which back-to-front is the opposite of the one the operator is
    posing -- and being printed last, they used to win this comparison and title
    the prompt with the wrong arm.
    """
    lower = text.lower()
    left = lower.rfind("left arm")
    right = lower.rfind("right arm")
    if left < 0 and right < 0:
        # Nothing spoken yet. Fall back to the model names so an early prompt
        # still says something, rather than defaulting to one side silently.
        left = lower.rfind("left_arm")
        right = lower.rfind("right_arm")
    return "prompt.leftArm" if left > right else "prompt.rightArm"


def _buttons(*items: tuple[str, str, str]) -> list[dict[str, str]]:
    return [{"value": value, "label": label, "kind": kind}
            for value, label, kind in items]


def _yes_no(yes: str, no: str) -> list[dict[str, str]]:
    return _buttons(("y", yes, "primary"), ("n", no, "secondary"))


# The model's q=0 arm pose is not described here in words. It is defined by
# calibration/model/xlerobot_calib.xml, and reference.zeroPose.note sends the
# operator there to look at it. The pose is a joint configuration, so it does
# not depend on the mounting -- both mountings pose the arms identically, and
# only the chassis they hang off has turned.



def zero_pose_reference() -> dict[str, Any]:
    """The rough-zero pose, shaped for the UI.

    A rendering of the model is unambiguous where a sentence is not, so the
    operator is sent to look at it rather than given the pose in words. There
    is deliberately no joint checklist here: describing the same pose twice
    invites the two from disagreeing, and it was the words that were wrong
    before, not the model.
    """
    return {
        "title": _text("reference.zeroPose.title"),
        "note": _text("reference.zeroPose.note"),
        "steps": [],
    }


# Renderings of POSITIVE in calibration/stages/stage2_senses.py. Those strings
# are checked against the model by verify_directions(), which rotates each
# joint and watches where the far end goes, so they cannot drift from the XML.
# These translations must track that table, not restate it from memory. Left
# and right share a description, so they share a key.
DIRECTION_KEYS = {
    "left_arm_shoulder_pan": "direction.shoulder_pan",
    "right_arm_shoulder_pan": "direction.shoulder_pan",
    "left_arm_shoulder_lift": "direction.shoulder_lift",
    "right_arm_shoulder_lift": "direction.shoulder_lift",
    "left_arm_elbow_flex": "direction.elbow_flex",
    "right_arm_elbow_flex": "direction.elbow_flex",
    "left_arm_wrist_flex": "direction.wrist_flex",
    "right_arm_wrist_flex": "direction.wrist_flex",
    "left_arm_wrist_roll": "direction.wrist_roll",
    "right_arm_wrist_roll": "direction.wrist_roll",
    "left_arm_gripper": "direction.gripper",
    "right_arm_gripper": "direction.gripper",
    "head_motor_1": "direction.head_motor_1",
    "head_motor_2": "direction.head_motor_2",
}


def positive_directions() -> dict[str, str]:
    return {joint: _text(key) for joint, key in DIRECTION_KEYS.items()}


SENSE_DIRECTION_NOTE = _text("reference.direction.note")


def direction_reference(joint: str | None, fallback: str,
                        mounting_name: str = NORMAL) -> dict[str, Any]:
    """The positive direction for one joint, as a UI reference block."""
    key = DIRECTION_KEYS.get(joint or "")
    # Fall back to the direction parsed out of the stage's own output, so a
    # joint missing from the table above still shows something truthful.
    detail = _text(key) if key else fallback
    return {
        "title": _text("reference.direction.title"),
        "note": _text("reference.direction.note"),
        # Named for the arm the operator can see, exactly as the title above it
        # is; the card sits beside that title and naming the two differently is
        # what made the reference look like it belonged to the other arm.
        "steps": [{"joint": _joint_label(joint or "", mounting_name) if joint
                   else _text("prompt.thisJoint"),
                   "detail": detail}],
    }


def _candidate(index: int, title: str, instruction: str,
               buttons: list[dict[str, str]], caution: str = "",
               reference: dict[str, Any] | None = None) -> tuple[int, dict[str, Any]]:
    prompt = {"title": title, "instruction": instruction,
              "buttons": buttons, "caution": caution}
    if reference:
        prompt["reference"] = reference
    return index, prompt


def describe(stage_key: str, log_text: str,
             mounting_name: str = NORMAL) -> dict[str, Any]:
    """Return the latest actionable prompt without exposing a free-form input."""
    text = log_text[-12000:]
    lower = text.lower()
    joint = _latest_joint(text)
    joint_label = (_joint_label(joint, mounting_name) if joint
                   else _text("prompt.thisJoint"))
    candidates: list[tuple[int, dict[str, Any]]] = []

    def say(key: str, **fields: Any) -> str:
        return _text(key, **fields)

    def add(marker: str, title: str, instruction: str,
            buttons: list[dict[str, str]], caution: str = "",
            reference: dict[str, Any] | None = None) -> None:
        index = lower.rfind(marker)
        if index >= 0:
            candidates.append(
                _candidate(index, title, instruction, buttons, caution, reference))

    # Every stage that can be re-run asks this before replacing what it saved,
    # via common.confirm_overwrite, so it is matched for all of them rather
    # than per stage. An unmatched prompt is not a cosmetic problem here: the
    # child is already blocked in input(), and describe() falling through to
    # the placeholder returns no buttons, which leaves the operator a page they
    # cannot answer and a stage that never starts its work. That is a hang, and
    # ending the run is the only way out of it.
    overwrite = re.search(r"these results already exist:\s*(.+)", text,
                          flags=re.IGNORECASE)
    if overwrite:
        add("overwrite them (y/n)",
            say("prompt.overwrite.title",
                results=overwrite.group(1).strip().rstrip(".")),
            say("prompt.overwrite.instruction"),
            _yes_no(say("prompt.overwrite.yes"), say("prompt.overwrite.no")),
            say("prompt.overwrite.caution"))

    if stage_key == "senses":
        direction_match = list(re.finditer(r"positive direction:\s*(.+?)\.", text,
                                           flags=re.IGNORECASE))
        direction = (direction_match[-1].group(1) if direction_match
                     else say("reference.direction.fallback"))
        reference = direction_reference(joint, direction, mounting_name)
        add("put the joint at the start position",
            say("prompt.senses.start.title", joint=joint_label),
            say("prompt.senses.start.instruction"),
            _buttons(("", say("prompt.senses.start.yes"), "primary"),
                     ("s", say("prompt.senses.skip"), "secondary")),
            reference=reference)
        add("now move it so that",
            say("prompt.senses.end.title", joint=joint_label),
            say("prompt.senses.end.instruction"),
            _buttons(("", say("prompt.senses.end.yes"), "primary"),
                     ("s", say("prompt.senses.skip"), "secondary")),
            say("prompt.senses.end.caution"),
            reference=reference)
        add("retry (y/n)", say("prompt.readFailed.title"),
            say("prompt.senses.readFailed.instruction"),
            _yes_no(say("prompt.readFailed.retry"),
                    say("prompt.senses.readFailed.no")))
        add("continue anyway (y/n)", say("prompt.busFault.title"),
            say("prompt.senses.busFault.instruction"),
            _yes_no(say("prompt.busFault.yes"), say("prompt.busFault.no")),
            say("prompt.senses.busFault.caution"))
        add("go through them again (y/n)", say("prompt.senses.again.title"),
            say("prompt.senses.again.instruction"),
            _yes_no(say("prompt.senses.again.yes"), say("prompt.senses.again.no")))
    elif stage_key == "arm_ranges":
        arm = say(_latest_arm(text))
        arm_start = say(_latest_arm(text) + "Cap")
        add("is the left arm posed (y/n)", say("prompt.ranges.poseLeft.title"),
            say("prompt.ranges.pose.instruction"),
            _yes_no(say("prompt.ranges.poseLeft.yes"), say("prompt.ranges.pose.no")),
            reference=zero_pose_reference())
        add("is the right arm posed (y/n)", say("prompt.ranges.poseRight.title"),
            say("prompt.ranges.pose.instruction"),
            _yes_no(say("prompt.ranges.poseRight.yes"), say("prompt.ranges.pose.no")),
            reference=zero_pose_reference())
        add("skip this arm entirely (y/n)", say("prompt.ranges.skip.title", arm=arm),
            say("prompt.ranges.skip.instruction"),
            _yes_no(say("prompt.ranges.skip.yes", arm=arm),
                    say("prompt.ranges.skip.no")),
            say("prompt.ranges.skip.caution"))
        add("accept these as the rough zero (y/n)",
            say("prompt.ranges.accept.title", arm=arm),
            say("prompt.ranges.accept.instruction"),
            _yes_no(say("prompt.ranges.accept.yes"), say("prompt.ranges.accept.no")))
        add("sweep the joints now", say("prompt.ranges.sweep.title", arm=arm_start),
            say("prompt.ranges.sweep.instruction"),
            _buttons(("", say("prompt.ranges.sweep.yes"), "primary"),
                     ("s", say("prompt.ranges.sweep.no"), "secondary")),
            say("prompt.ranges.sweep.caution"))
        add("keep sweeping this arm (y/n)",
            say("prompt.ranges.keep.title", arm=arm_start),
            say("prompt.ranges.keep.instruction"),
            _yes_no(say("prompt.ranges.keep.yes"), say("prompt.ranges.keep.no")))
        add("retry (y/n)", say("prompt.readFailed.title"),
            say("prompt.ranges.readFailed.instruction"),
            _yes_no(say("prompt.readFailed.retry"),
                    say("prompt.ranges.readFailed.no")))
        add("continue anyway (y/n)", say("prompt.busFault.title"),
            say("prompt.ranges.busFault.instruction"),
            _yes_no(say("prompt.busFault.yes"), say("prompt.busFault.no")),
            say("prompt.ranges.busFault.caution"))

    if not candidates:
        starting = {"senses": "prompt.startingSenses",
                    "arm_ranges": "prompt.startingRanges"}.get(
                        stage_key, "prompt.starting")
        return {"title": say(f"{starting}.title"),
                "instruction": say(f"{starting}.instruction"),
                "buttons": [], "caution": ""}
    return max(candidates, key=lambda item: item[0])[1]
