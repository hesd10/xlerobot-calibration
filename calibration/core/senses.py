"""Joint senses: which way a servo turns its joint, relative to the model's axis.

A servo's positive direction is a wiring and assembly fact. The model file
describes geometry and cannot know it, so the sign has to be measured on the
unit. There are fourteen of them and they are not guessable: 2^14 combinations,
and trying them against a residual does not work.

Why a residual cannot find them
-------------------------------
Measured on this robot's head, both pan senses fit the capture to 3.6mm. The one
that is wrong places the world board 1515mm above the floor instead of the 750mm
it was actually at, and has the camera looking at the ceiling while it is plainly
looking down at a table. The fit has no opinion; only an external fact separates
the branches. So the sense is recorded up front, before anything is solved, and
everything downstream is expressed in it.

What a sense means
------------------
`+1` when increasing encoder counts moves the joint along the model's own axis in
the positive direction, `-1` when it moves against it. Applied to the measured
angle before any kinematics, which keeps the axis constants exactly as the model
states them and confines the unit-specific part to this one table.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

# Counts of travel needed before a sign is trustworthy. Encoder noise is a count
# or two, and a joint nudged by hand easily wanders that much, so this is set far
# above the noise rather than just outside it.
MIN_TRAVEL_COUNTS = 40

# Every joint whose sense matters. Wheels are excluded: they have no zero and no
# pose, so their sense is a driving convention rather than a calibration input.
JOINTS: tuple[str, ...] = (
    "left_arm_shoulder_pan", "left_arm_shoulder_lift", "left_arm_elbow_flex",
    "left_arm_wrist_flex", "left_arm_wrist_roll", "left_arm_gripper",
    "right_arm_shoulder_pan", "right_arm_shoulder_lift", "right_arm_elbow_flex",
    "right_arm_wrist_flex", "right_arm_wrist_roll", "right_arm_gripper",
    "head_motor_1", "head_motor_2",
)


@dataclass
class Sense:
    """One joint's sense, with the evidence that produced it."""

    name: str
    sign: int
    # Raw counts before and after the demonstrated motion, and the travel between
    # them. Kept so a doubtful sign can be re-examined without redoing the move.
    raw_before: int | None = None
    raw_after: int | None = None
    travel_counts: int | None = None
    # Which named direction the operator was asked to move in.
    prompt: str = ""
    source: str = "demonstrated"

    def __post_init__(self) -> None:
        if self.sign not in (-1, 1):
            raise ValueError(f"{self.name}: sense must be -1 or +1, got {self.sign}")

    @property
    def weak(self) -> bool:
        """Was the travel too small for the sign to be trusted?"""
        return (self.travel_counts is not None
                and abs(self.travel_counts) < MIN_TRAVEL_COUNTS)

    def to_dict(self) -> dict:
        return {"name": self.name, "sign": self.sign,
                "raw_before": self.raw_before, "raw_after": self.raw_after,
                "travel_counts": self.travel_counts, "prompt": self.prompt,
                "source": self.source}

    @classmethod
    def from_dict(cls, d: dict) -> "Sense":
        known = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in d.items() if k in known})


@dataclass
class SenseSet:
    """The senses recorded for one robot, and when."""

    senses: dict[str, Sense] = field(default_factory=dict)
    recorded_at: str = ""

    def record(self, sense: Sense) -> None:
        self.senses[sense.name] = sense
        self.recorded_at = datetime.now().isoformat(timespec="seconds")

    def sign(self, joint: str) -> int:
        """The sense for a joint, refusing to invent one that was never measured.

        Defaulting to +1 for an unmeasured joint is the failure this module
        exists to prevent, so an absent entry is an error rather than a guess.
        """
        if joint not in self.senses:
            raise KeyError(
                f"no sense recorded for {joint!r}; run stage 2 before using it")
        return self.senses[joint].sign

    @property
    def missing(self) -> list[str]:
        return [j for j in JOINTS if j not in self.senses]

    @property
    def weak(self) -> list[str]:
        return [n for n, s in self.senses.items() if s.weak]

    @property
    def complete(self) -> bool:
        return not self.missing

    def to_dict(self) -> dict:
        return {"senses": {n: s.to_dict() for n, s in self.senses.items()},
                "recorded_at": self.recorded_at,
                "min_travel_counts": MIN_TRAVEL_COUNTS}

    @classmethod
    def from_dict(cls, d: dict) -> "SenseSet":
        return cls(senses={n: Sense.from_dict(v)
                           for n, v in (d.get("senses") or {}).items()},
                   recorded_at=d.get("recorded_at", ""))


def load(path: Path | None = None) -> SenseSet | None:
    """Read the recorded senses, or None if stage 2 has not run."""
    from . import storage
    if path is None:
        data = storage.load_result("senses")
        return SenseSet.from_dict(data) if data else None
    p = Path(path)
    if not p.is_file():
        return None
    return SenseSet.from_dict(json.loads(p.read_text()))


def require(joints: tuple[str, ...] | list[str]) -> SenseSet:
    """The recorded senses, or a clear refusal naming what is missing."""
    got = load()
    if got is None:
        raise SystemExit(
            "No joint senses recorded. Run:\n"
            "  python calibration/run.py --stage 2\n\n"
            "This cannot be skipped or guessed: a wrong sense fits the data just\n"
            "as well as the right one and silently mirrors the result.")
    absent = [j for j in joints if j not in got.senses]
    if absent:
        raise SystemExit(
            "These joints have no recorded sense:\n  "
            + "\n  ".join(absent)
            + "\n\nRe-run: python calibration/run.py --stage 2")
    return got
