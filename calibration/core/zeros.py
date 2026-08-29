"""Recorded joint zeros: raw encoder counts, and going back to them later.

Why raw counts and not offsets
------------------------------
A zero is stored as the absolute count the encoder reads at the zero posture, not
as an offset from anything. An offset is only meaningful relative to whatever the
firmware's homing offset happened to be at the time, and that can be rewritten by
any other tool; the raw count is a fact about the hardware. This matches
`core/servos.py`, which deliberately bypasses the firmware's calibration layer.

Why the zeros travel with the frame they define
-----------------------------------------------
Fixing a zero by convention displaces the body frame, and T_W^B absorbs the
displacement. That makes the recorded zero and the solved T_W^B a matched pair: a
count that no longer corresponds to the T_W^B in hand puts the camera in the wrong
place, by 13 mm for a 3-degree discrepancy on this robot, with nothing to warn you.

So `check_pairing` exists, and the head stage stores both together. Being out of
pair is recoverable rather than fatal -- the correction is closed-form, see
`head_model.shift_pan_zero` -- but it has to be noticed first.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from . import servos

# A servo reading this far from the recorded zero is a different posture, not
# noise. Two counts is the settling tolerance used during capture; ten leaves room
# for temperature drift and gearbox backlash without hiding a real move.
PAIRING_TOLERANCE_COUNTS = 10


@dataclass
class JointZero:
    """The encoder count at a joint's zero posture."""

    name: str
    raw: int
    # What convention this zero represents, for the report rather than the maths.
    source: str = "mechanical"
    note: str = ""

    def to_dict(self) -> dict:
        return {"name": self.name, "raw": int(self.raw),
                "source": self.source, "note": self.note}

    @classmethod
    def from_dict(cls, d: dict) -> "JointZero":
        return cls(name=d["name"], raw=int(d["raw"]),
                   source=d.get("source", "mechanical"), note=d.get("note", ""))

    def angle_of(self, raw: int, sign: int = 1) -> float:
        """Joint angle in radians for a live reading, wrap-aware."""
        return servos.raw_to_rad(raw, self.raw, sign)

    def count_for(self, angle: float, sign: int = 1) -> float:
        """Encoder count to command for a wanted angle."""
        return servos.rad_to_raw(angle, self.raw, sign)


@dataclass
class ZeroSet:
    """The zeros recorded by one stage, plus how they came to be."""

    joints: dict[str, JointZero] = field(default_factory=dict)
    # Each entry records a closed-form re-definition: which joint, how far, why.
    history: list[dict] = field(default_factory=list)

    def add(self, name: str, raw: int, source: str = "mechanical",
            note: str = "") -> JointZero:
        z = JointZero(name=name, raw=int(raw), source=source, note=note)
        self.joints[name] = z
        return z

    def to_dict(self) -> dict:
        return {"joints": {n: z.to_dict() for n, z in self.joints.items()},
                "history": list(self.history)}

    @classmethod
    def from_dict(cls, d: dict | None) -> "ZeroSet":
        if not d:
            return cls()
        return cls(joints={n: JointZero.from_dict(v)
                           for n, v in (d.get("joints") or {}).items()},
                   history=list(d.get("history") or []))

    def record_shift(self, name: str, delta_rad: float, reason: str,
                     sign: int = 1) -> None:
        """Move a zero and note why, keeping the count and the reason together.

        The caller is responsible for applying the matching closed-form transform
        to the results, which is what keeps the pair intact. This only records it.
        """
        if name not in self.joints:
            raise KeyError(f"no zero recorded for {name}")
        old = self.joints[name].raw
        new = int(round(servos.rad_to_raw(delta_rad, old, sign)))
        self.joints[name].raw = new
        self.joints[name].source = "derived"
        self.history.append({
            "joint": name, "from_raw": int(old), "to_raw": new,
            "delta_deg": float(np.rad2deg(delta_rad)), "reason": reason,
        })

    def describe(self) -> str:
        lines = []
        for name, z in sorted(self.joints.items()):
            extra = f"  ({z.note})" if z.note else ""
            lines.append(f"  {name:<16} raw {z.raw:>5}  {z.source}{extra}")
        for h in self.history:
            lines.append(f"  moved {h['joint']} by {h['delta_deg']:+.3f} deg "
                         f"({h['from_raw']} -> {h['to_raw']}): {h['reason']}")
        return "\n".join(lines) if lines else "  (no zeros recorded)"


def check_pairing(recorded: ZeroSet, live: dict[str, int],
                  tolerance: int = PAIRING_TOLERANCE_COUNTS) -> list[str]:
    """Which joints are no longer at the posture their zero was recorded at.

    Returns a human-readable complaint per joint. An empty list means the
    recorded zeros still describe this robot.

    This does NOT mean the robot must sit at its zero posture -- it may be
    anywhere. It compares the recorded count against itself across runs, catching
    the case where something else rewrote the encoder's frame of reference: a
    firmware homing offset changed, a servo was swapped, or a horn was refitted.
    """
    problems = []
    for name, z in sorted(recorded.joints.items()):
        if name not in live:
            problems.append(f"{name}: no live reading to compare against")
            continue
        # Both are counts in the same frame, so an unwrapped difference of zero is
        # what a healthy pairing looks like when the joint is at its zero.
        drift = abs(servos.unwrap_delta(int(live[name]) - z.raw))
        if drift > tolerance:
            problems.append(
                f"{name}: reads {live[name]}, zero recorded at {z.raw} "
                f"({drift} counts, {np.rad2deg(drift * 2 * np.pi / 4096):.1f} deg)")
    return problems


def is_paired(results: dict | None, zero_set: ZeroSet) -> bool:
    """True when a result was solved against exactly these zeros.

    A stage writes the zeros it used into its own result. If the two disagree,
    the transform in that result places the camera using a joint angle convention
    that no longer applies.
    """
    if not results:
        return False
    stored = ZeroSet.from_dict(results.get("zeros"))
    if set(stored.joints) != set(zero_set.joints):
        return False
    return all(stored.joints[n].raw == z.raw for n, z in zero_set.joints.items())


def pairing_advice(stage_number: str) -> str:
    """What to do about a broken pairing. Recoverable, so say so."""
    return (
        f"  The recorded zeros no longer match what stage {stage_number} solved\n"
        "  against. This is recoverable and needs no recapture: a change of zero\n"
        "  is an exact change of body frame, so the existing result can be\n"
        "  corrected in closed form. Re-run the head-zero stage to apply it.\n"
        "  Do NOT use the current result as-is: a 3-degree discrepancy puts the\n"
        "  camera 13 mm away from where it really is, with no visible symptom.")
