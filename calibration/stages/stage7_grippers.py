"""Stage 7: gripper opening angle."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import common

PLAN = """
The jaws are driven by rotary servos and the model's jaw joint is revolute, so the
output is an ANGLE, not a linear width. This reduces to a zero and a sign per
gripper. Hysteresis is ignored by choice.

  - drive each gripper to several openings including fully closed and fully open
  - record raw counts at each, and fit counts -> jaw angle
  - as a one-off sanity check, measure the physical opening at fully closed and
    fully open and compare against the model at the same angle. This does not
    enter the fit; it tests whether the XML jaw geometry is trustworthy, which
    matters because stage 5 solves the touch point in that same geometry

Gates: sample count, fit residual in degrees, range consistent with the XML limits.
"""


def main() -> int:
    try:
        common.require_results("senses")
    except common.Aborted:
        return 1
    return common.not_implemented("7", "Gripper opening angle", PLAN)


if __name__ == "__main__":
    raise SystemExit(main())
