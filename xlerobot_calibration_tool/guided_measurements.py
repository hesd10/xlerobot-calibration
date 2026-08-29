"""Parse confirmed measurements emitted by the legacy algorithms.

Stage 3 (senses) reports a travel direction per joint; stage 4 (arm_ranges)
reports a table of raw encoder counts for the arm being zeroed, and during a
sweep publishes a live travel snapshot. All of it is read back out of the legacy
stage rather than recomputed here, so what the operator sees and confirms is
exactly what the legacy stage recorded.
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from .guided_interaction import JOINT_PATTERN
from .i18n import joint_label
from .mounting import NORMAL


_JOINT_HEADER = re.compile(rf"^\s{{2}}({JOINT_PATTERN})\s*$", re.MULTILINE)
_ATTEMPT_START = re.compile(
    r"^\s*(?::\s*)?start:\s*(-?\d+)\s+counts\s*$", re.MULTILINE)
_END = re.compile(
    r"^\s*(?::\s*)?end:\s*(-?\d+)\s+counts\s*$", re.MULTILINE)
_MOVED = re.compile(
    r"^\s+moved\s+([+-]?\d+)\s+counts\s+\(([0-9]+(?:\.[0-9]+)?)\s+deg\)"
    r"\s+->\s+sense\s+([+-]1)\s*$", re.MULTILINE)


def _number(pattern: re.Pattern[str], section: str) -> int | None:
    match = pattern.search(section)
    return int(match.group(1)) if match else None


def extract_measurements(log_text: str,
                         mounting_name: str = NORMAL) -> list[dict[str, Any]]:
    """Extract only values printed by the legacy MotionTracker algorithm.

    The parser never derives travel from endpoint readings. In particular, the
    ``travel_counts`` value comes only from the legacy ``moved`` line, which is
    produced from MotionTracker.position after continuous unwrap_delta updates.
    """
    matches = list(_JOINT_HEADER.finditer(log_text))
    latest: dict[str, dict[str, Any]] = {}
    for index, match in enumerate(matches):
        joint = match.group(1)
        end = matches[index + 1].start() if index + 1 < len(matches) else len(log_text)
        section = log_text[match.end():end]
        attempts = list(_ATTEMPT_START.finditer(section))
        # A failed short move is followed by a fresh start prompt. Keep the
        # latest attempt only; the legacy algorithm does the same by retrying
        # inside measure_joint().
        attempt_start = attempts[-1].start() if attempts else 0
        attempt = section[attempt_start:]
        moved = _MOVED.search(attempt)
        record: dict[str, Any] = {
            "joint": joint,
            "label": joint_label(joint, mounting_name),
            "start_raw": _number(_ATTEMPT_START, attempt),
            "end_raw": _number(_END, attempt),
            "travel_counts": int(moved.group(1)) if moved else None,
            "travel_deg": float(moved.group(2)) if moved else None,
            "sense": int(moved.group(3)) if moved else None,
            "confirmed": moved is not None,
            "calculation": "continuous MotionTracker / unwrap_delta accumulation"
            if moved else "waiting for endpoint confirmation; no accumulated delta yet",
        }
        latest[joint] = record
    return list(latest.values())


# The rough-zero table printed by stage4_zeros.pose_arm(): a "joint / raw"
# header followed by one indented "<joint name> <count>" line per joint.
_ZERO_TABLE_HEADER = re.compile(r"^\s+joint\s+raw\s*$", re.MULTILINE)
_ZERO_ROW = re.compile(rf"^\s+({JOINT_PATTERN})\s+(-?\d+)\s*$", re.MULTILINE)
_ZERO_PROMPT = "accept these as the rough zero"


def extract_zero_readings(log_text: str,
                          mounting_name: str = NORMAL) -> list[dict[str, Any]]:
    """Extract the rough-zero encoder table the operator is being asked about.

    A table is only the answer to one question: the "accept these as the rough
    zero" prompt that stage4_zeros.pose_zero() prints directly beneath it. Once
    that question has been answered the readings are history, so they are
    dropped rather than carried forward.

    Reporting the latest table unconditionally is what put the previous arm's
    counts under the next arm's prompt. The log only grows, so the left arm's
    table stayed on screen through its whole sweep and through "is the right arm
    posed?", right up until the right arm printed a table of its own -- six rows
    of stale counts sitting under a question about a different arm, and under a
    re-pose that had not been read yet. Nothing marked them stale, and in a
    back-to-front mounting they even carried the opposite arm's stored names.

    Rows are taken verbatim from the legacy output, never recomputed.
    """
    headers = list(_ZERO_TABLE_HEADER.finditer(log_text))
    if not headers:
        return []
    section = log_text[headers[-1].end():]
    # The prompt below the table is what makes it current. Its absence means the
    # table has not been printed in full yet; its presence with an answer after
    # it means the question is closed.
    stop = section.lower().find(_ZERO_PROMPT)
    if stop < 0:
        return []
    if _answered(section[stop:]):
        return []
    section = section[:stop]
    out: list[dict[str, Any]] = []
    for match in _ZERO_ROW.finditer(section):
        joint = match.group(1)
        out.append({
            "joint": joint,
            "label": joint_label(joint, mounting_name),
            "raw": int(match.group(2)),
        })
    return out


def _answered(tail: str) -> bool:
    """Whether anything follows the prompt line other than the prompt itself.

    The stage writes the prompt without a trailing newline and blocks, so while
    it waits the prompt is the last thing in the log. Anything after it -- the
    echoed answer, the next heading -- means the operator has moved on.
    """
    _, _, after = tail.partition("\n")
    return bool(after.strip())


# The live sweep snapshot written by stage4_zeros.Tracker. Anything older than
# this is treated as absent: the file is removed when a sweep ends, so a
# lingering one means the writer died, and stale travel shown as live would tell
# the operator a joint is covered when nothing is being read.
LIVE_SNAPSHOT_MAX_AGE_S = 3.0


def read_live_ranges(data_dir: Path | str, now: float,
                     max_age_s: float = LIVE_SNAPSHOT_MAX_AGE_S,
                     mounting_name: str = NORMAL) -> dict[str, Any]:
    """Read the current sweep snapshot, or return an empty one.

    Values are passed through untouched. Every travel figure in the snapshot was
    accumulated by the legacy ``Travel`` object from continuous ``unwrap_delta``
    steps, so it already accounts for the 0/4095 encoder seam; recomputing any of
    it here from raw endpoints would reintroduce exactly the wrap bug the legacy
    tracker exists to avoid.
    """
    path = Path(data_dir) / "arm_range_live.json"
    try:
        snapshot = json.loads(path.read_text())
    except (OSError, ValueError):
        return {"joints": []}
    if not isinstance(snapshot, dict):
        return {"joints": []}
    updated_at = snapshot.get("updated_at")
    if not isinstance(updated_at, (int, float)):
        return {"joints": []}
    if now - float(updated_at) > max_age_s:
        return {"joints": []}
    joints = []
    for entry in snapshot.get("joints") or []:
        if not isinstance(entry, dict):
            continue
        name = str(entry.get("joint", ""))
        joints.append({**entry, "label": joint_label(name, mounting_name)})
    return {"arm": snapshot.get("arm", ""), "joints": joints,
            "updated_at": float(updated_at)}
