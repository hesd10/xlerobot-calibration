"""Guided entry point for the XLeRobot real-to-sim calibration.

Run this and follow the prompts:

    python calibration/run.py              show progress, offer the next stage
    python calibration/run.py --status     progress only
    python calibration/run.py --stage 1    run one stage by number or key
    python calibration/run.py --plan       the whole procedure with dependencies

The runner owns ordering and safety: it will not start a stage whose inputs are
missing, and it explains what to do about it instead of failing with a traceback.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from core import stages as reg  # noqa: E402
from core import storage  # noqa: E402

BAR = "=" * 72


def banner(text: str) -> None:
    print(f"\n{BAR}\n{text}\n{BAR}")


MARK = {"done": "[x]", "ready": "[ ]", "blocked": "[-]"}


def show_status() -> None:
    banner("XLeRobot calibration progress")
    for stage, state in reg.progress():
        flag = " (optional)" if stage.optional else ""
        print(f"  {MARK[state]} stage {stage.number:>2}  {stage.title}{flag}")
        if state == "blocked":
            missing = reg.missing_requirements(stage)
            print(f"         waiting on: {', '.join(missing)}")
        elif state != "done":
            partial = reg.partial_outputs(stage)
            if partial:
                print(f"         partly done: {', '.join(partial)}")

    nxt = reg.next_stage()
    print()
    if nxt is None:
        print("  Every stage is complete. Run stage 8 again any time to re-validate.")
    else:
        print(f"  Next: stage {nxt.number}, {nxt.title}")
        print(f"  Purpose: {nxt.purpose}")


def show_plan() -> None:
    banner("Full procedure")
    print("  Order matters. Intrinsics come first because every pose downstream")
    print("  is solved through PnP, and PnP with a wrong K returns a confidently")
    print("  wrong answer rather than an obviously broken one.\n")
    for stage in reg.STAGES:
        state = reg.stage_state(stage)
        flag = "  (optional)" if stage.optional else ""
        print(f"  {MARK[state]} Stage {stage.number}: {stage.title}{flag}")
        print(f"      {stage.purpose}")
        if stage.requires:
            print(f"      needs:     {', '.join(stage.requires)}")
        if stage.produces:
            print(f"      produces:  {', '.join(stage.produces)}")
        if stage.locks:
            print(f"      then keep fixed: {', '.join(stage.locks)}")
        print()


def show_stage_help(stage: reg.Stage) -> None:
    banner(f"Stage {stage.number}: {stage.title}")
    print(f"  {stage.purpose}\n")
    if stage.manual:
        for line in stage.manual.rstrip().splitlines():
            print(f"  {line}")
        print()


def find_stage(token: str) -> reg.Stage | None:
    """Match a stage by key or exact number.

    The number is matched exactly: stage "-1" must not be reachable by typing
    "1", or asking for intrinsics would silently run the prep stage instead.
    """
    token = token.strip().lower()
    for stage in reg.STAGES:
        if stage.matches(token):
            return stage
    return None


def run_stage(stage: reg.Stage, extra_args: list[str] | None = None) -> int:
    state = reg.stage_state(stage)
    if state == "blocked":
        missing = reg.missing_requirements(stage)
        print(f"\nStage {stage.number} cannot run yet. Missing: {', '.join(missing)}")
        for name in missing:
            owner = next((s for s in reg.STAGES if name in s.produces), None)
            if owner:
                print(f"  '{name}' comes from stage {owner.number} ({owner.title})")
        return 1

    if state == "done":
        print(f"\nStage {stage.number} is already done. Its outputs: "
              f"{', '.join(stage.produces)}")
        if input("  Run it again and overwrite? [y/N] ").strip().lower() != "y":
            return 0

    if stage.script is None:
        show_stage_help(stage)
        print("  This stage has no script; it is a manual step.")
        return 0

    script = HERE / stage.script
    if not script.is_file():
        show_stage_help(stage)
        print(f"  Not implemented yet: {stage.script}")
        return 1

    show_stage_help(stage)
    cmd = [sys.executable, "-u", str(script)] + list(extra_args or [])
    sys.stdout.flush()
    return subprocess.call(cmd)


def interactive() -> int:
    show_status()
    nxt = reg.next_stage()
    if nxt is None:
        return 0

    show_stage_help(nxt)
    if nxt.script and not (HERE / nxt.script).is_file():
        print(f"  Not implemented yet: {nxt.script}")
        return 1

    answer = input(f"  Start stage {nxt.number} now? [y/N] ").strip().lower()
    if answer != "y":
        print("  Nothing run. Use --stage to pick a specific one.")
        return 0
    return run_stage(nxt)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Guided XLeRobot real-to-sim calibration")
    parser.add_argument("--status", action="store_true", help="show progress only")
    parser.add_argument("--plan", action="store_true", help="show the full procedure")
    # Numbers are passed as strings so "-1" is not mistaken for a flag; argparse
    # needs --stage=-1 or --stage prep for the preparation stage.
    parser.add_argument("--stage", metavar="N",
                        help="run one stage by number or key, e.g. 1 or prep")
    parser.add_argument("--results", action="store_true", help="list saved results")
    parser.add_argument("--sessions", action="store_true", help="list capture sessions")
    args, extra = parser.parse_known_args()

    if args.plan:
        show_plan()
        return 0

    if args.results:
        banner("Saved results")
        found = False
        for stage in reg.STAGES:
            for name in stage.produces:
                data = storage.load_result(name)
                if data:
                    found = True
                    print(f"  {name:<26} saved {data.get('saved_at', '?')}")
        if not found:
            print("  none yet")
        return 0

    if args.sessions:
        banner("Capture sessions")
        paths = storage.list_sessions()
        if not paths:
            print("  none yet")
        for path in paths:
            try:
                print(storage.CaptureSession(path).describe())
            except Exception as exc:
                print(f"  {path.name}: unreadable ({exc})")
        return 0

    if args.stage:
        stage = find_stage(args.stage)
        if stage is None:
            print(f"No such stage: {args.stage}")
            print("  Valid: " + ", ".join(f"{s.number}/{s.key}" for s in reg.STAGES))
            return 2
        return run_stage(stage, extra)

    if args.status:
        show_status()
        return 0

    return interactive()


if __name__ == "__main__":
    sys.exit(main())
