"""Persistent workflow state with one-owner rerun authorization."""
from __future__ import annotations

import json
import os
import secrets
import shutil
import tempfile
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from . import mounting
from .i18n import (LocalizedFileExistsError, LocalizedRuntimeError,
                   LocalizedValueError, text)
from .registry import BY_KEY, STAGES, StageGuide
from .workspace import Workspace

# States that mean "this stage has run and its outputs are on disk". Stage 8 can
# finish having missed its accuracy gates, which is still a finished stage: it
# archives, re-runs and discards exactly like a clean pass, and differs only in
# what the dashboard badge says.
DONE_STATES = {"completed", "completed_large_error", "stale"}


@dataclass
class StageStatus:
    key: str
    number: int
    title: str
    state: str
    reason: str
    missing: list[str]
    outputs: list[str]
    action: str
    completion: str
    hardware: bool
    # A stable identifier for `reason`, so the interface can show it in
    # whichever language the operator picked.
    reason_key: str = ""


class WorkflowEngine:
    """Compute workflow state without touching hardware or legacy results."""

    def __init__(self, workspace: Workspace):
        self.workspace = workspace
        self.state_path = workspace.root / "workflow.json"
        self.state = self._load_state()
        self._invalidate_orphaned_run()

    def _load_state(self) -> dict[str, Any]:
        if self.state_path.is_file():
            import json
            return json.loads(self.state_path.read_text())
        state = {
            "schema_version": 1,
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "active_stage": None,
            "mounting": mounting.NORMAL,
            "runs": {},
        }
        self.workspace.save_json_atomic(self.state_path, state)
        return state

    def _save(self) -> None:
        self.workspace.save_json_atomic(self.state_path, self.state)

    def _invalidate_orphaned_run(self) -> None:
        key = self.state.get("active_stage")
        if not key:
            return
        run = self.state.get("runs", {}).get(key)
        if run and run.get("phase") in {"authorized", "running", "review"}:
            run["phase"] = "failed"
            run["error"] = text("err.runtimeLost")
            run["error_key"] = "err.runtimeLost"
            run["updated_at"] = datetime.now().isoformat(timespec="seconds")
            self.state["active_stage"] = None
            self._save()

    def _outputs_exist(self, stage: StageGuide) -> bool:
        if not stage.outputs or not all(
                self.workspace.result_path(name).is_file() for name in stage.outputs):
            return False
        run = self.state["runs"].get(stage.key, {})
        expected = run.get("input_fingerprints", {})
        return all(self.workspace.fingerprint(name) is not None for name in stage.outputs) and not any(
            expected.get(name) != self.workspace.fingerprint(name)
            for name in stage.requires if expected.get(name) is not None
        )

    def _missing(self, stage: StageGuide) -> list[str]:
        return [name for name in stage.requires
                if not self.workspace.result_path(name).is_file()]

    def _completed_with_large_error(self, stage: StageGuide) -> bool:
        """True when stage 8 finished but missed its accuracy gates.

        Stage 8 measures rather than gates, so a run that misses the thresholds
        is still a finished stage and its result is still saved. Refusing to
        accept it left the operator no way out but to end the session, after
        which the stage read Ready again and the report was gone. The dashboard
        distinguishes the two outcomes instead of the engine rejecting one.
        """
        if "validation" not in stage.outputs:
            return False
        path = self.workspace.result_path("validation")
        try:
            payload = json.loads(path.read_text())
        except (OSError, ValueError):
            return False
        return payload.get("passed") is not True

    def status(self, stage: StageGuide) -> StageStatus:
        run = self.state["runs"].get(stage.key, {})
        if run.get("phase") in {"authorized", "running", "review"}:
            phase = run["phase"]
            reason = {
                "authorized": "This run is confirmed; waiting for the stage service to start.",
                "running": "The stage is running; return to it to carry on.",
                "review": "This stage is finished; the result is being saved automatically.",
            }[phase]
            reason_key = f"reason.{phase}"
            state = phase
            missing = []
        elif self._outputs_exist(stage):
            missing = []
            if self._completed_with_large_error(stage):
                state = "completed_large_error"
                reason = ("The stage is complete and its result is saved, but "
                          "the measured error is too large to deploy.")
                reason_key = "reason.completedLargeError"
            else:
                state, reason = "completed", "The result is complete; view it, or archive it explicitly to re-run."
                reason_key = "reason.completed"
        else:
            missing = self._missing(stage)
            files_present = any(self.workspace.result_path(name).is_file()
                                for name in stage.outputs)
            run_inputs = self.state["runs"].get(stage.key, {}).get("input_fingerprints", {})
            inputs_changed = any(
                run_inputs.get(name) is not None
                and run_inputs.get(name) != self.workspace.fingerprint(name)
                for name in stage.requires
            )
            if missing:
                state = "blocked"
                reason = "Finish the earlier stages first."
                reason_key = "reason.blocked"
            elif files_present and inputs_changed:
                state = "stale"
                reason = "An earlier result has changed; archive the old outputs and run again."
                reason_key = "reason.stale"
            else:
                state = "ready"
                reason = "The prerequisites are met; ready to start."
                reason_key = "reason.ready"
        return StageStatus(
            stage.key, stage.number, stage.title, state, reason, missing,
            list(stage.outputs), stage.action, stage.completion, stage.hardware,
            reason_key)

    def overview(self) -> list[dict[str, Any]]:
        return [asdict(self.status(stage)) for stage in STAGES]

    # ---- how the robot is standing ---------------------------------------
    #
    # Every result is measured in a frame that depends on which way the chassis
    # faces, so this is a property of the whole run, not of any one stage. It is
    # declared up front and never inferred while running: an inferred mounting
    # cannot tell a genuinely back-to-front robot from a head pan zero set half a
    # turn out, and silently "correcting" the second would bake in the mistake.

    @property
    def mounting(self) -> str:
        """The declared mounting for this workspace, defaulting to normal."""
        return self.state.get("mounting", mounting.NORMAL)

    def mounting_change(self, target: str) -> dict[str, Any]:
        """What switching to `target` would cost, without doing it.

        The page asks before acting, so it needs the damage in advance.
        """
        mounting.check(target)
        done = [stage for stage in STAGES
                if self.status(stage).state in DONE_STATES]
        return {
            "current": self.mounting,
            "target": target,
            "changes": target != self.mounting,
            "discards": [stage.key for stage in done],
            "titles": [stage.title for stage in done],
        }

    def set_mounting(self, target: str) -> dict[str, Any]:
        """Declare the mounting, discarding results measured the other way.

        Results from the other mounting are archived rather than deleted: they
        are still a record of a real measurement session, and the operator may
        have switched by mistake.
        """
        mounting.check(target)
        if self.state.get("active_stage"):
            raise LocalizedRuntimeError("err.anotherRunning")
        plan = self.mounting_change(target)
        if not plan["changes"]:
            self.state["mounting"] = target
            self._save()
            return plan
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        for stage in STAGES:
            if stage.key in plan["discards"]:
                self._archive_outputs(stage, f"mounting_{stamp}")
                self.state["runs"].pop(stage.key, None)
        self.state["mounting"] = target
        self.state["mounting_changed_at"] = datetime.now().isoformat(
            timespec="seconds")
        self._save()
        return plan

    def recommended(self) -> StageStatus | None:
        statuses = [self.status(stage) for stage in STAGES]
        active = next((item for item in statuses
                       if item.state in {"authorized", "running", "review"}), None)
        if active:
            return active
        return next((item for item in statuses if item.state == "ready"), None)

    def authorize(self, key: str, rerun: bool = False) -> dict[str, Any]:
        stage = BY_KEY[key]
        status = self.status(stage)
        if status.state == "blocked":
            raise LocalizedValueError("err.requiresMissing",
                                      items=", ".join(status.missing))
        if status.state in {"authorized", "running", "review"}:
            return self.state["runs"][key]
        if status.state in DONE_STATES and not rerun:
            raise LocalizedFileExistsError("err.resultExists")
        if self.state.get("active_stage") not in (None, key):
            raise LocalizedRuntimeError("err.anotherRunning")
        archive = None
        if status.state in DONE_STATES:
            archive = self._archive_outputs(stage)
        input_fingerprints = {
            name: self.workspace.fingerprint(name) for name in stage.requires
        }
        run = {
            "run_id": f"{key}-{secrets.token_hex(6)}",
            "stage": key,
            "phase": "authorized",
            "authorized_at": datetime.now().isoformat(timespec="seconds"),
            "rerun": status.state in {"completed", "completed_large_error"},
            "archive": archive,
            "input_fingerprints": input_fingerprints,
            "confirmation_consumed": True,
        }
        self.state["runs"][key] = run
        self.state["active_stage"] = key
        self._save()
        return run

    def _archive_outputs(self, stage: StageGuide, reason: str = "") -> str:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        label = reason or stamp
        target = self.workspace.runs / f"archive_{stage.key}_{label}"
        target.mkdir(parents=True, exist_ok=True)
        for name in stage.outputs:
            path = self.workspace.result_path(name)
            if path.is_file():
                shutil.move(str(path), target / path.name)
        self.workspace.save_json_atomic(target / "manifest.json", {
            "kind": "mounting_change" if reason else "rerun_archive",
            "stage": stage.key,
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "outputs": list(stage.outputs), "committed": True,
        })
        return str(target)

    def set_phase(self, key: str, phase: str) -> None:
        if phase not in {"running", "review", "failed", "cancelled"}:
            raise ValueError(f"unsupported phase: {phase}")
        run = self.state["runs"].get(key)
        if not run:
            raise RuntimeError("stage has no authorized run")
        run["phase"] = phase
        run["updated_at"] = datetime.now().isoformat(timespec="seconds")
        if phase in {"failed", "cancelled"}:
            self.state["active_stage"] = None
        self._save()

    def fail_authorization(self, key: str, message: str) -> None:
        run = self.state["runs"].get(key)
        if run:
            run["phase"] = "failed"
            run["error"] = message
            run["updated_at"] = datetime.now().isoformat(timespec="seconds")
        self.state["active_stage"] = None
        self._save()

    @staticmethod
    def _validate_stage_outputs(key: str, outputs: dict[str, dict[str, Any]]) -> None:
        if key == "arms":
            touch = outputs["touch"]
            if touch.get("complete") is not True:
                raise LocalizedValueError("err.armsIncomplete")
            if not touch.get("arms", {}).get("left_arm") or not touch.get("arms", {}).get("right_arm"):
                raise LocalizedValueError("err.armsMissing")
        # Stage 8 measures; it does not gate. A run whose accuracy gates failed
        # is still a completed measurement, and refusing it left the operator
        # with no way to finish the stage: the only exit was to end the session,
        # after which the stage was Ready again and the report they had just
        # generated was gone. The verdict travels with the result instead, and
        # the dashboard says "Completed with a large error".

    def complete(self, key: str, outputs: dict[str, dict[str, Any]],
                 artifacts: dict[str, bytes] | None = None) -> None:
        stage = BY_KEY[key]
        run = self.state["runs"].get(key)
        if not run or not run.get("confirmation_consumed"):
            raise RuntimeError("stage completion requires dashboard authorization")
        if run.get("phase") != "review":
            raise LocalizedRuntimeError("err.notReviewState")
        if set(outputs) != set(stage.outputs):
            raise ValueError(f"output contract mismatch for {key}")
        self._validate_stage_outputs(key, outputs)
        run_dir = self.workspace.runs / run["run_id"]
        run_dir.mkdir(parents=True, exist_ok=True)
        artifacts = artifacts or {}
        for name, payload in outputs.items():
            body = dict(payload)
            body.setdefault("tool_metadata", {})
            body["tool_metadata"].update({
                "schema_version": 1,
                "tool_version": "0.1.0",
                "stage_key": key, "stage_number": stage.number,
                "run_id": run["run_id"], "accepted": True,
                "input_fingerprints": run["input_fingerprints"],
            })
            self.workspace.save_json_atomic(run_dir / f"{name}.json", body)
        for name, content in artifacts.items():
            if Path(name).name != name:
                raise ValueError(f"invalid artifact name: {name}")
            artifact_path = run_dir / name
            artifact_path.write_bytes(content)
        import hashlib
        output_fingerprints = {
            name: hashlib.sha256((run_dir / f"{name}.json").read_bytes()).hexdigest()
            for name in stage.outputs
        }
        self.workspace.save_json_atomic(run_dir / "manifest.json", {
            "kind": "stage_result", "schema_version": 1,
            "tool_version": "0.1.0", "stage": key, "stage_number": stage.number,
            "run_id": run["run_id"], "inputs": run["input_fingerprints"],
            "outputs": output_fingerprints,
            "artifacts": {
                name: hashlib.sha256((run_dir / name).read_bytes()).hexdigest()
                for name in artifacts
            },
            "committed": True,
        })
        activation = Path(tempfile.mkdtemp(prefix=f".activate-{key}-", dir=self.workspace.root))
        try:
            staged_results = activation / "results"
            staged_results.mkdir()
            for name in stage.outputs:
                shutil.copy2(run_dir / f"{name}.json", staged_results / f"{name}.json")
            for name in artifacts:
                shutil.copy2(run_dir / name, staged_results / name)
            backups: dict[Path, Path] = {}
            try:
                paths = [self.workspace.result_path(name) for name in stage.outputs]
                paths += [self.workspace.results / name for name in artifacts]
                for path in paths:
                    if path.exists():
                        backup = activation / "backup" / path.name
                        backup.parent.mkdir(parents=True, exist_ok=True)
                        os.replace(path, backup)
                        backups[path] = backup
                for source in staged_results.iterdir():
                    os.replace(source, self.workspace.results / source.name)
            except Exception:
                for path in paths:
                    if path.exists() and path not in backups:
                        path.unlink()
                for path, backup in backups.items():
                    os.replace(backup, path)
                raise
        finally:
            shutil.rmtree(activation, ignore_errors=True)
        run["phase"] = "completed"
        run["completed_at"] = datetime.now().isoformat(timespec="seconds")
        self.state["active_stage"] = None
        self._save()
