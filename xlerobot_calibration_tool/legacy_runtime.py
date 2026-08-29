"""Isolated runtime for the stable legacy calibration implementation.

Each run receives a private copy of ``calibration``. Stable algorithm source is
not modified, and all legacy writes land inside the run directory. Only one
runtime may be active, enforced by WorkflowEngine.
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import signal
import socket
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from urllib.error import URLError
from urllib.request import urlopen

from . import mounting
from .adapters import AdapterStatus
from .guided_interaction import describe as describe_interaction
from .guided_measurements import (
    extract_measurements, extract_zero_readings, read_live_ranges)
from .registry import BY_KEY
from .workspace import Workspace


WEB_STAGES = {
    "prepare": ("prepare_web.py", None),
    "intrinsics": ("stages/stage1_web.py", "HOST, PORT = \"127.0.0.1\", 8422"),
    "head": ("stages/stage23_head.py", "HOST, PORT = \"127.0.0.1\", 8423"),
    "arms": ("stages/stage5_fusion.py", "HOST, PORT = \"0.0.0.0\", 5005"),
    "verify": ("stages/stage8_verify.py", "HOST, PORT = \"0.0.0.0\", 5008"),
}


# The separator for page headings, held here rather than written inline. An
# escape inside an f-string's braces is a syntax error before Python 3.12, and
# pyproject declares support for 3.10, so the four call sites that pass this
# would stop the module importing at all on a version we claim to run on.
HEADING_DOT = " \u00b7 "


def _heading(key: str, separator: str = ": ") -> str:
    """The stage's browser-tab heading, numbered from the registry.

    The legacy pages carry their own numbering from an older ordering. These
    headings replace it, so they must follow the registry rather than repeat
    a number here: stages have been reordered once already, and every literal
    copy is a place for the two to drift apart.
    """
    stage = BY_KEY[key]
    return f"Stage {stage.number}{separator}{stage.title}"

DETERMINISTIC_STAGES = {
    "normalize": "stages/stage5b_head_zero.py",
}
INTERACTIVE_STAGES = {
    "senses": "stages/stage2_senses.py",
    "arm_ranges": "stages/stage4_zeros.py",
}


def _stage_measurements(stage_key: str, log_tail: str,
                        mounting_name: str = mounting.NORMAL) -> list[dict]:
    """Readings to show alongside the current prompt, per stage.

    Each interactive stage prints a different table, so the confirmation screen
    would otherwise show nothing for anything but senses -- leaving the operator
    to accept a rough zero they cannot see.
    """
    if stage_key == "senses":
        return extract_measurements(log_tail, mounting_name)
    if stage_key == "arm_ranges":
        return extract_zero_readings(log_tail, mounting_name)
    return []


@dataclass
class RuntimeHandle:
    run_id: str
    stage_key: str
    root: Path
    process: subprocess.Popen
    port: int | None
    log_path: Path
    interactive: bool = False


class LegacyRuntime:
    def __init__(self, workspace: Workspace, source_root: Path | str | None = None):
        self.workspace = workspace
        self.source_root = Path(source_root or Path(__file__).resolve().parent.parent).resolve()
        self.handle: RuntimeHandle | None = None
        self.input_lock = threading.Lock()

    @property
    def mounting(self) -> str:
        """The declared mounting, so prompts can name physical sides.

        Falls back to normal whenever the declaration cannot be read. A prompt
        is worth showing even from a runtime that has no workspace attached yet,
        and normal is the reading that leaves every name as the model stores it.
        """
        workspace = getattr(self, "workspace", None)
        if workspace is None or not workspace.state_file.is_file():
            return mounting.NORMAL
        try:
            declared = json.loads(
                workspace.state_file.read_text()).get("mounting")
        except (ValueError, OSError):
            return mounting.NORMAL
        return declared if declared in mounting.MOUNTINGS else mounting.NORMAL

    def _prepare(self, run_id: str) -> Path:
        root = self.workspace.root / "runtime" / run_id
        if root.exists():
            shutil.rmtree(root)
        source = self.source_root / "calibration"
        if not source.is_dir():
            raise FileNotFoundError(f"Cannot find the stable algorithm directory: {source}")
        shutil.copytree(source, root / "calibration", ignore=shutil.ignore_patterns(
            "results", "data", "__pycache__", "*.pyc"))
        prepare_source = self.source_root / "xlerobot_calibration_tool" / "prepare_web.py"
        if prepare_source.is_file():
            shutil.copy2(prepare_source, root / "calibration" / "prepare_web.py")
        self._apply_compatibility_patches(root / "calibration")
        shutil.copytree(self.workspace.results, root / "calibration" / "results",
                        dirs_exist_ok=True)
        board_result = self.workspace.result_path("board")
        if board_result.is_file():
            import json
            wrapped = json.loads(board_result.read_text())
            boards = wrapped.get("boards") if isinstance(wrapped, dict) else None
            if isinstance(boards, dict):
                (root / "calibration" / "board.json").write_text(
                    json.dumps(boards, ensure_ascii=False, indent=2) + "\n")
        refined_zeros = self.workspace.result_path("zeros_refined")
        if refined_zeros.is_file():
            shutil.copy2(refined_zeros, root / "calibration" / "results" / "zeros.json")
        # The declared mounting travels with the results it describes. A stage
        # that solves a frame has to know which way the chassis is facing, and
        # this isolated copy is all it can see.
        if self.workspace.state_file.is_file():
            shutil.copy2(self.workspace.state_file, root / "calibration" / "workflow.json")
        (root / "calibration" / "data").mkdir(parents=True, exist_ok=True)
        tools = self.source_root / "tools"
        if tools.is_dir():
            os.symlink(tools, root / "tools", target_is_directory=True)
        return root

    @staticmethod
    def _apply_compatibility_patches(calibration: Path) -> None:
        ranges_path = calibration / "core" / "ranges.py"
        source = ranges_path.read_text()
        marker = "tolerance: float = 1.0,"
        if marker not in source:
            raise RuntimeError("Cannot locate the range boundary tolerance declaration")
        ranges_path.write_text(source.replace(
            marker, "tolerance: float = 8.0,", 1))

        arms_path = calibration / "stages" / "stage5_fusion.py"
        if arms_path.is_file():
            source = arms_path.read_text()
            confirmation = (
                '    print("\\n  !!  The board and the base must stay exactly where '
                'they were in Stage 3")\n'
                '    if not common.confirm("Board and base have not moved", False):\n'
                '        return 1\n'
            )
            if confirmation not in source:
                raise RuntimeError(
                    "Cannot locate the duplicate safety confirmation in the arm stage")
            source = source.replace(
                confirmation,
                '    print("\\n  Board and base state were authorised at launch by the '
                'dashboard")\n',
                1,
            )
            source = source.replace("<title>Stage 5 Fusion</title>",
                                    f"<title>{_heading('arms')}</title>")
            source = source.replace(
                "<h1>Stage 5 Fusion &mdash; arm fusion calibration</h1>",
                f"<h1>{_heading('arms', HEADING_DOT)}</h1>")
            source = source.replace(
                "(relative to Stage 4)",
                f"(relative to the stage {BY_KEY['arm_ranges'].number} rough zeros)")
            arms_path.write_text(source)

        # The legacy pages carry stage numbers from an older ordering. Only the
        # headings are rewritten, so the numbers follow the registry; the rest
        # of each page is left exactly as its own stage wrote it.
        ui_replacements = {
            "stage1_web.py": (
                ("<title>Stage 1: camera intrinsics</title>",
                 f"<title>{_heading('intrinsics')}</title>"),
                ("<h1>Stage 1: camera intrinsics</h1>",
                 f"<h1>{_heading('intrinsics', HEADING_DOT)}</h1>"),
            ),
            "stage23_head.py": (
                ("<title>Stage: head, world frame and head camera</title>",
                 f"<title>{_heading('head')}</title>"),
                ("<h1>Stage: head, world frame and head camera</h1>",
                 f"<h1>{_heading('head', HEADING_DOT)}</h1>"),
            ),
            "stage8_verify.py": (
                ("<title>Stage 8: three-camera verification</title>",
                 f"<title>{_heading('verify')}</title>"),
                ("<h1>Stage 8: independent three-camera verification</h1>",
                 f"<h1>{_heading('verify', HEADING_DOT)}</h1>"),
            ),
        }
        stages_dir = calibration / "stages"
        for filename, replacements in ui_replacements.items():
            path = stages_dir / filename
            if not path.is_file():
                continue
            source = path.read_text()
            for old, new in replacements:
                source = source.replace(old, new)
            path.write_text(source)

    @staticmethod
    def _free_port() -> int:
        with socket.socket() as sock:
            sock.bind(("127.0.0.1", 0))
            return int(sock.getsockname()[1])

    @staticmethod
    def _patch_port(script: Path, marker: str, port: int) -> None:
        source = script.read_text()
        if marker not in source:
            raise RuntimeError(f"Cannot locate the legacy stage port declaration: {marker}")
        if marker.startswith("HOST"):
            replacement = f'HOST, PORT = "127.0.0.1", {port}'
        else:
            replacement = f"PORT = {port}"
        script.write_text(source.replace(marker, replacement, 1))

    def start(self, stage_key: str, run_id: str) -> AdapterStatus:
        if self.handle and self.handle.process.poll() is None:
            raise RuntimeError("A stage subprocess is already running")
        if (stage_key not in WEB_STAGES and stage_key not in DETERMINISTIC_STAGES
                and stage_key not in INTERACTIVE_STAGES):
            raise NotImplementedError("The structured hardware adapter for this stage is not finished yet")
        root = self._prepare(run_id)
        interactive = stage_key in INTERACTIVE_STAGES
        if stage_key in WEB_STAGES:
            script, marker = WEB_STAGES[stage_key]
            port = self._free_port()
            extra = (["--workspace", str(root), "--port", str(port)]
                     if stage_key == "prepare" else [])
            if marker is not None:
                self._patch_port(root / "calibration" / script, marker, port)
        elif stage_key in DETERMINISTIC_STAGES:
            script, port, extra = DETERMINISTIC_STAGES[stage_key], None, []
        else:
            script, port, extra = INTERACTIVE_STAGES[stage_key], None, []
        log_path = root / "runtime.log"
        log = log_path.open("w", encoding="utf-8")
        process = subprocess.Popen(
            [sys.executable, "-u", str(root / "calibration" / script), *extra],
            cwd=root, stdin=subprocess.PIPE if interactive else subprocess.DEVNULL,
            stdout=log, stderr=subprocess.STDOUT, text=True, bufsize=1,
            start_new_session=True,
        )
        log.close()
        self.handle = RuntimeHandle(run_id, stage_key, root, process, port, log_path,
                                    interactive)
        if port is not None:
            self._wait_for_web(port, process)
            return AdapterStatus("running", "The stage page has started", {"port": port}, True)
        if interactive:
            # The child reaches its first input prompt almost immediately. Give its
            # unbuffered log a brief chance to arrive so the first Dashboard render
            # already contains a concrete action instead of a generic placeholder.
            deadline = time.monotonic() + 1.0
            log_tail = self.log_tail(120)
            interaction = self._describe(stage_key, log_tail)
            while not interaction.get("buttons") and time.monotonic() < deadline:
                if process.poll() is not None:
                    break
                time.sleep(0.05)
                log_tail = self.log_tail(120)
                interaction = self._describe(stage_key, log_tail)
            return AdapterStatus("running", "The interactive step has started", {
                "stage": stage_key, "run_id": run_id,
                "interactive": True, "interaction": interaction,
                "measurements": _stage_measurements(stage_key, log_tail,
                                                    self.mounting),
                "live_ranges": self.live_ranges(),
                "log_tail": log_tail}, True)
        return AdapterStatus("running", "The deterministic computation is running", {}, True)

    def _describe(self, stage_key: str, log_tail: str) -> dict:
        """The current prompt, with a token identifying which step it is."""
        interaction = describe_interaction(stage_key, log_tail, self.mounting)
        interaction["token"] = self._interaction_token(interaction)
        return interaction

    @staticmethod
    def _interaction_token(interaction: dict) -> str:
        """A fingerprint of which step the operator is on.

        Derived from the prompt's content, so it changes when the operator is
        moved to a different step and not merely when a sentence is reworded.
        """
        content = "\x1f".join((
            str(interaction.get("title", "")),
            str(interaction.get("instruction", "")),
            repr(interaction.get("buttons", [])),
        ))
        return hashlib.sha256(content.encode()).hexdigest()[:16]

    def _current_interaction(self) -> tuple[dict, str]:
        if not self.handle:
            return {}, ""
        # Same mounting as _describe(), or the token recomputed here would never
        # match the one the browser was given and every answer would be refused
        # as stale.
        interaction = describe_interaction(
            self.handle.stage_key, self.log_tail(120), self.mounting)
        return interaction, self._interaction_token(interaction)

    def live_ranges(self) -> dict:
        """The active sweep's live travel table, if a sweep is running.

        Read from the run's own data directory rather than the stable source
        tree, since each run works on a private copy.
        """
        if not self.handle or self.handle.stage_key != "arm_ranges":
            return {"joints": []}
        return read_live_ranges(
            self.handle.root / "calibration" / "data", time.time(),
            mounting_name=self.mounting)

    def send_input(self, value: str, interaction_token: str = "") -> AdapterStatus:
        with self.input_lock:
            if not self.handle or not self.handle.interactive:
                raise RuntimeError("This stage does not accept text input")
            if self.handle.process.poll() is not None or self.handle.process.stdin is None:
                raise RuntimeError("The stage process has already exited")
            before, current_token = self._current_interaction()
            if not interaction_token or interaction_token != current_token:
                raise RuntimeError("The current step has moved on; follow the new prompt.")
            self.handle.process.stdin.write(value + "\n")
            self.handle.process.stdin.flush()

            # wait_until_still() runs after an answer is consumed. Do not return
            # the old prompt while that settling/read step is still in progress,
            # otherwise the next key press can be queued for the following input().
            deadline = time.monotonic() + 12.0
            while time.monotonic() < deadline:
                if self.handle.process.poll() is not None:
                    break
                interaction, token = self._current_interaction()
                if token != current_token:
                    return self.status()
                time.sleep(0.05)
            raise TimeoutError(
                f"The action was sent, but the stage did not advance from "
                f"\"{before.get('title', 'the current step')}\"; hold the joint "
                "still and try again.")

    def _wait_for_web(self, port: int, process: subprocess.Popen) -> None:
        deadline = time.monotonic() + 20.0
        while time.monotonic() < deadline:
            if process.poll() is not None:
                raise RuntimeError(self.log_tail())
            try:
                with urlopen(f"http://127.0.0.1:{port}/", timeout=0.3):
                    return
            except (URLError, TimeoutError):
                time.sleep(0.15)
        self.stop()
        raise TimeoutError(f"The stage service failed to start on port {port}")

    def _outputs_ready(self) -> bool:
        if not self.handle:
            return False
        stage = BY_KEY[self.handle.stage_key]
        results = self.handle.root / "calibration" / "results"
        def exists(name: str) -> bool:
            source_name = "zeros" if (self.handle.stage_key == "arms"
                                       and name == "zeros_refined") else name
            return (results / f"{source_name}.json").is_file()
        if not (bool(stage.outputs) and all(exists(name) for name in stage.outputs)):
            return False
        import json
        if self.handle.stage_key == "arms":
            touch = json.loads((results / "touch.json").read_text())
            return (touch.get("complete") is True
                    and all(touch.get("arms", {}).get(arm)
                            for arm in ("left_arm", "right_arm")))
        # Note the absence of a "verify" branch. Stage 8 measures rather than
        # gates: once its report is written the stage has done its work, and
        # missing the accuracy thresholds is a verdict about the robot, not a
        # sign the run is unfinished. So its outputs existing is the whole test.
        #
        # This used to require validation["passed"], which meant a large-error
        # run never reached review, was never committed, and left the dashboard
        # with no accept button to offer. The only ways out were "show the
        # stage list", which stranded the result in the runtime copy, and "end
        # this run", which discarded the captured frames as well -- so the one
        # outcome most worth keeping for diagnosis was the only one that could
        # not be saved. The engine already tells the two apart through its
        # completed_large_error state; it just needs the result committed in
        # order to read it.
        return True

    def status(self) -> AdapterStatus:
        if not self.handle:
            return AdapterStatus("idle", "No stage subprocess is running", {}, False)
        code = self.handle.process.poll()
        outputs_ready = self._outputs_ready()
        completion_safe = self.handle.port is not None or code == 0
        if outputs_ready and completion_safe:
            return AdapterStatus("review", "Stage outputs are complete; saving automatically", {
                "stage": self.handle.stage_key, "run_id": self.handle.run_id,
                "port": self.handle.port,
                "interactive": self.handle.interactive}, True)
        if code is None:
            log_tail = self.log_tail(120) if self.handle.interactive else None
            interaction = (self._describe(self.handle.stage_key, log_tail)
                           if self.handle.interactive else None)
            return AdapterStatus("running", "The stage is running", {
                "stage": self.handle.stage_key, "run_id": self.handle.run_id,
                "port": self.handle.port,
                "interactive": self.handle.interactive,
                "interaction": interaction,
                "measurements": _stage_measurements(self.handle.stage_key,
                                                    log_tail, self.mounting),
                "live_ranges": self.live_ranges(),
                "log_tail": log_tail}, True)
        stage = BY_KEY[self.handle.stage_key]
        results = self.handle.root / "calibration" / "results"
        missing = [name for name in stage.outputs
                   if not (results / f"{name}.json").is_file()]
        if code == 0:
            return AdapterStatus(
                "failed", "The stage program exited without producing complete results", {
                    "stage": self.handle.stage_key, "run_id": self.handle.run_id,
                    "missing_outputs": missing,
                    "log_tail": self.log_tail()}, False,
                "Missing outputs: " + ", ".join(missing))
        return AdapterStatus("failed", f"The stage program exited with status {code}", {
            "stage": self.handle.stage_key, "run_id": self.handle.run_id,
            "log_tail": self.log_tail()}, False, self.log_tail())

    def collect_outputs(self) -> tuple[dict[str, dict], dict[str, bytes]]:
        if not self.handle:
            raise RuntimeError("No runtime is active")
        import json
        stage = BY_KEY[self.handle.stage_key]
        calibration = self.handle.root / "calibration"
        results = calibration / "results"
        outputs = {}
        for name in stage.outputs:
            source_name = "zeros" if (self.handle.stage_key == "arms"
                                       and name == "zeros_refined") else name
            path = results / f"{source_name}.json"
            if not path.is_file():
                raise FileNotFoundError(f"The stage did not produce {source_name}.json")
            outputs[name] = json.loads(path.read_text())
        artifacts: dict[str, bytes] = {}
        if self.handle.stage_key == "verify":
            yaml_path = results / "robot.yaml"
            if yaml_path.is_file():
                artifacts["robot.yaml"] = yaml_path.read_bytes()
            xml_path = calibration / "model" / "xlerobot_calib_fitted.xml"
            process = subprocess.run(
                [sys.executable, str(calibration / "bake_xml.py"),
                 "--output", "model/xlerobot_calib_fitted.xml"],
                cwd=self.handle.root, capture_output=True, text=True, timeout=120)
            if process.returncode != 0:
                raise RuntimeError(f"MuJoCo XML export failed: {process.stderr or process.stdout}")
            artifacts["xlerobot_calib_fitted.xml"] = xml_path.read_bytes()
        return outputs, artifacts

    def proxy_base(self) -> str | None:
        if not self.handle or self.handle.process.poll() is not None or self.handle.port is None:
            return None
        return f"http://127.0.0.1:{self.handle.port}"

    def log_tail(self, lines: int = 30) -> str:
        if not self.handle or not self.handle.log_path.is_file():
            return "No run log"
        return "\n".join(self.handle.log_path.read_text(errors="replace").splitlines()[-lines:])

    def stop(self) -> None:
        handle = self.handle
        if not handle:
            return
        process = handle.process
        try:
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait(timeout=3)
        finally:
            stdin = getattr(process, "stdin", None)
            if stdin is not None:
                stdin.close()
            self.handle = None
