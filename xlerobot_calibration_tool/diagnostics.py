"""Startup diagnostics for the guided calibration application."""
from __future__ import annotations

import importlib.util
import subprocess
import sys
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .i18n import text


@dataclass(frozen=True)
class Check:
    """One startup check and its verdict.

    The wording is carried as a key plus its substitutions rather than a
    finished sentence, because a check runs long before the request that
    displays it, and one result may be rendered more than once.
    """

    key: str
    passed: bool
    message_key: str
    fix_key: str | None = None
    message_fields: dict[str, Any] = field(default_factory=dict)
    fix_fields: dict[str, Any] = field(default_factory=dict)

    @property
    def message(self) -> str:
        return text(self.message_key, **self._fields())

    @property
    def fix(self) -> str | None:
        if self.fix_key is None:
            return None
        return text(self.fix_key, **self.fix_fields)

    def _fields(self) -> dict[str, Any]:
        """Substitutions, with any that are themselves phrase keys resolved.

        A field may be a product name that stays as it is ("PyTorch") or a
        phrase key, so keys are resolved here rather than when the check was
        built.
        """
        return {name: (text(value)
                       if isinstance(value, str) and value.startswith("check.")
                       else value)
                for name, value in self.message_fields.items()}

    def rendered(self) -> dict[str, Any]:
        """What the browser needs: a verdict and two readable sentences."""
        return {
            "key": self.key,
            "passed": self.passed,
            "message": self.message,
            "fix": self.fix,
        }


def preflight(source_root: Path | str | None = None) -> list[Check]:
    root = Path(source_root or Path(__file__).resolve().parent.parent).resolve()
    algorithms = (root / "calibration" / "core").is_dir()
    buses = (root / "tools" / "config" / "buses.py").is_file()
    checks = [
        Check("legacy_algorithms", algorithms,
              "check.algorithmsFound" if algorithms else "check.algorithmsMissing",
              None if algorithms else "check.algorithmsFix"),
        Check("robot_tools", buses,
              "check.busFound" if buses else "check.busMissing",
              None if buses else "check.busFix"),
    ]
    # Every module here is one calibration cannot run without. lerobot and torch
    # used to sit alongside them, inherited from a preflight written for teleop
    # and VLA training, and were reported as failures on a machine that had no
    # use for either: this tool reads raw encoder counts through scservo_sdk and
    # never loads a policy. Two red rows the operator could do nothing useful
    # about is worse than silence, so they are simply not checked.
    modules = {
        "numpy": ("NumPy", "check.installLegacy"),
        "cv2": ("OpenCV contrib", "check.installLegacy"),
        "scipy": ("SciPy", "check.installLegacy"),
        "yaml": ("PyYAML", "check.installLegacy"),
        "scservo_sdk": ("Feetech SDK", "check.installFeetech"),
    }
    for module, (label, fix) in modules.items():
        found = importlib.util.find_spec(module) is not None
        checks.append(Check(
            module, found,
            "check.moduleFound" if found else "check.moduleMissing",
            None if found else fix,
            message_fields={"label": label}))
    return checks


def camera_identification(source_root: Path | str | None = None) -> Check:
    """Validate the saved role mapping without opening cameras."""
    root = Path(source_root or Path(__file__).resolve().parent.parent).resolve()
    tools = root / "tools"
    if not (tools / "config" / "cameras.py").is_file():
        return Check("camera_identification", False,
                     "check.cameraModuleMissing", "check.cameraFix")
    module_name = f"_xlerobot_cameras_{uuid.uuid4().hex}"
    spec = importlib.util.spec_from_file_location(
        module_name, tools / "config" / "cameras.py")
    if spec is None or spec.loader is None:
        return Check("camera_identification", False,
                     "check.cameraModuleUnloadable", "check.cameraFix")
    cameras = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(cameras)
        resolved = cameras.resolve(strict=False)
        roles = cameras.ROLES
    except Exception as exc:
        return Check("camera_identification", False,
                     "check.cameraReadFailed", "check.cameraIdentifyFix",
                     message_fields={"message": str(exc)})
    missing = [role for role in roles if role not in resolved]
    missing_devices = [f"{role}={resolved[role]}" for role in roles
                       if role in resolved and not Path(resolved[role]).exists()]
    if missing or missing_devices:
        # Both problems can be present at once; report the one the operator
        # must fix first, but never silently drop the other.
        detail = {}
        if missing:
            key = "check.cameraRolesMissing"
            detail = {"roles": ", ".join(missing)}
        if missing_devices and not missing:
            key = "check.cameraDevicesMissing"
            detail = {"devices": ", ".join(missing_devices)}
        elif missing and missing_devices:
            key = "check.cameraRolesAndDevices"
            detail = {"roles": ", ".join(missing),
                      "devices": ", ".join(missing_devices)}
        return Check("camera_identification", False, key,
                     "check.cameraIdentifyFix", message_fields=detail)
    return Check("camera_identification", True, "check.cameraOk",
                 message_fields={"mapping": ", ".join(
                     f"{role}={resolved[role]}" for role in roles)})



def _preflight_errors(output: str) -> list[str]:
    """Extract actionable failures without returning the verbose success log."""
    indicators = (
        "problem:", "error:", "warning:", "missing:", "unexpected ids:",
        "no servos responded", "could not open", "opened but no frames",
        "nearly black", "blurry", "import failed", "not available",
        "servo scan reported problems", "layout does not match",
    )
    errors: list[str] = []
    in_failed_summary = False
    for raw in output.splitlines():
        line = raw.strip()
        if not line or set(line) == {"="}:
            continue
        lower = line.lower()
        if lower == "summary":
            in_failed_summary = True
            continue
        is_summary_error = in_failed_summary and (line.startswith("-") or line.endswith(":"))
        if any(marker in lower for marker in indicators) or is_summary_error:
            cleaned = line.removeprefix("PROBLEM:").strip()
            if cleaned not in errors:
                errors.append(cleaned)
    return errors


def run_hardware_preflight(source_root: Path | str | None = None) -> Check:
    """Run the repository's read-only preflight and return only its verdict/errors."""
    root = Path(source_root or Path(__file__).resolve().parent.parent).resolve()
    script = root / "tools" / "checks" / "preflight.py"
    if not script.is_file():
        return Check("preflight", False, "check.preflightMissing",
                     "check.preflightMissingFix")
    try:
        completed = subprocess.run(
            [sys.executable, "-u", str(script)], cwd=root,
            capture_output=True, text=True, timeout=120)
    except subprocess.TimeoutExpired:
        return Check("preflight", False, "check.preflightTimeout",
                     "check.preflightTimeoutFix")
    combined = "\n".join(
        part for part in (completed.stdout or "", completed.stderr or "") if part)
    if completed.returncode == 0:
        return Check("preflight", True, "check.preflightPassed")
    errors = _preflight_errors(combined)
    if errors:
        # The preflight's own output is the detail here; it is program output
        # rather than interface wording, so it is passed through as it stands.
        return Check("preflight", False, "check.preflightDetail",
                     "check.preflightFix",
                     message_fields={"detail": "\n".join(f"- {line}"
                                                         for line in errors)})
    return Check("preflight", False, "check.preflightExit", "check.preflightFix",
                 message_fields={"code": completed.returncode})


def run_startup_checks(source_root: Path | str | None = None) -> list[Check]:
    """Every startup check, as verdicts rather than finished sentences.

    The hardware preflight opens the cameras and the servo bus, which takes
    seconds, so the result is worth keeping: one run can be re-rendered
    without probing the robot again.
    """
    root = Path(source_root or Path(__file__).resolve().parent.parent).resolve()
    return [*preflight(root), camera_identification(root),
            run_hardware_preflight(root)]


def render_checks(checks: list[Check]) -> dict:
    """Phrase an existing set of check results."""
    return {"passed": all(check.passed for check in checks),
            "checks": [check.rendered() for check in checks]}


def startup_payload(source_root: Path | str | None = None) -> dict:
    return render_checks(run_startup_checks(source_root))


def payload(source_root: Path | str | None = None) -> dict:
    """Compatibility endpoint for the original dependency-only diagnostics."""
    return render_checks(preflight(source_root))
