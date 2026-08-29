"""Stage adapter contracts.

Hardware modules are imported only inside adapter start methods. Importing the guided
application therefore never opens a camera, serial bus, or robot connection.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from .registry import StageGuide


@dataclass(frozen=True)
class AdapterStatus:
    phase: str
    message: str
    progress: dict
    can_continue: bool
    blocked_reason: str | None = None


class StageAdapter(Protocol):
    guide: StageGuide

    def start(self, run_id: str) -> AdapterStatus: ...
    def status(self) -> AdapterStatus: ...
    def stop(self) -> None: ...


class PreviewAdapter:
    """No-hardware adapter used for UI development and workflow smoke tests."""

    def __init__(self, guide: StageGuide):
        self.guide = guide
        self._status = AdapterStatus(
            phase="idle", message="Not started", progress={}, can_continue=False)

    def start(self, run_id: str) -> AdapterStatus:
        self._status = AdapterStatus(
            phase="preview",
            message="Hardware-free preview: the interface and the workflow can be inspected, but no robot device is read or written.",
            progress={"run_id": run_id, "mode": "preview"},
            can_continue=False,
            blocked_reason="Preview mode does not produce real calibration results.",
        )
        return self._status

    def status(self) -> AdapterStatus:
        return self._status

    def stop(self) -> None:
        self._status = AdapterStatus(
            phase="stopped", message="Preview finished", progress={}, can_continue=False)


class LegacyAlgorithmAdapter:
    """Boundary for stable legacy algorithms, intentionally not implemented inline.

    A concrete adapter must receive an isolated workspace and an already-consumed
    dashboard authorization. It may import numerical modules from ``calibration``
    only after start; it must never invoke legacy runner overwrite prompts.
    """

    def __init__(self, guide: StageGuide):
        self.guide = guide
        self._status = AdapterStatus(
            phase="idle", message="Not started", progress={}, can_continue=False)

    def start(self, run_id: str) -> AdapterStatus:
        raise NotImplementedError(
            "The real hardware adapter is enabled per stage once its compatibility "
         "has been verified; no hardware is accessed yet.")

    def status(self) -> AdapterStatus:
        return self._status

    def stop(self) -> None:
        self._status = AdapterStatus(
            phase="stopped", message="Stage stopped", progress={}, can_continue=False)
