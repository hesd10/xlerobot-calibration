"""Joint travel ranges, measured in encoder counts without losing track of turns.

Why a running total rather than two endpoint readings
-----------------------------------------------------
These are single-turn absolute encoders: 4096 counts, then back to 0. Differencing
two readings only works if the motion between them is under half a turn, and on
this robot several joints are not. Wrist roll spans 320 degrees, or 3641 counts,
so its two extremes can sit either side of the seam and a straight difference
reports the short way round -- a 320 degree range measured as 40.

What removes the ambiguity is posing the arm near its zero first. From there every
extreme is within half a turn of the start, so each incremental step is
unambiguous, and accumulating those steps recovers the true travel even when it
exceeds a full turn. This is why stage 4 asks for a rough zero pose before it
measures anything: not for accuracy, but to put the encoder somewhere its wrap
cannot be confused.

A range that reaches the full 0-4095 span is therefore a legitimate result for a
flexible joint, not a fault. The tracker reports total travel, which can exceed
4096, alongside the raw extremes.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from . import servos


@dataclass
class Travel:
    """One joint's accumulated travel, tracked from a starting reading.

    Positions are held as a continuous count relative to the start, so they are
    not confined to 0-4095 and a joint that turns more than once still reads
    monotonically.
    """

    name: str
    start_raw: int
    last_raw: int = 0
    position: float = 0.0
    lowest: float = 0.0
    highest: float = 0.0
    steps: int = 0

    def __post_init__(self) -> None:
        self.last_raw = int(self.start_raw)

    def update(self, raw: int) -> float:
        """Fold in a new reading, returning the continuous position.

        Each step is unwrapped against the *previous* reading rather than the
        start, so travel beyond half a turn accumulates correctly as long as the
        joint is sampled often enough to move less than half a turn between reads.
        """
        step = servos.unwrap_delta(int(raw) - self.last_raw)
        self.last_raw = int(raw)
        self.position += step
        self.lowest = min(self.lowest, self.position)
        self.highest = max(self.highest, self.position)
        self.steps += 1
        return self.position

    @property
    def span_counts(self) -> float:
        """Total travel seen, which may legitimately exceed one turn."""
        return self.highest - self.lowest

    @property
    def span_deg(self) -> float:
        return self.span_counts * 360.0 / servos.COUNTS_PER_TURN

    @property
    def wrapped(self) -> bool:
        """Did this joint cross the raw 0/4095 seam during measurement?

        Not a problem, but worth reporting: it is the case a naive endpoint
        difference would get wrong, so seeing it confirms the tracking is earning
        its keep.
        """
        return (self.start_raw + self.lowest < 0
                or self.start_raw + self.highest >= servos.COUNTS_PER_TURN)

    def raw_at(self, position: float) -> int:
        """The raw count corresponding to a continuous position."""
        return int(round(self.start_raw + position)) % servos.COUNTS_PER_TURN

    def to_dict(self) -> dict:
        return {"name": self.name, "start_raw": int(self.start_raw),
                "last_raw": int(self.last_raw), "position": float(self.position),
                "lowest": float(self.lowest), "highest": float(self.highest),
                "span_counts": float(self.span_counts),
                "span_deg": round(self.span_deg, 2),
                "raw_lowest": self.raw_at(self.lowest),
                "raw_highest": self.raw_at(self.highest),
                "wrapped": self.wrapped, "samples": int(self.steps)}

    @classmethod
    def from_dict(cls, d: dict) -> "Travel":
        t = cls(name=d["name"], start_raw=int(d["start_raw"]))
        t.last_raw = int(d.get("last_raw", t.start_raw))
        t.position = float(d.get("position", 0.0))
        t.lowest = float(d.get("lowest", 0.0))
        t.highest = float(d.get("highest", 0.0))
        t.steps = int(d.get("samples", 0))
        return t


@dataclass
class RangeSet:
    """Measured travel for a set of joints, plus the zero they were measured from."""

    travels: dict[str, Travel] = field(default_factory=dict)
    zero_raw: dict[str, int] = field(default_factory=dict)

    def begin(self, name: str, raw: int) -> Travel:
        t = Travel(name=name, start_raw=int(raw))
        self.travels[name] = t
        self.zero_raw.setdefault(name, int(raw))
        return t

    def update(self, readings: dict[str, int | None]) -> None:
        for name, raw in readings.items():
            if raw is None:
                continue
            if name not in self.travels:
                self.begin(name, raw)
            else:
                self.travels[name].update(raw)

    def to_dict(self) -> dict:
        return {"travels": {n: t.to_dict() for n, t in self.travels.items()},
                "zero_raw": dict(self.zero_raw)}

    @classmethod
    def from_dict(cls, d: dict | None) -> "RangeSet":
        if not d:
            return cls()
        return cls(travels={n: Travel.from_dict(v)
                            for n, v in (d.get("travels") or {}).items()},
                   zero_raw={n: int(v)
                             for n, v in (d.get("zero_raw") or {}).items()})


def position_from_raw(raw: int, zero_raw: int, travel: Travel,
                      tolerance: float = 1.0,
                      measured_zero_raw: int | None = None) -> float:
    """Resolve one raw reading into the unique measured continuous position."""
    if travel.span_counts >= servos.COUNTS_PER_TURN:
        raise ValueError(f"{travel.name}: travel covers more than one turn")
    measured_zero = int(zero_raw if measured_zero_raw is None else measured_zero_raw)
    start_from_measured = servos.unwrap_delta(travel.start_raw - measured_zero)
    base = servos.unwrap_delta(int(raw) - measured_zero)
    lo = start_from_measured + travel.lowest
    hi = start_from_measured + travel.highest
    candidates = [base + k * servos.COUNTS_PER_TURN
                  for k in (-1, 0, 1)
                  if lo - tolerance <= base + k * servos.COUNTS_PER_TURN <= hi + tolerance]
    if len(candidates) != 1:
        raise ValueError(f"{travel.name}: raw reading is outside its measured range")
    target_zero_from_measured = servos.unwrap_delta(int(zero_raw) - measured_zero)
    return float(candidates[0] - target_zero_from_measured)


class RangeAngleTracker:
    """Track absolute joint positions using measured sub-turn travel ranges."""

    def __init__(self, zero_raw: dict[str, int], measured: RangeSet):
        self.zero_raw = {n: int(v) for n, v in zero_raw.items()}
        self.measured = measured
        self.last_raw: dict[str, int] = {}
        self.positions: dict[str, float] = {}
        missing = [n for n in self.zero_raw
                   if n not in measured.travels or n not in measured.zero_raw]
        if missing:
            raise ValueError("missing measured ranges: " + ", ".join(missing))

    def reseed(self, readings: dict[str, int | None]) -> dict[str, float]:
        """Locate each current raw reading independently in its legal range."""
        for name, raw in readings.items():
            if name not in self.zero_raw or raw is None:
                continue
            self.positions[name] = position_from_raw(
                int(raw), self.zero_raw[name], self.measured.travels[name],
                measured_zero_raw=self.measured.zero_raw[name])
            self.last_raw[name] = int(raw)
        return dict(self.positions)

    def update(self, readings: dict[str, int | None]) -> dict[str, float]:
        """Accumulate adjacent samples after an absolute range-based seed."""
        for name, raw in readings.items():
            if name not in self.zero_raw or raw is None:
                continue
            raw = int(raw)
            if name not in self.last_raw:
                self.positions[name] = position_from_raw(
                    raw, self.zero_raw[name], self.measured.travels[name],
                    measured_zero_raw=self.measured.zero_raw[name])
            else:
                self.positions[name] += servos.unwrap_delta(raw - self.last_raw[name])
            self.last_raw[name] = raw
        return dict(self.positions)

    def angles(self, signs: dict[str, int]) -> dict[str, float]:
        scale = 2.0 * 3.141592653589793 / servos.COUNTS_PER_TURN
        return {n: signs[n] * p * scale for n, p in self.positions.items()
                if n in signs}


def angles_from_ranges(readings: dict[str, int], zero_raw: dict[str, int],
                       signs: dict[str, int], measured: RangeSet) -> dict[str, float]:
    """Resolve independent raw readings to angles using Stage 4 ranges."""
    tracker = RangeAngleTracker(zero_raw, measured)
    tracker.reseed(readings)
    return tracker.angles(signs)


def sense_agrees(travel: Travel, model_span_deg: float,
                 fraction: float = 0.6) -> bool:
    """Did this joint travel a plausible share of the range the model allows?

    A joint that barely moved was probably jammed, mis-wired, or the operator
    stopped early; either way its range is not usable as a bound.
    """
    return travel.span_deg >= fraction * model_span_deg
