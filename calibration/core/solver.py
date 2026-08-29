"""Nonlinear least squares for the calibration stages.

Wraps scipy.optimize.least_squares with the pieces every stage needs:

  - parameter blocks addressed by name, so a stage describes what it is solving
    rather than juggling offsets into a flat vector
  - blocks can be frozen, which is how gauge freedoms are handled: the head tilt
    zero and similar parameters are fixed by convention rather than fitted
  - robust loss by default, because manual touching and corner detection both
    produce occasional large errors that would otherwise dominate
  - covariance and rank analysis, so a weakly determined parameter is reported
    instead of hiding behind a small residual

The document specifies Ceres. This is the same mathematics in the language the
rest of the pipeline is written in: Lie-algebra parameterisation, robust loss,
sparse-free dense solve at the scale we have (tens of parameters).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import least_squares

DEFAULT_LOSS = "soft_l1"


@dataclass
class Block:
    """One named group of parameters."""

    name: str
    size: int
    initial: np.ndarray
    frozen: bool = False
    # Why a block is frozen, shown in the report so the choice is never silent.
    reason: str = ""
    # Per-element labels for readable output, e.g. ("rx", "ry", "rz").
    labels: tuple[str, ...] | None = None

    def element_labels(self) -> list[str]:
        if self.labels and len(self.labels) == self.size:
            return [f"{self.name}.{l}" for l in self.labels]
        if self.size == 1:
            return [self.name]
        return [f"{self.name}[{i}]" for i in range(self.size)]


class Problem:
    """A parameter layout plus a residual function.

    The residual function receives a dict of {block name: value array} so it never
    needs to know the packing order.
    """

    def __init__(self):
        self.blocks: list[Block] = []
        self._by_name: dict[str, Block] = {}

    def add(self, name: str, initial, frozen: bool = False, reason: str = "",
            labels: tuple[str, ...] | None = None) -> Block:
        if name in self._by_name:
            raise ValueError(f"duplicate parameter block '{name}'")
        value = np.atleast_1d(np.asarray(initial, dtype=float)).ravel()
        block = Block(name, len(value), value.copy(), frozen, reason, labels)
        self.blocks.append(block)
        self._by_name[name] = block
        return block

    def freeze(self, name: str, reason: str) -> None:
        """Hold a block fixed. Requires a reason; silent freezing hides gauges."""
        if not reason:
            raise ValueError("freezing a block requires a reason")
        self._by_name[name].frozen = True
        self._by_name[name].reason = reason

    @property
    def free_blocks(self) -> list[Block]:
        return [b for b in self.blocks if not b.frozen]

    @property
    def n_free(self) -> int:
        return sum(b.size for b in self.free_blocks)

    def pack(self, values: dict[str, np.ndarray] | None = None) -> np.ndarray:
        """Flatten the free blocks into the vector the optimiser sees."""
        out = []
        for block in self.free_blocks:
            v = block.initial if values is None else values[block.name]
            out.append(np.asarray(v, dtype=float).ravel())
        return np.concatenate(out) if out else np.zeros(0)

    def unpack(self, x: np.ndarray) -> dict[str, np.ndarray]:
        """Expand the optimiser vector back into every block, frozen included."""
        out: dict[str, np.ndarray] = {}
        i = 0
        for block in self.blocks:
            if block.frozen:
                out[block.name] = block.initial.copy()
            else:
                out[block.name] = np.asarray(x[i:i + block.size], dtype=float)
                i += block.size
        return out

    def free_labels(self) -> list[str]:
        labels: list[str] = []
        for block in self.free_blocks:
            labels.extend(block.element_labels())
        return labels

    def describe(self) -> str:
        lines = [f"  {'BLOCK':<24} {'SIZE':>4}  STATUS"]
        for block in self.blocks:
            status = f"frozen: {block.reason}" if block.frozen else "free"
            lines.append(f"  {block.name:<24} {block.size:>4}  {status}")
        lines.append(f"  {self.n_free} free parameters, "
                     f"{sum(b.size for b in self.blocks)} total")
        return "\n".join(lines)


@dataclass
class Solution:
    """Result of a solve, including the diagnostics that matter."""

    values: dict[str, np.ndarray]
    residuals: np.ndarray
    cost: float
    success: bool
    message: str
    n_iterations: int
    labels: list[str] = field(default_factory=list)
    jacobian: np.ndarray | None = None
    covariance: np.ndarray | None = None
    singular_values: np.ndarray | None = None
    # Free-parameter values in label order, as the optimiser left them.
    flat: np.ndarray = field(default_factory=lambda: np.zeros(0))

    @property
    def rms(self) -> float:
        """RMS residual in whatever units the residual function returned."""
        if self.residuals.size == 0:
            return float("nan")
        return float(np.sqrt(np.mean(self.residuals ** 2)))

    @property
    def max_abs(self) -> float:
        return float(np.max(np.abs(self.residuals))) if self.residuals.size else float("nan")

    @property
    def condition_number(self) -> float:
        s = self.singular_values
        if s is None or len(s) == 0 or s[-1] <= 0:
            return float("inf")
        return float(s[0] / s[-1])

    def sigma(self) -> np.ndarray | None:
        """One-sigma uncertainty per free parameter."""
        if self.covariance is None:
            return None
        return np.sqrt(np.clip(np.diag(self.covariance), 0.0, None))

    def weak_directions(self, threshold: float = 1e-6) -> list[tuple[float, list[tuple[str, float]]]]:
        """Parameter combinations the data barely constrains.

        Reported as (singular value, [(label, weight), ...]) so an unobservable
        pairing is visible as a mix rather than as one suspicious parameter.
        """
        if self.jacobian is None or not self.labels:
            return []
        _, s, vt = np.linalg.svd(self.jacobian, full_matrices=False)
        scale = s[0] if len(s) and s[0] > 0 else 1.0
        out = []
        for i, value in enumerate(s):
            if value / scale <= threshold or value <= threshold:
                mix = sorted(zip(self.labels, vt[i]), key=lambda t: -abs(t[1]))
                out.append((float(value), [(l, float(w)) for l, w in mix[:4]]))
        return out

    def report(self, unit: str = "", scale: float = 1.0) -> str:
        """Human-readable summary, including uncertainty and weak directions."""
        lines = [f"  converged: {self.success} ({self.message})",
                 f"  iterations: {self.n_iterations}",
                 f"  residual RMS: {self.rms * scale:.4f}{unit}",
                 f"  residual max: {self.max_abs * scale:.4f}{unit}"]
        if self.singular_values is not None and len(self.singular_values):
            lines.append(f"  condition number: {self.condition_number:.1f}")

        sigma = self.sigma()
        if sigma is not None and self.labels and len(self.flat) == len(self.labels):
            lines.append(f"\n    {'PARAMETER':<20} {'VALUE':>12} {'1-SIGMA':>12}")
            for i, label in enumerate(self.labels):
                lines.append(f"    {label:<20} {self.flat[i]:+12.6f} "
                             f"{sigma[i]:12.6f}")

        weak = self.weak_directions()
        if weak:
            lines.append("\n  WEAKLY DETERMINED directions (near-unobservable):")
            for value, mix in weak:
                terms = ", ".join(f"{l} {w:+.3f}" for l, w in mix)
                lines.append(f"    singular value {value:.3e}: {terms}")
            lines.append("    Consider freezing one of these by convention;"
                         " fitting both lets them trade off.")
        return "\n".join(lines)


def solve(problem: Problem, residual_fn, loss: str = DEFAULT_LOSS,
          f_scale: float | None = None, max_nfev: int | None = None,
          verbose: bool = False, x_scale="jac") -> Solution:
    """Fit the free blocks of `problem` so residual_fn returns small values.

    residual_fn(values: dict[str, np.ndarray]) -> 1-D array of residuals.

    f_scale sets the robust loss transition point and must be in the residual's
    units: below it errors are treated as Gaussian, above it as outliers. Passing
    None picks a value from the initial residual spread, which is better than a
    fixed default but worth overriding when the noise level is known.
    """
    x0 = problem.pack()
    if x0.size == 0:
        raise ValueError("nothing to solve: every block is frozen")

    def wrapped(x):
        r = np.asarray(residual_fn(problem.unpack(x)), dtype=float).ravel()
        if not np.all(np.isfinite(r)):
            # least_squares cannot recover from NaN; a large finite value lets it
            # back away from the bad region instead.
            r = np.where(np.isfinite(r), r, 1e6)
        return r

    r0 = wrapped(x0)
    if r0.size == 0:
        raise ValueError("residual function returned nothing")
    if r0.size < problem.n_free:
        raise ValueError(
            f"{r0.size} residuals cannot determine {problem.n_free} parameters; "
            f"collect more data or freeze some blocks")

    if f_scale is None:
        # Median absolute residual is a robust scale estimate; the floor stops a
        # near-perfect start from making every real error look like an outlier.
        f_scale = max(float(np.median(np.abs(r0))), 1e-9)

    result = least_squares(
        wrapped, x0, loss=loss, f_scale=f_scale, x_scale=x_scale,
        max_nfev=max_nfev, verbose=2 if verbose else 0)

    values = problem.unpack(result.x)
    jac = np.asarray(result.jac, dtype=float)
    residuals = np.asarray(result.fun, dtype=float)

    covariance = None
    singular_values = None
    if jac.size:
        _, s, _ = np.linalg.svd(jac, full_matrices=False)
        singular_values = s
        dof = max(len(residuals) - problem.n_free, 1)
        variance = float(residuals @ residuals) / dof
        # Pseudo-inverse of J^T J, tolerant of the rank deficiency that a genuine
        # gauge freedom produces.
        try:
            covariance = np.linalg.pinv(jac.T @ jac, rcond=1e-12) * variance
        except np.linalg.LinAlgError:
            covariance = None

    sol = Solution(
        values=values, residuals=residuals, cost=float(result.cost),
        success=bool(result.success), message=str(result.message),
        n_iterations=int(result.nfev), labels=problem.free_labels(),
        jacobian=jac, covariance=covariance, singular_values=singular_values,
        flat=np.asarray(result.x, dtype=float))
    return sol


def split_holdout(n: int, fraction: float = 0.25, seed: int = 0,
                  minimum: int = 3) -> tuple[np.ndarray, np.ndarray]:
    """Random fit/holdout split of n observations.

    Validation on data that took part in the fit always flatters the result, so
    every stage that can afford it should hold some back.
    """
    if n < minimum * 2:
        raise ValueError(f"{n} observations is too few to split; need "
                         f"{minimum * 2}")
    rng = np.random.default_rng(seed)
    order = rng.permutation(n)
    n_hold = max(minimum, int(round(n * fraction)))
    n_hold = min(n_hold, n - minimum)
    return np.sort(order[n_hold:]), np.sort(order[:n_hold])
