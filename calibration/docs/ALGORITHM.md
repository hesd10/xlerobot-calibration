# Algorithms

How each quantity is solved, which directions in parameter space are observable,
and how the ones that are not are fixed.

Every stage is a nonlinear least-squares problem on SE(3). That part is
unremarkable. What matters on this hardware is the second question: several
stages have parameter directions that are **exactly** unobservable — not weakly
determined, but provably invisible to any amount of data. Fitting them anyway
produces a confident, well-conditioned, wrong answer. Each is identified below
and held by convention instead.

## Contents

1. [Conventions](#1-conventions)
2. [Board detection and PnP](#2-board-detection-and-pnp)
3. [Stage 2 — intrinsics](#3-stage-2--intrinsics)
4. [Stage 3 — joint senses](#4-stage-3--joint-senses)
5. [Stage 4 — rough zeros and travel](#5-stage-4--rough-zeros-and-travel)
6. [Stage 5 — world frame and head](#6-stage-5--world-frame-and-head)
7. [Stage 6 — arm and wrist-camera fusion](#7-stage-6--arm-and-wrist-camera-fusion)
8. [Stage 7 — fixing the remaining gauges](#8-stage-7--fixing-the-remaining-gauges)
9. [Stage 8 — held-out validation](#9-stage-8--held-out-validation)
10. [Mounting: nominal and physical sides](#10-mounting-nominal-and-physical-sides)
11. [Numerical practice](#11-numerical-practice)

---

## 1. Conventions

### Frames

| Symbol | Frame |
|---|---|
| $W$ | World — **defined as the ChArUco board frame** |
| $B$ | Body / chassis |
| $A$ | Arm root, after the mounting correction |
| $G$ | Gripper (`Fixed_Jaw`), the last rigid link |
| $C$ | Camera, OpenCV convention: $+Z$ along the optical axis |

$T_{XY} \in SE(3)$ maps points from frame $Y$ into frame $X$:
$p_X = T_{XY}\, p_Y$. In code this is `T_X_Y`.

**Defining $W$ as the board frame is the single decision the whole procedure
rests on.** PnP measures $T_{CW_\text{board}}$; because $W$ *is* the board,
that same matrix is $T_{CW}$, with no intermediate transform to estimate.

### Lie algebra

$\mathfrak{se}(3)$ elements are 6-vectors $\xi = (\omega, u)$, **rotation
first**:

$$\xi = (\underbrace{\omega_x, \omega_y, \omega_z}_{\text{rotation}},\ \underbrace{u_x, u_y, u_z}_{\text{translation}})$$

with $\exp$ and $\log$ as in `core/se3.py`. Rotation uses Rodrigues:

$$\exp(\hat\omega) = I + \frac{\sin\theta}{\theta}\hat\omega + \frac{1-\cos\theta}{\theta^2}\hat\omega^2, \qquad \theta = \lVert\omega\rVert$$

and the translation goes through the left Jacobian $V$:

$$\exp(\xi) = \begin{bmatrix} \exp(\hat\omega) & V u \\ 0 & 1\end{bmatrix}, \qquad V = I + \frac{1-\cos\theta}{\theta^2}\hat\omega + \frac{\theta - \sin\theta}{\theta^3}\hat\omega^2$$

> **The ordering is load-bearing.** Rotation occupies indices 0–2, translation
> 3–5. The arm mount's yaw — the gauge direction held in Stage 6 — is index 2.
> Holding index 5 instead, a $z$ translation carrying no weight in the null
> space, leaves the freedom untouched and yields a 1 mm residual with **31° of
> zero error**. The tell: a convergence problem varies with the initial guess, a
> gauge problem does not.

### The pose residual

Every stage that compares a predicted camera pose against an observed one uses
the same right-invariant SE(3) residual:

$$r = \log\!\left(T_\text{obs}^{-1}\, T_\text{pred}\right) \in \mathbb{R}^6$$

zero exactly when the two poses agree.

### Weighting rotation against translation

A radian is dimensionless and a metre is not, so raw residuals let one block
dominate. Both camera stages rescale rotation by a lever arm $\lambda$:

$$\tilde r = (\lambda\,\omega,\ u)$$

Stage 5 uses $\lambda = 0.05$ m (`ROT_SCALE`, "1 rad ≈ 50 mm of image motion at
the working distance"); Stage 6 uses $\lambda = 0.1$ m (`ROTATION_LEVER_M`).
Without this the fit quietly favours whichever block has the larger numbers.

### Robust loss

All pose fits use `soft_l1` with scale $f$:

$$\rho(s) = 2\left(\sqrt{1 + s} - 1\right), \qquad s = (r/f)^2$$

$f = 2$ mm for the head, $5$ mm for fusion, $10$ mm for the standalone wrist
solve. This keeps a single misdetected view from dragging the solution while
staying quadratic for the bulk of the data.

### Held-out scoring

Every fit that can be scored on unseen data is. `solver.split_holdout` reserves
25% of views (minimum 3, fixed seed 0), and the **gates are applied to the
holdout**, never to the fit residual. A low fit residual with a high holdout
residual is the signature of postures that were too alike — the exact failure
that a fit-only score cannot see.

---

## 2. Board detection and PnP

A ChArUco board gives each detected corner a stable identity, so correspondences
are known rather than matched. For view $i$ with corners $\{(P_j, p_{ij})\}$ in
board and image coordinates, PnP solves

$$T_{CW}^{(i)} = \arg\min_{T} \sum_j \left\lVert \pi\!\left(K, d,\ T P_j\right) - p_{ij} \right\rVert^2$$

with $\pi$ the pinhole-plus-distortion projection.

### Two gates, for opposite reasons

```python
PNP_MIN_CORNERS  = 12     # below this the pose is weak
PNP_MAX_REPROJ_PX = 1.5   # above this the detection is wrong
PNP_GOOD_CORNERS = 20     # advisory
```

**Reprojection error does not measure pose quality.** With few corners, the pose
is poorly constrained *and* the reprojection error is small — there is little
left to disagree with. Measured on the Stage 5 capture:

| corners | mean pose error |
|---|---|
| < 12 | 8.7 mm |
| ≥ 20 | 3.9 mm |

Every one of the worst views was corner-starved while looking numerically
healthy. This is why corner count is gated *independently* of reprojection
error, and why it reappears in the Stage 8 diagnosis in the README.

### Board identity

The board's orientation and its `legacy` dictionary flag both change the
board-frame coordinates of every corner while leaving marker detection working.
The failure is silent until no ChArUco corner is ever interpolated.
`identify_board.py` settles it by fitting each candidate layout and comparing
reprojection: the correct one lands within a pixel, wrong ones are off by tens.

---

## 3. Stage 2 — intrinsics

Standard Zhang calibration per camera, solving $K$ and the distortion vector
$d = (k_1, k_2, p_1, p_2, k_3)$.

$$K = \begin{bmatrix} f_x & 0 & c_x \\ 0 & f_y & c_y \\ 0 & 0 & 1\end{bmatrix}$$

Gates (`core/gates.py`):

```python
INTRINSICS_RMS_MAX_PX      = 1.0    # 0.35 is "good"
INTRINSICS_MIN_VIEWS       = 15
INTRINSICS_MIN_HOLDOUT     = 5
INTRINSICS_HOLDOUT_RATIO_MAX = 2.0
INTRINSICS_MIN_COVERAGE    = 0.55
```

**Coverage is gated because the distortion terms are otherwise extrapolated.**
$k_1, k_2, k_3$ act radially, so they are only constrained where corners were
actually seen. A stack of centred frontal views yields a low RMS and a model that
is wrong at the frame edges — where the wrist cameras spend most of their time.
Coverage is measured as the fraction of image area spanned by detected corners
across all views, and the capture page shows which regions are still empty.

Locked afterwards: **lens focus and capture resolution.** These are manual-focus
modules with no software readback; intrinsics are valid only at the focus they
were solved at.

---

## 4. Stage 3 — joint senses

For each of the fourteen joints, the sense

$$s_j \in \{+1, -1\}$$

relates the servo's positive count direction to the model's positive joint axis.
The angle used everywhere downstream is

$$q_j = s_j \cdot \Delta n_j \cdot \frac{2\pi}{4096}$$

The operator moves each joint in a named direction by hand, torque off; raw
counts before and after give the sign. A joint must move at least
`DIRECTION_MIN_TRAVEL_COUNTS = 40` for the reading to beat encoder noise.

Senses are measured rather than fitted because a wrong sense is **not** a large
residual. On this unit, both head pan senses fit the capture to **3.56 mm**, with
every gate green. The wrong one placed the world board **1515 mm above the
floor** against a tape-measured 750 mm, and had the camera looking up at a board
that was plainly below it.

Flipping $s$ for a joint maps the whole problem to a mirrored one, and the
mirrored problem is a perfectly good fit to the data — it is a different,
self-consistent robot. Only an external fact separates them. With 14 joints there
are $2^{14} = 16384$ combinations, so searching is not an option either.

Hence: measured by a human, before anything is solved, never revisited.
`head_model.load_senses()` **refuses to fall back to a default** on real data, so
a robot whose senses were never measured stops rather than quietly calibrating a
mirror image of itself.

---

## 5. Stage 4 — rough zeros and travel

Records a starting count $n_j^0$ per joint and sweeps each joint to both ends.

**The zero is deliberately rough.** Stage 6 reaches the same answer from a guess
90° out (see §7.4), so precision here buys nothing.

What the stage is actually for is the **encoder**. These are single-turn absolute
encoders over 4096 counts, and several joints travel more than half a turn —
wrist roll covers 320°. A range measured as two endpoint readings can therefore
come out as the short way round. Two mechanisms prevent this:

1. Starting near the model's zero pose puts every extreme within half a turn.
2. The sweep is sampled **continuously**, so travel accumulates rather than being
   inferred from endpoints.

Angles are then resolved through the recorded legal range
(`ranges.angles_from_ranges`) rather than by shortest-arc unwrapping, which would
choose wrongly near the seam.

The wrap arithmetic itself is:

$$\Delta n = \left((n - n^0 + 2048) \bmod 4096\right) - 2048$$

used wherever a raw difference must be interpreted as a shortest signed
difference (`servos.unwrap_delta`). A joint whose measured range fills the whole
0–4095 span is a normal result for a flexible joint, not a fault.

---

## 6. Stage 5 — world frame and head

Clamp the base and the board. Sweep the head; observe the board.

### 6.1 Forward kinematics

Base → pan → tilt → camera:

$$T_{BC}(q_p, q_t) = A_\text{pan}(s_p q_p)\; A_\text{tilt}(s_t q_t)\; T_{\text{tilt},C}$$

Each joint rotates about a **line**, not the frame origin, so for axis $a$
through point $c$:

$$A(a, \theta, c) = \begin{bmatrix} R & c - Rc \\ 0 & 1 \end{bmatrix}, \qquad R = \exp(\hat a\,\theta)$$

The predicted measurement, in the same form PnP reports:

$$T_{CW}^\text{pred} = \left(T_{WB}\, T_{BC}(q_p, q_t)\right)^{-1}$$

### 6.2 The tilt axis is not what the XML says

`PAN_AXIS` is $+z$; `TILT_AXIS` is $(0, -1, 0)$ — **not** the `axis="0 1 0"`
written in the XML.

MuJoCo expresses a joint axis in its own body's frame. `head_tilt_link` carries
`quat="0 0 0 1"`, a half turn about $z$. Rotated back into the pan link's
coordinates, where this module works, the axis is $-y$.

Taking the XML number at face value flips the sense of every tilt rotation.
Conditioning stays healthy; measured holdout error went from **5.7 mm to 92 mm**.

### 6.3 Which parameters are observable

The natural parameter set is 17: $T_{WB}$ (6), two joint zeros (2), the pan axis
position (3), the camera mount (6).

**Five directions are exactly unobservable**, with singular values eleven orders
of magnitude below the largest:

| freedom | why |
|---|---|
| pan zero ↔ base yaw | turning the head looks like turning the base |
| tilt zero ↔ camera pitch | nodding looks like tilting the camera |
| axis position (3) ↔ base translation | moving the axis looks like moving the base |

This is **gauge freedom, not weak data** — no amount of extra capture removes it.
A single fixed board simply cannot distinguish "the head turned" from "the base
turned".

So the zeros are held at the current mechanical position and the axis position is
taken from the XML, leaving **12 well-conditioned parameters**:

$$p = \left(\log T_{WB},\ \log T_{\text{tilt},C}\right) \in \mathbb{R}^{12}$$

**Fixing them at the wrong value costs nothing.** $T_{WB}$ absorbs the error, and
predictions at poses that took no part in the fit are unaffected to machine
precision: the body frame shifts, and every frame attached to it shifts with it.

### 6.4 Changing the convention afterwards, exactly

Because rotations about one axis compose, $A(q) = A(q-\delta)\,A(\delta)$, so
moving a joint zero by $\delta$ is exactly a change of the neighbouring fixed
transform:

$$T_{WB} \leftarrow T_{WB}\, A_\text{pan}(\delta) \qquad\text{(pan zero)}$$
$$T_{\text{tilt},C} \leftarrow A_\text{tilt}(\delta)\, T_{\text{tilt},C} \qquad\text{(tilt zero)}$$

Pan's surplus rotation is absorbed *before* the chain, into $T_{WB}$; tilt's is
absorbed *after*, into the camera mount, because tilt is the last joint before
the camera. Both are exact, so Stage 7 can redefine either zero with **no
recapture and no re-solve**.

### 6.5 Multi-start

The objective has a spurious basin in which the camera lands ~0.8 m from the tilt
joint instead of the correct ~0.04 m. A single start converged on 44% of real
sessions; multi-start reached **78% (7 of 9)**.

Candidates are built by anchoring on each view: at that view's posture,
$T_{WB} = T_{WC}^{(i)} \, T_{BC}^{(i)-1}$ given a nominal mount. Thoroughness
escalates only on failure — 12 guesses (3 pitches × 4 views), then 48, then
exhaustive.

**A plausible lever arm alone does not stop the search.** A candidate must also
clear every solution-dependent gate: holdout RMS, worst holdout view,
holdout/fit ratio ≤ 3, and condition number. Otherwise the search stops at a
geometrically sensible but poorly fitting solution.

### 6.6 Gates

```python
HEAD_MIN_VIEWS          = 20
HEAD_PAN_SWEEP_MIN_DEG  = 30.0   # total, max - min
HEAD_RESIDUAL_MAX_MM    = 6.0    # on the holdout
```

Pan sweep is gated because it is what pins the vertical axis. From a sensitivity
study with 0.5 mm PnP noise: a 60° total sweep locates the axis to ~1 mm, 30° to
only ~10 mm. Sweeps are **total**, not per-side, because the board's own width
eats asymmetrically into the budget. The usable range follows from the calibrated
field of view:

$$\Delta_\text{pan} = \gamma\left(\frac{\text{fov}_x}{2} - \arctan\frac{w_\text{board}}{2 D}\right), \qquad \gamma = 0.85$$

On this robot — 86° measured fov, a 200 mm board at 600 mm — that is about ±28°,
so 30° total is comfortably reachable.

---

## 7. Stage 6 — arm and wrist-camera fusion

Per arm, one solve recovers the arm mounting, four joint zeros and the wrist
camera mount from wrist-camera views of the fixed board. A camera view gives a
full 6-DoF pose per observation, which is what makes the joint zeros and the
camera mount separable in a single fit.

### 7.1 Forward kinematics

$$T_{WC}^\text{pred} = T_{WB}\; T_{BA}\; T_{AG}(q)\; T_{GC}$$

- $T_{WB}$ — from Stage 5, held fixed
- $T_{BA}$ — the arm mounting correction, solved
- $T_{AG}(q)$ — MuJoCo FK through the five arm joints
- $T_{GC}$ — the wrist camera mount on `Fixed_Jaw`, solved

True angles are the tracked angles plus the zero corrections being solved:

$$q_j = \tilde q_j + \delta_j$$

Angles come from the capture's **continuous tracker**, not from re-deriving them
from a single raw count, which would pick the shortest arc at the 4095/0 seam. A
capture lacking tracked angles is rejected rather than reinterpreted.

### 7.2 Parameters and the gauge

16 parameters, **15 free**:

$$p = (\underbrace{\log T_{BA}}_{6},\ \underbrace{\delta_\text{pan}, \delta_\text{lift}, \delta_\text{elbow}, \delta_\text{flex}}_{4},\ \underbrace{\log T_{GC}}_{6})$$

Two directions are deliberately excluded:

**(a) Arm mount yaw ↔ shoulder pan zero.** Index 2 of the mount's
$\mathfrak{se}(3)$ vector. Rotating the first shoulder joint and yawing the whole
arm move the gripper identically, so no data separates them. Singular value
measured at $4.5\times10^{-8}$ of the largest. The mount's yaw is held at zero
and the pan zero absorbs it. Stage 7 then pins the pan zero physically (§8.2).

**(b) Wrist roll zero ↔ camera roll.** Held at Stage 4's rough value here, and
resolved in §8.3. The gauge is exact:

$$T_{GC} \mapsto R_\text{roll}(-\delta)\,T_{GC}, \qquad q_\text{roll} \mapsto q_\text{roll} + \delta$$

leaves **every** predicted camera pose unchanged, verified numerically at
0.000 mm / 0.0000°. The lever arm does not help: the roll joint rotates the whole
gripper — camera included — about that axis, and the mount's own rotation about
the same axis cancels it term for term. This is the same coupling that makes the
head's tilt zero unsolvable.

### 7.3 How good the initial guess must be

Once the gauge is fixed, barely at all. With 1 mm of placement error and 0.1° of
joint noise, an initial zero guess wrong by **±90°** still recovers zeros to
0.23° and gripper position to 0.41 mm.

This is why Stage 4 exists to bound joint ranges and keep the encoder off its
wrap seam, **not** to supply an accurate zero.

The guess is correspondingly unfussy: identity mounting, zero corrections, the
XML nominal camera mount.

### 7.4 Conditioning

With 10–15 diverse postures the condition number runs ~50–200. Above 1000
indicates insufficient posture diversity or a gimbal-lock configuration.

**View count alone is not sufficient.** Repeated nearby poses leave joint zeros
undetermined however many are captured. The capture page tracks the spread of
each joint and of camera height, and asks for variation in `wrist_flex`
specifically, since it is the weakest direction.

The advisory capture thresholds come from a learning-curve study
(`holdout_study.py`) on a run whose holdout RMS was high: the curve was still
falling at 10–11 views for 15 parameters, and views with fewer than 40 corners
showed **2.5× the median error**. Hence the advice to collect 14–20 views with
40+ corners each. These are advisory, not gates — a thin capture is allowed
through and caught afterwards by the holdout score.

---

## 8. Stage 7 — fixing the remaining gauges

No capture. Closed-form arithmetic on results already in hand, always recomputed
from the Stage 5 and Stage 6 sources so corrections never accumulate.

Three conventions are pinned, each by a *physical* fact rather than an arbitrary
choice.

### 8.1 Body forward, from arm symmetry

The head zero cannot be measured against the board — that is the gauge of §6.3.
The robot's own build symmetry supplies what the board cannot: the two arm roots
are mirror images in the XML, so the head zero that makes the **solved** arm
roots symmetric is the one facing straight ahead.

With solved root positions $p_L, p_R$ in the base frame, the line joining them is
lateral, so the sagittal direction is its perpendicular:

$$d = (p_L - p_R)_{xy}, \qquad n = (-d_y, d_x), \qquad \psi = \operatorname{atan2}(n_y, n_x)$$

Of the two perpendiculars, the one with larger $x$ is taken. The correction is
applied as a pan-zero shift via the exact identity of §6.4.

Two subtleties:

- The returned angle is measured from the robot's **backward** direction, since
  the model's forward is $-x$. This is harmless: the value is used only as a
  relative correction, and a symmetric pair of mounts gives the same correction
  from either perpendicular. Only consistency between runs matters, which taking
  the larger $x$ guarantees.
- **Yaw and roll come from symmetry; pitch cannot.** Rotating about the sideways
  axis maps the mirror plane onto itself, leaving the symmetry untouched. Pitch
  is taken instead from the direction of the arm-root midpoint, matched to the
  model.
- This **cannot** repair a base frame that is a half turn out, because such a
  frame turns both mounts together and leaves them symmetric. Only the
  model-anchored check in `frames.wrong_side_report` catches that.

### 8.2 Shoulder pan zero, from the forearm link

Stage 6 left the pan zero floating (§7.2a). The forearm breaks the tie.

In the model at $q = 0$, the `Lower_Arm → Wrist_Pitch_Roll` link lies exactly in
the sagittal plane, pointing along $-y$ for the left arm and $+y$ for the right.
Its heading in the XY plane turns **degree for degree with shoulder pan** while
being completely unaffected by shoulder lift and elbow flex, which rotate about
axes parallel to the link.

So "this link points straight out to the side" is an observable that fixes the
pan zero and nothing else. Solve for the shift $\delta$ with

$$h(\delta) = \operatorname{atan2}(\ell_x, \ell_y) \big|_{q_\text{pan} = \delta} \;\overset{!}{=}\; h_\text{ideal}, \qquad h_\text{ideal} = \begin{cases}180° & \text{left}\\ 0° & \text{right}\end{cases}$$

by Brent's method on $\pm 30°$.

The longer of the two candidate links is used (135 mm vs 116 mm): the same
angular error moves its endpoint further, so the heading is less sensitive to
model noise.

> **The bracket needs more than a sign change.** Heading error is wrapped to
> $(-180°, 180°]$, so near an error of 180° the wrap flips the sign at the seam.
> The endpoints then straddle a seam containing no root, `brentq` converges on
> the seam itself, and reports a near-zero shift for an arm that is a half turn
> out — exactly what a back-to-front robot produces. Requiring **both** endpoints
> to be small keeps the bracket on the continuous stretch around the real root.

### 8.3 Wrist roll zero, from the horizontal optical axis

The gauge orbit of §7.2b is exact, so sliding along it costs nothing in fit
error. The physical constraint that stops the slide: **at the XML zero
configuration, each wrist camera's optical axis is horizontal in the chassis XY
plane.**

Two steps, deliberately not combined:

1. Fit the 6 observable parameters with the gauge direction projected out.
   Well conditioned (condition number ~20). This nails everything the images can
   see, and leaves "where zero sits" undetermined.
2. Slide along the gauge orbit until the optical-axis-at-XML-zero is horizontal
   and points into the operator's stated quadrant — III for the left arm,
   I for the right.

Because step 2 moves along an exact invariance, **every observed-view prediction
is preserved bit for bit**. Folding the constraint into step 1 as a penalty would
let it fight the pose fit and degrade conditioning for nothing.

The gauge direction is derived rather than guessed. Predictions are invariant
under $T_{GC} \to R_\text{roll}(-\delta) T_{GC}$ with $q_\text{roll} \to +\delta$;
under the local parameterisation $T_{GC} = T_\text{nom}\exp(\xi)$ the left
multiplication maps to a local perturbation through the adjoint:

$$g = \left(-\operatorname{Ad}_{T_\text{nom}^{-1}} \begin{bmatrix} a \\ 0\end{bmatrix},\; 1\right), \qquad a = (0,-1,0)$$

normalised, with $\operatorname{Ad}_{T^{-1}} = \begin{bmatrix} R^\top & -R^\top\hat t \\ 0 & R^\top\end{bmatrix}$.
A QR completion of $g$ to an orthonormal basis gives the 6 free directions.

### 8.4 Head tilt zero

Same idea, one dimension: find the nearest tilt shift making the head's optical
axis horizontal and forward-facing, by scanning $[-\pi, \pi]$ on a 1441-point
grid for sign changes of the axis $z$-component and refining each with `brentq`.
Candidates are filtered to forward-facing, and the smallest $|\delta|$ wins.
Shifts beyond 45° raise rather than silently accept.

This check is **frame-relative**: the operator sets the pan zero facing the
board, which makes the axis forward by construction. It confirms the solve is
self-consistent; it is *not* evidence that the frame itself is right.

---

## 9. Stage 8 — held-out validation

**Fits nothing.** Every parameter is frozen. The operator poses the robot by hand
into fresh postures, torque off, and each camera observes the board.

For each sample: predict the camera pose from the frozen model and the measured
joint angles, observe it via PnP, and report

$$\Delta = T_\text{obs}^{-1} T_\text{pred}, \qquad e_t = \lVert \Delta_{t}\rVert, \quad e_R = \lVert\log \Delta_R\rVert$$

Head angles come from the paired final zero records through `unwrap_delta`; arm
angles are resolved through Stage 4's legal ranges.

Reported per camera: RMS, p95 and max of both errors, the per-axis **bias**
(a systematic offset looks quite different from noise), the PnP reprojection
error, and the **predicted pixel RMS** — board points projected through the
*predicted* pose, which is the number that reflects how the calibration would
behave in use.

### Shared drift

If the board or base moved between calibration and validation, every camera is
wrong in the same way. Stage 8 estimates one shared world correction $D$ from all
samples:

$$D_i = T^{(i)}_\text{obs}\,T^{(i)-1}_\text{pred}, \qquad D = \operatorname{mean}_{SE(3)}\{D_i\}$$

using an iterative Lie mean:

$$\mu \leftarrow \mu \exp\left(\frac{1}{N}\sum_i \log\left(\mu^{-1} D_i\right)\right)$$

Drift-corrected errors are reported **alongside** the raw ones, never instead of
them. A large $D$ is a diagnosis — "the setup moved" — not a correction to be
deployed.

### Gates

```python
FINAL_TCP_MAX_MM = 8.0   # 3.0 is "good"
```

plus a rotation gate and a minimum sample count, per camera. Results are **always
written**, whether or not gates pass, and the UI reports "complete, but the error
is too large to deploy". Discarding a failed validation would discard exactly the
evidence needed to diagnose it.

---

## 10. Mounting: nominal and physical sides

The robot can be mounted normally or back-to-front. This distinction caused more
wrong-but-plausible results than any other single thing in the project, so it is
stated precisely.

**Nominal left/right is invariant.** It is fixed by a camera's or motor's name
and the port it is plugged into. No pose changes it. Every saved file, every
model role and the baked XML are keyed by it.

**Physical left/right varies.** It is the side an operator sees a thing on.

Turning the robot back-to-front rotates the body 180°, but the working side stays
on the same side of the room. Therefore:

| mounting | physical left maps to |
|---|---|
| normal | nominal left |
| back-to-front | nominal **right** |

The two agree under normal mounting and are exactly opposite under flipped. *If a
change makes them agree under both, or disagree under both, it is wrong.*

Anything that asks a human to look and point observes the **physical** side, so
it is stored physically (`left_wrist_physical`) and folded onto the nominal role
exactly once, by `cameras.resolve()`. Anything displayed to an operator is
converted back, so a label names the side they can actually point at.

### The head pan zero is the same rule

The stored pan zero is nominal: an encoder reading on `head_motor_1`. Where the
head physically points is what varies. The operator sets the coarse zero facing
the board — the only posture that sees anything — and the board is on the working
side in both mountings. So:

| mounting | head facing the board is |
|---|---|
| normal | chassis-front, model $q = 0$ |
| back-to-front | chassis-back, model $q = \pi$ |

Same physical posture, opposite nominal value.

This correspondence is fixed by the mounting and nothing else, so **every stage
derives it locally** from `head_model.mounting_pan_offset()`. It is never
recorded, never passed between stages, and no result file has to agree about it.
Adding a dependency on an upstream stage for it is a regression even if the
numbers happen to come out right.

Apply it exactly once per conversion, where an encoder count becomes a model
angle:

- **Stage 5** applies it when solving the head, so the saved $T_{WB}$ is already
  correct.
- **Stage 8** applies it when predicting camera poses from joint angles.
- **Stages 6 and 7 must not.** They never read a pan angle; they consume the
  finished $T_{WB}$. Applying it again adds a second half turn on top of Stage
  5's, putting the pan a full turn out and rotating every head prediction.

> **The symptom is unmistakable:** head camera rotation RMS near 180°, constant
> in pan and tilt, with **both wrist cameras unaffected** — they reach the board
> through $T_{BA}$ rather than through the head's pan angle. An error that *does*
> track the joint angles is a zero or a sense, not this.

---

## 11. Numerical practice

**Optimiser.** `scipy.optimize.least_squares`, `method='trf'`, `loss='soft_l1'`,
`x_scale='jac'`, `xtol = ftol = 1e-14`. Jacobians are finite-differenced;
the parameter counts are small enough that analytic ones would buy little.

**Conditioning is recorded, not just used.** $\kappa = \sigma_1/\sigma_n$ from the
SVD of the final Jacobian is stored with every result, so a direction the data
never pinned down shows up explicitly instead of hiding behind a small residual.
`MAX_CONDITION_NUMBER = 1e6`; `MIN_SINGULAR_VALUE = 1e-6` marks a direction that
should be fixed by convention rather than fitted.

**Gauge fixing, two ways.** Where the gauge direction is an axis of the
parameterisation (arm mount yaw), that index is simply dropped from the free set.
Where it is oblique (wrist roll), it is projected out with a QR-completed
orthonormal basis. Both keep the *reported* parameter vector full-length, so
callers never handle a reduced vector.

**Diagnosing a wrong answer.** In rough order of how often each has been the
cause here:

| symptom | likely cause |
|---|---|
| error constant under joint motion, ≈180° | mounting/pan-zero applied twice or not at all (§10) |
| error tracks joint angles | a joint sense or zero (§4) |
| good fit, bad holdout | postures too alike |
| result unchanged by the initial guess | gauge freedom, not convergence (§1) |
| plausible residual, implausible geometry | mirrored solve — check senses against an external fact |
| one bad view dominating | corner-starved detection (§2) |

The fourth row is worth restating, because it is the cheapest test available: **a
convergence problem varies with the initial guess; a gauge problem does not.**
