# S7C-C1: causal retarded-time EKF baseline

## Scope

This stage estimates one source state from asynchronous, calibrated bearing
events produced by three stationary stations. It is an **EKF baseline under a
strict constant-velocity model**, not a model for manoeuvres, process noise,
real audio, wind, reflections, or smoothing with future data.

All vectors use the common right-handed ENU world frame. Distances are metres,
times seconds, velocities metres per second, and tangent residuals angular-arc
radians. Reception timestamps are already synchronized. Availability time is
transport metadata and never enters the propagation equation.

## State and deterministic transition

At the central processing epoch `T`,

\[
x(T)=\begin{bmatrix}q(T)\\v(T)\end{bmatrix},\qquad
F(\Delta t)=\begin{bmatrix}I_3&\Delta t I_3\\0&I_3\end{bmatrix}.
\]

The C1 prior is

\[
x^- = F x^+_{prev},\qquad P^- = F P^+_{prev}F^T,
\qquad Q=0.
\]

This is exact only for constant velocity. In particular, a stochastic
acceleration model cannot recover the uncertain past emission position from
the current `q,v` alone; that extension needs an explicit trajectory-history
model and is outside C1.

## Retarded bearing and residual convention

For station centroid `r_j`, physical reception time `t_r`, and candidate state
at epoch `T`, the prediction solves

\[
q(t_e)=q(T)+v(T)(t_e-T),\qquad
t_r=t_e+\frac{\|q(t_e)-r_j\|}{c},\qquad
u=\frac{q(t_e)-r_j}{\|q(t_e)-r_j\|}.
\]

The estimator recomputes `t_e` from the candidate state. Simulator truth
`t_e` is never part of `BearingMeasurement` and is unavailable online. The
implementation reuses the analytic `retarded_bearing_residual` and its tested
Jacobian. `BearingMeasurement.tangent_frame` is immutable: residual, Jacobian,
calibration bias and covariance all stay in either the declared `prediction`
or `measurement` frame. An old covariance must not be relabelled without
recalibration or justified transport.

For residual `e(x)` and `H=de/dx`,

\[
e(x^-+\delta x)\simeq e(x^-)+H\delta x,
\quad S=HP^-H^T+R,
\quad K=P^-H^T S^{-1},
\quad x^+=x^- - K e(x^-).
\]

The minus sign follows from differentiating the residual itself. Linear
systems are solved directly; `S^{-1}` is notation only. Covariance uses the
Joseph form

\[
P^+=(I-KH)P^-(I-KH)^T+KRK^T.
\]

One measurement update uses one linearization. Agreement with the nonlinear
batch optimum is therefore only expected locally; exact equality is not an
acceptance criterion.

## Initialization and causal event handling

The filter returns `not_initialized` until the available prefix passes fixed
pre-evaluation gates: at least six unique valid measurements, local rank six,
scaled condition number at most `1e9`, maximum angular residual at most
`0.1 rad`, and scaled projected-KKT residual at most `1e-6`. The existing
retarded-time batch estimator supplies both the initial state and its local
linearization covariance at the current processing epoch. No truth, future
full batch, or artificial small covariance is used.

Events are replayed by `CausalBearingEventStream`. Processing time never moves
backward. Exact duplicates do not update twice. Initialization events are
marked processed and are not reused. A late event is evaluated against the
current state but retains its own physical reception time; this is justified
only because the trajectory is strictly constant velocity. If a previously
used identity is later quarantined by a conflicting payload, the current
publication is invalidated and the next call attempts batch reinitialization
from the cleaned available prefix. Earlier immutable publications are not
rewritten.

## Supported covariance domain and diagnostics

C1 requires positive-definite `R` and positive-definite initialized `P`.
Singular `R` returns `unsupported_singular_covariance` without changing the
posterior. No eigenvalue is replaced by epsilon and no nullspace component is
discarded. Exact constraints from positive-semidefinite observations remain
available in the independent batch estimator but are intentionally not yet an
EKF feature.

Each attempted update reports pre-update NIS with two degrees of freedom,
runtime, covariance rank/condition, symmetry error and minimum eigenvalue.
The evaluator—not the estimator—computes post-update state NEES with six
degrees of freedom. These chi-square comparisons are local Gaussian
diagnostics, not proof of a globally Gaussian nonlinear posterior. Temporal
NIS/NEES samples inside one sequence are dependent. Coverage intervals use
whole-sequence final outcomes.

## Validation protocol fixed before the full run

- seed: `20260908`, structured as
  `SeedSequence([base_seed, configuration_index, sequence_index, stream_id])`;
- separate truth, bearing-noise, and delivery streams;
- 24 configurations: informative/poorly-conditioned geometry, stationary/
  uniform-oblique motion, ordered/reordered delivery, and anisotropic standard
  deviations `(0.03°,0.05°)`, `(0.10°,0.18°)`, `(0.30°,0.50°)`;
- four independent whole sequences per configuration, 96 total;
- 15 asynchronous bearings per sequence;
- matched methods: EKF, causal-prefix nonlinear batch, and propagation from
  their common initial batch without later updates; offline full-record batch
  is reported separately as noncausal.

Four sequences per cell provide only wide binomial intervals. Time samples
within a sequence are retained for diagnostics but never counted as
independent trials.
