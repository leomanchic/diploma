# S7C-C manoeuvre-bearing history filter: frozen mathematical contract

## Scope and preserved variants

The new variant is explicitly opt-in. C1, published D2 and
`confirmed_recovery` remain separate, reproducible strict-CV (`Q=0`)
implementations. This variant consumes direct, calibrated bearing-level
events from three known station poses in ENU; it does not consume truth,
acoustic waveforms, wind, reflections or future events. The S7C-D event
contract correction at `d281d730` is closed, not a new substage here.

## Continuous and discrete dynamics

Let `x=[q,v]` with `q` in m and `v` in m/s. The stochastic filter model is

\[
dq=v\,dt,\qquad dv=L\,dW,\qquad Q_c=LL^T.
\]

`Qc` has units **m²/s³**. Integrating acceleration white noise over a causal
interval `h>=0` gives

\[
F(h)=\begin{bmatrix}I&hI\\0&I\end{bmatrix},\quad
Q_d(h)=\begin{bmatrix}
h^3Q_c/3&h^2Q_c/2\\h^2Q_c/2&hQ_c
\end{bmatrix}.
\]

The blocks of `Qd` have units m², m²/s and m²/s². This is uncertainty in the
filter, **not** the prescribed deterministic acceleration/turn used to make
synthetic truth. `Qc=0` is the strict-CV limit. The first implementation
supports either `Qc=0` or positive-definite `Qc`; nonzero singular spectral
density requires a separate constrained bridge derivation and is rejected.

## Bounded augmented history and retarded observation

The causal processor stores ordered node epochs `t0<...<tn`, an augmented
state `X=[x(t0),...,x(tn)]` and its **full joint covariance**. Forward
augmentation uses `F,Qd`, retains every cross block, and prunes old nodes by
marginalization (submatrix selection), never by treating them as independent.
The declared history age must be at least
`maximum_range_m/c + maximum_transport_delay_s + history_step_s`.
Measurement events beyond this bound fail with `emission_outside_history`;
neither true range nor true emission time is provided online. For `Qc>0`, the
buffer starts at causal confirmation. An observation whose emission predates
that epoch is rejected, not falsely reverse-projected through stochastic
motion. In the exact `Qc=0` limit only, deterministic CV rebaselining seeds
history before confirmation by the declared physical age; this reproduces
the accepted strict-CV update. Earlier events already consumed by the
strict-CV confirmation batch are never replayed.

For each measurement, the predicted emission epoch solves

\[
t_r=t_e+\|q(t_e)-p_{station}\|/c.
\]

Availability gates access only; the physical reception timestamp enters the
equation. Within a stochastic interval of length `h`, at `s=t-t0`, let
`Qs=Qd(s)`, `Qh=Qd(h)` and `C=Qs F(h-s)^T`. For positive-definite `Qc`, the
integrated-Wiener Gaussian bridge conditioned on both endpoint nodes is

\[
A_1=CQ_h^{-1},\quad A_0=F(s)-A_1F(h),\quad
E[x(t)|x_0,x_1]=A_0x_0+A_1x_1,\quad
B=Q_s-CQ_h^{-1}C^T.
\]

The code uses linear solves, not an explicit inverse. At the predicted
emission epoch it **inserts a bridge node** with covariance `B` and cross
covariances `A0 P(left,*) + A1 P(right,*)`. This makes the nominal emission
state part of the augmented posterior; a delayed observation changes the
current state through retained cross covariance. Repeated events in one
interval are not treated as independent bridge draws. The piecewise
conditional-mean interpolation between nodes remains an approximation for a
retarded root that shifts after linearization; its discretization bias must
be checked by reducing `history_step_s`. The full nonlinear posterior is not
claimed Gaussian or a CRLB.

For a fixed interpolated epoch, define `A=dq(t_e)/dX`, `v_e=dq/dt`,
`n=(q(t_e)-p)/R`, `d=1+n^T v_e/c`. Implicit differentiation gives

\[
dt_e/dX=-n^T A/(c d),\quad
du/dX=\frac{I-nn^T}{R}\left(A+v_e\,dt_e/dX\right).
\]

The local spherical tangent residual derivative multiplies this direction
Jacobian, preserving the declared `tangent_frame`, calibration bias and
residual-sign update `X^+=X^- - P H^T S^{-1}e`. A positive-definite bearing
`R` is required. Covariance uses the joint Joseph update. Pre-update NIS
uses the complete joint `P`; a rejected event does not alter state or P.

## Causal lifecycle and limits

The opt-in tracker reuses the existing tentative/confirmation construction
until a clean CV prefix confirms. **Starting directly inside a manoeuvre is
not validated.** It then runs the stochastic history filter. Event groups are
resolved in availability order independent of external publication cadence;
exact duplicates do not update twice. A conflicting active construction,
initialization or accepted-update ID invalidates the active generation,
while historical conflicts remain auditable. All admissible events in a large
availability group are considered; an intra-group reset classifies its
remainder as excluded. Previously published outputs never change.

An output-only prediction to a requested timestamp includes `Qd` but does
not mutate the event posterior. No true position/range is silently inserted.
The history covariance and coverage are empirical diagnostics, not a signal
CRLB. Real audio, temporally correlated bearing errors, sound-source physical
calibration, variable medium properties and field tests remain outside this
stage.
