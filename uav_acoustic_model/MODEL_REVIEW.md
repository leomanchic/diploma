# Physical and mathematical model review — 2026-09-07

Base: `c281523d478169e7611b1a5b1c9003ac5dcdbb71`,
`feature/s7c-retarded-bearing-model`. The review began on 2026-09-06.

## Confirmed fixes

1. **Calibration-frame discontinuity.** The previous fusion residual switched
   charts when a candidate or observation had horizontal norm below `1e-7`,
   while leaving anisotropic covariance and bias unchanged. With measured
   direction proportional to `[0, .01, 1]`, covariance `diag(1e-8,1e-4)` and
   candidate proportional to `[x,0,1]`, changing x from `.999999e-7` to
   `1.000001e-7` changed zero-bias cost from `0.9999343385` to `9999.3333844`.
   An immutable `tangent_frame` now selects one residual/Jacobian convention
   for the entire measurement. The event payload comparison includes it.
2. **Retarded-time cancellation.** For radial receding motion with
   `v=343*(1-1e-14)`, current range 100 m and reception time zero, the old
   quadratic-root evaluation had relative error `0.00130435` against the
   independent linear equation `t_emit=-100/(343+v)`. The nonnegative-dot
   branch now uses the rationalized root; the approaching branch is unchanged.

## Calibration migration

The default `tangent_frame="prediction"` preserves existing calibration
tables and means. Its azimuth chart excludes a predicted local pole; this is
now an explicit domain error. The residual and Jacobian use the same domain.

For a pole-safe measurement use `tangent_frame="measurement"` with bias and
covariance fitted from `measurement_anchored_tangent_residual(true, measured)`
on calibration data. Do not relabel an existing anisotropic covariance or
nonzero mean without refitting or a justified transport of the distribution.
The static and dynamic estimators accept both declared conventions. No truth
field was added to the online contract. Existing statistical studies retain
their historical convention. New regression tests include nonzero bias,
anisotropic covariance, noisy zenith, analytic/finite-difference Jacobians and
the static limit of the dynamic model.

## Physical scope

The implemented forward signal is `x_m(t)=A_m(t)*s(t_emit,m)`, with A equal to
one or inverse range. It exactly models retarded-time kinematics under the
homogeneous stationary-medium assumptions, with interpolation error in the
sampled signal. It is not a complete calibrated pressure model.

One explicit physical reference is a compact volume-flow monopole Q:

\[
(c^{-2}\partial_t^2-\Delta)\Phi=Q(t)\delta^{(3)}(x-q(t)),
\quad p=\rho_0\partial_t\Phi,
\quad\Phi=\frac{Q(t_e)}{4\pi R(t_e)[1+u(t_e)^Tv(t_e)/c]}.
\]

The reference requires a causal outgoing field, smooth source and trajectory,
subsonic speed with a margin, and receivers separated from the source. Its
pressure includes time derivatives of amplitude and geometry. This review
does not introduce a new pressure generator, wind, reflections or tracking.
Absolute SPL, detection range and real-environment robustness remain outside
the validated claims. The corresponding full Russian model specification
updates the existing `bdipl_formal_model_2026-09-02.md` document, revision
2026-09-07.

Model credibility is assessed through explicit intended use, assumptions,
verification, independent validation and uncertainty; see
[NASA-STD-7009B](https://standards.nasa.gov/sites/default/files/standards/NASA/B/1/NASA-STD-7009B-Final-3-5-2024.pdf).
This is not a certification against that standard.

## Verification

Baseline: 337 tests passed in 103.15 seconds.
After fixes: 345 tests passed in 105.16 seconds.

Notebook gate: **14/14 passed, no error outputs**. Unmodified code cells ran
through IPython because network-kernel startup is blocked. After the first
long process was interrupted, the GCC statistical suite was completed
using 8 worker processes and checkpoints: all 70 original configuration
functions ran with 1000 calibration and 2000 evaluation trials each
(210,000 independent realizations). The original suite aggregation and
every notebook cell/assertion then ran with exact result-cache replay,
keyed by source hash and complete bound arguments. No trial counts,
seeds, criteria, or source cells were changed. Recovery runtime: 457.44 s.
Some other notebooks read historical CSV results; executing all notebooks
does not mean every historical simulation was rerun. Verification-copy
code matches final code apart from non-executable docstring changes.
