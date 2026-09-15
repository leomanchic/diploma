# Frozen S7C-C manoeuvre-bearing comparison protocol

This protocol is fixed before evaluation. It compares the accepted opt-in
`confirmed_recovery` (strict CV, Q=0) with only one explicitly enabled
integrated-Wiener augmented-history variant. Both see identical direct-bearing
events per sequence; evaluator truth is never passed to either filter.

## Physical and stochastic grid

- Known three-station `informative` and `poorly_conditioned` geometries from
  the D1 stress study; independent prescribed truth trajectories: constant
  velocity control, acceleration segment, smooth turn. Every path has a CV
  prefix through 5 s and a manoeuvre in [5, 9] s. Starting inside a manoeuvre
  is not tested.
- Forty-five nominal reception centers (15 per station) at the existing D1
  grid; independent Gaussian tangent errors with standard deviations
  (0.3, 0.5) degrees; no acoustic frontend, outliers, or transport drops.
  Availability delay is uniformly between 0.01 and 0.42 s. The direct-bearing
  covariance is declared positive definite, and tangent frame is the
  measurement direction.
- Truth initial position has an independent per-sequence 2 m Gaussian jitter.
  All other random components use disjoint SeedSequence mechanism streams.
  Development seed: 20260915; final evaluation seed: 20260916. Development
  uses two independent **base blocks** per geometry and runs each block in
  three matched truth modes. Evaluation uses eight independent base blocks per
  geometry, with three matched truth modes each: **16 independent base blocks,
  48 dependent trajectory-runs**. Within each geometry/truth group the eight
  sequence blocks are independent, but runs of different truth modes with the
  same index share noise/jitter and must not be pooled as 48 independent
  trials. Over-time outputs from one sequence are also dependent.
- Fixed evaluation publications every 0.5 s from 0 to 12.5 s plus 14.5 s.
  Three periods are `pre` (<5 s), `during` ([5,9] s), `post` (>9 s).

## Filter selection and history bounds

The development candidate set is isotropic `Qc = alpha I` with
`alpha in {0.05, 0.25, 1.0} m^2/s^3`. Select the candidate minimizing pooled
development *during-manoeuvre* position RMSE across **both** geometries and
**both** manoeuvre kinds, with failed publications assigned a frozen 50 m
diagnostic availability penalty in the development selection score, not an
unreported disappearance. If candidate coverage falls below 0.5, exclude it.
Break ties toward smaller `alpha`. The 50 m value is a selection guardrail,
not a required operational error bound.
Freeze the selected `alpha` before generating evaluation sequences. Do not
retune it after looking at evaluation.

History uses 0.25 s maximum step and 2.0 s retained window. Its declared
physical bounds are maximum source range 250 m and maximum transport delay
0.6 s: `250/343 + 0.6 + 0.25 < 2.0 s`. An observation with emission before
retained support is explicitly rejected. Output-only publication prediction
includes `Qd` without mutating event posterior. Published outputs are causal,
immutable and independent of external publication cadence.

## Fixed evaluator metrics

Per geometry, trajectory, variant and period: conditional position/velocity
RMSE, P95, maximum; valid and confirmed fractions with explicit publication
denominators; six-dimensional 95% chi-square ellipsoid coverage conditional
on valid and unconditional valid-and-covered; count and fraction of rejected
clean bearings during manoeuvre; reset count and recovery duration; first
confirmation and first response after manoeuvre onset; runtime per sequence;
peak retained history nodes and bytes. Reporting of confidence intervals uses
paired bootstrap resampling of whole independent sequences (500 draws,
  fixed seed 20260917), never within-sequence time points. Cross-mode pooled
  intervals, if ever needed, must resample base blocks together across modes.
  Coverage is an
empirical diagnostic, not a calibrated signal-level CRLB.

First run deterministic tests and one-sequence smoke per truth kind. Only
after those PASS run development, freeze alpha, and then evaluation once.
