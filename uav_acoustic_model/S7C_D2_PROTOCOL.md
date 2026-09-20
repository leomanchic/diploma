# S7C-D2 robust-bearing protocol

## Frozen scope

S7C-D2 compares four explicitly selected causal strict-constant-velocity
estimators. The accepted C1 default is unchanged. All variants retain the
state `x(T)=[q(T),v(T)]`, exact retarded-time bearing physics, known constant
`c=343 m/s`, `Q=0`, positive-definite observation/initial-state covariances,
availability-time causality, the residual-Jacobian correction sign and Joseph
covariance update.

The four variants are:

1. `c1_baseline`: unchanged C1 initialization and no NIS rejection;
2. `robust_initialization_only`: deterministic consensus initialization and
   no NIS rejection;
3. `nis_gate_only`: unchanged C1 initialization and pre-update NIS rejection;
4. `robust_combined`: consensus initialization and pre-update NIS rejection.

No truth, outlier flag, true error or future event is available to any
variant. No adaptive `Q/R`, robust loss after initialization, smoothing,
manoeuvre dynamics, acoustic frontend, wind or reflections are introduced.

## Deterministic consensus initialization

At an availability group with `n>=6` eligible observations, candidate subsets
are generated reproducibly from sorted event IDs. The complete prefix is first.
For prefixes of at most eight observations, every leave-one/two-out subset
retaining at least six is then tried. For larger prefixes, bounded six-event
hypotheses are drawn by a local RNG whose seed is the SHA-256 digest of the
sorted observable event IDs. It neither touches global RNG state nor uses
truth/outlier labels. At most 128 candidates are evaluated. Each candidate is
fitted by the existing nonlinear retarded-time batch estimator at the current
processing epoch.

For every valid, locally rank-six, sufficiently conditioned and KKT-consistent
candidate, all currently eligible observations are scored using spherical
tangent residuals and their declared positive-definite covariance:

`d_i^2 = r_i(x)^T R_i^{-1} r_i(x)`.

The frozen consensus threshold is
`chi2_2(0.995) = 10.596634733096073`. The first hypothesis in the deterministic
order producing at least six inliers is refitted with all of those inliers and
accepted only if that refinement passes the original C1 rank, condition,
maximum angular residual and scaled-KKT gates. Otherwise the search continues.
This bounded first-valid rule was fixed after a pre-evaluation performance run
showed that exhaustive repeated large-prefix fits were computationally
unbounded; that aborted run wrote no evaluation CSV and did not change either
threshold or the held-out sample.

On success, every event in the successful prefix is classified exactly once:
consensus inliers become initialization events; non-inliers receive persistent
reason `robust_initialization_consensus_outlier`. Both sets are marked
processed, so no prefix event can later become an EKF update. On failure the
publication reports `initialization_failed:robust_consensus_not_found` plus
the attempted diagnostic; no observation is permanently classified until a
successful initialization exists. No epsilon loading or artificial covariance
shrinkage is permitted.

## Pre-update NIS gate

For a positive-definite two-dimensional bearing observation, before changing
state or covariance compute

`S = H P^- H^T + R`, `NIS = r^T S^{-1} r`.

The frozen update threshold is `chi2_2(0.99) = 9.210340371976184`. If the NIS
exceeds the threshold, the update is not applied, prior state/covariance are
retained exactly, and the event receives persistent reason
`pre_update_nis_gate`. The finite pre-update NIS remains in diagnostics for
accepted and rejected attempts. Chi-square interpretation is only a local
Gaussian approximation; post-gate accepted-only NIS is a selected
distribution and is not described as raw chi-square.

## Evaluation design frozen before results

- evaluation base seed: `20260912`, distinct from D1 `20260910` and smoke
  `20260911`;
- geometries and physical profile grid: the two D1 geometries and all nine D1
  profiles without parameter changes;
- 100 independent whole base sequences per geometry, hence 200 independent
  base blocks and 1,800 paired profile-runs;
- all four variants receive identical immutable event objects inside each
  profile; profile-level truth/noise/loss/delay/outlier components retain the
  D1 structured `SeedSequence` construction;
- evaluation epochs remain `0,0.5,...,12.5,14.5 s`, with `14.5 s` draining
  the maximum transport delay;
- smoke uses two sequences per geometry and seed `20260911`; evaluation size
  is not changed after inspecting results.

The comparison reports initialization success/time, final validity,
conditional position/velocity RMSE and linear P95, conditional 95% state
coverage and unconditional `valid AND covered`, failure reasons, event counts,
all defined pre-update NIS, runtime and paired whole-sequence bootstrap
differences relative to C1. Evaluator-only outlier labels classify persistent
robust rejections into false rejection of clean delivered events and detected
outliers; missed delivered outliers are those consumed by initialization or an
applied update. Unclassified events and failed initialization remain explicit
and are not silently counted as correct decisions.

Fraction intervals use Wilson 95% intervals over whole sequences. Variant
differences use a deterministic paired bootstrap over whole base sequences,
2,000 resamples and percentile endpoints. Events, epochs and paired variants
within a sequence are dependent and are never called independent trials.

## Acceptance gates

- default construction remains numerically identical to accepted C1;
- consensus and NIS mechanisms pass separate and combined deterministic tests;
- rejected NIS updates preserve state and covariance with `rtol=0, atol=0`;
- initialization/applied/rejected event identities are disjoint and no event
  is reused;
- frequent, sparse and one-shot publication schedules agree at a common epoch;
- clean, early/late outlier, insufficient-consensus, rank-deficient, dropout
  and delayed cases have explicit outcomes;
- all saved denominators, partitions, seeds and paired units pass structural
  audit; covariance coverage is measured rather than assumed calibrated;
- full pytest, all committed notebooks, nbformat/error/unexecuted-cell audit,
  `pip check` and `git diff --check` pass before S7C-D2 is marked Done.

Acceptance means that the estimator variants and comparison are implemented
and reported correctly. It does not require robust variants to dominate C1.
