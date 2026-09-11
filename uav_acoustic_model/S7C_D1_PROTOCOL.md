# S7C-D1 stress-benchmark protocol

## Scope frozen before evaluation

S7C-D1 measures the limits of the accepted strict-constant-velocity C1 EKF.
It does not change the estimator. The state is `x(T)=[q(T),v(T)]`, `Q=0`,
the homogeneous stationary medium has known `c=343 m/s`, and all observations
are direct bearing-level measurements with positive-definite nominal tangent
covariance. Reception time enters the exact retarded-time equation;
availability time only controls causal access.

No NIS gate, robust loss, adaptive covariance, process noise, manoeuvre model,
UKF, smoothing, wind, reflections, or acoustic frontend is introduced. An
outlier is therefore a controlled violation of the Gaussian observation model,
not a larger covariance reported to the filter.

## Independent units and fixed matrix

- Base seed: `20260910`.
- Geometries: `informative`, `poorly_conditioned`.
- Motion in the statistical study: `uniform_oblique`, with
  `v=[7,-3,1.5] m/s` and a random initial position around
  `[70,55,40] m` generated once per base block.
- Nominal tangent standard deviations: azimuthal arc `0.3 deg`, elevation arc
  `0.5 deg`; the covariance passed to all methods is their squared value in
  radians.
- 100 independent whole sequences per geometry: 200 independent base blocks.
- Nine profiles per block: 1,800 profile-runs. Profiles within a block are
  paired and dependent; they are not 1,800 independent trials.
- Three reception grids contain 15 samples per station: starts
  `0.60/0.75/0.90 s`, common step `0.80 s`, hence 45 potential events.
- External evaluation epochs are `0,0.5,...,12.5 s`, followed by `14.5 s` to
  drain the maximum two-second transport delay. The 14.5-s publication is a
  different epoch and never rewrites the saved 12.5-s result.

The fixed profiles are:

| profile | random loss | deterministic gap | delay | outliers |
|---|---:|---|---|---|
| `nominal` | 0 | none | `U[0.01,0.42] s` | none |
| `dropout_20` | 0.20/event | none | nominal | none |
| `dropout_50` | 0.50/event | none | nominal | none |
| `one_station_gap` | 0 | S1 reception in `[3,7] s` | nominal | none |
| `all_station_gap` | 0 | all stations reception in `[3,7] s` | nominal | none |
| `long_delay` | 0 | none | `U[0.01,2.00] s` | none |
| `outlier_mild` | 0 | none | nominal | probability 0.05, length 5 deg |
| `outlier_strong` | 0 | none | nominal | probability 0.10, length 20 deg |
| `mixed` | 0.20/event | none | `U[0.01,2.00] s` | probability 0.05, length 20 deg |

Gap endpoints are inclusive in physical reception time. Losses are absent
events and are never passed to an estimator. No sequence is removed because
it initializes late or never initializes.

## Bearing generator

For true local direction `u`, let `B(u)` be the accepted orthonormal tangent
basis. The nominal error is

`eta ~ N(0, diag(deg2rad([0.3,0.5])**2))`.

For an outlier, independently draw `alpha ~ U[0,2*pi)` and add

`o = deg2rad(L) [cos(alpha), sin(alpha)]`.

The measurement is produced once by the spherical exponential map

`Exp_u(B(u).T @ (eta + o))`.

The map is evaluated directly for every finite tangent length, including the
extremely unlikely non-injective lengths beyond `pi`; no outcome-dependent
resampling is permitted. `o=0` outside the pre-generated outlier mask. The
nominal positive-definite covariance, not the contaminated distribution, is
passed to the estimator. Truth/outlier labels remain evaluator-only.

## Structured random streams

Each base block uses

`SeedSequence([20260910, geometry_index, sequence_index, mechanism_id])`

with distinct mechanism IDs for `truth`, `nominal_bearing_noise`,
`transport_loss_uniform`, `transport_delay_uniform`, `outlier_mask_uniform`,
`outlier_direction`. All 45 variates for every mechanism are generated before
profile selection. Truth and nominal noise are intentionally shared by all
nine profiles; common loss/mask uniforms give paired nested probabilities;
the same delay uniform is affinely mapped to the nominal or long interval.
This intended pairing is recorded in provenance and is not a seed collision.

## Compared methods and causal access

1. `retarded_ekf`: the accepted C1 implementation, unchanged.
2. `causal_prefix_batch`: a fresh nonlinear retarded-time batch using only the
   prefix available at that external epoch.
3. `initial_batch_no_updates`: exact CV propagation of the EKF's first accepted
   initialization batch state and covariance. It uses
   `publication.initialization_batch`, not an already updated publication.

The same event objects feed all three methods. The full-record batch is stored
only at 14.5 s as `offline_full_record_noncausal`. It is not a causal method
and is not used to revise the 12.5-s publication.

## Saved grain and metrics

Four CSV artifacts are fixed:

- `retarded_ekf_stress_epoch_summary.csv`: geometry/profile/method/epoch
  aggregates over 100 independent sequences;
- `retarded_ekf_stress_sequence_results.csv`: one row per
  geometry/profile/method/base sequence, with separate 12.5-s and 14.5-s
  outcomes and gap diagnostics;
- `retarded_ekf_stress_profile_summary.csv`: final profile comparison and
  paired differences from nominal;
- `retarded_ekf_stress_seed_provenance.csv`: one row per independent base
  block and all six mechanism seeds.

Metrics include valid and initialization fractions, total/valid denominators,
failure reasons, conditional position/velocity RMSE and linear P95,
conditional 95% state coverage among valid results, unconditional
`valid AND covered` fraction, NEES (6 dof), pre-update NIS (2 dof), potential/
delivered/lost/initialization/update/rejected/quarantined/unprocessed event
counts, gap-end and first-post-gap errors, total processing and update runtime,
and covariance symmetry/minimum eigenvalue/condition diagnostics. Temporal
samples within a sequence and profile-runs within a paired block are labelled
dependent.

At every external epoch the aggregate also stores minimum/mean/maximum counts
per sequence for the 45 potential events, complete-profile delivered/lost
events, causally available events, and method-specific used, initialization,
update, rejected, quarantined, and remaining-unprocessed events. Complete-
profile delivered/lost counts describe transport outcomes fixed for the whole
sequence; `causally_available_event_count` is the prefix permitted at that
epoch. These names prevent a future event from being mistaken for a currently
available one.

All P95 values use NumPy `method="linear"`. Fraction intervals use Wilson 95%
intervals over whole independent sequences. Profile-vs-nominal fraction
differences use a deterministic paired whole-block bootstrap with seed
`20260910`, 2,000 resamples, and percentile endpoints; frames/epochs are never
bootstrap units. The bootstrap is descriptive at `n=100`, not a promise of
few-percent precision.

Potential events partition exactly into delivered and lost. For the drained
EKF stream, delivered events partition into initialization, update, rejected,
quarantined, and remaining-unprocessed categories. Initialization/update are
historical disjoint counters; rejected/quarantined remain separately audited.

For gap profiles, `gap_end=7.0 s`. The report stores the delay to the first
subsequently used observation and the errors at the gap end and first external
epoch after that use. This is labelled `post_gap_first_used`; it is not called
accuracy recovery and has no success threshold.

## Pre-evaluation gates and tolerances

These gates test implementation integrity, not favourable stress performance:

- deterministic rerun: identical generated truth, masks, directions, delays,
  event IDs and non-runtime summaries (`rtol=0`, `atol=0` where numeric);
- CV event-free propagation: state `atol=2e-12`, covariance `atol=2e-10`,
  always with `rtol=0`;
- frequent/sparse publication invariance at a common epoch: state
  `atol=3e-10`, covariance `atol=3e-9`, `rtol=0`, identical IDs/outcomes;
- covariance symmetry error at most `1e-10`; a valid covariance must have
  finite strictly positive minimum eigenvalue;
- delay bounds inclusive within `2e-15 s`; gap boundaries use exact decimal
  reception grids and inclusive comparison;
- loss probability controls 0 and 1 yield exactly 45 and 0 delivered events;
- no-update origin IDs equal initialization IDs and exclude later EKF updates;
- aggregation fixture checks invalid denominators, multiple failure reasons,
  conditional/unconditional coverage and linear-vs-higher P95 distinction.

No bound is imposed on stress-case RMSE, P95, NIS, NEES or coverage. Bad
results are retained. Smoke uses two sequences per geometry and is stored only
under a temporary/smoke output directory. The final 100-sequence sample is run
once after smoke passes; its values do not tune estimator or protocol.

Stationary and genuinely rank-deficient one-station radial cases are separate
deterministic controls. They are not members of the 1,800 statistical runs and
must not acquire artificial range information.
