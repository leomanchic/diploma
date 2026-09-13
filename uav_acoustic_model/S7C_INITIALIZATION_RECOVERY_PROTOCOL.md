# S7C-D initialization confirmation and recovery protocol

## Scope and preserved baselines

This corrective S7C-D experiment preserves the published C1 and D2 combined
implementations as separate reproducible variants.  The new
`confirmed_recovery` variant retains strict constant velocity, `Q=0`, exact
retarded-time bearing physics, known `c=343 m/s` and positive-definite `R/P`.
It adds no manoeuvre model, adaptive `Q/R`, smoothing, acoustic frontend or
environmental propagation model.

The estimator receives only `BearingMeasurement`, station poses and causal
availability metadata.  True state and D1 outlier labels are evaluator-only.

## Frozen tentative-confirmation rules

1. A preliminary hypothesis is fitted from exactly six currently available
   bearings.  At most 16 deterministic candidate subsets are compared.  The
   order is derived from observable event IDs; candidates are ranked by the
   number of tangent-residual inliers, truncated residual score, scaled
   condition number and event-ID signature.
2. The hypothesis remains `tentative` and is not a valid confirmed estimate.
   Only events arriving after hypothesis creation can confirm it.
3. Confirmation requires at least three predictive-NIS inliers from at least
   two stations with at least `0.5 s` reception-time span.  The predictive
   statistic uses `S=H P_h H^T+R`; the fixed threshold is
   `chi2_2(0.995)=10.596634733096073`.
4. A tentative hypothesis is rejected after two contradictory confirmation
   events, eight confirmation events or `3.0 s` of observed confirmation
   opportunity.  At most eight hypotheses are attempted per state generation.
   Silence alone does not increment a contradiction counter.
5. On confirmation, all preliminary inliers are batch-refitted.  Membership
   and refit are iterated to a stable inlier set, the original C1 rank,
   conditioning, angular-residual and projected-KKT gates are reapplied, and
   residual scores are recomputed from the returned final state.  Preliminary,
   confirmation and final scores are stored separately.

Only the candidate pool used to construct a tentative hypothesis is limited to
the most recent 18 unconsumed events.  This bound never truncates confirmed EKF
updates: every admissible event in an availability group is processed in
deterministic `(available_timestamp_s, event_id)` order.  Construction and
confirmation events become one batch initialization and are never replayed as
EKF updates.

## Frozen loss-of-consistency and recovery rules

Confirmed updates retain the D2 pre-update gate
`chi2_2(0.99)=9.210340371976184`.  Four consecutive available measurements
rejected by that gate trigger loss of consistency only when they span at least
two stations and `0.5 s` of reception time.  Missing packets do not enter the
counter.  An accepted update clears the streak.

At a trigger, the publication is explicitly `questionable`, the old state and
covariance are discarded, and a new generation begins.  Triggering events are
not reused.  Only events arriving strictly after the reset enter the fresh
recovery buffer.  The new hypothesis passes the same tentative confirmation
rules.  No old posterior is combined with the recovery batch; no covariance
inflation or artificial regularization is used.  Earlier publications are
immutable.  Event IDs, generation, statistical role, exclusions, reset reason
and recovery duration are retained.

The event-stream quarantine is checked before every availability group.  A
conflict in a tentative construction event rejects that hypothesis.  A
conflict in any initialization or accepted-update event of the active
generation immediately invalidates its state and starts fresh causal recovery
without the conflicting payload.  A conflict in a historical generation is
retained as an audit diagnostic but cannot rewrite old publications or reset a
newer state.  Exact duplicates remain harmless.  If invalidation occurs partway
through an availability group, every still-unprocessed event in that group is
explicitly excluded as `recovery_group_excluded_after_reset`; only events with
strictly later availability may seed the new generation.

## Development and held-out evaluation

Parameters above were selected before evaluation using development seed
`20260913`, ten independent sequences per geometry, the two D1 geometries and
all nine D1 profiles.  The two published D2 failures at seed `20260912`
(`informative/57` and `poorly_conditioned/37`) are deterministic regressions,
not held-out evidence.

Held-out evaluation is frozen as:

- seed `20260914`, disjoint from D1 `20260910`, D2 smoke/evaluation
  `20260911/20260912` and development `20260913`;
- 100 independent whole sequences per geometry, two geometries and nine
  profiles: 200 independent base blocks and 1,800 paired profile-runs;
- three variants on the exact same event objects: `c1_baseline`,
  `d2_combined_published`, `confirmed_recovery`;
- fixed evaluation epochs `0,0.5,...,12.5,14.5 s`;
- 2,000 paired bootstrap resamples over whole sequences.  Events and epochs
  within one sequence are dependent and are not independent trials.

Report final conditional position/velocity RMSE, linear P95 and maxima,
diagnostic position-error fractions above 10/50 m, conditional and
unconditional 95% state coverage, final validity, confirmed-epoch fraction,
first confirmation time, resets, evaluator-only false resets, completed and
censored recovery time, event partitions, failure reasons and runtime.  The
10/50 m thresholds are diagnostics, not operational requirements.

Acceptance does not require the new method to dominate D2.  Invalid periods
and lower confirmed availability must remain in the report; covariance after
selection is measured and is not assumed calibrated.
