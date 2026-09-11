# Project instructions

- Keep all distances in metres, time values in seconds, and internal angles in radians.
- Use `u = [cos(elevation) cos(phi), cos(elevation) sin(phi), sin(elevation)]` from array centre to source.
- Define every oriented pair as `(i, j)` with `tau_ij = T_i - T_j`.
- The far-field row for `(i, j)` is `(r_j - r_i)^T / c`; the default sound speed is `343 m/s`.
- Default observations are `M-1` linearly independent pairs relative to microphone 0; under independent TOA errors they are statistically correlated through the shared reference.
- Keep `sigma_toa` (one microphone TOA error) distinct from `sigma_tdoa` (an abstract independent measured-TDOA error). Full TOA-induced pair sets require the propagated covariance and a pseudoinverse/projected treatment.
- Never represent a fractional acoustic delay with an integer sample shift.
- A positive fractional delay means `y(t) = x(t-delay)` and moves a waveform
  to the right. Keep delay units explicit (seconds versus samples), use
  zero-extended boundaries, and report the valid comparison region.
- Direction/distance source coordinates are relative to the microphone-array
  centroid: `q = centroid(r) + R*u`. Plane relative TOAs are
  `-(r_m-centroid).T*u/c`, preserving the established TDOA sign.
- Keep the exact spherical norm, the second-order distance expansion, and the
  plane-wave model as separate implementations and tests.
- Do not hide rank deficiency with an ordinary inverse or report a finite CRLB in an unobservable parameter direction.
- Planar arrays must remain labelled as mirror-ambiguous even where their local upper-hemisphere CRLB is finite.
- Run `python -m pytest` and execute every committed notebook before accepting mathematical changes.
- The deterministic propagation generator has been independently validated;
  GCC-PHAT may use it, but frequency-domain delay remains the high-accuracy
  reference until cross-generator GCC agreement is checked.
- Signal-level GCC Monte Carlo uses an explicitly labelled diagnostic noise
  model. Do not conflate per-channel additive-noise SNR with `sigma_toa` or
  `sigma_tdoa`, and do not claim a signal-level CRLB without deriving it.
- The current stage permits an exact retarded-time kinematic model in a
  homogeneous stationary medium for subsonic source motion and a
  sequence of independent frame-wise GCC/WLS and equal-weight far-field SRP-PHAT
  bearings. Do not call this tracking, and do not add EKF/UKF, SRP-Harmonics,
  reflections, wind, or correlated background without a later validated stage.
- Sequential validation must synthesize one continuous source/channel/noise
  realization and extract overlapping frame views from it. Never resynthesize
  overlapping frames independently, never count overlapping frames as
  independent trials, and never provide truth or future DOA estimates to an
  estimator. Call the output "sequential independent bearings, not tracking".
- Bearing errors must use the spherical log-map in an orthonormal tangent
  basis and have units of angular-arc radians. Never replace them with an
  unwrapped coordinate subtraction; reject the non-unique antipodal log-map
  explicitly.
- Fit bearing covariance only from calibration sequences. Calibration and
  evaluation require disjoint sequence/source/noise seeds, while all methods
  inside one sequence receive the same stream. Overlap frames are dependent
  residual samples, never independent trials or bootstrap units.
- Online quality metadata may use only signal/estimator observables. Truth and
  angular error may be used for offline correlation diagnostics but never to
  form a quality score. Do not call an uncalibrated score a probability.
- S7A is a calibrated bearing measurement benchmark, not tracking and not a
  signal-level CRLB. Do not add EKF/UKF/alpha-beta filtering during S7A.
- Multi-station world coordinates use a right-handed ENU frame: `x=East`,
  `y=North`, `z=Up`. A `StationPose.position_world_m` is the array centroid;
  local microphone coordinates must be centroid-relative and
  `r_world=p_station+Q_local_to_world@r_local`, with `Q` a proper rotation.
- The online `BearingMeasurement` contract must remain truth-free. Never add
  true direction/position, angular error, future estimates, or true emission
  time to it. Calibration bias enters the spherical tangent residual; never
  subtract raw azimuth/elevation coordinates.
- One station measures bearing, not range. Static multi-station position
  covariance is only a local Gaussian linearization benchmark. Do not hide
  position-information rank deficiency with epsilon, an ordinary inverse, a
  ground constraint, or an unreported `z>=0` bound.
- A singular tangent bearing covariance defines a degenerate Gaussian, not an
  unweighted residual component. Whiten only its positive-eigenvalue subspace
  and enforce every zero-eigenvalue component as an exact equality constraint.
  Report incompatible deterministic constraints as invalid; never replace a
  zero eigenvalue by epsilon or silently discard its nullspace.
- A preliminary exact-constraint least-squares solve is only an initializer
  whenever constrained stochastic optimization is possible. Decide
  compatibility from the final constraint residual. Report both raw
  `||g_Z||`, where `g_Z=Z.T @ grad(J)`, and the dimensionless diagnostic
  `0.5*sqrt(g_Z.T @ solve(Z.T @ I @ Z, g_Z))`; the latter must pass. Optimizer
  exit status is diagnostic: neither `xtol` success alone guarantees
  acceptance nor `success=False` alone invalidates an otherwise finite,
  feasible, forward-ray, observable and KKT-consistent solution.
- S7B fuses only bearings referring to one static source state/time and first
  validates fusion with direct bearing-level noise. For future dynamics,
  `t_receive,k=t_emit,k+||q(t_emit,k)-p_k||/c`; equal reception timestamps can
  correspond to different emission times. Do not intersect asynchronous
  moving-source rays and call it dynamic localization. Central causal
  retarded-time 3-D tracking belongs to S7C.
- S7C-A is only the retarded-time bearing measurement foundation for a
  constant-velocity 6-D state. Use synchronized reception timestamps in the
  physical equation; availability is causal metadata, and station clock
  offset/drift must not be applied a second time. Do not add a state update,
  EKF/UKF/particle filter, process noise, tracking Monte Carlo, signal-level
  GCC/SRP fusion, wind, reflections, or SRP-Harmonics during S7C-A.
- Distinguish one-station observability cases explicitly. One instantaneous
  bearing has no range; a radial constant-velocity temporal stack remains
  rank deficient. Under the ideal exact retarded-time model, known finite
  sound speed and strictly constant non-radial velocity can give formal local
  rank 6, but the scale information is weak and vanishes in the independently
  derived instantaneous `c -> infinity` limit, which has rank 5. Do not turn
  this formal finite-speed rank into a claim of robust single-station ranging.
- During S7C-B, call the output a batch estimate of one constant-velocity
  state, not tracking. `reception_center_timestamp_s` enters the retarded-time
  physics; `available_timestamp_s` only gates causal access. An event may
  affect estimates published after it arrives, but never rewrite an earlier
  published result. Deduplicate exact event identities, reject conflicting
  payloads explicitly, select one estimator variant per experiment, and keep
  scenario truth outside the causal processor and estimator contracts.
- Treat every PSD observation covariance as a degenerate Gaussian: whiten its
  positive eigenspace and enforce the nullspace exactly. A residual outside
  that support has infinite normalized squared error; chi-square diagnostics
  use the positive covariance rank, with rank zero handled explicitly.
- Far-field TDOA WLS must check interior, hard-case, and elevation-boundary
  candidates of the unit-vector quadratic problem. Record the numerical
  global-search contract; never equate one optimizer `success` flag with a
  proof of the best admissible solution.
- Near a local bearing pole, use a fixed measurement-anchored tangent frame
  with covariance and calibration bias in the same coordinates. Keep the
  antipodal log-map ambiguity explicit and do not hide physical rank loss.
- GCC lag candidates and refinement must stay inside the exact continuous
  delay interval. When zero-padding a one-sided real spectrum for an
  oversampled inverse FFT, preserve the single-bin Nyquist weight and verify
  against an independent direct spectral sum.
- Moving-source metadata must distinguish equal-reception delay differences
  from same-emission event TDOA. Never extrapolate a finite-support trajectory
  silently. When the emitted band edge is known, enforce
  `f_emit,max * max(dt_emit/dt_receive) < fs/2`; do not infer an arbitrary
  waveform's band automatically.
- Validation random streams must use structured `SeedSequence` coordinates
  and separate child streams for distinct stochastic mechanisms. Include the
  run seed in provenance/IDs and test for collisions across shifted base seeds,
  configurations, and sequence indices beyond 1000.
- S7C-C1 is an EKF baseline only under strict constant velocity, known constant
  sound speed, direct bearing-level observations and `Q=0`. The filter state is
  `[q(T),v(T)]`; physical reception time enters the retarded measurement while
  availability only gates causal access. Initialize only from an accepted
  causal batch prefix and never reuse initialization events.
- In S7C-C1 accept only positive-definite tangent `R` and initialized `P`.
  Report singular observations as `unsupported_singular_covariance` without
  epsilon loading or pseudoinverse; retain the batch estimator's separate PSD
  exact-constraint semantics. Use the residual-Jacobian sign
  `x_post=x_prior-K*e` and Joseph covariance update. Do not claim manoeuvre
  support, process noise, smoothing, real-audio tracking, or completion of all
  S7C-C work from this limited baseline.
- In S7C-C1, `advance_to(T)` must replay every complete availability group up
  to `T` before publishing. External publication frequency must not select the
  initialization prefix or repartition measurement updates. Resolve a whole
  equal-time group before estimation and keep event-free publication
  propagation separate from the internal event posterior.
- Preserve the raw event journal but construct the C1 eligible set separately.
  Reject unknown stations and non-positive-definite tangent covariances per
  event with persistent IDs/reasons; never let one unsupported observation
  poison later eligible initialization or invalidate an otherwise valid state.
  Conflict invalidation/recovery must remain auditable. C1 final P95 uses
  explicit linear interpolation over one final error per independent sequence.
- S7C-D1 is a validation-only stress benchmark of the unchanged strict-CV C1
  baseline. Keep `Q=0`, the existing update, initialization gates, and nominal
  positive-definite tangent `R`; do not add outlier rejection, NIS gating,
  adaptive covariance, smoothing, or manoeuvre dynamics during D1.
- Generate D1 truth, nominal bearing noise, transport loss, delay, outlier
  mask, and outlier direction from separate structured seed streams before a
  profile is selected. The nine profiles within one base block are paired, not
  independent trials; confidence intervals and bootstrap units are whole base
  sequences. Missing events must not enter the estimator, and outlier truth
  must remain evaluator-only metadata.
- D1 causal results use fixed external epochs and may consume only events whose
  availability time has passed. The no-update baseline originates from the
  immutable first accepted EKF initialization batch, not from a state that has
  already received updates. Label the full-record batch as offline and
  noncausal. Report conditional error/coverage together with all-sequence
  denominators and explicit failure reasons; use linear P95.
