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
