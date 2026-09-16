# S7C three-station audio-tracking pilot protocol

This protocol is frozen before the held-out evaluation. It is an integration
pilot, not a rare-tail qualification or a field-validation claim.

## Fixed physical and signal configuration

- ENU station centroids: `S0=(0,0,0) m`, `S1=(100,0,5) m`,
  `S2=(10,90,-2) m`.
- Every station uses the accepted four-microphone tetrahedral array with
  `0.20 m` maximum aperture and its complete world-coordinate microphone set.
- One common moving-source waveform feeds all twelve microphones through the
  exact retarded-time kinematic model in a homogeneous stationary medium.
- Signal: random broadband, `300--10000 Hz`; `fs=48000 Hz`.
- Duration: `2.0 s`; reception interval starts at `0.5 s`.
- Frame length: `1024`; hop length: `512` (50% overlap).
- Bearing estimators run on every frame. The tracker consumes every thirty-
  second frame from every station (`2.9296875 Hz/station`, deterministic frame
  indices `0,32,64,...`); all-frame bearing accuracy remains reported. This causal,
  truth-free decimation was frozen after pre-evaluation feasibility runs of
  dense augmented-history updates at every frame, stride four and stride
  sixteen each remained longer than one hour, before any evaluation CSV or
  error metric existed.
- SNR: `-6 dB` and `10 dB`, defined separately from the full clean/noise RMS
  of each continuous station recording.
- Trajectories: constant velocity, a finite constant-acceleration segment and
  a smooth turn. Manoeuvres start at `1.0 s`, after an initial CV interval,
  and finish at `1.5 s`.
- `Qc = I m^2/s^3`, fixed by the accepted S7C-C development experiment. It is
  not retuned from the audio evaluation.
- Tracker history is `0.85 s`, with a declared maximum range of `150 m`,
  maximum transport delay of `0.10 s` and history step `0.25 s`. The required
  implementation span is `150/343 + 0.10 + 0.25 = 0.7873 s`; the remaining
  `0.0627 s` is margin. The
  generated trajectories stay below `110 m` from every station. These bounds
  were fixed after a pre-evaluation computational-feasibility run of a `2 s`
  history exceeded one hour and about `2 GB` process RSS, before any result
  CSV or evaluation-error metric existed. `Qc`, gates and estimator equations
  were not changed. Together with the declared tracker-frame stride this is a
  bounded offline integration pilot, not a real-time performance claim.

## Split, seeds and estimators

- Calibration base seed: `20260918`; evaluation base seed: `20260919`;
  smoke base seed: `20260917`.
- For every `(trajectory, SNR)` cell: 1 independent calibration sequence and
  1 independent evaluation sequence (6 sequences in each split). Frames
  within a sequence overlap and
  are dependent; they are never counted as independent trials.
- Calibration is separate for station, estimator, SNR and trajectory. The
  calibration mean tangent residual and sample covariance are frozen before
  the corresponding evaluation cell is processed.
- Compared audio bearing variants: all-six-pair GCC/WLS and equal-weight
  SRP-PHAT. Both consume the identical station frame.
- No true state, range, velocity, error, outlier label or true emission time
  enters a bearing estimator or tracker.

## Causal timing contract

- Bearing physics uses the reception-frame centre timestamp.
- Availability is `frame_end + 0.010 s modeled processing delay + station
  delivery delay`, where delivery delays are `0.000/0.015/0.030 s` for
  `S0/S1/S2`.
- Measured computation runtime is reported independently and does not alter
  simulated availability.
- Events are processed only after availability. Earlier publications are not
  rewritten. The retarded-time tracker solves emission time from the bearing
  reception timestamp and estimated history; no true emission time is passed.
- The confirmation span is `0.05 s`, the allowance is 180 events with up to
  20 individual confirmation failures, and the initialization buffer is 360
  events. These settings are fixed from the `10.667 ms` frame cadence and the
  short pilot duration before evaluation, not from evaluation errors;
  per-event NIS, consensus, recovery and process-noise thresholds are unchanged.

## Predeclared pilot interpretation

- Report bearing coverage/error, tracker confirmed availability, final valid
  fraction, conditional position/velocity errors, empirical covariance
  coverage, failures, resets, runtime and history memory.
- Temporal lag-one and aligned inter-station tangent-error correlations are
  diagnostics. The present filter still assumes independent measurement
  errors; measured correlations are therefore an explicit limitation.
- No parameter is changed after viewing evaluation results. The deliberately
  small pilot cannot qualify rare tails or sequence-level confidence
  intervals. Improvement over
  synthetic direct-bearing experiments is not an acceptance requirement.
