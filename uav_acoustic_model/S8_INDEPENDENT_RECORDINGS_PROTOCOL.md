# S8 independent-recording protocol

This protocol was frozen before the independent-session evaluation run.

## Scope and source-data unit

The source-data unit is an original recording session, identified jointly by
`recording_id`, `session_id` and `origin_asset_id`. Fragments, files and
transcodes sharing a session or origin asset count once. Calibration uses
Freesound assets `683298` and `263022`; evaluation uses assets `383904` and
`321687`. The manifest audit requires at least two unique sessions and origins
in each split and no session/origin overlap.

Every recording is an approximation to an emitted source waveform. It already
contains the source microphone response, its original environment, background
and source motion. Re-propagation after level normalization does not validate
absolute SPL, detection range or field performance.

## Frozen paired design

- geometry: the existing informative three-tetrahedral-station ENU scene;
- trajectory: existing `constant_velocity`, with deterministic session-index
  offset; the recorded and broadband member of a pair use the same trajectory;
- duration: `2.0 s`, reception start `0.5 s`. A second pre-evaluation
  feasibility run showed that `3.0 s` low-SNR data could enter a prohibitive
  combinatorial search over unconfirmed initialization hypotheses before any
  CSV or error metric existed. The shortened stream leaves 9 causal tracker
  events (6 construction plus at most 3 confirmation events) and therefore
  preserves an honest possibility of explicit initialization failure;
- sampling/frame/hop: `48 kHz`, `1024/512` samples;
- tracker stride: `64` (`1.46484375 Hz/station`); GCC/SRP bearing metrics
  still use every frame. A pre-evaluation feasibility run at stride `32` was
  stopped before CSV or error metrics because the first held-out low-SNR
  recovery optimization remained prohibitively slow. `Qc`, NIS, covariance
  and estimator algorithms were not changed;
- SNR: `-6 dB` and `10 dB`;
- estimators: all-six equal GCC/WLS and equal-weight SRP-PHAT on identical
  frames;
- emitted band edge: `10 kHz`, passed to the existing Doppler/Nyquist guard;
- tracker: unchanged stochastic-history retarded-time EKF, `Qc=I m²/s³`,
  unchanged NIS, confirmation and recovery thresholds;
- split seeds: calibration `20260923`, evaluation `20260924`, smoke
  `20260925`, expanded through structured `SeedSequence` coordinates;
- count: two source sessions × two SNR values × two source models in each
  split: 8 continuous calibration and 8 continuous evaluation streams.

Recorded and random-broadband members of each pair use the same base sequence
seed. Thus each station receives the same standard-normal AWGN realization;
only its scale changes to realize the requested SNR for that clean stream.
Trajectory, duration, reception timestamps, frame timestamps and station
delivery schedule are identical within a pair.

## Calibration and evaluation contract

Recorded and broadband data receive separate calibrations. Each calibration
is pooled over both calibration sessions and both SNR values and produces one
fixed bias/covariance per `(station_id, estimator_variant, source_model)`.
Evaluation lookup uses only those observable keys; trajectory, SNR, recording
identity and truth are not calibration-selection keys. Evaluation sessions do
not enter either calibration.

Reporting is primarily per original evaluation session. Frames and
publications inside a session are dependent observations, not independent
trials. Aggregate tables state the independent session count explicitly. With
only two evaluation sessions per cell, this is a limited held-out benchmark;
it is not a rare-tail qualification and does not support narrow population
confidence intervals.

The historical single-session files `results/recorded_source_*.csv` remain
unchanged. New outputs use the `independent_recordings_*.csv` prefix.
