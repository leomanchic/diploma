# S8 recorded-source integration protocol

This protocol was frozen before running the recorded-source evaluation.

## Scope

The experiment reuses one real recording as an approximation to the emitted
source waveform in the already validated homogeneous-medium propagation and
three-station tracking chain. The recording already contains its original
microphone response, indoor environment, background and possible source
motion. Its level is normalized. Therefore the experiment does **not**
validate absolute SPL, detection range, outdoor propagation, field accuracy or
rare statistical tails.

Only one recording session is available. The disjoint calibration interval
`[2,8) s` and evaluation interval `[12,18) s` prevent sample overlap, but do
not create independent source sessions. Results are consequently labelled
`single_session_integration_demonstration`, not held-out source validation.

## Frozen inputs

- manifest: `data/recorded_sources/manifest.json`, schema 1;
- recording: `freesound-683298-sadiquecat-mavic-mini-2`, CC0-1.0;
- decoding: HQ OGG preview, channel mean, `96→48 kHz` polyphase resampling,
  DC removal, exact FFT mask above `10 kHz`, peak normalization to `0.95`;
- physical geometry: existing informative three-tetrahedral-station ENU scene;
- trajectory: constant velocity;
- duration: `3.0 s`, reception start `0.5 s`;
- SNR: `-6 dB` and `10 dB` independent per-station continuous AWGN;
- frame/hop: `1024/512`, tracker stride `32`;
- estimators: all-six equal GCC/WLS and equal-weight SRP-PHAT on identical
  frames;
- calibration: one pooled bias/covariance per
  `(station_id, estimator_variant)`, fitted only from the calibration interval;
- tracker: unchanged stochastic-history retarded-time EKF,
  `Qc=I m²/s³`, unchanged NIS/confirmation/recovery settings;
- seeds: calibration `20260920`, evaluation `20260921`, smoke `20260922`,
  expanded through structured `SeedSequence` coordinates.

## Acceptance and reporting

The smoke chain must produce valid GCC and SRP bearings from the recorded
waveform. Calibration and evaluation noise seeds must be disjoint. Evaluation
must not enter calibration. GCC/SRP frame-key sets must match exactly. Report
all-frame bearing error, causal initialization/update/failure accounting,
conditional tracking error, coverage and runtime. Compare matching cells with
the frozen broadband pilot, but do not interpret one source session as a
population estimate or use evaluation errors to change parameters.

