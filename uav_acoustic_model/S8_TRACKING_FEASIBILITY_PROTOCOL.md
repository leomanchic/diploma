# S8 tracking-feasibility correction — frozen before renewed evaluation

This is a correction of the S8 independent-recordings pilot, not a new tracker
or a retuning against held-out position errors. The prior 2.0 s outputs in
`results/independent_recordings_*.csv` and their notebook remain historical.
They establish a bearing-level comparison but **cannot** attribute the absence
of confirmed tracks to recorded sound: even exact retarded bearings on that
schedule cannot satisfy the confirmation reception-span requirement.

## Positive scheduling and source-support gates

- Geometry, physical CV trajectory, station delivery delays, 48 kHz sampling,
  1024/512 frame/hop and tracker stride 64 are unchanged.
- Exact retarded bearings with the old 2.0 s schedule have 3 reception-frame
  groups / 9 events: 6 construct the hypothesis, while the last 3 are one
  common reception time. Confirmed = false, accepted updates = 0.
- With duration **4.5 s**, the exact-bearing control has 7 frame groups / 21
  events. It confirms at processing time **2.5793125 s** and produces **9
  accepted updates across 3 later reception-frame groups** (2 attempted
  updates are rejected as `emission_outside_history`). This control runs
  before processing new audio; no NIS or confirmation threshold was relaxed.
- All four declared 6 s clips have 288000 resampled samples. The 4.5 s
  reception interval needs at most 222169 source samples, including the
  existing 256-sample-or-larger FIR guards. At least 65831 samples remain.
  The original interval is neither looped nor extended by repetition;
  synthesis still checks every channel's valid interpolation region.

## Frozen paired evaluation

- Source/session split, Freesound assets, calibration-only pooling by
  `(station_id, estimator_variant, source_model)`, two SNR levels (-6, 10 dB),
  methods (all-6 GCC/WLS, equal-weight SRP-PHAT), trajectory, station poses,
  `Qc=I m²/s³`, NIS and confirmation thresholds remain as in
  [S8_INDEPENDENT_RECORDINGS_PROTOCOL.md](S8_INDEPENDENT_RECORDINGS_PROTOCOL.md).
- Duration changes to **4.5 s** for both calibration and evaluation. All-frame
  bearing metrics remain; the tracker receives every 64th frame. Pair members
  retain identical trajectory, duration, frame/reception/availability times,
  structured seed and standard-normal noise draws.
- Before the renewed evaluation, a replay of the historical
  `383904`/recorded/-6 dB/GCC bearing stream showed 12 nonlinear batch fits
  consuming **71.43 s** of **71.44 s** tracker runtime. The historical unbounded
  runtime was **141.31 s**. The cost is repeated initial-hypothesis fitting,
  not waveform synthesis or the acoustic frontend.
- The opt-in budget is **4 nonlinear batch optimizations per initialization
  generation**, counting candidate fits and confirmation/refinement fits.
  Geometric rank prechecks do not consume the budget. At exhaustion the
  generation becomes invalid with `computational_budget_exceeded` and reports
  fit count and fit wall runtime. Baseline C1/D2/recovery and other audio
  studies keep their prior unlimited default. This work cap is chosen from the
  historical cost replay and the exact control's 2-fit requirement, not from
  renewed evaluation accuracy. It does not promise a wall-clock bound for one
  optimizer launch; the measured runtime is reported.
  A post-evaluation reproducibility replay of that same historical stream
  with the already frozen four-fit cap took 14.388 s, including 14.386 s
  inside nonlinear fits. This replay did not change the cap.
- Counts and seeds remain two original sessions per split × two SNR × two
  source models: 8 continuous streams per split and 16 method-session
  evaluation runs. Calibration/evaluation base seeds remain 20260923/20260924.
  Calibration is fit again because the continuous waveform duration changes.
- New outputs use `results/s8_tracking_feasibility_*.csv`; historical outputs
  are not overwritten. Conditional position/velocity errors and conditional
  covariance coverage exist only when estimates are valid. Confirmed/valid
  publication fractions and unconditional valid-and-covered fractions remain
  separate. Missing estimates stay missing. Failure reasons distinguish
  statistical/geometric failures from budget exhaustion. Original sessions,
  not their frames or SNR repeats, are the source-data units.
- Time availability/coverage are recomputed solely from saved causal
  publications into `s8_tracking_feasibility_time_coverage.csv`. Each
  publication state is held until the next one; the denominator is the span
  from first to last available publication, not the full audio duration.
  Report available-time fraction, conditional covered time among available
  time, and unconditional available-and-covered time separately. The final
  publication has no inferred post-publication duration.

This remains a small integration pilot, not field validation, absolute SPL or
detection-range evidence. The budget can sacrifice availability, and the
temporal/interstation dependence of errors is not represented by the tracker.
