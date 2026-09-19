# S8 calibration-transfer diagnostic addendum

This addendum authorizes **calibration-only** reprocessing after the saved-data
audit at commit `33952f01d59b8774f7abc69254b998bb7f0d0027`. It does not
reopen the viewed evaluation split or select a new operational calibration.

## Frozen input and processing contract

- Reuse the two manifest-declared calibration recording/session IDs, the
  recorded-source approximation and paired random-broadband control, SNR
  `-6` and `10 dB`, the existing `4.5 s` signal, 48 kHz, 1024/512 frame/hop,
  three tetrahedral station poses, GCC and SRP frontends, structured seeds,
  source support checks and source/noise pairing from
  `S8_TRACKING_FEASIBILITY_PROTOCOL.md`.
- Call only `generate_paired_sequences(..., split="calibration")` for the
  four `(session, SNR)` coordinates. Do not synthesize evaluation audio,
  replay evaluation acoustics, run a tracker here, or rerun old Monte Carlo.
- Persist one offline truth-derived tangent residual per calibration frame,
  including source/session/recording ID, station, method, SNR, frame and seed
  provenance. These residuals must never enter an online estimator as truth.
- Refit the pre-existing pooled station/method/source calibration from these
  rows and verify bias, covariance and frame counts against the committed
  `s8_tracking_feasibility_calibration.csv`. Do not overwrite that table.

## Diagnostics fixed before inspection

- Report source-session × station × method × SNR counts, mean tangent bias,
  sample covariance, median/P95/P99 geodesic error and fractions above
  `5/10/30°`. Frames are dependent; original sessions are the data units.
- For each original source session, pool its two predeclared SNRs with the
  same valid-frame weighting as the existing calibration. Decompose pooled
  sample covariance exactly into within-session and between-session scatter.
  Report the between-session trace fraction.
- Quantify influence of the largest 5% geodesic residuals by comparing raw
  and 95%-trimmed mean/covariance. Also report the share beyond 30° and the
  resulting mean shift when such frames are excluded. Trimming is **only a
  diagnostic**: it does not change the published bias/R or tracker inputs.
- Perform two directed leave-one-original-session-out checks per
  station/method/source: fit mean/R using both SNRs of one session, then
  diagnose each SNR of the other. Use centered spherical tangent residuals
  and the training covariance. No evaluation residual may enter these fits.
  Show cross-session bias mismatch, normalized squared error quantiles and
  Gaussian chi-square(2) exceedance only as diagnostics, not calibration
  acceptance or population inference.
- Compare these calibration-only findings with the already frozen post-hoc
  evaluation replay, without changing `Qc`, NIS/confirmation thresholds,
  physics, initialization budget or the operational bias/R.

Two original calibration sessions cannot support a reliable estimate of
between-session population variability or confidence intervals. This is a
limited transfer diagnostic, not a new algorithm or field validation.
