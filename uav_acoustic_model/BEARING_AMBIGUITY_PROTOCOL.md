# Experimental observable bearing gate — frozen protocol v1

Base: 4ef0fb1c7487b16479b429c215d63dc7f6ca28e0. This document and its JSON
manifest are frozen before any new synthesis. Historical harmonic diagnostics
motivate the question only; none of their labels train thresholds. The stopped
15/24 matrix remains stopped. All new results have separate IDs/directories.

## Physical experiment and split

Read the existing spatial recording for development, radial for evaluation.
Use their entire previously declared reception windows, translated to initial
200 and 700 m. Original ENU station/microphone geometry, exact retarded-time
propagation, 1/r, continuous channel/noise arrays, fs=48 kHz, frame/hop/stride,
SNR_ref=+10 dB at 100 m, broadband calibration bias/R, Qc, history,
construction/three-station confirmation, and every optimization bound remain
as in UNSEEN_MANOEUVRES_PROTOCOL_MANIFEST.json. No flight or old run repeated.

Eight streams per split: broadband / nonstationary harmonic × 200/700 m ×
replicate 0/1. Source banks are shared across distances within split/class/
replicate; added noise also shared across distances. These are dependent
synthetic transfers, not independent real UAV sessions. Seeds use
SeedSequence([20261003, split_index, class_index, replicate, mechanism])
and uint64 state (mechanism 0=source, 1=noise). No seed overlap is allowed.

Broadband: original random_bandlimited_signal, 300–10000 Hz, taper=0,
normalize complete bank once. Harmonic: integrate f0(t)=f_base +
f_mod*sin(2π*f_rate*t) + chirp*t; sum k^(-power)*sin(k*phase+random_phase_k),
multiply envelope 0.65+0.25*sin(2π*env_rate*t+0.4)+0.1*sin(2π*0.9*t),
normalize complete bank once. Development: (f_base=480 Hz, f_mod=95 Hz,
f_rate=.09 Hz, chirp=1.8 Hz/s, harmonics=6, power=1.0, env_rate=.31 Hz).
Evaluation: (620, 75, .11, -1.2, 4, 1.3, .43), respectively. Band guard
remains the existing 10000 Hz emitted maximum. Noise stays fixed-reference.

## Exactly two observable features per method

Both take only the current complete audio frame and current raw estimator
outputs. No truth, future, range, velocity or source-class inputs.

1. Spectral participation p=(sum P)^2/(B*sum P^2), where P is the sum of
channel power of mean-removed Hann-windowed frame FFT, restricted to 200–10000
Hz, and B is the number of these bins. p in (0,1]. This measures occupancy,
not probability or SNR; noise may increase it on unreliable frames.
2. Separated peak margin m=(s_peak-s_comp)/D. D=max(score)-min(score)
for the complete admissible curve/map. Undefined if D<=1e-12.
GCC: recompute exactly the existing six GCC diagnostics (interpolation=2,
200–10000 Hz, floor=1e-8, same delay bounds). s_peak is the grid maximum;
competitor is highest score at least 2 samples from that maximum. Take the
minimum six-pair margin. A maximum at an endpoint remains eligible.
SRP: use the same equal-pair cubic interpolation of these GCC curves on the
original upper hemisphere; diagnostic grid is 5° azimuth/elevation, including
zenith. s_peak is score at the raw refined SRP direction; competitor is the
highest grid value >=20° geodesically away. Clip margins to [0,1]. A coarse
map does not prove global uniqueness or detect a nearby competing peak.

Silence RMS<=1e-10, fewer than 3 in-band bins, nonfinite spectra, missing
competitor, invalid pair, or flat map => explicit undefined feature reason.
Enabled gate rejects undefined features and invalid frontend bearings; gate
off returns the exact original BearingMeasurement object. Gate decision
accepts iff p>=p_min AND m>=m_min, thresholds common to distances/classes.

Save actual six GCC curves and SRP grids/selected maxima for each station at
frame 0, frame 32 and the final frame (9 diagnostic frames per stream), chosen
before seeing errors. Save all-frame feature/decision journals and raw bearings.

## Development selection, fixed before labels

Evaluate only Cartesian threshold grid:
p_min=[0,.02,.05,.10,.20,.35]; m_min=[0,.05,.10,.20,.35,.50].
Macro-average equal-weight stream/station cells, original-valid denominator.
Feasible if retention>=70% AND conditional retention of good (<=5°) bearings
>=95% in every source class (across development cells). Minimize macro
fraction of >10° errors passed / all original valid. Ties: maximize total
retention, then lower p_min, then lower m_min. If no configuration feasible,
freeze p_min=m_min=0 with an explicit development failure; undefined scores
still rejected in enabled mode. Report entire grid and per-cell tradeoffs.
No bootstrap or iid-frame significance tests. Do not label score probability.

Freeze a separate gate manifest with development result SHA, feature source
SHA, processing SHA, selected thresholds and selection rule BEFORE evaluation
synthesis/access. Evaluation cannot alter thresholds or tracker settings.

## Reporting and tracking success criterion

Per stream/method/station, all frames and frame_stride=32 separately: original
valid denominator, accepted/rejected, conditional median/P95, accepted error
fractions >5/10/30°, >5/10/30 passed/all original valid, good false rejection
conditional on good and /all valid, physical missing-series durations (from
frame start through end of consecutive rejected frames), availability from
>=2 and 3 stations in causal trailing 3 s confirmation-age windows, plus
instantaneous selected frame groups. Absence at beginning/end included.

Eight preselected replays ONLY: evaluation 200 m replicate0, both classes ×
GCC/SRP × gate off/on, three-station confirmation. Identical raw events and
publication epochs. Persist rejected events/reasons as invalid; no rejected
ID may occur in construction, confirmation, initialization or update roles.
Report first confirmation, valid/confirmed publications, conditional RMSE/P95,
>10/50 m tails, failures, accepted/rejected updates and fits across generations.
Gate success requires >=25% relative reduction of >10° passed/all valid in
evaluation for BOTH methods, <=5 percentage points good false rejection,
>=70% retention, no loss of confirmations, <=5 percentage points loss of
tracking availability, and no increase of >50 m valid publications/all epochs.
Accuracy improvements with starvation are failure. Mixed result reported as
such; never enable gate by default or tune R/filter/budgets to compensate.

## Execution bounds and stop

One external worker at a time, at most 16 syntheses and 8 replays; audio worker
<=1200 s, replay<=600 s, summed external worker wall budget<=21600 s. Parent
timeout kills process group. Ledger records attempt BEFORE start, outcome,
stdout/stderr, checkpoints and SHA completion. No automatic retries after an
incomplete attempt, no limit increases. Completed files verified before skip.
On timeout, resource/process error or integrity failure STOP all remaining
heavy work, retain partial journals/checkpoints and full intended denominators.
First initialize/freeze protocol and commit, then 8 development workers,
freeze gate and commit, then 8 evaluation workers, then 8 preselected replays.
Report exact attempts/completions. Notebook uses saved artifacts only.

## Verification

Truth-free gate contract, no future access, opt-out identity/baseline numerical
parity, rejected exclusion all tracker stages, reason/count consistency,
split seeds/harmonic parameters, silence/insufficient spectrum/undefined score.
Targeted pytest, full pytest, new executed notebook, pip check, diff check,
Linux/Windows CI. Historical inputs, IDs and integrity checks stay unchanged.
