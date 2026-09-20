# S7C-C1 review fixes

## Scope

Corrective review of the strict constant-velocity retarded-time EKF baseline.
The scope remains direct bearing-level observations, known constant sound
speed, `Q=0`, one EKF linearization per accepted event, and no smoothing or
future access. This is not a manoeuvre model or signal-level tracking result.

## Review matrix

| Finding | Change | Evidence |
|---|---|---|
| Unknown station raised `KeyError` after partial time advancement | Validate station IDs before batch/update; retain per-event `unknown_station_id`; continue with eligible events | Targeted before/after-initialization and control-stream regressions |
| Singular tangent `R` poisoned every later initialization attempt | Preserve raw journal, but filter a separate positive-definite C1 eligible set; retain `unsupported_singular_covariance` per event | Mixed/only-singular, duplicate and recovery regressions |
| `advance_to` result depended on external call frequency | Internally replay each complete availability group, keep event posterior separate from event-free publication prediction | Frequent/sparse/one-shot, no-event, equal-time and conflict/recovery regressions compare state, `P` and IDs |
| P95 conventions were mixed | Use explicit NumPy `method="linear"`; label aggregation grain in CSV | Fixed-sample regression and recomputation from sequence rows |

An individual event rejection is not the same as lack of initialization. A
valid state can coexist with rejected observations. A conflict involving a
previously used identity invalidates the state until a later availability
group supports reinitialization; this is distinct from both cases above.

## Verification record

The implementation was frozen before the fresh study and final gates.

- Targeted review tests: `37 passed`.
- Extended retarded selection: `74 passed, 304 deselected`.
- Full suite: `378 passed in 98.76s`.
- Fresh S7C-C1 study: 4,416 publication rows, 384 sequence-method rows and
  96 aggregate rows from 96 independent whole sequences (24 configurations,
  four sequences/configuration, 15 events/sequence).
- Fresh S7C-B reference study: 576 sequence rows and 144 aggregate rows.
- `pip check`: `No broken requirements found`.
- `git diff --check`: pass.
- All 15 committed notebooks executed to completion. Saved-object audit:
  96 code cells, zero nbformat failures, error outputs, unexecuted non-empty
  cells and missing cell IDs. The two retarded notebooks were executed and
  saved in place; unrelated notebooks were executed in memory so existing
  user changes were not overwritten.

The commands used from the project root were:

```powershell
& ".\.venv\Scripts\python.exe" -m pytest -q
& ".\.venv\Scripts\python.exe" -m validation.retarded_ekf_study
& ".\.venv\Scripts\python.exe" -m validation.retarded_batch_study
& ".\.venv\Scripts\python.exe" -m jupyter nbconvert --to notebook `
  --execute --inplace --ExecutePreprocessor.timeout=1200 `
  ".\notebooks\retarded_batch_validation.ipynb"
& ".\.venv\Scripts\python.exe" -m jupyter nbconvert --to notebook `
  --execute --inplace --ExecutePreprocessor.timeout=1200 `
  ".\notebooks\retarded_ekf_validation.ipynb"
& ".\.venv\Scripts\python.exe" -m pip check
git diff --check
```

The remaining committed notebooks were executed with fresh kernels through
`nbclient.NotebookClient(timeout=3600)` and validated with `nbformat` without
rewriting their files.

### Fresh numerical result

All `96/96` EKF sequences initialized and ended valid; final-state 95%
coverage is `93/96 = 0.96875`. The final independent-sequence metrics are:

| Method | position RMSE / P95 (m) | velocity RMSE / P95 (m/s) |
|---|---:|---:|
| retarded EKF | `0.424889226 / 0.856009559` | `0.186864851 / 0.359536159` |
| causal-prefix batch | `0.424831873 / 0.853387347` | `0.186713039 / 0.360116149` |
| initial batch, no updates | `2.741750935 / 6.008923959` | `0.803557065 / 1.827238321` |

P95 is NumPy `method="linear"`, computed from one final error per independent
whole sequence. The separately labelled temporal P95 uses dependent
publication samples and is not an independent-trial confidence statistic.
The prior larger P95 values used the discrete `higher` convention; RMSE,
per-sequence errors, NIS/NEES, coverage and seeds did not change.

Against reviewed commit `7004372`, all existing non-runtime S7C-C1 sequence
values are exactly unchanged (`max abs difference = 0`) for final position
and velocity error, NEES, mean NIS and NIS coverage. Fresh S7C-B non-runtime
fields are also exactly unchanged; only measured runtime fields vary.

## Deliberate limitations

- strict constant velocity and `Q=0` only;
- positive-definite tangent `R` only in EKF updates;
- no UKF, smoothing, manoeuvres, acoustic frontend, wind or reflections;
- dependent temporal NIS/NEES samples are not independent trials;
- offline full-record batch remains a noncausal reference.
