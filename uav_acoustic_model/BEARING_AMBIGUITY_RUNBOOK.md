# Observable bearing gate: reproduce saved results

From the repository's `uav_acoustic_model` directory with the pinned dev
requirements installed. This experimental adapter lives in `analysis/`; run
from the checkout. It is not installed or enabled by the core package.

## Inspect published artifacts (no synthesis/replay)

```bash
python -m analysis.bearing_ambiguity_study verify
python -m analysis.bearing_ambiguity_study report
python -m jupyter nbconvert --execute --to notebook --inplace notebooks/bearing_ambiguity_gate.ipynb
```

`report` recalculates derived tables from saved bearings/features and frozen
thresholds. `verify` validates protocol, code, recording and completion hashes.
For a completed study, all-frame and stride tables would be separate.
The accepted technical failure has no raw bearings, thresholds or tracking results.
Its default `verify/report` commands audit the preserved failure only.
Use `--output results/bearing_ambiguity_retry_001` to inspect the authorized retry. `gate_decisions.csv.gz` contains explicit reasons. Curves
are `.npz` files (numpy `load(..., allow_pickle=False)`) with actual six GCC
curves and SRP diagnostic grid, selected maximum and separated competitor.

## Authorized retry — 2026-10-03

The original attempt remains immutable. The user subsequently authorized a
separate retry (BEARING_AMBIGUITY_RETRY_AUTHORIZATION.json), using the same
scientific protocol, thresholds grid, seeds and limits. Its preflight worker
time is carried into the total 6-hour budget. Use a NEW output directory. Do not overwrite published recordings/results.
Freeze/commit implementation and protocol first. Execute these stages in order:

```bash
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
OUT="$PWD/results/bearing_ambiguity_retry_001"
python -m analysis.bearing_ambiguity_study initialize --output "$OUT" \
  --authorization BEARING_AMBIGUITY_RETRY_AUTHORIZATION.json
python -m analysis.bearing_ambiguity_study development --output "$OUT"
python -m analysis.bearing_ambiguity_study freeze --output "$OUT"
# Review and commit the gate manifest before evaluation.
python -m analysis.bearing_ambiguity_study evaluation --output "$OUT"
python -m analysis.bearing_ambiguity_study tracking --output "$OUT"
python -m analysis.bearing_ambiguity_study report --output "$OUT"
python -m analysis.bearing_ambiguity_study verify --output "$OUT"
```

Windows PowerShell: set `$env:OPENBLAS_NUM_THREADS='1'`,
`$env:OMP_NUM_THREADS='1'` and `$OUT=Join-Path $PWD 'results/bearing_ambiguity_local_001'`.
The external worker supervisor handles process termination on both platforms.
The initializer refuses an existing directory. Published completed stages can
be verified and skipped; incomplete/stopped attempts cannot be retried. A new
experiment needs its own authorization and a different directory.
Do not directly call internal `audio-worker` / `replay-worker`: they are only
for the supervisor, which freezes attempts before starting and enforces limits.

External bounds: 1200 s/audio, 600 s/replay, 21600 s summed worker wall.
An exclusive worker lock prevents simultaneous supervisors. A stopped ledger
blocks retries. All intended denominators are in the initial manifest; partial
checkpoints/journals and stdout/stderr are retained. Limits never auto-increase.
No command here resumes the old stopped study or performs a new flight.

## Online adapter

```python
from analysis.bearing_ambiguity_gate import GateConfig, frame_features, apply_gate
# raw_srp_direction is the estimate from the same complete frame.
features, diagnostics = frame_features(frame, microphones_local_m, fs_hz,
                                       raw_srp_direction)
# Explicit opt-in; thresholds must come from the frozen gate manifest.
config = GateConfig(enabled=True, minimum_participation=p_min,
                    minimum_margin=m_min)
eligible_measurement = apply_gate(raw_measurement,
                                 features[raw_measurement.estimator_variant], config)
```

`GateConfig()` disables the gate and returns the exact original object.
Rejected measurements use the existing invalid-bearing contract; their
identity/time/reason remain in the journal. No truth, future frames, range,
source label or velocity is an argument. No covariance/filter adjustment.
