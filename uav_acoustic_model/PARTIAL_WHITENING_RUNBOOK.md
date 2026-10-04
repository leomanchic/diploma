# Normalized partial whitening: bounded S8 experiment

Run from `uav_acoustic_model` using Python 3.12 and pinned `requirements.txt`.
Production GCC/SRP remain unchanged. The experiment is explicitly invoked
through `analysis.partial_whitening_study`; ambiguity gate is disabled.
Do not rerun historical audio/Monte Carlo/flight studies.

## Inspect saved results without synthesis or replay

```bash
python -m analysis.partial_whitening_study verify
python -m analysis.partial_whitening_study report
python -m jupyter nbconvert --execute --to notebook --inplace notebooks/partial_whitening.ipynb
```

The notebook reads saved results only. Review `PARTIAL_WHITENING_REPORT.md`,
`results/partial_whitening_001/review.json`, all/stride cell tables and raw
`.npz` weights/curves/maps. Errors without valid estimates stay undefined.

## Frozen execution order

These stages document the authorized experiment. Initialization requires a
new output directory and refuses to overwrite existing results. No automatic
retry after technical stop; a further experiment needs a separate protocol.

```bash
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
python -m pytest tests/test_partial_whitening.py -q
# Commit the implementation and initial protocol BEFORE calculations.
python -m analysis.partial_whitening_study initialize
python -m analysis.partial_whitening_study development
python -m analysis.partial_whitening_study freeze
# Commit selection_manifest.json + code and development artifacts separately.
python -m analysis.partial_whitening_study evaluation
python -m analysis.partial_whitening_study tracking
python -m analysis.partial_whitening_study report
python -m analysis.partial_whitening_study verify
```

Windows PowerShell: `$env:OPENBLAS_NUM_THREADS='1'` and
`$env:OMP_NUM_THREADS='1'`. The parent supervisor handles timeouts on both OSes.
One worker lock; ≤16 audio, ≤8 replay computations, 1200/600 s per worker,
21600 s total wall. Interrupted/failed attempts retain partial checkpoints,
complete stdout/stderr and full planned schedule; they cannot silently restart.
Internal `audio-worker`/`replay-worker` are only for the supervisor.

Development compares β=1/0.75/0.5; evaluation only baseline/frozen choice.
Selected β=1 reuses verified baseline tracker results instead of another run.
Source banks are shared across paired distances; these are dependent synthetic
cases on known recorded trajectories. Baseline calibration R is reused,
not recalibrated on evaluation. Prospective calibration and stable harmonic
tracking are separate questions in the report.
