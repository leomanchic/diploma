"""Read-only source audit and descriptive tables after the frozen cost stop.

This module never resumes synthesis or tracking. Incomplete streams remain
visible and never enter the completed-stream metric denominators.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np

from analysis.localization_error_attribution import _read_csv_gz, _write_csv, _write_json
from analysis.robust_track_confirmation import METHODS, VARIANTS
from analysis.unseen_manoeuvres_and_sources import (
    DEFAULT_OUTPUT, _directory, _load, paired_shared, summarize_track,
)
from validation.gazebo_experiment import sha256


def audit(output: Path = DEFAULT_OUTPUT) -> dict:
    output = Path(output)
    manifest = _load(output)
    stop_path = output / 'technical_stop.json'
    stop = json.loads(stop_path.read_text(encoding='utf-8'))
    if stop['status'] != 'stopped_by_frozen_technical_limit':
        raise ValueError('not a frozen technical stop')
    if stop['limit_wall_s_per_stream'] != manifest['technical_limits']['max_wall_s_per_stream']:
        raise ValueError('technical stop limit differs from the frozen manifest')
    rows: list[dict] = []
    pairs: list[dict] = []
    stream_states: list[dict] = []
    artifact_hashes: dict[str, str] = {}
    wall_s = 0.0
    peak_rss = 0
    for spec in manifest['specs']:
        directory = _directory(output, spec)
        experiment_path = directory / 'experiment.json'
        if not experiment_path.exists():
            stream_states.append({'index': spec['index'], 'run_id': spec['run_id'],
                                  'status': 'unstarted', 'bearing_saved': False,
                                  'saved_tracker_count': 0})
            continue
        experiment = json.loads(experiment_path.read_text(encoding='utf-8'))
        artifact_hashes[f'{directory.relative_to(output).as_posix()}/experiment.json'] = sha256(experiment_path)
        if experiment['run_id'] != spec['run_id']:
            raise ValueError(f'run identity mismatch at index {spec["index"]}')
        status = experiment['status']
        if status not in ('complete', 'running'):
            raise ValueError(f'unexpected stream status: {status}')
        bearing_path = directory / 'bearing_manifest.json'
        bearing_saved = bearing_path.exists()
        if bearing_saved:
            bearing = json.loads(bearing_path.read_text(encoding='utf-8'))
            if bearing['run_id'] != spec['run_id'] or (
                bearing['bearing_records_sha256'] != sha256(directory / 'bearing_records.csv.gz')
            ):
                raise ValueError(f'bearing artifact mismatch at index {spec["index"]}')
            for name in ('bearing_manifest.json', 'bearing_records.csv.gz'):
                artifact_hashes[f'{directory.relative_to(output).as_posix()}/{name}'] = sha256(directory / name)
        tracks_saved = sorted(directory.glob('tracking_*.csv.gz'))
        stream_states.append({'index': spec['index'], 'run_id': spec['run_id'],
                              'status': status, 'bearing_saved': bearing_saved,
                              'saved_tracker_count': len(tracks_saved)})
        if status != 'complete':
            if (directory / 'summary.json').exists():
                raise ValueError(f'incomplete stream has final summary: {spec["index"]}')
            for path in sorted(directory.glob('*.csv.gz')):
                artifact_hashes[f'{directory.relative_to(output).as_posix()}/{path.name}'] = sha256(path)
            continue
        if len(tracks_saved) != 4 or not bearing_saved:
            raise ValueError(f'completed stream is missing a tracker or bearings: {spec["index"]}')
        for name, digest in experiment['result_sha256'].items():
            if sha256(directory / name) != digest:
                raise ValueError(f'completed stream artifact changed: {spec["index"]}/{name}')
            artifact_hashes[f'{directory.relative_to(output).as_posix()}/{name}'] = digest
        summary = json.loads((directory / 'summary.json').read_text(encoding='utf-8'))
        if summary['run_id'] != spec['run_id'] or summary['bearing_manifest_sha256'] != sha256(bearing_path):
            raise ValueError(f'completed stream summary mismatch: {spec["index"]}')
        wall_s += float(summary['stream_wall_s'])
        peak_rss = max(peak_rss, int(summary['stream_sampled_peak_rss_bytes']))
        for result in summary['method_variants']:
            method, variant = result['estimator_variant'], result['confirmation_variant']
            suffix = f'{method}_{variant}.csv.gz'
            derived = summarize_track(
                _read_csv_gz(directory / f'tracking_{suffix}'),
                _read_csv_gz(directory / f'updates_{suffix}'),
                _read_csv_gz(directory / f'batch_fits_{suffix}'),
                _read_csv_gz(directory / f'lifecycle_{suffix}'),
                reception_start_s=0.0,
            )
            for key, value in derived.items():
                saved = result[key]
                if isinstance(value, float):
                    matches = np.isclose(value, saved, rtol=1e-10, atol=1e-9)
                else:
                    matches = value == saved
                if not matches:
                    raise ValueError(f'stale metric: {spec["index"]}/{method}/{variant}/{key}')
            rows.append(result)
        for method in METHODS:
            left = _read_csv_gz(directory / f'tracking_{method}_baseline.csv.gz')
            right = _read_csv_gz(directory / f'tracking_{method}_three_station_confirmation.csv.gz')
            pairs.append({'run_id': spec['run_id'], 'index': spec['index'],
                          'trajectory': spec['trajectory'], 'source_class': spec['source_class'],
                          'distance_m': spec['distance_m'], 'replicate': spec['replicate'],
                          'estimator_variant': method, **paired_shared(left, right)})
    counts = Counter(item['status'] for item in stream_states)
    if counts['complete'] == 24:
        raise ValueError('matrix complete; use full aggregate and verify instead')
    if stop['trigger_index'] not in [item['index'] for item in stream_states if item['status'] == 'running']:
        raise ValueError('technical trigger is not an interrupted stream')
    if len(rows) != 4 * counts['complete'] or len(pairs) != 2 * counts['complete']:
        raise ValueError('completed-stream metric count mismatch')
    _write_csv(output / 'partial_run_summary.csv', rows, list(rows[0]))
    _write_csv(output / 'partial_paired_shared.csv', pairs, list(pairs[0]))
    audit_result = {
        'schema_version': 1, 'status': 'technical_stop_partial_audit',
        'planned_audio_stream_count': 24, 'planned_tracker_run_count': 96,
        'completed_audio_stream_count': counts['complete'],
        'saved_audio_stream_count': sum(item['bearing_saved'] for item in stream_states),
        'completed_tracker_run_count': len(rows),
        'partial_tracker_artifact_count': sum(item['saved_tracker_count'] for item in stream_states
                                              if item['status'] == 'running'),
        'completed_stream_wall_s_sum': wall_s, 'completed_stream_wall_s_max': max(
            float(json.loads((_directory(output, spec) / 'summary.json').read_text())['stream_wall_s'])
            for spec in manifest['specs'] if (_directory(output, spec) / 'summary.json').exists()),
        'completed_stream_peak_rss_bytes_max': peak_rss,
        'states': stream_states,
        'technical_stop_sha256': sha256(stop_path),
        'evaluation_manifest_sha256': sha256(output / 'evaluation_manifest.json'),
        'artifact_sha256': artifact_hashes,
        'tables_sha256': {name: sha256(output / name) for name in
                          ('partial_run_summary.csv', 'partial_paired_shared.csv')},
        'accuracy_result_is_complete': False,
    }
    _write_json(output / 'partial_audit.json', audit_result)
    return {key: value for key, value in audit_result.items() if key not in ('states', 'artifact_sha256')}


def verify(output: Path = DEFAULT_OUTPUT) -> dict:
    output = Path(output)
    previous = json.loads((output / 'partial_audit.json').read_text(encoding='utf-8'))
    for name, digest in previous['tables_sha256'].items():
        if sha256(output / name) != digest:
            raise ValueError(f'partial table changed: {name}')
    if sha256(output / 'evaluation_manifest.json') != previous['evaluation_manifest_sha256']:
        raise ValueError('evaluation manifest changed')
    if sha256(output / 'technical_stop.json') != previous['technical_stop_sha256']:
        raise ValueError('technical stop marker changed')
    for name, digest in previous['artifact_sha256'].items():
        if sha256(output / name) != digest:
            raise ValueError(f'saved artifact changed: {name}')
    return {'status': 'verified_partial', 'completed_audio_stream_count':
            previous['completed_audio_stream_count'], 'completed_tracker_run_count':
            previous['completed_tracker_run_count']}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('audit', 'verify'))
    parser.add_argument('--output', type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    print(json.dumps(audit(args.output) if args.action == 'audit' else verify(args.output),
                     sort_keys=True))


if __name__ == '__main__':
    main()
