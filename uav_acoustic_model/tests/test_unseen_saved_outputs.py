"""Regression for the preserved, explicitly incomplete technical-stop result."""
from __future__ import annotations

import csv
import json
from collections import defaultdict

from analysis.unseen_manoeuvres_and_sources import DEFAULT_OUTPUT, _directory, _load
from analysis.unseen_partial_audit import verify


def test_technical_stop_never_presents_partial_result_as_full_matrix():
    manifest = _load(DEFAULT_OUTPUT)
    snapshot = json.loads((DEFAULT_OUTPUT / 'partial_audit.json').read_text())
    assert len(manifest['specs']) == snapshot['planned_audio_stream_count'] == 24
    assert snapshot['planned_tracker_run_count'] == 96
    assert snapshot['accuracy_result_is_complete'] is False
    assert snapshot['completed_audio_stream_count'] == 15
    assert snapshot['saved_audio_stream_count'] == 18
    assert snapshot['completed_tracker_run_count'] == 60
    assert snapshot['partial_tracker_artifact_count'] == 5
    assert not (DEFAULT_OUTPUT / 'evaluation_summary.json').exists()
    assert verify(DEFAULT_OUTPUT)['status'] == 'verified_partial'


def test_saved_complete_streams_share_audio_and_pair_noise_without_dropping_failures():
    manifest = _load(DEFAULT_OUTPUT)
    state = json.loads((DEFAULT_OUTPUT / 'partial_audit.json').read_text())['states']
    complete_indexes = {item['index'] for item in state if item['status'] == 'complete'}
    completed = [spec for spec in manifest['specs'] if spec['index'] in complete_indexes]
    assert len(completed) == 15
    noise_by_pair = defaultdict(set)
    for spec in completed:
        directory = _directory(DEFAULT_OUTPUT, spec)
        summary = json.loads((directory / 'summary.json').read_text())
        bearing = json.loads((directory / 'bearing_manifest.json').read_text())
        assert summary['run_id'] == bearing['run_id'] == spec['run_id']
        assert summary['audio_synthesis_count'] == bearing['audio_synthesis_count'] == 1
        assert len(summary['method_variants']) == 4
        assert len({(item['estimator_variant'], item['confirmation_variant'])
                    for item in summary['method_variants']}) == 4
        noise_by_pair[(spec['trajectory'], spec['replicate'])].add(
            bearing['standardized_noise_sha256'])
    assert all(len(digests) == 1 for digests in noise_by_pair.values())
    with (DEFAULT_OUTPUT / 'partial_run_summary.csv').open(newline='', encoding='utf-8') as file:
        rows = list(csv.DictReader(file))
    assert len(rows) == 60
    assert {int(row['index']) for row in rows} == complete_indexes
    for method in {row['estimator_variant'] for row in rows}:
        for variant in {row['confirmation_variant'] for row in rows}:
            subset = [row for row in rows if row['estimator_variant'] == method
                      and row['confirmation_variant'] == variant]
            assert len(subset) == 15
            assert all(int(row['publication_count']) >= int(row['valid_publication_count'])
                       >= int(row['nominal_5m_count']) for row in subset)
            assert all(int(row['total_executed_optimizations']) >=
                       int(row['final_generation_optimizations']) for row in subset)
    with (DEFAULT_OUTPUT / 'partial_paired_shared.csv').open(newline='', encoding='utf-8') as file:
        pairs = list(csv.DictReader(file))
    assert len(pairs) == 30
    assert {int(row['index']) for row in pairs} == complete_indexes
