"""Smoke and split-integrity tests for the S7C-B validation study."""

import json

from dataclasses import fields

import numpy as np

from model.measurements import BearingMeasurement
from model.bearing_events import measurements_are_exact_duplicates
from validation.retarded_batch_study import (
    RetardedBatchStudyConfig,
    default_retarded_batch_configurations,
    generate_retarded_batch_scenario,
    run_retarded_batch_configuration,
    run_retarded_batch_study,
)


def test_fixed_configuration_matrix_has_multiple_physical_axes():
    configs = default_retarded_batch_configurations(sequence_count=2)
    assert len(configs) == 24
    assert {item.geometry for item in configs} == {"wide", "compact"}
    assert {item.motion for item in configs} == {
        "stationary",
        "oblique_slow",
        "oblique_fast",
    }
    assert {item.angular_noise_std_deg for item in configs} == {0.05, 0.2}
    assert {item.delivery_schedule for item in configs} == {
        "ordered",
        "reordered_dropout",
    }


def test_all_sequence_noise_and_delivery_seeds_are_disjoint_and_reproducible():
    configs = default_retarded_batch_configurations(sequence_count=2)
    scenarios = [
        generate_retarded_batch_scenario(config, index)
        for config in configs
        for index in range(config.sequence_count)
    ]
    sequence_seeds = [item.sequence_seed for item in scenarios]
    noise_seeds = [item.bearing_noise_seed for item in scenarios]
    delivery_seeds = [item.delivery_seed for item in scenarios]
    assert len(sequence_seeds) == len(set(sequence_seeds))
    assert len(noise_seeds) == len(set(noise_seeds))
    assert len(delivery_seeds) == len(set(delivery_seeds))
    assert set(noise_seeds).isdisjoint(delivery_seeds)
    repeated = generate_retarded_batch_scenario(configs[7], 1)
    original = scenarios[7 * 2 + 1]
    assert repeated.sequence_seed == original.sequence_seed
    np.testing.assert_array_equal(
        repeated.truth_state.vector, original.truth_state.vector
    )
    assert len(repeated.events) == len(original.events)
    for repeated_event, original_event in zip(
        repeated.events, original.events, strict=True
    ):
        assert repeated_event.dropped == original_event.dropped
        assert repeated_event.drop_reason == original_event.drop_reason
        assert measurements_are_exact_duplicates(
            repeated_event.measurement, original_event.measurement
        )


def test_scenario_truth_is_separate_from_truth_free_measurement_contract():
    names = {item.name for item in fields(BearingMeasurement)}
    assert names.isdisjoint(
        {
            "truth_state",
            "true_position_world_m",
            "true_velocity_world_mps",
            "true_emission_time_s",
            "angular_error",
        }
    )
    scenario = generate_retarded_batch_scenario(
        RetardedBatchStudyConfig(
            "wide", "oblique_slow", 0.05, "ordered", sequence_count=1
        ),
        0,
    )
    assert all(
        isinstance(item.measurement, BearingMeasurement) for item in scenario.events
    )


def test_smoke_configuration_reports_dependent_prefixes_and_matching_final_batch():
    config = RetardedBatchStudyConfig(
        "wide", "oblique_slow", 0.05, "reordered_dropout", sequence_count=2
    )
    rows = run_retarded_batch_configuration(config)
    assert len(rows) == 12
    assert {row["independent_unit"] for row in rows} == {"whole_sequence"}
    assert all(row["prefixes_within_sequence_are_dependent"] for row in rows)
    for sequence_index in range(2):
        sequence_rows = [
            row for row in rows if row["sequence_index"] == sequence_index
        ]
        offline = next(row for row in sequence_rows if row["mode"] == "offline_full_record")
        final = next(
            row
            for row in sequence_rows
            if row["mode"] == "causal_prefix" and row["prefix_index"] == 4
        )
        assert offline["used_measurement_count"] == final["used_measurement_count"]
        assert offline["valid"] == final["valid"]
        assert offline["position_error_m"] == final["position_error_m"]
        assert offline["velocity_error_mps"] == final["velocity_error_mps"]
        assert final["dropped_count"] == 2
        assert final["duplicate_count"] == 1
        journal = json.loads(final["event_journal_json"])
        assert len(journal) == final["available_event_log_count"]
        assert {item["action"] for item in journal} >= {
            "accepted",
            "duplicate_exact",
            "excluded_dropped",
        }
        assert len(json.loads(final["accepted_event_ids_json"])) == final[
            "used_measurement_count"
        ]


def test_smoke_study_writes_sequence_and_aggregate_csv(tmp_path):
    rows, summaries = run_retarded_batch_study(tmp_path, sequence_count=1)
    assert len(rows) == 24 * 6
    assert len(summaries) == 24 * 6
    assert (tmp_path / "retarded_batch_sequence_results.csv").is_file()
    assert (tmp_path / "retarded_batch_summary.csv").is_file()
    assert {row["independent_sequence_count"] for row in summaries} == {1}
    assert {row["dependent_prefix_count_per_sequence"] for row in summaries} == {5}
