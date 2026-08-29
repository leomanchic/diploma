"""Causality, truth timestamps, invalid semantics, and sequence reporting."""

from collections import defaultdict

import numpy as np

from model.geometry import array_centroid, comparison_arrays
from validation.moving_source_study import MOVING_STUDY_METHODS
from validation.sequential_doa_study import (
    SequentialStudyConfig,
    default_sequential_configurations,
    run_sequential_configuration,
    trajectory_for_sequence,
)


def _without_runtime(row):
    excluded = {
        "shared_gcc_frontend_runtime_s",
        "estimator_backend_runtime_s",
        "algorithm_runtime_s",
        "estimate_available_reception_time_s",
        "total_emission_to_available_latency_s",
    }
    return {key: value for key, value in row.items() if key not in excluded}


def test_sequential_timestamps_are_causal_monotone_and_use_centroid_emission_truth():
    config = default_sequential_configurations()[1]
    rows, summaries, _ = run_sequential_configuration(config, 1)
    reference = [r for r in rows if r["estimator_variant"] == "reference_3_gcc_wls"]
    reception = np.asarray([r["frame_center_reception_time_s"] for r in reference])
    emission = np.asarray([r["centroid_emission_time_s"] for r in reference])
    assert np.all(np.diff(reception) > 0.0)
    assert np.all(np.diff(emission) > 0.0)
    assert np.all(emission < reception)
    positions = comparison_arrays()[config.geometry]
    trajectory = trajectory_for_sequence(config, positions)
    for row in reference[::10]:
        displacement = trajectory.q(row["centroid_emission_time_s"]) - array_centroid(positions)
        expected = displacement / np.linalg.norm(displacement)
        np.testing.assert_allclose(
            [row["truth_x"], row["truth_y"], row["truth_z"]], expected, atol=2e-15
        )
    assert all(r["truth_definition"].startswith("array_centroid_emission_time") for r in rows)
    assert all(s["sequential_independent_bearings_not_tracking"] is True for s in summaries)


def test_azimuth_wrap_is_reported_without_a_360_degree_jump_error():
    config = default_sequential_configurations()[5]
    rows, summaries, _ = run_sequential_configuration(config, 5)
    reference = [r for r in rows if r["estimator_variant"] == "reference_3_gcc_wls"]
    azimuth_deg = np.asarray([r["truth_azimuth_deg"] for r in reference])
    assert np.any(azimuth_deg > 350.0)
    assert np.any(azimuth_deg < 10.0)
    unwrapped = np.unwrap(np.deg2rad(azimuth_deg))
    assert np.max(np.abs(np.diff(unwrapped))) < np.deg2rad(1.0)
    assert all(s["azimuth_wrap_359_to_0_present"] is True for s in summaries)
    assert max(float(s["conditional_rmse_deg"]) for s in summaries) < 0.2


def test_dropout_invalid_frames_never_become_arbitrary_bearings():
    config = default_sequential_configurations()[-1]
    rows, summaries, _ = run_sequential_configuration(config, 6)
    invalid = [row for row in rows if not row["valid"]]
    assert invalid
    assert {row["frame_index"] for row in invalid} == {18, 19, 20, 21, 22}
    for row in invalid:
        assert row["estimate_x"] is None
        assert row["estimate_y"] is None
        assert row["estimate_z"] is None
        assert row["estimate_azimuth_deg"] is None
        assert row["estimate_elevation_deg"] is None
        assert row["geodesic_angular_error_deg"] is None
    assert all(s["invalid_frame_count"] == 5 for s in summaries)


def test_all_methods_receive_identical_frames_and_no_future_samples_or_doa():
    config = SequentialStudyConfig("short_stationary", "stationary", 20.0, duration_s=0.04)
    rows, _, _ = run_sequential_configuration(config, 40)
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["frame_index"]].append(row)
        assert row["maximum_reception_sample_used"] == row["frame_end_sample_inclusive"]
        assert row["future_samples_used"] is False
        assert row["future_doa_estimates_used"] is False
        assert row["truth_used_by_estimator"] is False
    for group in grouped.values():
        assert {row["estimator_variant"] for row in group} == set(MOVING_STUDY_METHODS)
        assert len({row["frame_content_sha256"] for row in group}) == 1


def test_estimator_interface_is_not_given_true_direction():
    calls = []

    def spy(frame, positions, sampling_rate_hz):
        calls.append((frame.shape, positions.shape, sampling_rate_hz))
        return {
            method: {
                "direction": np.full(3, np.nan),
                "valid": False,
                "boundary": False,
                "shared_gcc_frontend_runtime_s": 0.0,
                "estimator_backend_runtime_s": 0.0,
                "total_runtime_s": 0.0,
                "gcc_frontend_pair_count": 6,
                "estimator_backend_pair_count": 3 if method == "reference_3_gcc_wls" else 6,
            }
            for method in MOVING_STUDY_METHODS
        }

    config = SequentialStudyConfig("truth_isolation", "transverse", 20.0, duration_s=0.04)
    rows, _, _ = run_sequential_configuration(config, 41, frame_estimator=spy)
    assert len(calls) == len(rows) // 3
    assert all(shape == (4, 1024) for shape, _, _ in calls)
    assert all(row["truth_used_by_estimator"] is False for row in rows)
    assert all(row["valid"] is False for row in rows)


def test_same_seed_reproduces_sequential_bearings_excluding_measured_runtime():
    config = SequentialStudyConfig("reproducible", "receding", 10.0, duration_s=0.04)
    first, _, first_stream = run_sequential_configuration(config, 42)
    second, _, second_stream = run_sequential_configuration(config, 42)
    np.testing.assert_array_equal(first_stream.channels, second_stream.channels)
    assert [_without_runtime(row) for row in first] == [_without_runtime(row) for row in second]


def test_stationary_sequence_has_constant_truth_and_expected_frame_count():
    config = default_sequential_configurations()[0]
    rows, summaries, _ = run_sequential_configuration(config, 0)
    assert len(rows) == 43 * 3
    assert all(summary["frame_count"] == 43 for summary in summaries)
    assert all(summary["frame_count_is_independent_trial_count"] is False for summary in summaries)
    assert all(summary["maximum_truth_doa_change_from_first_deg"] == 0.0 for summary in summaries)
    assert all(summary["coverage"] == 1.0 for summary in summaries)
    assert max(float(summary["conditional_rmse_deg"]) for summary in summaries) < 0.2
