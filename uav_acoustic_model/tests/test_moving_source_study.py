"""Deterministic and smoke gates for independent frame-wise DOA."""

from types import SimpleNamespace

import numpy as np

from model.geometry import comparison_arrays
from validation.moving_source_study import (
    MOVING_STUDY_METHODS,
    MovingStudyConfig,
    default_moving_study_configurations,
    frame_truth_at_reception,
    gcc_boundary_flags,
    run_deterministic_moving_gate,
    run_moving_configuration,
    run_moving_smoke_gate,
    trajectory_for_configuration,
)


def test_reference_3_boundary_ignores_boundary_hit_on_unused_pair():
    diagnostics = [SimpleNamespace(boundary_hit=False) for _ in range(6)]
    diagnostics[4].boundary_hit = True
    flags = gcc_boundary_flags(diagnostics)
    assert flags["reference_3_gcc_wls"] is False
    assert flags["all_6_equal_gcc_wls"] is True


def test_requested_moving_study_grid_is_complete():
    configs = default_moving_study_configurations()
    assert len(configs) == 2 * 3 * 5 * 3 * 4 * 2 * 3
    assert {config.speed_mps for config in configs} == {0, 5, 10, 20, 30}
    assert {config.distance_m for config in configs} == {10, 25, 50}
    assert {config.frame_length for config in configs} == {256, 512, 1024, 2048}
    assert {config.motion for config in configs} == {"approach", "recede", "transverse"}
    assert {config.geometry for config in configs} == {"square", "tetrahedral"}
    assert {config.signal_model for config in configs} == {
        "random_broadband",
        "deterministic_multisine",
    }
    assert {config.snr_db for config in configs} == {-6, 5, 20}


def test_frame_truth_uses_centroid_emission_time_not_reception_position():
    positions = comparison_arrays()["tetrahedral"]
    config = MovingStudyConfig(
        "tetrahedral", "transverse", 30, 10, 2048, "random_broadband", 10
    )
    trajectory = trajectory_for_configuration(config, positions)
    reception = config.distance_m / 343.0 + 0.02
    truth = frame_truth_at_reception(reception, positions, trajectory)
    assert truth.emission_time_s < reception
    np.testing.assert_allclose(truth.source_position_m, trajectory.q(truth.emission_time_s))
    assert np.linalg.norm(truth.source_position_m - trajectory.q(reception)) > 0.5


def test_deterministic_then_smoke_gates_pass_and_report_all_estimators():
    deterministic = run_deterministic_moving_gate()
    smoke = run_moving_smoke_gate()
    assert {row["estimator_variant"] for row in deterministic} == set(MOVING_STUDY_METHODS)
    assert {row["estimator_variant"] for row in smoke} == set(MOVING_STUDY_METHODS)
    assert max(row["moving_conditional_rmse_deg"] for row in deterministic) < 0.1
    assert all(row["moving_coverage"] == 1.0 for row in smoke)
    assert all(row["truth_definition"].startswith("array_centroid_emission_time") for row in smoke)
    assert all(row["independent_framewise_doa_not_tracking"] is True for row in smoke)


def test_zero_speed_paired_baseline_is_identical_under_common_noise():
    config = MovingStudyConfig(
        "square", "approach", 0, 25, 512, "deterministic_multisine", 5
    )
    rows = run_moving_configuration(config, 91_000, trial_count=4)
    for row in rows:
        assert row["common_random_numbers"] is True
        assert row["moving_coverage"] == row["static_coverage"]
        assert row["moving_boundary_hit_fraction"] == row["static_boundary_hit_fraction"]
        assert abs(row["motion_induced_excess_rmse_deg"]) < 1e-13
        assert abs(row["doa_change_within_frame_deg"]) < 1e-10
        assert abs(row["max_tdoa_change_within_frame_us"]) < 1e-10
        assert abs(row["doppler_factor"] - 1.0) < 1e-15


def test_moving_configuration_schema_counts_and_seeded_metrics_are_reproducible():
    config = MovingStudyConfig(
        "tetrahedral", "recede", 10, 25, 256, "random_broadband", -6
    )
    first = run_moving_configuration(config, 91_001, trial_count=3)
    second = run_moving_configuration(config, 91_001, trial_count=3)
    assert len(first) == len(second) == 3
    nondeterministic = {
        "mean_moving_shared_gcc_frontend_runtime_s",
        "mean_moving_estimator_backend_runtime_s",
        "mean_moving_total_runtime_per_estimate_s",
        "mean_static_shared_gcc_frontend_runtime_s",
        "mean_static_estimator_backend_runtime_s",
        "mean_static_total_runtime_per_estimate_s",
    }
    for left, right in zip(first, second, strict=True):
        assert left["moving_successful_trial_count"] + left["moving_unsuccessful_trial_count"] == 3
        assert left["static_successful_trial_count"] + left["static_unsuccessful_trial_count"] == 3
        assert left["gcc_frontend_pair_count"] == 6
        assert left["estimator_backend_pair_count"] == (
            3 if left["estimator_variant"] == "reference_3_gcc_wls" else 6
        )
        np.testing.assert_allclose(
            left["mean_moving_total_runtime_per_estimate_s"],
            left["mean_moving_shared_gcc_frontend_runtime_s"]
            + left["mean_moving_estimator_backend_runtime_s"],
            rtol=1e-12,
        )
        assert np.isfinite(left["mean_effective_moving_snr_db"])
        assert abs(left["mean_effective_static_snr_db"] - config.snr_db) < 0.5
        assert {k: v for k, v in left.items() if k not in nondeterministic} == {
            k: v for k, v in right.items() if k not in nondeterministic
        }


def test_clean_signal_seed_is_shared_between_snr_levels_but_noise_seed_is_not():
    low = MovingStudyConfig(
        "tetrahedral", "recede", 10, 25, 256, "random_broadband", -6
    )
    high = MovingStudyConfig(
        "tetrahedral", "recede", 10, 25, 256, "random_broadband", 20
    )
    low_rows = run_moving_configuration(low, 91_010, trial_count=1)
    high_rows = run_moving_configuration(high, 91_011, trial_count=1)
    assert {row["signal_seed"] for row in low_rows} == {
        row["signal_seed"] for row in high_rows
    }
    assert {row["noise_seed"] for row in low_rows} != {
        row["noise_seed"] for row in high_rows
    }
