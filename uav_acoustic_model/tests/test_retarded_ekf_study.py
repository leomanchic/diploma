"""Smoke, seed and sequence-grain checks for the S7C-C1 study."""

from dataclasses import fields, replace

import numpy as np

from estimators.retarded_ekf import CausalRetardedTimeEKF
from model.bearing_events import measurements_are_exact_duplicates
from model.retarded_bearing import retarded_bearing_residual
from validation.retarded_ekf_study import (
    RetardedEKFStudyConfig,
    default_retarded_ekf_configurations,
    generate_retarded_ekf_scenario,
    retarded_ekf_seed_provenance,
    run_retarded_ekf_configuration,
    summarize_retarded_ekf_sequences,
)


def test_protocol_matrix_is_fixed_before_evaluation():
    configs = default_retarded_ekf_configurations(sequence_count=2)
    assert len(configs) == 24
    assert {item.geometry for item in configs} == {
        "informative",
        "poorly_conditioned",
    }
    assert {item.motion for item in configs} == {"stationary", "uniform_oblique"}
    assert {
        (item.azimuth_noise_std_deg, item.elevation_noise_std_deg)
        for item in configs
    } == {(0.03, 0.05), (0.10, 0.18), (0.30, 0.50)}
    assert {item.delivery_schedule for item in configs} == {"ordered", "reordered"}


def test_seed_streams_are_reproducible_disjoint_and_collision_free_beyond_1000():
    identifiers = []
    generated = []
    for base_seed in (20260908, 20260909):
        for configuration_index in (0, 1):
            for sequence_index in range(1001):
                ids, seeds = retarded_ekf_seed_provenance(
                    base_seed, configuration_index, sequence_index
                )
                identifiers.extend(ids)
                generated.extend(seeds)
    assert len(identifiers) == len(set(identifiers))
    assert len(generated) == len(set(generated))


def test_same_seed_reproduces_sequence_and_changed_seed_preserves_physics():
    config = RetardedEKFStudyConfig(
        "informative", "uniform_oblique", 0.1, 0.18, "reordered", 1
    )
    first = generate_retarded_ekf_scenario(config, 0)
    repeated = generate_retarded_ekf_scenario(config, 0)
    alternate = generate_retarded_ekf_scenario(replace(config, base_seed=20260910), 0)
    np.testing.assert_array_equal(first.truth_state.vector, repeated.truth_state.vector)
    assert first.truth_seed == repeated.truth_seed
    for left, right in zip(first.events, repeated.events, strict=True):
        assert measurements_are_exact_duplicates(left, right)
    assert alternate.config.geometry == first.config.geometry
    assert alternate.config.motion == first.config.motion
    assert alternate.truth_seed != first.truth_seed
    assert not np.array_equal(alternate.truth_state.vector, first.truth_state.vector)


def test_truth_is_separate_and_declared_prediction_frame_is_used_consistently():
    scenario = generate_retarded_ekf_scenario(
        RetardedEKFStudyConfig(
            "informative", "stationary", 0.03, 0.05, "ordered", 1
        ),
        0,
    )
    assert all(item.tangent_frame == "prediction" for item in scenario.events)
    estimator_fields = {item.name for item in fields(CausalRetardedTimeEKF)} if hasattr(CausalRetardedTimeEKF, "__dataclass_fields__") else set()
    assert "truth_state" not in estimator_fields
    assert not hasattr(CausalRetardedTimeEKF, "truth_state")


def test_generated_tangent_residual_is_exactly_the_declared_anisotropic_draw():
    config = RetardedEKFStudyConfig(
        "informative", "stationary", 0.1, 0.18, "ordered", 1
    )
    scenario = generate_retarded_ekf_scenario(config, 0)
    covariance = np.diag(
        np.deg2rad(
            [config.azimuth_noise_std_deg, config.elevation_noise_std_deg]
        )
        ** 2
    )
    expected = np.random.default_rng(
        scenario.bearing_noise_seed
    ).multivariate_normal(np.zeros(2), covariance)
    measurement = scenario.events[0]
    actual = retarded_bearing_residual(
        scenario.truth_state, scenario.stations[0], measurement
    )
    np.testing.assert_allclose(actual, expected, atol=5e-16, rtol=0.0)


def test_smoke_configuration_compares_three_causal_methods_and_offline_reference():
    config = RetardedEKFStudyConfig(
        "informative", "uniform_oblique", 0.1, 0.18, "reordered", 1
    )
    publications, sequences = run_retarded_ekf_configuration(config)
    assert len(publications) == 46
    assert len(sequences) == 4
    assert {row["method"] for row in sequences} == {
        "retarded_ekf",
        "causal_prefix_batch",
        "initial_batch_no_updates",
        "offline_full_record_noncausal",
    }
    assert all(row["independent_unit"] == "whole_sequence" for row in sequences)
    assert all(row["time_samples_within_sequence_are_dependent"] for row in sequences)
    ekf = next(row for row in sequences if row["method"] == "retarded_ekf")
    assert ekf["initialization_succeeded"]
    assert ekf["final_valid"]
    assert np.isfinite(ekf["mean_measurement_nis"])
    summaries = summarize_retarded_ekf_sequences(sequences)
    assert len(summaries) == 4
    assert all(
        row["coverage_interval_unit"] == "whole_sequence_final_outcome"
        for row in summaries
    )


def test_poorly_conditioned_smoke_is_reported_without_deleting_failures():
    config = RetardedEKFStudyConfig(
        "poorly_conditioned", "uniform_oblique", 0.3, 0.5, "reordered", 1
    )
    _, sequences = run_retarded_ekf_configuration(config)
    assert len(sequences) == 4
    assert all("failure_reason_counts" in row for row in sequences)
    assert all(row["dependent_publication_count"] > 0 for row in sequences)
