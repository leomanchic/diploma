"""Deterministic gates for the frozen S7C-D1 stress protocol."""

import json

import numpy as np

from estimators.retarded_ekf import (
    CausalRetardedTimeEKF,
    propagate_constant_velocity_estimate,
)
from model.bearing_events import bearing_event_id
from model.dynamic_state import ConstantVelocityState
from model.bearing_statistics import tangent_residual
from model.measurements import BearingMeasurement
from model.retarded_bearing import (
    predict_retarded_bearing,
    stack_retarded_bearing_observability,
)
from validation.retarded_ekf_stress_study import (
    EVALUATION_TIMES_S,
    GAP_END_S,
    GAP_START_S,
    StressProfile,
    _wilson_interval,
    default_stress_profiles,
    generate_stress_base_block,
    generate_stress_scenario,
    run_stress_sequence,
    stress_seed_provenance,
    summarize_independent_outcomes,
    spherical_exp_map_from_tangent,
)


def test_protocol_has_exactly_the_nine_frozen_profiles():
    assert tuple(profile.name for profile in default_stress_profiles()) == (
        "nominal",
        "dropout_20",
        "dropout_50",
        "one_station_gap",
        "all_station_gap",
        "long_delay",
        "outlier_mild",
        "outlier_strong",
        "mixed",
    )
    assert EVALUATION_TIMES_S[:3] == (0.0, 0.5, 1.0)
    assert EVALUATION_TIMES_S[-2:] == (12.5, 14.5)


def test_loss_extremes_gap_boundaries_and_delay_bounds_are_exact():
    block = generate_stress_base_block("informative", 0)
    delivered = generate_stress_scenario(block, StressProfile("loss_zero"))
    lost = generate_stress_scenario(
        block, StressProfile("loss_one", dropout_probability=1.0)
    )
    assert len(delivered.events) == 45
    assert len(lost.events) == 0
    assert sum(item.delivered for item in lost.event_truth) == 0

    gap = generate_stress_scenario(
        block, StressProfile("gap", gap_scope="all_stations")
    )
    for item in gap.event_truth:
        expected = not (GAP_START_S <= item.reception_timestamp_s <= GAP_END_S)
        assert item.delivered is expected
    boundary_times = {
        item.reception_timestamp_s
        for item in gap.event_truth
        if item.reception_timestamp_s in {GAP_START_S, GAP_END_S}
    }
    assert boundary_times == {GAP_START_S, GAP_END_S}

    long_delay = generate_stress_scenario(
        block, StressProfile("delay", delay_min_s=0.01, delay_max_s=2.0)
    )
    delays = np.asarray(
        [item.available_timestamp_s - item.reception_timestamp_s for item in long_delay.event_truth]
    )
    assert np.min(delays) >= 0.01 - 2e-15
    assert np.max(delays) <= 2.0 + 2e-15


def test_common_random_components_are_profile_independent_and_mechanisms_disjoint():
    block = generate_stress_base_block("poorly_conditioned", 3)
    nominal = generate_stress_scenario(block, default_stress_profiles()[0])
    mixed = generate_stress_scenario(block, default_stress_profiles()[-1])
    np.testing.assert_allclose(
        [item.nominal_tangent_error_rad for item in nominal.event_truth],
        [item.nominal_tangent_error_rad for item in mixed.event_truth],
        rtol=0.0,
        atol=0.0,
    )
    assert len(set(block.provenance.generated_seeds)) == 6
    assert len(set(block.provenance.identifiers)) == 6
    for measurement in mixed.events:
        assert "outlier" not in json.dumps(dict(measurement.quality_metadata)).lower()


def test_outlier_is_one_spherical_exp_map_in_declared_tangent_coordinates():
    direction = np.asarray([0.8, 0.4, 0.4472135954999579])
    direction /= np.linalg.norm(direction)
    offset = np.deg2rad([12.0, -7.0])
    measured = spherical_exp_map_from_tangent(direction, offset)
    residual = tangent_residual(direction, measured)
    np.testing.assert_allclose(residual, offset, rtol=0.0, atol=2e-15)
    assert np.linalg.norm(measured) == 1.0


def test_seed_grid_has_no_collisions_and_generation_is_reproducible():
    seeds = set()
    identifiers = set()
    sequence_indices = (*range(100), 1001, 2048)
    for base_seed in (20260910, 20260911):
        for geometry_index in range(2):
            for sequence_index in sequence_indices:
                provenance = stress_seed_provenance(
                    base_seed, geometry_index, sequence_index
                )
                seeds.update(provenance.generated_seeds)
                identifiers.update(provenance.identifiers)
    expected_count = 2 * 2 * len(sequence_indices) * 6
    assert len(seeds) == expected_count
    assert len(identifiers) == expected_count

    first = generate_stress_base_block("informative", 7)
    second = generate_stress_base_block("informative", 7)
    np.testing.assert_allclose(
        first.nominal_tangent_errors_rad,
        second.nominal_tangent_errors_rad,
        rtol=0.0,
        atol=0.0,
    )
    np.testing.assert_allclose(first.loss_uniforms, second.loss_uniforms, rtol=0.0, atol=0.0)
    first_scenario = generate_stress_scenario(first, default_stress_profiles()[-1])
    second_scenario = generate_stress_scenario(second, default_stress_profiles()[-1])
    assert tuple(item.event_id for item in first_scenario.event_truth) == tuple(
        item.event_id for item in second_scenario.event_truth
    )
    np.testing.assert_allclose(
        [item.direction_local for item in first_scenario.events],
        [item.direction_local for item in second_scenario.events],
        rtol=0.0,
        atol=0.0,
    )


def test_no_observation_cv_prediction_is_exact_with_cross_covariances():
    state = ConstantVelocityState([10.0, -4.0, 8.0], [2.0, -1.0, 0.5], 1.0)
    factor = np.asarray(
        [
            [2.0, 0.0, 0.0, 0.2, 0.0, 0.0],
            [0.1, 1.8, 0.0, 0.0, -0.1, 0.0],
            [0.0, 0.2, 1.5, 0.0, 0.0, 0.15],
            [0.0, 0.0, 0.0, 0.5, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.1, 0.6, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.1, 0.7],
        ]
    )
    covariance = factor @ factor.T
    propagated, propagated_covariance, transition = propagate_constant_velocity_estimate(
        state, covariance, 5.0
    )
    expected_transition = np.eye(6)
    expected_transition[:3, 3:] = 4.0 * np.eye(3)
    np.testing.assert_allclose(transition, expected_transition, rtol=0.0, atol=0.0)
    np.testing.assert_allclose(
        propagated.vector,
        expected_transition @ state.vector,
        rtol=0.0,
        atol=2e-12,
    )
    np.testing.assert_allclose(
        propagated_covariance,
        expected_transition @ covariance @ expected_transition.T,
        rtol=0.0,
        atol=2e-10,
    )


def test_stress_stream_never_exposes_future_events_and_schedule_is_invariant():
    scenario = generate_stress_scenario(
        generate_stress_base_block("informative", 1),
        default_stress_profiles()[5],
    )
    events = tuple(sorted(scenario.events, key=lambda item: item.available_timestamp_s)[:15])
    final_time = max(item.available_timestamp_s for item in events)
    frequent = CausalRetardedTimeEKF(
        scenario.base_block.stations, events, estimator_variant="direct_bearing"
    )
    for current_time in sorted({item.available_timestamp_s for item in events}):
        frequent_result = frequent.advance_to(current_time)
        assert all(
            item.available_timestamp_s <= current_time
            for item in frequent_result.prefix.measurements
        )
    sparse = CausalRetardedTimeEKF(
        scenario.base_block.stations, events, estimator_variant="direct_bearing"
    )
    sparse_result = sparse.advance_to(final_time)
    assert frequent_result.valid and sparse_result.valid
    np.testing.assert_allclose(
        frequent_result.state.vector, sparse_result.state.vector, rtol=0.0, atol=3e-10
    )
    np.testing.assert_allclose(
        frequent_result.covariance_state,
        sparse_result.covariance_state,
        rtol=0.0,
        atol=3e-9,
    )
    assert frequent_result.initialization_event_ids == sparse_result.initialization_event_ids
    assert frequent_result.applied_event_ids == sparse_result.applied_event_ids


def test_one_station_radial_control_remains_rank_deficient():
    station = generate_stress_base_block("informative", 0).stations[0]
    state = ConstantVelocityState([80.0, 50.0, 40.0], [8.0, 5.0, 4.0], 0.0)
    measurements = []
    covariance = np.diag(np.deg2rad([0.3, 0.5]) ** 2)
    for frame_index, reception_time in enumerate([1.0, 2.0, 3.0, 4.0]):
        prediction = predict_retarded_bearing(state, station, reception_time)
        measurements.append(
            BearingMeasurement(
                station.station_id,
                "s7cd1-radial-control",
                frame_index,
                reception_time,
                reception_time + 0.01,
                prediction.direction_local,
                covariance,
                np.zeros(2),
                "direct_bearing",
                tangent_frame="prediction",
            )
        )
    diagnostic = stack_retarded_bearing_observability(state, [station], measurements)
    assert diagnostic.rank == 4
    assert np.isinf(diagnostic.condition_number)

    stationary = ConstantVelocityState([80.0, 50.0, 40.0], np.zeros(3), 0.0)
    stationary_measurements = []
    for frame_index, reception_time in enumerate([1.0, 2.0, 3.0, 4.0]):
        prediction = predict_retarded_bearing(stationary, station, reception_time)
        stationary_measurements.append(
            BearingMeasurement(
                station.station_id,
                "s7cd1-stationary-control",
                frame_index,
                reception_time,
                reception_time + 0.01,
                prediction.direction_local,
                covariance,
                np.zeros(2),
                "direct_bearing",
                tangent_frame="prediction",
            )
        )
    stationary_diagnostic = stack_retarded_bearing_observability(
        stationary, [station], stationary_measurements
    )
    assert stationary_diagnostic.rank < 6
    assert np.isinf(stationary_diagnostic.condition_number)


def test_no_update_baseline_uses_initialization_batch_not_updated_publication():
    block = generate_stress_base_block("informative", 0)
    full = generate_stress_scenario(block, default_stress_profiles()[0])
    # Nine early events are enough to initialize and leave later updates while
    # keeping this regression faster than a full 45-event stress run.
    early_events = tuple(sorted(full.events, key=lambda item: item.available_timestamp_s)[:9])
    delivered_ids = {item.event_id for item in full.event_truth if item.event_id in {bearing_event_id(event) for event in early_events}}
    scenario = type(full)(full.base_block, full.profile, full.sequence_id, early_events, tuple(item for item in full.event_truth if item.event_id in delivered_ids))
    epochs, rows = run_stress_sequence(scenario)
    by_method = {row["method"]: row for row in rows}
    ekf_ids = json.loads(by_method["retarded_ekf"]["ekf_initialization_event_ids_json"])
    applied_ids = json.loads(by_method["retarded_ekf"]["ekf_applied_event_ids_json"])
    origin_ids = json.loads(by_method["initial_batch_no_updates"]["no_update_origin_event_ids_json"])
    assert sorted(origin_ids) == sorted(ekf_ids)
    assert by_method["retarded_ekf"]["method_update_event_count"] > 0
    assert by_method["initial_batch_no_updates"]["method_update_event_count"] == 0
    assert set(origin_ids).isdisjoint(applied_ids)
    for epoch in (item for item in epochs if item["method"] == "retarded_ekf"):
        partition = sum(
            int(epoch[field])
            for field in (
                "method_initialization_event_count",
                "method_update_event_count",
                "method_rejected_event_count",
                "stream_quarantined_event_count",
                "method_remaining_unprocessed_available_count",
            )
        )
        assert partition == int(epoch["causally_available_event_count"])


def test_all_dropped_sequence_is_explicitly_invalid_without_fake_state():
    block = generate_stress_base_block("informative", 0)
    scenario = generate_stress_scenario(
        block, StressProfile("all_dropped", dropout_probability=1.0)
    )
    epochs, rows = run_stress_sequence(scenario)
    assert scenario.events == ()
    assert all(not bool(row["at_14_5_s_valid"]) for row in rows)
    assert all(int(row["delivered_event_count"]) == 0 for row in rows)
    assert all(str(row["at_14_5_s_failure_reason"]) for row in rows)
    causal_epochs = [row for row in epochs if row["method"] != "offline_full_batch"]
    assert causal_epochs
    assert all(not bool(row["valid"]) for row in causal_epochs)
    assert all(np.isnan(float(row["position_error_m"])) for row in causal_epochs)
    assert all(int(row["potential_event_count"]) == 45 for row in epochs)
    assert all(int(row["profile_delivered_event_count"]) == 0 for row in epochs)
    assert all(int(row["profile_lost_event_count"]) == 45 for row in epochs)
    assert all(int(row["causally_available_event_count"]) == 0 for row in epochs)


def test_aggregation_keeps_invalid_denominator_reasons_coverage_and_linear_p95():
    rows = [
        {
            "valid": True,
            "position": 0.0,
            "velocity": 0.0,
            "covered": True,
            "reason": "",
            "initialization_succeeded": True,
        },
        {
            "valid": True,
            "position": 1.0,
            "velocity": 2.0,
            "covered": False,
            "reason": "",
            "initialization_succeeded": True,
        },
        {
            "valid": True,
            "position": 9.0,
            "velocity": 4.0,
            "covered": True,
            "reason": "",
            "initialization_succeeded": True,
        },
        {
            "valid": False,
            "position": np.nan,
            "velocity": np.nan,
            "covered": False,
            "reason": "not_initialized",
            "initialization_succeeded": False,
        },
        {
            "valid": False,
            "position": np.nan,
            "velocity": np.nan,
            "covered": False,
            "reason": "different_failure",
            "initialization_succeeded": False,
        },
    ]
    result = summarize_independent_outcomes(
        rows,
        valid_field="valid",
        position_error_field="position",
        velocity_error_field="velocity",
        covered_field="covered",
        failure_reason_field="reason",
    )
    assert result["valid_count"] == 3
    assert result["total_sequence_denominator"] == 5
    assert result["conditional_state_95_coverage_fraction"] == 2 / 3
    assert result["unconditional_valid_and_covered_fraction"] == 2 / 5
    assert result["conditional_position_p95_m"] == np.percentile(
        [0.0, 1.0, 9.0], 95, method="linear"
    )
    assert result["conditional_position_p95_m"] != np.percentile(
        [0.0, 1.0, 9.0], 95, method="higher"
    )
    assert json.loads(result["failure_reason_counts"]) == {
        "different_failure": 1,
        "not_initialized": 1,
    }


def test_wilson_interval_stays_inside_probability_support_at_endpoints():
    low_zero, high_zero = _wilson_interval(0, 100)
    low_one, high_one = _wilson_interval(100, 100)
    assert low_zero == 0.0
    assert 0.0 < high_zero < 1.0
    assert 0.0 < low_one < 1.0
    assert high_one == 1.0
