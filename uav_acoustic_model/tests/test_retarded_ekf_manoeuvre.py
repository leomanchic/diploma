"""Deterministic gates for retarded-time updates on correlated history."""

from dataclasses import replace

import numpy as np

from estimators.retarded_ekf_manoeuvre import (
    AugmentedMotionHistory,
    CausalManoeuvreRetardedTimeEKF,
    HistoryObservationError,
    ManoeuvreHistoryConfig,
)
from estimators.retarded_ekf_recovery import CausalConfirmedRetardedTimeEKF
from model.bearing_events import bearing_event_id
from model.bearing_statistics import tangent_basis
from model.dynamic_state import ConstantVelocityState
from model.geometry import direction_angles, tetrahedral_array
from model.measurements import BearingMeasurement
from model.retarded_bearing import predict_retarded_bearing
from model.station import StationPose
from simulation.manoeuvre_trajectory import BenchmarkManoeuvreTrajectory
from simulation.moving_source import emission_time_residual, solve_emission_time
from validation.retarded_ekf_stress_study import (
    default_stress_profiles,
    generate_stress_base_block,
    generate_stress_scenario,
)
from validation.manoeuvre_tracking_study import (
    DEVELOPMENT_SEED, EVALUATION_SEED, generate_manoeuvre_scenario,
    run_paired_sequence,
)


def _station():
    return StationPose("S0", [0.0, 0.0, 0.0], np.eye(3), tetrahedral_array())


def _measurement(station, state, reception=1.0, frame_index=0):
    direction = predict_retarded_bearing(state, station, reception).direction_local
    return BearingMeasurement(
        station.station_id, "manoeuvre-test", frame_index,
        reception, reception + 0.2, direction,
        np.diag(np.deg2rad([0.3, 0.5])**2), np.zeros(2),
        "direct_bearing", tangent_frame="measurement",
    )


def _history(qc=0.25, step=0.25):
    state = ConstantVelocityState([70.0, 55.0, 40.0], [7.0, -3.0, 1.5], 0.0)
    config = ManoeuvreHistoryConfig(np.eye(3) * qc, history_step_s=step)
    history = AugmentedMotionHistory(state, np.eye(6) * 0.5, config)
    history.propagate_to(1.5)
    return state, history


def test_history_emission_solver_matches_independent_cv_solver_and_jacobian():
    station = _station()
    state, history = _history(qc=0.0)
    measurement = _measurement(station, state)
    expected = predict_retarded_bearing(state, station, 1.0).emission_time_s
    actual = history.emission_time(station, 1.0)
    assert abs(actual - expected) < 3e-13
    prediction = history.predict_measurement(station, measurement)
    analytic = prediction.jacobian_augmented
    numeric = np.zeros_like(analytic)
    original = history.mean.copy()
    for index in range(original.size):
        epsilon = 2e-5 if index % 6 < 3 else 2e-6
        history.mean[index] = original[index] + epsilon
        plus = history.predict_measurement(station, measurement).residual_tangent_rad
        history.mean[index] = original[index] - epsilon
        minus = history.predict_measurement(station, measurement).residual_tangent_rad
        numeric[:, index] = (plus - minus) / (2 * epsilon)
        history.mean[index] = original[index]
    np.testing.assert_allclose(analytic, numeric, rtol=0.0, atol=2e-8)


def test_delayed_measurement_changes_current_state_via_joint_cross_covariance():
    station = _station()
    state, history = _history(qc=0.25)
    measurement = _measurement(station, state, reception=1.0)
    phi, elevation = direction_angles(measurement.direction_local)
    tangent = tangent_basis(phi, elevation).T @ np.deg2rad([0.2, 0.1])
    angle = np.linalg.norm(tangent)
    offset = np.cos(angle) * measurement.direction_local + np.sin(angle) * tangent / angle
    measurement = replace(measurement, direction_local=offset)
    prior_current, prior_p = history.latest_state()
    predicted = history.predict_measurement(station, measurement, insert_emission_node=True)
    h = predicted.jacobian_augmented
    r = measurement.covariance_tangent_rad2
    s = h @ history.covariance @ h.T + r
    prior_joint_p = history.covariance.copy()
    gain = np.linalg.solve(s, h @ prior_joint_p).T
    joseph_factor = np.eye(prior_joint_p.shape[0]) - gain @ h
    independent_p = joseph_factor @ prior_joint_p @ joseph_factor.T + gain @ r @ gain.T
    independent_current = prior_current.vector - (
        history.covariance[-6:, :] @ h.T @ np.linalg.solve(s, predicted.residual_tangent_rad)
    )
    result = history.update_bearing(station, measurement, maximum_pre_update_nis=1e9)
    current, posterior_p = history.latest_state()
    assert result.update_applied and result.emission_time_s < result.reception_time_s
    np.testing.assert_allclose(current.vector, independent_current, rtol=0.0, atol=3e-11)
    np.testing.assert_allclose(posterior_p, independent_p[-6:, -6:], rtol=0.0, atol=3e-11)
    assert np.linalg.norm(current.vector - prior_current.vector) > 1e-6
    assert np.linalg.eigvalsh(posterior_p)[0] > 0.0
    assert np.linalg.norm(posterior_p - posterior_p.T) < 1e-12


def test_history_rejects_emission_before_window_without_using_true_range():
    station = _station()
    state, history = _history(qc=0.25)
    history.propagate_to(6.0)
    measurement = _measurement(station, state, reception=1.0)
    result = history.update_bearing(station, measurement, maximum_pre_update_nis=1e9)
    assert not result.update_applied
    assert result.failure_reason == "emission_outside_history"


def test_q_zero_history_update_matches_strict_cv_current_marginal():
    station = _station()
    state, history = _history(qc=0.0)
    measurement = _measurement(station, state)
    from estimators.retarded_ekf import update_retarded_ekf
    current_before, p_before = history.latest_state()
    strict = update_retarded_ekf(
        current_before, p_before, station, measurement,
        maximum_pre_update_nis=1e9,
    )
    augmented = history.update_bearing(station, measurement, maximum_pre_update_nis=1e9)
    current_after, p_after = history.latest_state()
    assert strict.update_applied and augmented.update_applied
    np.testing.assert_allclose(current_after.vector, strict.posterior_state.vector, rtol=0.0, atol=1e-9)
    np.testing.assert_allclose(p_after, strict.posterior_covariance, rtol=0.0, atol=1e-9)


def test_manoeuvre_truth_continuity_and_independent_emission_solver():
    trajectory = BenchmarkManoeuvreTrajectory(
        [70, 55, 40], [7, -3, 1.5], "smooth_turn"
    )
    station = _station()
    for boundary in (5.0, 9.0):
        assert np.linalg.norm(trajectory.q(boundary - 1e-8) - trajectory.q(boundary + 1e-8)) < 3e-7
        assert np.linalg.norm(trajectory.v(boundary - 1e-8) - trajectory.v(boundary + 1e-8)) < 3e-7
    assert np.max(np.linalg.norm(trajectory.v(np.linspace(0, 13, 51)), axis=1)) < 343.0
    emission = solve_emission_time(7.0, station.position_world_m, trajectory)
    assert abs(float(emission_time_residual(emission, 7.0, station.position_world_m, trajectory))) < 3e-13


def test_opt_in_schedule_invariance_and_event_use_contract_on_nominal_sequence():
    block = generate_stress_base_block("informative", 2, base_seed=20260913)
    profile = next(item for item in default_stress_profiles() if item.name == "nominal")
    scenario = generate_stress_scenario(block, profile)
    config = ManoeuvreHistoryConfig(np.eye(3) * 0.25)
    frequent = CausalManoeuvreRetardedTimeEKF(
        block.stations, scenario.events, estimator_variant="direct_bearing",
        history_config=config,
    )
    timestamps = sorted({item.available_timestamp_s for item in scenario.events})
    for timestamp in timestamps:
        frequent.advance_to(timestamp)
    final = frequent.advance_to(14.5)
    one_shot = CausalManoeuvreRetardedTimeEKF(
        block.stations, scenario.events, estimator_variant="direct_bearing",
        history_config=config,
    ).advance_to(14.5)
    assert final.valid == one_shot.valid
    assert final.event_uses == one_shot.event_uses
    used = [item.event_id for item in final.event_uses]
    assert len(used) == len(set(used))
    if final.valid:
        np.testing.assert_allclose(final.state.vector, one_shot.state.vector, rtol=0.0, atol=2e-9)
        np.testing.assert_allclose(final.covariance_state, one_shot.covariance_state, rtol=0.0, atol=2e-9)


def test_history_refinement_converges_for_smooth_turn_truth():
    station = _station()
    trajectory = BenchmarkManoeuvreTrajectory([70, 55, 40], [7, -3, 1.5], "smooth_turn")
    reception = 7.3
    truth_emission = solve_emission_time(reception, station.position_world_m, trajectory)
    true_direction = station.world_to_local_direction(
        (trajectory.q(truth_emission) - station.position_world_m)
        / np.linalg.norm(trajectory.q(truth_emission) - station.position_world_m)
    )
    measurement = BearingMeasurement(
        "S0", "refinement", 0, reception, reception + 0.1,
        true_direction, np.eye(2) * 1e-5, np.zeros(2), "direct_bearing",
        tangent_frame="measurement",
    )
    errors = []
    for step in (1.0, 0.5, 0.25):
        history = AugmentedMotionHistory(
            ConstantVelocityState(trajectory.q(4.5), trajectory.v(4.5), 4.5),
            np.eye(6),
            ManoeuvreHistoryConfig(
                np.eye(3) * 0.25, history_step_s=step, history_window_s=4.0
            ),
        )
        history.propagate_to(7.5)
        for node, epoch in enumerate(history.times_s):
            history.mean[6 * node : 6 * node + 3] = trajectory.q(epoch)
            history.mean[6 * node + 3 : 6 * node + 6] = trajectory.v(epoch)
        errors.append(np.linalg.norm(history.predict_measurement(station, measurement).residual_tangent_rad))
    assert errors[2] < errors[1] < errors[0]


def test_q_zero_full_cv_sequence_reproduces_confirmed_recovery():
    scenario = generate_manoeuvre_scenario(
        "informative", "constant_velocity", 0, DEVELOPMENT_SEED
    )
    rows, sequences = run_paired_sequence(scenario, 0.0)
    assert sequences[0]["accepted_update_count"] == sequences[1]["accepted_update_count"]
    assert sequences[0]["rejected_event_count"] == sequences[1]["rejected_event_count"]
    for timestamp in sorted({item["processing_time_s"] for item in rows}):
        pair = [item for item in rows if item["processing_time_s"] == timestamp]
        assert pair[0]["valid"] == pair[1]["valid"]
        if pair[0]["valid"]:
            assert abs(pair[0]["position_error_m"] - pair[1]["position_error_m"]) < 2e-8


def test_manoeuvre_event_stream_pairing_and_seed_isolation():
    first = generate_manoeuvre_scenario(
        "informative", "smooth_turn", 0, DEVELOPMENT_SEED
    )
    repeat = generate_manoeuvre_scenario(
        "informative", "smooth_turn", 0, DEVELOPMENT_SEED
    )
    evaluation = generate_manoeuvre_scenario(
        "informative", "smooth_turn", 0, EVALUATION_SEED
    )
    assert first.event_ids == repeat.event_ids
    assert set(first.event_ids).isdisjoint(evaluation.event_ids)
    for left, right in zip(first.events, repeat.events, strict=True):
        np.testing.assert_allclose(left.direction_local, right.direction_local, rtol=0, atol=0)
    assert any(not np.array_equal(left.direction_local, right.direction_local)
               for left, right in zip(first.events, evaluation.events, strict=True))
    assert all(event.available_timestamp_s > event.reception_center_timestamp_s
               for event in first.events)
    assert len(first.events) == len(set(first.event_ids)) == 45


def test_conflicting_active_event_invalidates_manoeuvre_history():
    block = generate_stress_base_block("informative", 2, base_seed=20260913)
    profile = next(item for item in default_stress_profiles() if item.name == "nominal")
    scenario = generate_stress_scenario(block, profile)
    config = ManoeuvreHistoryConfig(np.eye(3) * 0.25)
    preliminary = CausalManoeuvreRetardedTimeEKF(
        block.stations, scenario.events, estimator_variant="direct_bearing",
        history_config=config,
    ).advance_to(13.0)
    assert preliminary.valid and preliminary.active_state_event_ids
    target_id = preliminary.active_state_event_ids[0]
    target = next(item for item in scenario.events if bearing_event_id(item) == target_id)
    conflicting = replace(
        target, available_timestamp_s=14.0,
        direction_local=np.asarray([0.0, 0.0, 1.0]),
    )
    processor = CausalManoeuvreRetardedTimeEKF(
        block.stations, (*scenario.events, conflicting),
        estimator_variant="direct_bearing", history_config=config,
    )
    assert processor.advance_to(13.0).valid
    final = processor.advance_to(14.5)
    assert not final.valid and not final.confirmed
    assert final.history_node_count == 0
    assert target_id in final.conflicted_event_ids
    assert final.failure_reason == "conflicted_active_state_event_requires_recovery"


def test_large_availability_group_accounts_for_every_event_in_history_variant():
    block = generate_stress_base_block("informative", 2, base_seed=20260913)
    profile = next(item for item in default_stress_profiles() if item.name == "nominal")
    scenario = generate_stress_scenario(block, profile)
    ordered = sorted(
        scenario.events,
        key=lambda item: (item.available_timestamp_s, bearing_event_id(item)),
    )
    batched = tuple(
        item if index < 15 else replace(item, available_timestamp_s=14.0)
        for index, item in enumerate(ordered)
    )
    final = CausalManoeuvreRetardedTimeEKF(
        block.stations, batched, estimator_variant="direct_bearing",
        history_config=ManoeuvreHistoryConfig(np.eye(3) * 0.25),
    ).advance_to(14.5)
    accounted = (set(final.initialization_event_ids) |
                 set(final.applied_event_ids) | set(final.rejected_event_ids))
    assert accounted == set(final.prefix.accepted_event_ids)
    assert len(final.event_uses) == len(set(item.event_id for item in final.event_uses))


def test_joint_history_covariance_stays_symmetric_psd_after_manoeuvre_sequence():
    scenario = generate_manoeuvre_scenario(
        "informative", "smooth_turn", 0, DEVELOPMENT_SEED
    )
    estimator = CausalManoeuvreRetardedTimeEKF(
        scenario.stations, scenario.events, estimator_variant="direct_bearing",
        history_config=ManoeuvreHistoryConfig(
            np.eye(3), history_window_s=2.0, maximum_transport_delay_s=0.6
        ),
    )
    final = estimator.advance_to(14.5)
    assert final.valid and estimator._history is not None
    joint_p = estimator._history.covariance
    np.testing.assert_allclose(joint_p, joint_p.T, rtol=0.0, atol=1e-12)
    assert np.linalg.eigvalsh(joint_p)[0] > 0.0


def test_rejected_pre_update_nis_does_not_change_current_state_or_covariance():
    station = _station()
    state, history = _history(qc=0.25)
    measurement = _measurement(station, state, reception=1.0)
    direction = measurement.direction_local
    phi, elevation = direction_angles(direction)
    tangent = tangent_basis(phi, elevation).T @ np.deg2rad([15.0, 0.0])
    angle = np.linalg.norm(tangent)
    measurement = replace(
        measurement,
        direction_local=np.cos(angle) * direction + np.sin(angle) * tangent / angle,
    )
    prior, p_prior = history.latest_state()
    result = history.update_bearing(station, measurement, maximum_pre_update_nis=9.21034)
    after, p_after = history.latest_state()
    assert result.failure_reason == "pre_update_nis_gate"
    assert not result.update_applied and np.isfinite(result.pre_update_nis)
    np.testing.assert_allclose(after.vector, prior.vector, rtol=0.0, atol=0.0)
    np.testing.assert_allclose(p_after, p_prior, rtol=0.0, atol=2e-15)


def test_exact_duplicate_used_event_does_not_reset_or_update_twice():
    block = generate_stress_base_block("informative", 2, base_seed=20260913)
    profile = next(item for item in default_stress_profiles() if item.name == "nominal")
    scenario = generate_stress_scenario(block, profile)
    config = ManoeuvreHistoryConfig(np.eye(3) * 0.25)
    clean = CausalManoeuvreRetardedTimeEKF(
        block.stations, scenario.events, estimator_variant="direct_bearing",
        history_config=config,
    ).advance_to(14.5)
    target_id = clean.initialization_event_ids[0]
    duplicate = next(item for item in scenario.events if bearing_event_id(item) == target_id)
    repeated = CausalManoeuvreRetardedTimeEKF(
        block.stations, (*scenario.events, duplicate),
        estimator_variant="direct_bearing", history_config=config,
    ).advance_to(14.5)
    assert clean.valid and repeated.valid
    assert not repeated.conflicted_event_ids
    assert any(item.action == "duplicate_exact" for item in repeated.prefix.journal)
    np.testing.assert_allclose(repeated.state.vector, clean.state.vector, rtol=0.0, atol=2e-11)
    np.testing.assert_allclose(repeated.covariance_state, clean.covariance_state, rtol=0.0, atol=2e-11)
