"""Deterministic acceptance tests for the strict-CV retarded-time EKF."""

from dataclasses import replace

import numpy as np
import pytest

import estimators.retarded_ekf as ekf_module
from estimators.retarded_ekf import (
    CausalRetardedTimeEKF,
    RetardedEKFInitializationCriteria,
    joseph_residual_update,
    propagate_constant_velocity_estimate,
    update_retarded_ekf,
)
from estimators.retarded_state_batch import estimate_retarded_constant_velocity_batch
from model.bearing_events import ScheduledBearingEvent, bearing_event_id
from model.dynamic_state import ConstantVelocityState
from model.geometry import direction_angles, tetrahedral_array
from model.measurements import BearingMeasurement
from model.retarded_bearing import predict_retarded_bearing
from model.retarded_bearing import retarded_bearing_residual
from model.station import StationPose
from model.bearing_statistics import tangent_basis


def _stations(kind="wide"):
    if kind == "wide":
        positions = ([0.0, 0.0, 0.0], [100.0, 0.0, 5.0], [10.0, 90.0, -2.0])
    elif kind == "degenerate":
        positions = ([0.0, 0.0, 0.0],) * 3
    else:
        raise ValueError(kind)
    return tuple(
        StationPose(f"S{index}", position, np.eye(3), tetrahedral_array())
        for index, position in enumerate(positions)
    )


def _exp_map(direction, tangent_offset):
    phi, elevation = direction_angles(direction)
    world_tangent = tangent_basis(phi, elevation).T @ np.asarray(tangent_offset)
    angle = float(np.linalg.norm(world_tangent))
    if angle == 0.0:
        return np.asarray(direction)
    return np.cos(angle) * direction + np.sin(angle) * world_tangent / angle


def _measurements(
    state,
    stations,
    *,
    covariance=None,
    noise=None,
    availability_shift=0.02,
    tangent_frame="prediction",
):
    covariance = (
        np.diag(np.deg2rad([0.12, 0.2]) ** 2)
        if covariance is None
        else np.asarray(covariance)
    )
    noise = np.zeros(2) if noise is None else np.asarray(noise)
    times = ([0.8, 1.8, 2.8, 3.8], [1.05, 2.05, 3.05, 4.05], [1.3, 2.3, 3.3, 4.3])
    values = []
    for station_index, (station, reception_times) in enumerate(
        zip(stations, times, strict=True)
    ):
        for frame_index, reception in enumerate(reception_times):
            predicted = predict_retarded_bearing(state, station, reception)
            measured = _exp_map(predicted.direction_local, noise)
            values.append(
                BearingMeasurement(
                    station.station_id,
                    "ekf-test",
                    frame_index,
                    reception,
                    reception + availability_shift + 0.005 * station_index,
                    measured,
                    covariance,
                    np.zeros(2),
                    "direct",
                    tangent_frame=tangent_frame,
                )
            )
    return values


def test_cv_transition_and_covariance_rebase_are_exact():
    state = ConstantVelocityState([10.0, -4.0, 2.0], [3.0, 1.0, -0.5], 2.0)
    matrix = np.arange(36.0).reshape(6, 6) / 100.0
    covariance = matrix @ matrix.T + np.eye(6)
    propagated, actual_covariance, transition = propagate_constant_velocity_estimate(
        state, covariance, 5.5
    )
    expected_transition = np.eye(6)
    expected_transition[:3, 3:] = 3.5 * np.eye(3)
    np.testing.assert_allclose(transition, expected_transition, atol=0.0, rtol=0.0)
    np.testing.assert_allclose(propagated.position_at_reference_world_m, [20.5, -0.5, 0.25], atol=1e-15, rtol=0.0)
    np.testing.assert_allclose(actual_covariance, expected_transition @ covariance @ expected_transition.T, atol=2e-14, rtol=0.0)


def test_residual_sign_and_joseph_form_match_independent_linear_gaussian_solution():
    prior = np.asarray([1.0, -0.5])
    covariance = np.asarray([[2.0, 0.3], [0.3, 1.0]])
    design = np.asarray([[1.0, 2.0]])
    observed = np.asarray([0.2])
    residual = design @ prior - observed
    observation = np.asarray([[0.4]])
    result = joseph_residual_update(
        prior, covariance, residual, design, observation
    )
    precision = np.linalg.solve(covariance, np.eye(2)) + design.T @ np.linalg.solve(observation, design)
    rhs = np.linalg.solve(covariance, prior) + design.T @ np.linalg.solve(observation, observed)
    expected_covariance = np.linalg.solve(precision, np.eye(2))
    expected_state = np.linalg.solve(precision, rhs)
    np.testing.assert_allclose(result.posterior_vector, expected_state, atol=2e-15, rtol=0.0)
    np.testing.assert_allclose(result.posterior_covariance, expected_covariance, atol=2e-15, rtol=0.0)
    assert result.covariance_symmetry_error < 1e-15
    assert result.covariance_minimum_eigenvalue > 0.0


@pytest.mark.parametrize("tangent_frame", ["prediction", "measurement"])
def test_update_uses_analytic_retarded_jacobian_and_declared_tangent_frame(
    monkeypatch, tangent_frame
):
    stations = _stations()
    truth = ConstantVelocityState([52.0, 39.0, 31.0], [4.0, -1.0, 0.5], 2.0)
    measurement = _measurements(
        truth, stations, noise=[2e-4, -1e-4], tangent_frame=tangent_frame
    )[0]
    calls = []
    original = ekf_module.retarded_bearing_residual_jacobian

    def recording(*args, **kwargs):
        calls.append(args[2].tangent_frame)
        return original(*args, **kwargs)

    monkeypatch.setattr(ekf_module, "retarded_bearing_residual_jacobian", recording)
    result = update_retarded_ekf(
        truth,
        np.diag([4.0, 4.0, 4.0, 0.5, 0.5, 0.5]),
        stations[0],
        measurement,
    )
    assert result.valid
    assert calls == [tangent_frame]
    assert result.residual_jacobian_state.shape == (2, 6)
    assert result.normalized_innovation_squared >= 0.0


def test_singular_observation_covariance_is_explicitly_unsupported_without_update():
    stations = _stations()
    state = ConstantVelocityState([50.0, 40.0, 30.0], [5.0, -2.0, 1.0], 2.0)
    measurement = _measurements(
        state, stations, covariance=np.diag([1e-6, 0.0])
    )[0]
    covariance = np.diag([4.0, 4.0, 4.0, 1.0, 1.0, 1.0])
    result = update_retarded_ekf(state, covariance, stations[0], measurement)
    assert not result.valid
    assert not result.update_applied
    assert result.failure_reason == "unsupported_singular_covariance"
    np.testing.assert_array_equal(result.posterior_state.vector, state.vector)
    np.testing.assert_array_equal(result.posterior_covariance, covariance)


def test_singular_covariance_cannot_enter_c1_batch_initialization():
    stations = _stations()
    truth = ConstantVelocityState([50.0, 40.0, 30.0], [5.0, -2.0, 1.0])
    measurements = _measurements(
        truth, stations, covariance=np.diag([1e-6, 0.0])
    )
    result = CausalRetardedTimeEKF(
        stations, measurements, estimator_variant="direct"
    ).advance_to(max(item.available_timestamp_s for item in measurements))
    assert not result.valid
    assert not result.initialized
    assert result.failure_reason == "unsupported_singular_covariance"


def test_nonlinear_update_uses_residual_sign_that_reduces_local_bearing_error():
    stations = _stations()
    truth = ConstantVelocityState([50.0, 40.0, 30.0], [5.0, -2.0, 1.0], 2.0)
    measurement = _measurements(truth, stations)[0]
    prior = ConstantVelocityState([52.0, 38.0, 31.0], [4.8, -1.8, 0.9], 2.0)
    before = np.linalg.norm(retarded_bearing_residual(prior, stations[0], measurement))
    result = update_retarded_ekf(
        prior,
        np.diag([9.0, 9.0, 9.0, 1.0, 1.0, 1.0]),
        stations[0],
        measurement,
    )
    after = np.linalg.norm(
        retarded_bearing_residual(result.posterior_state, stations[0], measurement)
    )
    assert result.valid
    assert after < before


def test_causal_initialization_uses_only_prefix_and_never_reuses_its_events():
    stations = _stations()
    truth = ConstantVelocityState([50.0, 40.0, 30.0], [5.0, -2.0, 1.0])
    measurements = _measurements(truth, stations)
    ordered = sorted(measurements, key=lambda item: item.available_timestamp_s)
    cutoff = ordered[7].available_timestamp_s
    future = [
        replace(item, direction_local=_exp_map(item.direction_local, [0.05, -0.03]))
        if item.available_timestamp_s > cutoff
        else item
        for item in measurements
    ]
    first = CausalRetardedTimeEKF(
        stations, measurements, estimator_variant="direct"
    ).advance_to(cutoff)
    second = CausalRetardedTimeEKF(
        stations, future, estimator_variant="direct"
    ).advance_to(cutoff)
    assert first.valid and second.valid
    np.testing.assert_array_equal(first.state.vector, second.state.vector)
    np.testing.assert_array_equal(first.covariance_state, second.covariance_state)
    assert set(first.initialization_event_ids) == set(first.prefix.accepted_event_ids)
    assert first.applied_event_ids == ()
    processor = CausalRetardedTimeEKF(
        stations, measurements, estimator_variant="direct"
    )
    initialized = processor.advance_to(cutoff)
    repeated = processor.advance_to(cutoff)
    assert repeated.update_diagnostics == ()
    assert repeated.initialization_event_ids == initialized.initialization_event_ids
    np.testing.assert_array_equal(repeated.state.vector, initialized.state.vector)


def test_late_delivery_updates_current_state_without_reversing_filter_time():
    stations = _stations()
    truth = ConstantVelocityState([50.0, 40.0, 30.0], [5.0, -2.0, 1.0])
    measurements = _measurements(truth, stations)
    ordered = sorted(measurements, key=lambda item: item.available_timestamp_s)
    cutoff = ordered[7].available_timestamp_s
    late_original = ordered[8]
    late = replace(late_original, available_timestamp_s=cutoff + 2.0)
    events = [late if item is late_original else item for item in measurements]
    processor = CausalRetardedTimeEKF(stations, events, estimator_variant="direct")
    initialized = processor.advance_to(cutoff)
    assert initialized.valid
    updated = processor.advance_to(cutoff + 2.0)
    assert updated.state.reference_time_s == cutoff + 2.0
    assert any(item.event_id == bearing_event_id(late) for item in updated.update_diagnostics)
    assert all(item.prior_state.reference_time_s == cutoff + 2.0 for item in updated.update_diagnostics)


def test_exact_duplicate_is_not_double_counted_and_publications_are_immutable():
    stations = _stations()
    truth = ConstantVelocityState([50.0, 40.0, 30.0], [5.0, -2.0, 1.0])
    measurements = _measurements(truth, stations)
    duplicate = measurements[-1]
    processor = CausalRetardedTimeEKF(
        stations, measurements + [duplicate], estimator_variant="direct"
    )
    ordered = sorted(measurements, key=lambda item: item.available_timestamp_s)
    first = processor.advance_to(ordered[7].available_timestamp_s)
    saved = first.state.vector.copy()
    final = processor.advance_to(max(item.available_timestamp_s for item in measurements))
    assert sum(entry.action == "duplicate_exact" for entry in final.prefix.journal) == 1
    assert len(final.applied_event_ids) == len(measurements) - len(first.initialization_event_ids)
    np.testing.assert_array_equal(first.state.vector, saved)


def test_late_conflict_of_used_event_invalidates_future_publication_not_past_one():
    stations = _stations()
    truth = ConstantVelocityState([50.0, 40.0, 30.0], [5.0, -2.0, 1.0])
    measurements = _measurements(truth, stations)
    ordered = sorted(measurements, key=lambda item: item.available_timestamp_s)
    cutoff = ordered[7].available_timestamp_s
    used = ordered[0]
    conflict = replace(
        used,
        available_timestamp_s=cutoff + 1.0,
        direction_local=_exp_map(used.direction_local, [0.01, 0.0]),
    )
    processor = CausalRetardedTimeEKF(
        stations, measurements + [conflict], estimator_variant="direct"
    )
    published = processor.advance_to(cutoff)
    saved = published.state.vector.copy()
    conflicted = processor.advance_to(cutoff + 1.0)
    assert not conflicted.valid
    assert conflicted.failure_reason == "conflicted_used_event_requires_reinitialization"
    assert conflicted.state is None
    np.testing.assert_array_equal(published.state.vector, saved)
    recovered = processor.advance_to(cutoff + 1.0)
    assert recovered.valid
    assert recovered.initialized
    assert bearing_event_id(used) not in recovered.initialization_event_ids


def test_non_positive_definite_prior_covariance_is_rejected_without_regularization():
    stations = _stations()
    truth = ConstantVelocityState([50.0, 40.0, 30.0], [5.0, -2.0, 1.0], 2.0)
    measurement = _measurements(truth, stations)[0]
    with pytest.raises(ValueError, match="positive definite"):
        update_retarded_ekf(
            truth,
            np.diag([1.0, 1.0, 1.0, 1.0, 1.0, 0.0]),
            stations[0],
            measurement,
        )


def test_rank_deficient_scene_remains_not_initialized():
    stations = _stations("degenerate")
    truth = ConstantVelocityState([50.0, 40.0, 30.0], [5.0, -2.0, 1.0])
    measurements = _measurements(truth, stations)
    processor = CausalRetardedTimeEKF(
        stations,
        measurements,
        estimator_variant="direct",
        initialization_criteria=RetardedEKFInitializationCriteria(
            maximum_scaled_condition_number=1e12
        ),
    )
    result = processor.advance_to(max(item.available_timestamp_s for item in measurements))
    assert not result.valid
    assert result.failure_reason.startswith("initialization_failed:")
    assert result.state is None


def test_filter_initialization_matches_independent_batch_and_does_not_change_batch():
    stations = _stations()
    truth = ConstantVelocityState([50.0, 40.0, 30.0], [5.0, -2.0, 1.0])
    measurements = _measurements(truth, stations)
    processing_time = max(item.available_timestamp_s for item in measurements)
    before = estimate_retarded_constant_velocity_batch(
        stations, measurements, reference_time_s=processing_time
    )
    publication = CausalRetardedTimeEKF(
        stations, measurements, estimator_variant="direct"
    ).advance_to(processing_time)
    after = estimate_retarded_constant_velocity_batch(
        stations, measurements, reference_time_s=processing_time
    )
    assert before.valid and publication.valid and after.valid
    np.testing.assert_array_equal(before.state.vector, after.state.vector)
    np.testing.assert_array_equal(before.covariance_state_linearization, after.covariance_state_linearization)
    np.testing.assert_array_equal(publication.state.vector, before.state.vector)


def test_dropped_and_invalid_events_never_enter_initialization_or_updates():
    stations = _stations()
    truth = ConstantVelocityState([50.0, 40.0, 30.0], [5.0, -2.0, 1.0])
    measurements = _measurements(truth, stations)
    invalid = BearingMeasurement.invalid(
        station_id="S0",
        sequence_id="ekf-test",
        frame_index=99,
        reception_center_timestamp_s=4.5,
        available_timestamp_s=4.6,
        estimator_variant="direct",
        invalid_reason="silence",
    )
    dropped = ScheduledBearingEvent(
        replace(measurements[-1], frame_index=98),
        dropped=True,
        drop_reason="network_drop",
    )
    processor = CausalRetardedTimeEKF(
        stations, measurements + [invalid, dropped], estimator_variant="direct"
    )
    result = processor.advance_to(10.0)
    assert result.valid
    actions = {entry.action for entry in result.prefix.journal}
    assert "excluded_invalid" in actions
    assert "excluded_dropped" in actions
    assert all("|98|" not in identity and "|99|" not in identity for identity in result.initialization_event_ids)
