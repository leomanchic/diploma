"""Acceptance tests for exact retarded-time constant-velocity batch WLS."""

from dataclasses import replace

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from estimators.bearing_triangulation import triangulate_bearings_spherical_wls
from estimators.retarded_state_batch import (
    assemble_retarded_batch_system,
    estimate_causal_prefix_batches,
    estimate_offline_retarded_batch,
    estimate_retarded_constant_velocity_batch,
)
from model.bearing_statistics import tangent_basis
from model.dynamic_state import (
    ConstantVelocityState,
    constant_velocity_transition_jacobian,
)
from model.geometry import direction_angles, tetrahedral_array
from model.measurements import BearingMeasurement
from model.retarded_bearing import predict_retarded_bearing
from model.station import StationPose


def _stations(kind="wide"):
    if kind == "wide":
        positions = ([0.0, 0.0, 0.0], [100.0, 0.0, 5.0], [10.0, 90.0, -2.0])
    elif kind == "rotated":
        positions = ([-30.0, 5.0, 4.0], [65.0, -20.0, 0.0], [20.0, 75.0, 8.0])
    else:
        raise ValueError(kind)
    return [
        StationPose(str(index), position, np.eye(3), tetrahedral_array())
        for index, position in enumerate(positions)
    ]


def _measurements(
    state,
    stations,
    *,
    covariance=None,
    availability_offset=0.02,
    sequence="batch",
):
    covariance = (
        np.diag(np.deg2rad([0.1, 0.15]) ** 2)
        if covariance is None
        else np.asarray(covariance)
    )
    times = ([0.8, 1.8, 2.8, 3.8], [1.05, 2.05, 3.05, 4.05], [1.3, 2.3, 3.3, 4.3])
    result = []
    for station_index, (station, station_times) in enumerate(zip(stations, times)):
        for frame, reception in enumerate(station_times):
            prediction = predict_retarded_bearing(state, station, reception)
            result.append(
                BearingMeasurement(
                    station.station_id,
                    sequence,
                    frame,
                    reception,
                    reception + availability_offset + 0.01 * station_index,
                    prediction.direction_local,
                    covariance,
                    np.zeros(2),
                    "direct",
                )
            )
    return result


def _exp_map(direction, offset):
    phi, elevation = direction_angles(direction)
    tangent = tangent_basis(phi, elevation).T @ np.asarray(offset)
    angle = np.linalg.norm(tangent)
    if angle == 0.0:
        return np.asarray(direction)
    return np.cos(angle) * direction + np.sin(angle) * tangent / angle


@pytest.mark.parametrize(
    "geometry,state",
    [
        ("wide", ConstantVelocityState([50.0, 40.0, 30.0], [5.0, -2.0, 1.0], 0.0)),
        ("rotated", ConstantVelocityState([35.0, 30.0, 55.0], [-7.0, 4.0, -1.5], 0.0)),
    ],
)
def test_noiseless_well_conditioned_scenes_recover_state_from_nontruth_initialization(
    geometry, state
):
    stations = _stations(geometry)
    measurements = _measurements(state, stations)
    deliberately_wrong = ConstantVelocityState(
        state.position_at_reference_world_m + [12.0, -9.0, 7.0],
        state.velocity_world_mps + [-3.0, 2.5, 1.0],
        0.0,
    )
    result = estimate_retarded_constant_velocity_batch(
        stations,
        measurements,
        reference_time_s=0.0,
        initial_state=deliberately_wrong,
    )
    assert result.valid, result.failure_reason
    np.testing.assert_allclose(
        result.state.position_at_reference_world_m,
        state.position_at_reference_world_m,
        atol=2e-8,
        rtol=0.0,
    )
    np.testing.assert_allclose(
        result.state.velocity_world_mps,
        state.velocity_world_mps,
        atol=2e-8,
        rtol=0.0,
    )
    assert result.maximum_angular_residual_rad < 2e-10
    assert result.local_observability_rank == 6


@pytest.mark.parametrize("reference_time_s", [0.0, 100.0])
def test_future_events_cannot_change_current_prefix_initialization_or_diagnostics(
    reference_time_s,
):
    stations = _stations()
    state = ConstantVelocityState([50.0, 40.0, 30.0], [5.0, -2.0, 1.0])
    events = _measurements(state, stations)
    cutoff = 2.5
    altered = []
    for item in events:
        if item.available_timestamp_s > cutoff:
            direction = _exp_map(item.direction_local, [0.02, -0.01])
            altered.append(replace(item, direction_local=direction))
        else:
            altered.append(item)
    first = estimate_causal_prefix_batches(
        stations,
        events,
        [cutoff],
        estimator_variant="direct",
        reference_time_s=reference_time_s,
    )[0]
    second = estimate_causal_prefix_batches(
        stations,
        altered,
        [cutoff],
        estimator_variant="direct",
        reference_time_s=reference_time_s,
    )[0]
    assert first.estimate.valid, first.estimate.failure_reason
    assert second.estimate.valid, second.estimate.failure_reason
    assert first.prefix == second.prefix
    assert first.estimate.used_event_ids == second.estimate.used_event_ids
    assert first.estimate.initialization_rank == second.estimate.initialization_rank
    np.testing.assert_array_equal(first.estimate.state.vector, second.estimate.state.vector)
    assert first.estimate.objective == second.estimate.objective


def test_offline_and_final_causal_prefix_match_for_identical_event_set():
    stations = _stations()
    state = ConstantVelocityState([50.0, 40.0, 30.0], [5.0, -2.0, 1.0])
    events = _measurements(state, stations)
    offline_prefix, offline = estimate_offline_retarded_batch(
        stations,
        events,
        estimator_variant="direct",
        reference_time_s=0.0,
    )
    final_time = max(item.available_timestamp_s for item in events)
    causal = estimate_causal_prefix_batches(
        stations,
        events,
        [final_time],
        estimator_variant="direct",
        reference_time_s=0.0,
    )[0]
    assert offline_prefix.accepted_event_ids == causal.prefix.accepted_event_ids
    np.testing.assert_array_equal(offline.state.vector, causal.estimate.state.vector)
    assert offline.objective == causal.estimate.objective


def test_availability_schedule_does_not_change_fixed_state_physical_system():
    stations = _stations()
    state = ConstantVelocityState([50.0, 40.0, 30.0], [5.0, -2.0, 1.0])
    original = _measurements(state, stations)
    delayed = [replace(item, available_timestamp_s=item.available_timestamp_s + 10.0) for item in original]
    first = assemble_retarded_batch_system(stations, original, state)
    second = assemble_retarded_batch_system(stations, delayed, state)
    np.testing.assert_array_equal(first.residuals_tangent_rad, second.residuals_tangent_rad)
    np.testing.assert_array_equal(first.residual_jacobian_state, second.residual_jacobian_state)


def test_fixed_zero_velocity_batch_matches_static_spherical_solver():
    stations = _stations()
    state = ConstantVelocityState([45.0, 35.0, 25.0], np.zeros(3))
    all_measurements = _measurements(state, stations)
    measurements = [item for item in all_measurements if item.frame_index == 0]
    static = triangulate_bearings_spherical_wls(stations, measurements)
    dynamic = estimate_retarded_constant_velocity_batch(
        stations,
        measurements,
        reference_time_s=0.0,
        fixed_velocity_world_mps=np.zeros(3),
    )
    assert static.valid and dynamic.valid
    np.testing.assert_allclose(
        dynamic.state.position_at_reference_world_m,
        static.position_world_m,
        atol=2e-10,
        rtol=0.0,
    )
    np.testing.assert_array_equal(dynamic.state.velocity_world_mps, np.zeros(3))


def test_whitened_batch_jacobian_matches_central_finite_difference():
    stations = _stations()
    truth = ConstantVelocityState([50.0, 40.0, 30.0], [5.0, -2.0, 1.0])
    measurements = _measurements(truth, stations)
    candidate = ConstantVelocityState([52.0, 38.0, 31.0], [4.5, -1.5, 0.7])
    analytic = assemble_retarded_batch_system(
        stations, measurements, candidate
    ).whitened_jacobian_state
    columns = []
    steps = np.asarray([2e-4] * 3 + [2e-5] * 3)
    for axis, step in enumerate(steps):
        delta = np.zeros(6)
        delta[axis] = step
        plus = ConstantVelocityState(
            candidate.vector[:3] + delta[:3],
            candidate.vector[3:] + delta[3:],
        )
        minus = ConstantVelocityState(
            candidate.vector[:3] - delta[:3],
            candidate.vector[3:] - delta[3:],
        )
        plus_residual = assemble_retarded_batch_system(
            stations, measurements, plus
        ).whitened_residuals
        minus_residual = assemble_retarded_batch_system(
            stations, measurements, minus
        ).whitened_residuals
        columns.append((plus_residual - minus_residual) / (2.0 * step))
    numerical = np.stack(columns, axis=1)
    np.testing.assert_allclose(analytic, numerical, rtol=2e-6, atol=2e-7)


@pytest.mark.parametrize("reference_time_s", [0.0, 100.0])
def test_zero_and_rank_one_covariances_preserve_exact_constraint_semantics(
    reference_time_s,
):
    stations = _stations()
    truth = ConstantVelocityState([50.0, 40.0, 30.0], [5.0, -2.0, 1.0])
    for covariance, expected_dimension in (
        (np.zeros((2, 2)), 24),
        (np.diag([np.deg2rad(0.1) ** 2, 0.0]), 12),
    ):
        measurements = _measurements(truth, stations, covariance=covariance)
        result = estimate_retarded_constant_velocity_batch(
            stations, measurements, reference_time_s=reference_time_s
        )
        assert result.valid, result.failure_reason
        assert result.exact_constraint_residuals.size == expected_dimension
        assert result.constraint_max_abs_rad < 2e-13
        assert result.constraints_satisfied
        np.testing.assert_allclose(
            result.state.position_at(0.0),
            truth.position_at(0.0),
            atol=3e-9,
            rtol=0.0,
        )
        np.testing.assert_allclose(
            result.state.velocity_world_mps,
            truth.velocity_world_mps,
            atol=3e-9,
            rtol=0.0,
        )
        assert result.local_observability_rank == 6


def test_small_positive_variance_converges_to_exact_constraint_solution():
    stations = _stations()
    truth = ConstantVelocityState([50.0, 40.0, 30.0], [5.0, -2.0, 1.0])
    exact_measurements = _measurements(
        truth, stations, covariance=np.diag([np.deg2rad(0.1) ** 2, 0.0])
    )
    exact = estimate_retarded_constant_velocity_batch(
        stations, exact_measurements, reference_time_s=0.0
    )
    nearly_exact = _measurements(
        truth,
        stations,
        covariance=np.diag([np.deg2rad(0.1) ** 2, 1e-16]),
    )
    limiting = estimate_retarded_constant_velocity_batch(
        stations, nearly_exact, reference_time_s=0.0
    )
    assert exact.valid and limiting.valid
    np.testing.assert_allclose(
        limiting.state.vector, exact.state.vector, atol=3e-8, rtol=0.0
    )
    assert np.all(np.isfinite(limiting.covariance_state_linearization))


def test_global_rigid_transform_station_permutation_and_reference_rebase_are_invariant():
    stations = _stations()
    state = ConstantVelocityState([50.0, 40.0, 30.0], [5.0, -2.0, 1.0])
    measurements = _measurements(state, stations)
    baseline = estimate_retarded_constant_velocity_batch(
        stations, measurements, reference_time_s=0.0
    )
    rotation = Rotation.from_euler("zyx", [0.4, -0.2, 0.1]).as_matrix()
    translation = np.asarray([300.0, -80.0, 25.0])
    transformed_stations = [
        StationPose(
            item.station_id,
            rotation @ item.position_world_m + translation,
            rotation @ item.rotation_local_to_world,
            item.microphone_positions_local_m,
        )
        for item in stations
    ]
    transformed = estimate_retarded_constant_velocity_batch(
        list(reversed(transformed_stations)),
        list(reversed(measurements)),
        reference_time_s=0.0,
    )
    assert baseline.valid and transformed.valid
    np.testing.assert_allclose(
        transformed.state.position_at_reference_world_m,
        rotation @ baseline.state.position_at_reference_world_m + translation,
        atol=2e-8,
        rtol=0.0,
    )
    np.testing.assert_allclose(
        transformed.state.velocity_world_mps,
        rotation @ baseline.state.velocity_world_mps,
        atol=2e-8,
        rtol=0.0,
    )


def test_requested_reference_epochs_preserve_trajectory_velocity_and_covariance():
    stations = _stations("wide")
    truth = ConstantVelocityState([50.0, 40.0, 30.0], [5.0, -2.0, 1.0], 0.0)
    measurements = _measurements(truth, stations)
    epochs = (0.0, 2.0, 30.0, 100.0)
    results = {}
    for epoch in epochs:
        result = estimate_retarded_constant_velocity_batch(
            stations, measurements, reference_time_s=epoch
        )
        assert result.valid, (epoch, result.failure_reason)
        assert result.local_observability_rank == 6
        np.testing.assert_allclose(
            result.state.position_at(0.0),
            truth.position_at(0.0),
            atol=2e-8,
            rtol=0.0,
        )
        np.testing.assert_allclose(
            result.state.velocity_world_mps,
            truth.velocity_world_mps,
            atol=2e-8,
            rtol=0.0,
        )
        results[epoch] = result

    baseline_covariance = results[0.0].covariance_state_linearization
    for epoch in epochs[1:]:
        transition = constant_velocity_transition_jacobian(epoch)
        expected = transition @ baseline_covariance @ transition.T
        np.testing.assert_allclose(
            results[epoch].covariance_state_linearization,
            expected,
            atol=2e-10,
            rtol=5e-10,
        )


def test_insufficient_radial_antipodal_and_incompatible_exact_cases_are_invalid():
    stations = _stations()
    state = ConstantVelocityState([50.0, 40.0, 30.0], [5.0, -2.0, 1.0])
    measurements = _measurements(state, stations)
    insufficient = estimate_retarded_constant_velocity_batch(
        stations, measurements[:2], reference_time_s=0.0
    )
    assert not insufficient.valid
    assert insufficient.failure_reason == "insufficient_measurements"

    antipodal_measurements = list(measurements)
    antipodal_measurements[0] = replace(
        antipodal_measurements[0],
        direction_local=-antipodal_measurements[0].direction_local,
    )
    antipodal = estimate_retarded_constant_velocity_batch(
        stations,
        antipodal_measurements,
        reference_time_s=0.0,
        initial_state=state,
    )
    assert not antipodal.valid
    assert "AntipodalDirectionError" in antipodal.failure_reason

    radial_station = [stations[0]]
    radial_state = ConstantVelocityState([80.0, 50.0, 40.0], [8.0, 5.0, 4.0])
    radial = _measurements(radial_state, radial_station)
    degenerate = estimate_retarded_constant_velocity_batch(
        radial_station, radial, reference_time_s=0.0
    )
    assert not degenerate.valid
    assert degenerate.failure_reason == "insufficient_local_observability"

    exact = _measurements(state, stations, covariance=np.zeros((2, 2)))
    exact[0] = replace(
        exact[0], direction_local=_exp_map(exact[0].direction_local, [0.03, -0.02])
    )
    incompatible = estimate_retarded_constant_velocity_batch(
        stations,
        exact,
        reference_time_s=0.0,
        constraint_tolerance_rad=1e-8,
    )
    assert not incompatible.valid
    assert incompatible.failure_reason == "incompatible_exact_constraints"
