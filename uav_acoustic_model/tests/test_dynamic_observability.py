"""Local rank diagnostics for asynchronous retarded bearing rows."""

import numpy as np

from model.dynamic_state import ConstantVelocityState
from model.geometry import tetrahedral_array
from model.measurements import BearingMeasurement
from model.retarded_bearing import (
    predict_retarded_bearing,
    predicted_local_direction_jacobian,
    retarded_bearing_residual,
    stack_retarded_bearing_observability,
)
from model.station import StationPose
from validation.retarded_bearing_validation import (
    stacked_instantaneous_world_direction_jacobian,
)


def _stations(scale=1.0):
    return [
        StationPose("A", scale * np.array([0.0, 0.0, 0.0]), np.eye(3), tetrahedral_array()),
        StationPose("B", scale * np.array([120.0, 0.0, 3.0]), np.eye(3), tetrahedral_array()),
        StationPose("C", scale * np.array([20.0, 100.0, -2.0]), np.eye(3), tetrahedral_array()),
    ]


def _ideal_measurements(state, stations, times):
    result = []
    for frame, reception in enumerate(times):
        for station in stations:
            prediction = predict_retarded_bearing(state, station, reception)
            result.append(
                BearingMeasurement(
                    station.station_id,
                    "observability",
                    frame,
                    reception,
                    reception + 0.01 + 0.001 * len(result),
                    prediction.direction_local,
                    np.diag([1e-5, 2e-5]),
                    np.zeros(2),
                    "direct",
                )
            )
    return result


def test_one_station_radial_motion_retains_two_unobservable_state_directions():
    # Pure radial constant velocity keeps the bearing fixed.  Even four
    # reception epochs leave two exact local state directions unobservable.
    state = ConstantVelocityState([80.0, 50.0, 40.0], [8.0, 5.0, 4.0], 0.0)
    stations = _stations()[:1]
    measurements = _ideal_measurements(state, stations, [1.0, 2.0, 3.0, 4.0])
    diagnostic = stack_retarded_bearing_observability(state, stations, measurements)
    assert diagnostic.rank == 4
    assert np.isinf(diagnostic.condition_number)


def test_one_station_nonradial_retarded_stack_is_formally_rank_six_and_fd_verified():
    state = ConstantVelocityState([80.0, 50.0, 40.0], [8.0, -3.0, 1.0], 0.0)
    station = _stations()[0]
    times = [1.0, 2.0, 3.0, 4.0]
    measurements = _ideal_measurements(state, [station], times)
    diagnostic = stack_retarded_bearing_observability(
        state, [station], measurements
    )

    step = 2e-4
    numerical_columns = []
    for axis in range(6):
        delta = np.zeros(6)
        delta[axis] = step
        plus_state = ConstantVelocityState(
            state.vector[:3] + delta[:3],
            state.vector[3:] + delta[3:],
            state.reference_time_s,
        )
        minus_state = ConstantVelocityState(
            state.vector[:3] - delta[:3],
            state.vector[3:] - delta[3:],
            state.reference_time_s,
        )
        plus = np.concatenate(
            [retarded_bearing_residual(plus_state, station, item) for item in measurements]
        )
        minus = np.concatenate(
            [retarded_bearing_residual(minus_state, station, item) for item in measurements]
        )
        numerical_columns.append((plus - minus) / (2.0 * step))
    numerical = np.stack(numerical_columns, axis=1)
    maximum_mismatch = float(np.max(np.abs(diagnostic.jacobian - numerical)))

    assert diagnostic.rank == 6
    assert 1e-8 < diagnostic.singular_values[-1] < 3e-8
    assert 2e6 < diagnostic.condition_number < 4e6
    assert maximum_mismatch < diagnostic.singular_values[-1] * 1e-3


def test_independent_instantaneous_limit_has_scale_null_direction_and_rank_five():
    state = ConstantVelocityState([80.0, 50.0, 40.0], [8.0, -3.0, 1.0], 0.0)
    station = _stations()[0]
    jacobian = stacked_instantaneous_world_direction_jacobian(
        state, station, [1.0, 2.0, 3.0, 4.0]
    )
    singular_values = np.linalg.svd(jacobian, compute_uv=False)
    tolerance = max(jacobian.shape) * np.finfo(float).eps * singular_values[0]
    rank = int(np.count_nonzero(singular_values > tolerance))
    scale_direction = np.concatenate(
        (
            state.position_at_reference_world_m - station.position_world_m,
            state.velocity_world_mps,
        )
    )
    assert rank == 5
    np.testing.assert_allclose(jacobian @ scale_direction, 0.0, atol=2e-16)


def test_finite_sound_speed_jacobians_converge_to_instantaneous_limit_without_forced_rank():
    state = ConstantVelocityState([80.0, 50.0, 40.0], [8.0, -3.0, 1.0], 0.0)
    station = _stations()[0]
    times = [1.0, 2.0, 3.0, 4.0]
    instantaneous = stacked_instantaneous_world_direction_jacobian(
        state, station, times
    )
    smallest = []
    distances = []
    for sound_speed in (343.0, 3430.0, 34300.0):
        retarded = np.vstack(
            [
                predicted_local_direction_jacobian(
                    state, station, reception, sound_speed=sound_speed
                )
                for reception in times
            ]
        )
        smallest.append(float(np.linalg.svd(retarded, compute_uv=False)[-1]))
        distances.append(float(np.linalg.norm(retarded - instantaneous, ord=2)))
    assert smallest[0] > smallest[1] > smallest[2] > 0.0
    assert distances[0] > distances[1] > distances[2] > 0.0
    np.testing.assert_allclose(
        np.asarray(smallest[:-1]) / np.asarray(smallest[1:]),
        10.0,
        rtol=0.02,
    )
    np.testing.assert_allclose(
        np.asarray(distances[:-1]) / np.asarray(distances[1:]),
        10.0,
        rtol=0.02,
    )


def test_three_stations_and_temporal_bearings_can_reach_rank_six():
    state = ConstantVelocityState([80.0, 50.0, 40.0], [8.0, -3.0, 1.0], 0.0)
    stations = _stations()
    measurements = _ideal_measurements(state, stations, [1.0, 2.5, 4.0])
    diagnostic = stack_retarded_bearing_observability(state, stations, measurements)
    assert diagnostic.rank == 6
    assert np.isfinite(diagnostic.condition_number)
    assert np.max(np.abs(diagnostic.residual_tangent_rad)) < 1e-15
    assert diagnostic.jacobian.shape == (18, 6)


def test_short_window_and_poor_station_geometry_are_visible_in_conditioning():
    state = ConstantVelocityState([300.0, 120.0, 60.0], [10.0, 2.0, 0.0], 0.0)
    good_stations = _stations()
    poor_stations = [
        StationPose(f"P{i}", [10.0 * i, 0.01 * i, 0.0], np.eye(3), tetrahedral_array())
        for i in range(3)
    ]
    good = stack_retarded_bearing_observability(
        state, good_stations, _ideal_measurements(state, good_stations, [1.0, 3.0, 5.0])
    )
    poor = stack_retarded_bearing_observability(
        state, poor_stations, _ideal_measurements(state, poor_stations, [1.0, 1.001, 1.002])
    )
    assert good.rank == 6
    assert poor.condition_number > good.condition_number * 100.0


def test_observability_metadata_retains_station_frame_and_both_timestamps():
    state = ConstantVelocityState([60.0, 40.0, 20.0], [4.0, 1.0, 0.0])
    stations = _stations()
    measurements = _ideal_measurements(state, stations, [1.0, 2.0])
    diagnostic = stack_retarded_bearing_observability(state, stations, measurements)
    assert diagnostic.station_ids == tuple(item.station_id for item in measurements)
    assert diagnostic.frame_indices == tuple(item.frame_index for item in measurements)
    np.testing.assert_array_equal(
        diagnostic.reception_timestamps_s,
        [item.reception_center_timestamp_s for item in measurements],
    )
    assert np.all(diagnostic.emission_timestamps_s < diagnostic.reception_timestamps_s)
    assert np.all(diagnostic.available_timestamps_s >= diagnostic.reception_timestamps_s)
