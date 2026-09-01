"""Local rank diagnostics for asynchronous retarded bearing rows."""

import numpy as np

from model.dynamic_state import ConstantVelocityState
from model.geometry import tetrahedral_array
from model.measurements import BearingMeasurement
from model.retarded_bearing import (
    predict_retarded_bearing,
    stack_retarded_bearing_observability,
)
from model.station import StationPose


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


def test_one_station_cannot_observe_full_position_and_velocity():
    state = ConstantVelocityState([80.0, 50.0, 40.0], [8.0, -3.0, 1.0], 0.0)
    stations = _stations()[:1]
    measurements = _ideal_measurements(state, stations, [1.0])
    diagnostic = stack_retarded_bearing_observability(state, stations, measurements)
    assert diagnostic.rank < 6
    assert np.isinf(diagnostic.condition_number)


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
