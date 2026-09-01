"""Deterministic physical and timing tests for retarded bearing prediction."""

import numpy as np
import pytest
from dataclasses import fields
from scipy.spatial.transform import Rotation

from estimators.bearing_triangulation import (
    bearing_residual,
    bearing_residual_jacobian,
)
from model.bearing_statistics import AntipodalDirectionError
from model.dynamic_state import ConstantVelocityState
from model.geometry import tetrahedral_array
from model.measurements import BearingMeasurement
from model.retarded_bearing import (
    available_bearing_measurements,
    predict_retarded_bearing,
    predict_retarded_bearing_measurement,
    retarded_bearing_residual,
    retarded_bearing_residual_jacobian,
    retarded_equation_residual_s,
)
from model.station import StationPose


def _station(
    position=(0.0, 0.0, 0.0), rotation=np.eye(3), **clock
) -> StationPose:
    return StationPose("S0", position, rotation, tetrahedral_array(0.2), **clock)


def _measurement(station, direction, reception=2.0, available=2.01, bias=None):
    return BearingMeasurement(
        station.station_id,
        "dynamic",
        0,
        reception,
        available,
        direction,
        np.diag(np.deg2rad([0.2, 0.3]) ** 2),
        np.zeros(2) if bias is None else bias,
        "direct_bearing",
    )


def test_analytic_and_general_numerical_emission_times_agree():
    state = ConstantVelocityState([80.0, -20.0, 35.0], [-24.0, 8.0, 2.0], 0.3)
    station = _station([5.0, 3.0, -1.0])
    analytic = predict_retarded_bearing(state, station, 4.0, emission_solver="analytic")
    numerical = predict_retarded_bearing(state, station, 4.0, emission_solver="numerical")
    assert abs(analytic.emission_time_s - numerical.emission_time_s) < 2e-13
    assert abs(retarded_equation_residual_s(state, station, analytic)) < 2e-12
    assert analytic.emission_time_s < analytic.reception_time_s


def test_zero_velocity_reduces_to_static_bearing_and_position_jacobian():
    state = ConstantVelocityState([30.0, 12.0, 8.0], np.zeros(3), 0.0)
    station = _station()
    prediction = predict_retarded_bearing(state, station, 1.0)
    measurement = _measurement(station, prediction.direction_local)
    np.testing.assert_allclose(
        prediction.direction_world,
        state.position_at_reference_world_m / np.linalg.norm(state.position_at_reference_world_m),
        atol=2e-16,
    )
    np.testing.assert_allclose(
        retarded_bearing_residual(state, station, measurement),
        bearing_residual(state.position_at_reference_world_m, station, measurement),
        atol=2e-16,
    )
    np.testing.assert_allclose(
        retarded_bearing_residual_jacobian(state, station, measurement)[:, :3],
        bearing_residual_jacobian(state.position_at_reference_world_m, station, measurement),
        rtol=2e-13,
        atol=2e-16,
    )


def test_availability_and_station_clock_metadata_do_not_change_physics():
    state = ConstantVelocityState([90.0, 20.0, 15.0], [-10.0, 4.0, 1.0], 0.0)
    first_station = _station(clock_offset_s=0.4, clock_drift_s_per_s=2e-5)
    second_station = _station(clock_offset_s=-7.0, clock_drift_s_per_s=-8e-4)
    prediction = predict_retarded_bearing(state, first_station, 3.0)
    early = _measurement(first_station, prediction.direction_local, 3.0, 3.01)
    late = _measurement(second_station, prediction.direction_local, 3.0, 20.0)
    first = predict_retarded_bearing_measurement(state, first_station, early)
    second = predict_retarded_bearing_measurement(state, second_station, late)
    assert first.emission_time_s == second.emission_time_s
    np.testing.assert_array_equal(first.direction_local, second.direction_local)
    assert available_bearing_measurements([early, late], 3.5) == (early,)
    assert available_bearing_measurements([early, late], 20.0) == (early, late)


def test_online_measurement_contract_has_no_truth_emission_or_future_fields():
    names = {item.name for item in fields(BearingMeasurement)}
    forbidden = {
        "true_direction",
        "truth_direction",
        "true_position",
        "angular_error",
        "emission_time_s",
        "future_estimate",
    }
    assert names.isdisjoint(forbidden)


def test_reception_timestamp_changes_moving_source_prediction():
    state = ConstantVelocityState([40.0, 10.0, 5.0], [0.0, 20.0, 0.0], 0.0)
    station = _station()
    first = predict_retarded_bearing(state, station, 1.0)
    second = predict_retarded_bearing(state, station, 2.0)
    assert second.emission_time_s > first.emission_time_s
    assert np.linalg.norm(second.direction_world - first.direction_world) > 0.05


def test_global_rigid_transform_preserves_local_prediction_and_range():
    rotation = Rotation.from_euler("zyx", [0.7, -0.2, 0.1]).as_matrix()
    translation = np.array([500.0, -120.0, 30.0])
    local_rotation = Rotation.from_euler("xyz", [0.2, -0.1, 0.4]).as_matrix()
    station = _station([4.0, -2.0, 1.0], local_rotation)
    state = ConstantVelocityState([60.0, 25.0, 15.0], [8.0, -3.0, 2.0], 0.2)
    baseline = predict_retarded_bearing(state, station, 3.0)
    transformed_station = StationPose(
        "S0",
        rotation @ station.position_world_m + translation,
        rotation @ station.rotation_local_to_world,
        station.microphone_positions_local_m,
    )
    transformed_state = ConstantVelocityState(
        rotation @ state.position_at_reference_world_m + translation,
        rotation @ state.velocity_world_mps,
        state.reference_time_s,
    )
    transformed = predict_retarded_bearing(transformed_state, transformed_station, 3.0)
    np.testing.assert_allclose(transformed.direction_local, baseline.direction_local, atol=5e-15)
    np.testing.assert_allclose(transformed.direction_world, rotation @ baseline.direction_world, atol=5e-15)
    assert transformed.range_m == pytest.approx(baseline.range_m, abs=2e-13)


def test_common_time_coordinate_shift_preserves_physics():
    state = ConstantVelocityState([50.0, 10.0, 8.0], [6.0, 2.0, -1.0], 1.0)
    station = _station()
    baseline = predict_retarded_bearing(state, station, 4.0)
    shift = 1000.0
    shifted_state = ConstantVelocityState(
        state.position_at_reference_world_m,
        state.velocity_world_mps,
        state.reference_time_s + shift,
    )
    shifted = predict_retarded_bearing(shifted_state, station, 4.0 + shift)
    np.testing.assert_allclose(shifted.direction_local, baseline.direction_local, atol=3e-14)
    assert shifted.emission_time_s == pytest.approx(baseline.emission_time_s + shift, abs=2e-13)


def test_invalid_subsonic_coincidence_pole_and_antipode_cases_are_explicit():
    station = _station()
    with pytest.raises(ValueError, match=r"\|v\| < sound_speed"):
        predict_retarded_bearing(
            ConstantVelocityState([10.0, 0.0, 0.0], [343.0, 0.0, 0.0]), station, 1.0
        )
    with pytest.raises(ValueError, match="intersects|coincide"):
        predict_retarded_bearing(ConstantVelocityState(np.zeros(3), np.zeros(3)), station, 1.0)

    pole_state = ConstantVelocityState([0.0, 0.0, 30.0], np.zeros(3))
    pole_prediction = predict_retarded_bearing(pole_state, station, 1.0)
    pole_measurement = _measurement(station, pole_prediction.direction_local)
    with pytest.raises(ValueError, match="pole"):
        retarded_bearing_residual_jacobian(pole_state, station, pole_measurement)

    ordinary_state = ConstantVelocityState([30.0, 5.0, 10.0], np.zeros(3))
    ordinary = predict_retarded_bearing(ordinary_state, station, 1.0)
    antipodal = _measurement(station, -ordinary.direction_local)
    with pytest.raises(AntipodalDirectionError):
        retarded_bearing_residual(ordinary_state, station, antipodal)
    with pytest.raises(AntipodalDirectionError):
        retarded_bearing_residual_jacobian(ordinary_state, station, antipodal)


def test_infinite_sound_speed_limit_is_instantaneous_geometry():
    state = ConstantVelocityState([50.0, -10.0, 7.0], [25.0, 8.0, -2.0], 0.0)
    station = _station([3.0, 4.0, 1.0])
    reception = 2.0
    prediction = predict_retarded_bearing(state, station, reception, 1e12)
    instantaneous = state.position_at(reception) - station.position_world_m
    instantaneous /= np.linalg.norm(instantaneous)
    np.testing.assert_allclose(prediction.direction_world, instantaneous, atol=2e-10)
    assert reception - prediction.emission_time_s < 1e-9
