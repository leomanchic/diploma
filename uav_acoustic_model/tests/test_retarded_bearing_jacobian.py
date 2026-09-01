"""Analytic/finite-difference audit of the retarded bearing Jacobians."""

import numpy as np
from scipy.spatial.transform import Rotation

from model.bearing_statistics import tangent_basis
from model.dynamic_state import (
    ConstantVelocityState,
    constant_velocity_transition_jacobian,
    rebase_constant_velocity_state,
)
from model.geometry import direction_angles, tetrahedral_array
from model.measurements import BearingMeasurement
from model.retarded_bearing import (
    emission_time_jacobian_wrt_state,
    predict_retarded_bearing,
    predicted_local_direction_jacobian,
    retarded_bearing_residual,
    retarded_bearing_residual_jacobian,
)
from model.station import StationPose


FD_SEED = 20260901


def _state_from_vector(vector, reference):
    return ConstantVelocityState(vector[:3], vector[3:], reference)


def _exp_map(direction, coordinates):
    phi, elevation = direction_angles(direction)
    tangent = tangent_basis(phi, elevation).T @ np.asarray(coordinates)
    theta = np.linalg.norm(tangent)
    return np.cos(theta) * direction + np.sin(theta) * tangent / theta


def _measurement(station, prediction, offset=(0.003, -0.002)):
    return BearingMeasurement(
        station.station_id,
        "fd",
        0,
        prediction.reception_time_s,
        prediction.reception_time_s + 0.02,
        _exp_map(prediction.direction_local, offset),
        np.diag([1e-5, 2e-5]),
        np.array([2e-4, -1e-4]),
        "direct",
    )


def _central(function, state, steps):
    vector = state.vector
    columns = []
    for axis, step in enumerate(steps):
        delta = np.zeros(6)
        delta[axis] = step
        plus = function(_state_from_vector(vector + delta, state.reference_time_s))
        minus = function(_state_from_vector(vector - delta, state.reference_time_s))
        columns.append((np.asarray(plus) - np.asarray(minus)) / (2.0 * step))
    return np.stack(columns, axis=-1)


def test_emission_time_direction_and_residual_jacobians_match_central_differences():
    station = StationPose(
        "S",
        [3.0, -8.0, 1.0],
        Rotation.from_euler("zyx", [0.4, -0.2, 0.1]).as_matrix(),
        tetrahedral_array(),
    )
    state = ConstantVelocityState([90.0, 20.0, 35.0], [-12.0, 7.0, 3.0], 0.4)
    reception = 3.0
    prediction = predict_retarded_bearing(state, station, reception)
    measurement = _measurement(station, prediction)
    steps = np.array([1e-4] * 3 + [1e-4] * 3)
    numerical_time = _central(
        lambda candidate: predict_retarded_bearing(candidate, station, reception).emission_time_s,
        state,
        steps,
    )
    numerical_direction = _central(
        lambda candidate: predict_retarded_bearing(candidate, station, reception).direction_local,
        state,
        steps,
    )
    numerical_residual = _central(
        lambda candidate: retarded_bearing_residual(candidate, station, measurement),
        state,
        steps,
    )
    np.testing.assert_allclose(
        emission_time_jacobian_wrt_state(state, station, reception),
        numerical_time,
        rtol=2e-7,
        atol=2e-10,
    )
    np.testing.assert_allclose(
        predicted_local_direction_jacobian(state, station, reception),
        numerical_direction,
        rtol=2e-7,
        atol=2e-10,
    )
    np.testing.assert_allclose(
        retarded_bearing_residual_jacobian(state, station, measurement),
        numerical_residual,
        rtol=2e-6,
        atol=2e-10,
    )


def test_rebase_jacobian_obeys_state_coordinate_chain_rule():
    station = StationPose("S", [2.0, 1.0, 0.0], np.eye(3), tetrahedral_array())
    state = ConstantVelocityState([60.0, 20.0, 15.0], [7.0, -2.0, 1.0], 0.0)
    reception = 4.0
    prediction = predict_retarded_bearing(state, station, reception)
    measurement = _measurement(station, prediction)
    dt = 1.3
    rebased = rebase_constant_velocity_state(state, dt)
    old_jacobian = retarded_bearing_residual_jacobian(state, station, measurement)
    new_jacobian = retarded_bearing_residual_jacobian(rebased, station, measurement)
    transition = constant_velocity_transition_jacobian(dt)
    np.testing.assert_allclose(old_jacobian, new_jacobian @ transition, rtol=2e-12, atol=2e-14)


def test_randomized_1000_scene_fd_audit_including_near_sonic_stress():
    rng = np.random.default_rng(FD_SEED)
    maximum = {
        "time_abs": 0.0,
        "direction_abs": 0.0,
        "residual_abs": 0.0,
        "relative": 0.0,
    }
    for trial in range(1000):
        rotation = Rotation.random(random_state=rng).as_matrix()
        station_position = rng.uniform(-100.0, 100.0, 3)
        station = StationPose("S", station_position, rotation, tetrahedral_array())
        direction = rng.normal(size=3)
        direction /= np.linalg.norm(direction)
        if abs((rotation.T @ direction)[2]) > 0.97:
            direction = np.cross(direction, rotation[:, 2])
            direction /= np.linalg.norm(direction)
        range_m = rng.uniform(10.0, 500.0)
        velocity_direction = rng.normal(size=3)
        velocity_direction /= np.linalg.norm(velocity_direction)
        speed = rng.uniform(0.0, 60.0) if trial < 950 else rng.uniform(0.75, 0.9) * 343.0
        velocity = speed * velocity_direction
        reference = rng.uniform(-2.0, 2.0)
        reception = reference + rng.uniform(0.2, 3.0)
        position_at_reception = station_position + range_m * direction
        position_at_reference = position_at_reception - velocity * (reception - reference)
        state = ConstantVelocityState(position_at_reference, velocity, reference)
        prediction = predict_retarded_bearing(state, station, reception)
        local = prediction.direction_local
        if np.hypot(local[0], local[1]) <= 0.05:
            continue
        measurement = _measurement(station, prediction, rng.normal(0.0, 0.004, 2))
        steps = np.array([2e-4] * 3 + [2e-4] * 3)
        analytic_time = emission_time_jacobian_wrt_state(state, station, reception)
        numeric_time = _central(
            lambda candidate: predict_retarded_bearing(candidate, station, reception).emission_time_s,
            state,
            steps,
        )
        analytic_direction = predicted_local_direction_jacobian(state, station, reception)
        numeric_direction = _central(
            lambda candidate: predict_retarded_bearing(candidate, station, reception).direction_local,
            state,
            steps,
        )
        analytic_residual = retarded_bearing_residual_jacobian(state, station, measurement)
        numeric_residual = _central(
            lambda candidate: retarded_bearing_residual(candidate, station, measurement),
            state,
            steps,
        )
        for key, analytic, numeric in (
            ("time_abs", analytic_time, numeric_time),
            ("direction_abs", analytic_direction, numeric_direction),
            ("residual_abs", analytic_residual, numeric_residual),
        ):
            mismatch = np.abs(analytic - numeric)
            maximum[key] = max(maximum[key], float(np.max(mismatch)))
            mask = np.abs(numeric) > 1e-7
            if np.any(mask):
                maximum["relative"] = max(
                    maximum["relative"],
                    float(np.max(mismatch[mask] / np.abs(numeric[mask]))),
                )
    assert maximum["time_abs"] < 2e-8, maximum
    assert maximum["direction_abs"] < 2e-8, maximum
    assert maximum["residual_abs"] < 2e-8, maximum
    assert maximum["relative"] < 2e-5, maximum
