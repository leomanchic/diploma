"""Independent counterexamples from the formal model review, 2026-09-06."""

from dataclasses import replace

import numpy as np
import pytest

from estimators.bearing_triangulation import bearing_residual, bearing_residual_jacobian
from model.bearing_events import measurements_are_exact_duplicates
from model.dynamic_state import ConstantVelocityState
from model.geometry import tetrahedral_array
from model.measurements import BearingMeasurement
from model.retarded_bearing import retarded_bearing_residual, retarded_bearing_residual_jacobian
from model.station import StationPose
from simulation.moving_source import constant_velocity_emission_time, emission_time_residual
from simulation.trajectory import ConstantVelocityTrajectory


@pytest.mark.parametrize("margin", [1e-6, 1e-10, 1e-12, 1e-14])
def test_radial_receding_time_agrees_with_independent_linear_equation(margin):
    # For t_r=0 and a radial receding source: (c+v)*(-t_e)=R.
    c, distance = 343.0, 100.0
    v = c * (1.0 - margin)
    trajectory = ConstantVelocityTrajectory([distance, 0, 0], [v, 0, 0])
    actual = constant_velocity_emission_time(0.0, np.zeros(3), trajectory)
    np.testing.assert_allclose(actual, -distance / (c + v), rtol=5e-15, atol=0)
    assert abs(emission_time_residual(actual, 0.0, np.zeros(3), trajectory)) < 1e-15


def _scene(frame):
    station = StationPose("S", [0, 0, 0], np.eye(3), tetrahedral_array())
    y = np.array([0., .01, 1.])
    y /= np.linalg.norm(y)
    measurement = BearingMeasurement(
        "S", "formal-review", 0, 0., 0., y,
        np.diag([1e-8, 1e-4]), np.array([2e-5, -3e-5]), "direct",
        tangent_frame=frame,
    )
    return station, measurement


@pytest.mark.parametrize("frame", ["prediction", "measurement"])
def test_fixed_calibration_frame_removes_candidate_threshold_discontinuity(frame):
    station, measurement = _scene(frame)
    positions = [np.array([x, 0, 1.]) for x in (0.999999e-7, 1.000001e-7)]
    residuals = [bearing_residual(q, station, measurement) for q in positions]
    costs = [r @ np.linalg.solve(measurement.covariance_tangent_rad2, r) for r in residuals]
    np.testing.assert_allclose(costs[0], costs[1], rtol=1e-8)
    np.testing.assert_allclose(residuals[0], residuals[1], atol=1e-11)
    for q, static in zip(positions, residuals):
        dynamic = retarded_bearing_residual(ConstantVelocityState(q, np.zeros(3)), station, measurement)
        np.testing.assert_allclose(dynamic, static, atol=1e-15)


def test_measurement_frame_anisotropic_biased_jacobian_at_noisy_zenith():
    station, measurement = _scene("measurement")
    q = np.array([0., 0., 1.])
    step = 1e-6
    numeric = np.column_stack([
        (bearing_residual(q + step * axis, station, measurement)
         - bearing_residual(q - step * axis, station, measurement)) / (2 * step)
        for axis in np.eye(3)
    ])
    analytic = bearing_residual_jacobian(q, station, measurement)
    np.testing.assert_allclose(analytic, numeric, rtol=1e-7, atol=1e-9)
    state = ConstantVelocityState(q, np.zeros(3))
    np.testing.assert_allclose(
        retarded_bearing_residual_jacobian(state, station, measurement)[:, :3],
        analytic, atol=1e-14,
    )
    legacy = replace(measurement, tangent_frame="prediction")
    with pytest.raises(ValueError, match="tangent frame.*pole"):
        bearing_residual(q, station, legacy)


def test_covariance_frame_is_part_of_event_payload():
    _, measurement = _scene("measurement")
    assert not measurements_are_exact_duplicates(
        measurement, replace(measurement, tangent_frame="prediction")
    )
    with pytest.raises(ValueError, match="tangent_frame"):
        replace(measurement, tangent_frame="automatic_candidate_switch")
