"""Tests for SI-valued, strictly subsonic source trajectories."""

import numpy as np
import pytest

from simulation.trajectory import (
    CircularTrajectory,
    ConstantVelocityTrajectory,
    PiecewiseLinearTrajectory,
    StationaryTrajectory,
)


def test_stationary_and_constant_velocity_scalar_and_vector_api():
    stationary = StationaryTrajectory([1.0, 2.0, 3.0])
    np.testing.assert_allclose(stationary.q([-2.0, 4.0]), [[1, 2, 3], [1, 2, 3]])
    np.testing.assert_array_equal(stationary.v(1.5), np.zeros(3))
    moving = ConstantVelocityTrajectory([1, -2, 3], [4, 5, -6], reference_time_s=2)
    np.testing.assert_allclose(moving.q(3.5), [7.0, 5.5, -6.0])
    np.testing.assert_allclose(moving.v([0.0, 1.0]), [[4, 5, -6], [4, 5, -6]])
    np.testing.assert_array_equal(moving.a(9.0), np.zeros(3))


def test_circular_trajectory_geometry_derivatives_and_speed():
    trajectory = CircularTrajectory(
        center_m=[1.0, -2.0, 0.5], radius_m=5.0, angular_speed_rad_s=2.0
    )
    times = np.linspace(-0.3, 0.7, 9)
    positions = trajectory.q(times)
    velocities = trajectory.v(times)
    accelerations = trajectory.a(times)
    np.testing.assert_allclose(np.linalg.norm(positions - trajectory.center_m, axis=1), 5.0)
    np.testing.assert_allclose(np.linalg.norm(velocities, axis=1), 10.0)
    np.testing.assert_allclose(
        accelerations, -4.0 * (positions - trajectory.center_m), atol=2e-14
    )
    step = 1e-6
    numerical = (trajectory.q(times + step) - trajectory.q(times - step)) / (2 * step)
    np.testing.assert_allclose(numerical, velocities, rtol=3e-10, atol=3e-9)


def test_piecewise_linear_interpolation_velocity_support_and_extrapolation():
    times = [0.0, 2.0, 5.0]
    positions = [[0, 0, 0], [2, 4, 0], [2, 7, 6]]
    trajectory = PiecewiseLinearTrajectory(times, positions)
    np.testing.assert_allclose(trajectory.q([1.0, 3.5]), [[1, 2, 0], [2, 5.5, 3]])
    np.testing.assert_allclose(trajectory.v([1.0, 3.5]), [[1, 2, 0], [0, 1, 2]])
    np.testing.assert_array_equal(trajectory.a([1.0, 3.5]), np.zeros((2, 3)))
    with pytest.raises(ValueError, match="outside"):
        trajectory.q(-0.1)
    extended = PiecewiseLinearTrajectory(times, positions, extrapolate=True)
    np.testing.assert_allclose(extended.q([-1.0, 6.0]), [[-1, -2, 0], [2, 8, 8]])


@pytest.mark.parametrize(
    "factory",
    [
        lambda: ConstantVelocityTrajectory([0, 0, 0], [343.0, 0, 0]),
        lambda: CircularTrajectory([0, 0, 0], 7.0, 49.0),
        lambda: PiecewiseLinearTrajectory([0, 1], [[0, 0, 0], [343, 0, 0]]),
    ],
)
def test_sonic_or_supersonic_trajectories_are_rejected(factory):
    with pytest.raises(ValueError, match=r"\|v\| < sound_speed"):
        factory()


def test_invalid_trajectory_inputs_are_rejected():
    with pytest.raises(ValueError):
        StationaryTrajectory([1, 2])
    with pytest.raises(ValueError):
        CircularTrajectory([0, 0, 0], -1, 1)
    with pytest.raises(ValueError):
        PiecewiseLinearTrajectory([0, 0], [[0, 0, 0], [1, 0, 0]])
