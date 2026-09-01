"""Tests for direction geometry and both TDOA models."""

import numpy as np
import pytest

from model.geometry import (
    DEFAULT_APERTURE,
    DEFAULT_SOUND_SPEED,
    all_pairs,
    aperture,
    comparison_arrays,
    direction_angles,
    direction_vector,
    geodesic_angle_between_directions,
)
from model.tdoa import far_field_tdoa, spherical_tdoa


@pytest.mark.parametrize(
    ("phi", "elevation"),
    [(0.0, 0.0), (1.2, 0.4), (-2.4, 0.9), (np.pi, np.pi / 2.0)],
)
def test_direction_vector_has_unit_norm(phi, elevation):
    assert np.linalg.norm(direction_vector(phi, elevation)) == pytest.approx(1.0, abs=1e-15)


def test_direction_angle_round_trip():
    expected = (1.1, 0.35)
    actual = direction_angles(direction_vector(*expected))
    assert actual == pytest.approx(expected, abs=1e-14)


def test_geodesic_angle_retains_sub_sqrt_epsilon_resolution():
    angle = geodesic_angle_between_directions(
        np.array([1.0, 0.0, 0.0]), np.array([1.0, 1e-12, 0.0])
    )
    assert angle == pytest.approx(1e-12, rel=1e-12)


def test_spherical_tdoa_is_antisymmetric():
    positions = comparison_arrays()["tetrahedral"]
    source = np.asarray([3.0, -2.0, 4.0])
    tau_ij = spherical_tdoa(source, positions, [(1, 3)])[0]
    tau_ji = spherical_tdoa(source, positions, [(3, 1)])[0]
    assert tau_ij == pytest.approx(-tau_ji, abs=1e-15)


def test_tdoa_cycle_closure():
    positions = comparison_arrays()["square"]
    source = np.asarray([2.5, 1.0, 3.0])
    tau_ij, tau_jk, tau_ki = spherical_tdoa(source, positions, [(0, 1), (1, 2), (2, 0)])
    assert tau_ij + tau_jk + tau_ki == pytest.approx(0.0, abs=2e-18)


@pytest.mark.parametrize("array_name", ["linear", "L-shaped", "rectangle 3:1", "square", "tetrahedral"])
def test_far_field_delay_respects_geometric_bound(array_name):
    positions = comparison_arrays()[array_name]
    pairs = all_pairs(len(positions))
    delays = far_field_tdoa(1.37, 0.53, positions, pairs)
    spherical_delays = spherical_tdoa(
        4.0 * direction_vector(1.37, 0.53), positions, pairs
    )
    bounds = np.asarray(
        [np.linalg.norm(positions[first] - positions[second]) / DEFAULT_SOUND_SPEED for first, second in pairs]
    )
    assert np.all(np.abs(delays) <= bounds + 1e-18)
    assert np.all(np.abs(spherical_delays) <= bounds + 1e-18)


def test_sign_convention_for_source_to_the_right():
    distance = DEFAULT_APERTURE
    positions = np.asarray([[-distance / 2.0, 0.0, 0.0], [distance / 2.0, 0.0, 0.0]])
    pair = [(0, 1)]
    assert far_field_tdoa(0.0, 0.0, positions, pair)[0] == pytest.approx(
        distance / DEFAULT_SOUND_SPEED
    )
    assert spherical_tdoa([100.0, 0.0, 0.0], positions, pair)[0] > 0.0


def test_spherical_model_converges_to_far_field_as_range_grows():
    positions = comparison_arrays()["tetrahedral"]
    pairs = all_pairs(len(positions))
    phi, elevation = 0.73, 0.41
    unit_direction = direction_vector(phi, elevation)
    far_delays = far_field_tdoa(phi, elevation, positions, pairs)
    ranges = DEFAULT_APERTURE * np.asarray([5.0, 10.0, 20.0, 40.0])
    errors = np.asarray(
        [
            np.max(np.abs(spherical_tdoa(radius * unit_direction, positions, pairs) - far_delays))
            for radius in ranges
        ]
    )
    assert np.all(np.diff(errors) < 0.0)
    assert errors[-1] < errors[0] / 6.0


def test_all_comparison_arrays_have_four_mics_and_requested_aperture():
    for positions in comparison_arrays().values():
        assert positions.shape == (4, 3)
        assert aperture(positions) == pytest.approx(DEFAULT_APERTURE, rel=1e-14)
