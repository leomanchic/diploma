"""Tests for ideal-data direction recovery."""

import numpy as np
import pytest

from estimators.wls_doa import UnobservableGeometryError, estimate_doa_wls
from model.geometry import all_pairs, comparison_arrays, direction_vector
from model.tdoa import far_field_tdoa, tdoa_covariance_from_toa


def angular_separation(first, second):
    return float(np.arccos(np.clip(np.dot(first, second), -1.0, 1.0)))


@pytest.mark.parametrize(
    ("phi", "elevation"), [(0.35, 0.2), (2.7, 0.75), (-2.4, 1.0)]
)
def test_ideal_tdoa_recovers_direction_for_tetrahedron(phi, elevation):
    positions = comparison_arrays()["tetrahedral"]
    delays = far_field_tdoa(phi, elevation, positions)
    estimate = estimate_doa_wls(delays, positions)
    assert estimate.success
    assert not estimate.mirror_ambiguous
    assert angular_separation(estimate.direction, direction_vector(phi, elevation)) < 2e-8


def test_ideal_planar_tdoa_recovers_upper_hemisphere_branch():
    phi, elevation = 1.4, 0.55
    positions = comparison_arrays()["square"]
    estimate = estimate_doa_wls(far_field_tdoa(phi, elevation, positions), positions)
    assert estimate.mirror_ambiguous
    assert angular_separation(estimate.direction, direction_vector(phi, elevation)) < 2e-8
    mirrored_delays = far_field_tdoa(phi, -elevation, positions)
    np.testing.assert_allclose(mirrored_delays, far_field_tdoa(phi, elevation, positions), atol=1e-18)


def test_linear_array_is_rejected_as_unobservable():
    positions = comparison_arrays()["linear"]
    delays = far_field_tdoa(0.8, 0.4, positions)
    with pytest.raises(UnobservableGeometryError):
        estimate_doa_wls(delays, positions)


def test_wls_accepts_all_pairs_with_singular_tdoa_covariance():
    phi, elevation = -0.7, 0.6
    positions = comparison_arrays()["tetrahedral"]
    pairs = all_pairs(len(positions))
    delays = far_field_tdoa(phi, elevation, positions, pairs)
    toa_covariance = np.eye(len(positions)) * (50e-6 / np.sqrt(2.0)) ** 2
    covariance = tdoa_covariance_from_toa(toa_covariance, pairs)
    estimate = estimate_doa_wls(delays, positions, pairs, tdoa_covariance=covariance)
    assert angular_separation(estimate.direction, direction_vector(phi, elevation)) < 2e-8
