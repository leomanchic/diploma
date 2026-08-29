"""Known-range exact spherical WLS tests."""

import numpy as np
import pytest

from estimators.wls_doa import estimate_doa_spherical_wls, estimate_doa_wls
from model.geometry import all_pairs, comparison_arrays
from model.tdoa import directional_spherical_tdoa


@pytest.mark.parametrize("geometry", ["square", "tetrahedral"])
@pytest.mark.parametrize("distance_m", [5.0, 10.0, 50.0])
def test_exact_spherical_wls_recovers_ideal_direction_at_known_range(geometry, distance_m):
    positions = comparison_arrays()[geometry]
    pairs = all_pairs(4)
    phi, elevation = np.deg2rad([47.0, 31.0])
    tdoa = directional_spherical_tdoa(
        phi, elevation, distance_m, positions, pairs
    )
    estimate = estimate_doa_spherical_wls(
        tdoa,
        positions,
        distance_m,
        pairs,
        sigma_tdoa=1e-6,
    )
    expected = np.asarray(
        [np.cos(elevation) * np.cos(phi), np.cos(elevation) * np.sin(phi), np.sin(elevation)]
    )
    np.testing.assert_allclose(estimate.direction, expected, atol=2e-11, rtol=0.0)


def test_plane_estimator_retains_nonzero_near_field_model_bias():
    positions = comparison_arrays()["tetrahedral"]
    pairs = all_pairs(4)
    phi, elevation = np.deg2rad([47.0, 31.0])
    tdoa = directional_spherical_tdoa(phi, elevation, 5.0, positions, pairs)
    plane = estimate_doa_wls(tdoa, positions, pairs, sigma_tdoa=1e-6)
    exact = estimate_doa_spherical_wls(tdoa, positions, 5.0, pairs, sigma_tdoa=1e-6)
    truth = np.asarray(
        [np.cos(elevation) * np.cos(phi), np.cos(elevation) * np.sin(phi), np.sin(elevation)]
    )
    plane_error = np.rad2deg(np.arccos(np.clip(plane.direction @ truth, -1.0, 1.0)))
    exact_error = np.rad2deg(np.arccos(np.clip(exact.direction @ truth, -1.0, 1.0)))
    assert plane_error > 1e-3
    assert exact_error < 1e-6
