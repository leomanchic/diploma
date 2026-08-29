"""Tests for analytic derivatives, information, and CRLB handling."""

import numpy as np
import pytest

from model.geometry import all_pairs, comparison_arrays, geometry_rank, reference_pairs
from model.jacobian import far_field_tdoa_jacobian, numerical_tdoa_jacobian
from model.statistics import (
    DEFAULT_SIGMA_TDOA,
    DegenerateInformationError,
    conditional_angular_crlb,
    conditional_crlb,
    fisher_information,
    matrix_diagnostics,
)
from model.tdoa import tdoa_covariance_from_toa


@pytest.mark.parametrize("array_name", ["square", "tetrahedral"])
@pytest.mark.parametrize(("phi", "elevation"), [(0.2, 0.15), (2.1, 0.7), (-1.3, 1.0)])
def test_analytic_jacobian_matches_central_difference(array_name, phi, elevation):
    positions = comparison_arrays()[array_name]
    analytic = far_field_tdoa_jacobian(phi, elevation, positions)
    numerical = numerical_tdoa_jacobian(phi, elevation, positions, step=1e-6)
    np.testing.assert_allclose(analytic, numerical, rtol=2e-9, atol=2e-13)


def test_fisher_information_is_symmetric_positive_semidefinite():
    positions = comparison_arrays()["square"]
    jacobian = far_field_tdoa_jacobian(0.8, 0.45, positions)
    information = fisher_information(jacobian, sigma_tdoa=DEFAULT_SIGMA_TDOA)
    np.testing.assert_allclose(information, information.T, atol=1e-14)
    assert np.min(np.linalg.eigvalsh(information)) >= -1e-12


def test_array_geometry_ranks_identify_linear_planar_and_spatial_arrays():
    arrays = comparison_arrays()
    assert geometry_rank(arrays["linear"]) == 1
    for name in ("L-shaped", "rectangle 3:1", "square"):
        assert geometry_rank(arrays[name]) == 2
    assert geometry_rank(arrays["tetrahedral"]) == 3


def test_information_diagnostics_detect_degenerate_geometries():
    arrays = comparison_arrays()
    line_information = fisher_information(far_field_tdoa_jacobian(0.7, 0.4, arrays["linear"]))
    planar_horizon_information = fisher_information(
        far_field_tdoa_jacobian(0.7, 0.0, arrays["square"])
    )
    spatial_information = fisher_information(
        far_field_tdoa_jacobian(0.7, 0.0, arrays["tetrahedral"])
    )
    assert matrix_diagnostics(line_information).rank == 1
    assert np.isinf(matrix_diagnostics(line_information).condition_number)
    assert matrix_diagnostics(planar_horizon_information).rank == 1
    assert matrix_diagnostics(spatial_information).rank == 2


def test_degenerate_crlb_raises_without_calling_ordinary_inverse(monkeypatch):
    positions = comparison_arrays()["linear"]
    information = fisher_information(far_field_tdoa_jacobian(0.7, 0.4, positions))

    def forbidden_inverse(*args, **kwargs):
        raise AssertionError("np.linalg.inv must not be called")

    monkeypatch.setattr(np.linalg, "inv", forbidden_inverse)
    with pytest.raises(DegenerateInformationError):
        conditional_crlb(information)


def test_finite_crlb_is_symmetric_positive_definite():
    positions = comparison_arrays()["tetrahedral"]
    information = fisher_information(far_field_tdoa_jacobian(-0.9, 0.35, positions))
    bound = conditional_crlb(information)
    np.testing.assert_allclose(bound, bound.T, atol=1e-15)
    assert np.min(np.linalg.eigvalsh(bound)) > 0.0


def test_coordinate_aware_angular_crlb_uses_local_spherical_metric():
    elevation = np.deg2rad(60.0)
    information = np.diag([4.0, 1.0])
    result = conditional_angular_crlb(information, elevation)
    expected = np.sqrt(np.cos(elevation) ** 2 * 0.25 + 1.0)
    assert result.full_rank
    assert result.angular_rms_rad == pytest.approx(expected)
    assert result.angular_rms_deg == pytest.approx(np.rad2deg(expected))


def test_degenerate_angular_crlb_reports_eigensystem_and_null_direction():
    information = np.asarray([[2.0, 2.0], [2.0, 2.0]])
    result = conditional_angular_crlb(information, np.deg2rad(30.0))
    assert not result.full_rank
    assert result.covariance is None
    assert result.angular_rms_rad is None
    assert result.angular_rms_deg is None
    assert result.eigenvalues.shape == (2,)
    assert result.unobservable_directions.shape == (1, 2)
    assert result.unobservable_tangent_directions.shape == (1, 2)
    assert np.linalg.norm(result.unobservable_tangent_directions[0]) == pytest.approx(1.0)
    np.testing.assert_allclose(
        information @ result.unobservable_directions[0], np.zeros(2), atol=1e-14
    )


def test_all_pair_pseudoinverse_matches_reference_pair_information():
    positions = comparison_arrays()["tetrahedral"]
    all_selected_pairs = all_pairs(len(positions))
    linearly_independent_pairs = reference_pairs(len(positions))
    sigma_toa = DEFAULT_SIGMA_TDOA / np.sqrt(2.0)
    toa_covariance = np.eye(len(positions)) * sigma_toa**2
    all_covariance = tdoa_covariance_from_toa(toa_covariance, all_selected_pairs)
    reference_covariance = tdoa_covariance_from_toa(
        toa_covariance, linearly_independent_pairs
    )
    all_information = fisher_information(
        far_field_tdoa_jacobian(0.4, 0.6, positions, all_selected_pairs),
        tdoa_covariance=all_covariance,
        allow_singular_covariance=True,
    )
    reference_information = fisher_information(
        far_field_tdoa_jacobian(0.4, 0.6, positions, linearly_independent_pairs),
        tdoa_covariance=reference_covariance,
    )
    np.testing.assert_allclose(all_information, reference_information, rtol=2e-12, atol=2e-12)
