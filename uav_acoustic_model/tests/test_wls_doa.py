"""Tests for ideal-data direction recovery."""

import numpy as np
import pytest

from estimators.wls_doa import UnobservableGeometryError, estimate_doa_wls
from model.geometry import (
    all_pairs,
    baselines,
    comparison_arrays,
    direction_vector,
    reference_pairs,
    tetrahedral_array,
)
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


def test_far_field_wls_rejects_a_worse_successful_local_minimum():
    positions = tetrahedral_array()
    delays = np.asarray(
        [-0.0003974371512447112, -0.00042413768745976175, 0.00024386392339719736]
    )
    covariance = np.asarray(
        [
            [7.915899293289159e-10, 1.3782009220384502e-09, -7.387515036320398e-10],
            [1.3782009220384502e-09, 3.1044605429030584e-09, -2.470059099763395e-09],
            [-7.387515036320398e-10, -2.470059099763395e-09, 5.026725371748956e-09],
        ]
    )
    default = estimate_doa_wls(delays, positions, tdoa_covariance=covariance)
    independent_start = estimate_doa_wls(
        delays, positions, tdoa_covariance=covariance, initial_angles=(0.0, 1.0)
    )
    assert default.success and independent_start.success
    assert default.weighted_cost == pytest.approx(independent_start.weighted_cost, rel=1e-10)
    assert default.weighted_cost < 110.0


def test_zero_variance_tdoa_component_is_an_exact_constraint():
    positions = tetrahedral_array()
    pairs = reference_pairs(4)
    design = baselines(positions, pairs) / 343.0
    delays = design @ direction_vector(0.7, 0.4)
    delays[0] += 1e-4
    covariance = np.diag([0.0, 1e-10, 1e-10])
    estimate = estimate_doa_wls(
        delays, positions, pairs, tdoa_covariance=covariance
    )
    assert estimate.success
    predicted = design @ estimate.direction
    assert predicted[0] == pytest.approx(delays[0], abs=1e-11)


def test_incompatible_zero_covariance_tdoa_returns_explicit_invalid():
    positions = tetrahedral_array()
    pairs = reference_pairs(4)
    delays = np.zeros(3)
    delays[0] = 1.0  # far beyond the physical unit-vector range
    estimate = estimate_doa_wls(
        delays, positions, pairs, tdoa_covariance=np.zeros((3, 3))
    )
    assert not estimate.success
    assert estimate.invalid_reason == "incompatible_exact_constraints"
