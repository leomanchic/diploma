"""Covariance algebra and representation-invariance tests."""

import numpy as np
import pytest

from estimators.wls_doa import estimate_doa_wls
from model.geometry import (
    all_pairs,
    comparison_arrays,
    incidence_matrix,
    reference_pairs,
)
from model.jacobian import far_field_tdoa_jacobian
from model.statistics import fisher_information
from model.tdoa import (
    cycle_constraint_matrix,
    far_field_tdoa,
    independent_tdoa_covariance,
    tdoa_covariance_from_independent_toa,
)


M = 4
SIGMA_TOA = 23e-6


def _relative_matrix_error(first, second):
    return np.linalg.norm(first - second) / np.linalg.norm(first)


def _toa_information(positions, pairs, phi=0.7, elevation=0.45):
    covariance = tdoa_covariance_from_independent_toa(M, pairs, SIGMA_TOA)
    jacobian = far_field_tdoa_jacobian(phi, elevation, positions, pairs)
    return fisher_information(
        jacobian,
        tdoa_covariance=covariance,
        allow_singular_covariance=len(pairs) > M - 1,
    )


def test_reference_tdoa_covariance_from_iid_toa_has_expected_correlations():
    covariance = tdoa_covariance_from_independent_toa(
        M, reference_pairs(M, reference=0), SIGMA_TOA
    )
    normalised = covariance / SIGMA_TOA**2
    np.testing.assert_allclose(np.diag(normalised), np.full(M - 1, 2.0), atol=1e-14)
    off_diagonal = normalised[~np.eye(M - 1, dtype=bool)]
    np.testing.assert_allclose(off_diagonal, np.ones_like(off_diagonal), atol=1e-14)
    standard_deviations = np.sqrt(np.diag(covariance))
    correlation = covariance / np.outer(standard_deviations, standard_deviations)
    np.testing.assert_allclose(
        correlation[~np.eye(M - 1, dtype=bool)],
        np.full((M - 1) * (M - 2), 0.5),
        atol=1e-14,
    )


def test_full_pair_covariance_rank_and_cycle_nullspace():
    pairs = all_pairs(M)
    incidence = incidence_matrix(pairs, M)
    covariance = tdoa_covariance_from_independent_toa(M, pairs, SIGMA_TOA)
    cycles = cycle_constraint_matrix(pairs, M)
    pair_count = len(pairs)

    assert np.linalg.matrix_rank(covariance / SIGMA_TOA**2, tol=1e-10) == M - 1
    assert cycles.shape == (pair_count - M + 1, pair_count)
    assert np.linalg.matrix_rank(cycles) == pair_count - M + 1
    np.testing.assert_allclose(cycles @ incidence, 0.0, atol=1e-14)
    np.testing.assert_allclose(covariance @ cycles.T / SIGMA_TOA**2, 0.0, atol=1e-14)

    _, _, right_vectors = np.linalg.svd(covariance / SIGMA_TOA**2)
    numerical_nullspace = right_vectors[M - 1 :].T
    cycle_projector = cycles.T @ np.linalg.solve(cycles @ cycles.T, cycles)
    numerical_projector = numerical_nullspace @ numerical_nullspace.T
    np.testing.assert_allclose(cycle_projector, numerical_projector, atol=2e-14)


def test_independent_measured_tdoa_model_is_distinct_from_toa_induced_model():
    pairs = all_pairs(M)
    sigma_tdoa = np.sqrt(2.0) * SIGMA_TOA
    direct_covariance = independent_tdoa_covariance(len(pairs), sigma_tdoa)
    toa_induced = tdoa_covariance_from_independent_toa(M, pairs, SIGMA_TOA)
    cycles = cycle_constraint_matrix(pairs, M)
    assert np.linalg.matrix_rank(direct_covariance) == len(pairs)
    assert np.linalg.matrix_rank(toa_induced / SIGMA_TOA**2) == M - 1
    assert np.linalg.norm(direct_covariance @ cycles.T) > 0.0
    np.testing.assert_allclose(toa_induced @ cycles.T / SIGMA_TOA**2, 0.0, atol=1e-14)


@pytest.mark.parametrize("array_name", ["square", "tetrahedral"])
def test_fisher_information_is_invariant_to_equivalent_tdoa_representations(array_name):
    positions = comparison_arrays()[array_name]
    baseline = _toa_information(positions, all_pairs(M))

    for reference in range(M):
        candidate = _toa_information(positions, reference_pairs(M, reference))
        assert _relative_matrix_error(baseline, candidate) < 2e-12

    oriented_pairs = tuple(
        (second, first) if index % 2 else (first, second)
        for index, (first, second) in enumerate(all_pairs(M))
    )
    assert _relative_matrix_error(baseline, _toa_information(positions, oriented_pairs)) < 2e-12

    translated = positions + np.asarray([12.5, -4.0, 3.25])
    assert _relative_matrix_error(baseline, _toa_information(translated, all_pairs(M))) < 2e-12

    permutation = np.asarray([2, 0, 3, 1])
    permuted = positions[permutation]
    assert _relative_matrix_error(baseline, _toa_information(permuted, all_pairs(M))) < 2e-12


@pytest.mark.parametrize("array_name", ["square", "tetrahedral"])
def test_wls_is_invariant_to_equivalent_tdoa_representations(array_name):
    positions = comparison_arrays()[array_name]
    phi, elevation = np.deg2rad([47.0, 34.0])
    toa_errors = SIGMA_TOA * np.asarray([0.25, -0.8, 1.1, -0.35])

    def estimate(candidate_positions, pairs, candidate_toa_errors):
        incidence = incidence_matrix(pairs, M)
        measured = far_field_tdoa(phi, elevation, candidate_positions, pairs) + (
            incidence @ candidate_toa_errors
        )
        covariance = tdoa_covariance_from_independent_toa(M, pairs, SIGMA_TOA)
        return estimate_doa_wls(
            measured,
            candidate_positions,
            pairs,
            tdoa_covariance=covariance,
            initial_angles=(phi + 0.05, elevation - 0.04),
        ).direction

    baseline = estimate(positions, all_pairs(M), toa_errors)
    for reference in range(M):
        candidate = estimate(positions, reference_pairs(M, reference), toa_errors)
        np.testing.assert_allclose(candidate, baseline, rtol=0.0, atol=2e-9)

    oriented_pairs = tuple(
        (second, first) if index % 2 else (first, second)
        for index, (first, second) in enumerate(all_pairs(M))
    )
    np.testing.assert_allclose(
        estimate(positions, oriented_pairs, toa_errors), baseline, rtol=0.0, atol=2e-9
    )
    np.testing.assert_allclose(
        estimate(positions + [5.0, -2.0, 7.0], all_pairs(M), toa_errors),
        baseline,
        rtol=0.0,
        atol=2e-9,
    )

    permutation = np.asarray([2, 0, 3, 1])
    np.testing.assert_allclose(
        estimate(positions[permutation], all_pairs(M), toa_errors[permutation]),
        baseline,
        rtol=0.0,
        atol=2e-9,
    )
