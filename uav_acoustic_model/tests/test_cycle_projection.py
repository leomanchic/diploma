"""Weighted cycle-consistency projection tests."""

import numpy as np
import pytest

from estimators.cycle_projection import (
    DisconnectedPairGraphError,
    project_tdoa_cycles,
)
from model.geometry import all_pairs, incidence_matrix

M = 4
PAIRS = all_pairs(M)
B = incidence_matrix(PAIRS, M)


def test_ideal_tdoa_is_unchanged_by_cycle_projection():
    arrival_times = np.asarray([1.2, -0.7, 0.4, 2.1]) * 1e-4
    ideal = B @ arrival_times
    result = project_tdoa_cycles(ideal, PAIRS, M)
    np.testing.assert_allclose(result.consistent_tdoa, ideal, atol=2e-19)
    np.testing.assert_allclose(result.residual, 0.0, atol=2e-19)
    assert result.cycle_residual_before < 2e-19
    assert result.cycle_residual_after < 2e-19


def test_projection_enforces_cycles_for_inconsistent_measurements():
    measured = np.asarray([2.0, -1.0, 0.5, 1.7, -0.9, 2.4]) * 1e-4
    result = project_tdoa_cycles(measured, PAIRS, M)
    assert result.cycle_residual_before > 1e-5
    assert result.cycle_residual_after < 2e-19
    np.testing.assert_allclose(result.consistent_tdoa, B @ result.arrival_times, atol=0.0)


def test_weighted_residual_is_minimal_over_arrival_times():
    measured = np.asarray([2.0, -1.0, 0.5, 1.7, -0.9, 2.4]) * 1e-4
    mixing = np.asarray(
        [
            [1.0, 0.2, 0.0, 0.0, 0.0, 0.0],
            [0.1, 1.3, 0.1, 0.0, 0.0, 0.0],
            [0.0, 0.1, 0.9, 0.2, 0.0, 0.0],
            [0.0, 0.0, 0.2, 1.1, 0.1, 0.0],
            [0.0, 0.0, 0.0, 0.1, 0.8, 0.2],
            [0.0, 0.0, 0.0, 0.0, 0.2, 1.2],
        ]
    )
    weights = mixing @ mixing.T + 0.3 * np.eye(6)
    result = project_tdoa_cycles(measured, PAIRS, M, weights=weights)
    generator = np.random.default_rng(20260828)
    for _ in range(100):
        perturbation = generator.normal(0.0, 2e-5, size=M)
        candidate = result.arrival_times + perturbation
        residual = measured - B @ candidate
        candidate_cost = float(residual @ weights @ residual)
        assert result.weighted_cost <= candidate_cost + 2e-22


def test_pair_orientation_does_not_change_physical_projection():
    measured = np.asarray([2.0, -1.0, 0.5, 1.7, -0.9, 2.4]) * 1e-4
    generator = np.random.default_rng(17)
    matrix = generator.normal(size=(6, 6))
    weights = matrix @ matrix.T + np.eye(6)
    baseline = project_tdoa_cycles(measured, PAIRS, M, weights=weights)
    signs = np.asarray([1.0, -1.0, 1.0, -1.0, -1.0, 1.0])
    oriented_pairs = tuple(
        pair if sign > 0 else (pair[1], pair[0])
        for pair, sign in zip(PAIRS, signs, strict=True)
    )
    oriented = project_tdoa_cycles(
        signs * measured,
        oriented_pairs,
        M,
        weights=signs[:, None] * weights * signs[None, :],
    )
    np.testing.assert_allclose(oriented.arrival_times, baseline.arrival_times, atol=2e-19)
    np.testing.assert_allclose(
        oriented.consistent_tdoa, signs * baseline.consistent_tdoa, atol=2e-19
    )
    assert oriented.weighted_cost == pytest.approx(baseline.weighted_cost, abs=2e-24)


def test_disconnected_pair_graph_is_detected_explicitly():
    with pytest.raises(DisconnectedPairGraphError, match="disconnected"):
        project_tdoa_cycles([1e-5, -2e-5], ((0, 1), (2, 3)), M)


def test_diagonal_confidence_weights_are_supported():
    measured = np.asarray([2.0, -1.0, 0.5, 1.7, -0.9, 2.4]) * 1e-4
    result = project_tdoa_cycles(
        measured, PAIRS, M, weights=np.asarray([1.0, 2.0, 4.0, 3.0, 1.5, 0.8])
    )
    assert np.isfinite(result.weighted_cost)
    assert result.cycle_residual_after < 2e-19
