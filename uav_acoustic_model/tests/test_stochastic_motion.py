"""Exact discrete integrated-Wiener and Gaussian-bridge mathematics."""

import numpy as np
import pytest

from model.stochastic_motion import (
    acceleration_spectral_density, integrated_wiener_bridge,
    integrated_wiener_transition,
)


def test_integrated_wiener_blocks_and_semigroup():
    qc = np.asarray([[0.3, 0.02, 0.0], [0.02, 0.5, 0.01], [0.0, 0.01, 0.2]])
    a = 0.4
    b = 0.7
    f, q = integrated_wiener_transition(a + b, qc)
    fa, qa = integrated_wiener_transition(a, qc)
    fb, qb = integrated_wiener_transition(b, qc)
    np.testing.assert_allclose(f[:3, 3:], (a + b) * np.eye(3), rtol=0.0, atol=1e-15)
    np.testing.assert_allclose(q[:3, :3], (a + b)**3 / 3 * qc, rtol=0.0, atol=1e-15)
    np.testing.assert_allclose(q[:3, 3:], (a + b)**2 / 2 * qc, rtol=0.0, atol=1e-15)
    np.testing.assert_allclose(q[3:, 3:], (a + b) * qc, rtol=0.0, atol=1e-15)
    np.testing.assert_allclose(f, fb @ fa, rtol=0.0, atol=2e-15)
    np.testing.assert_allclose(q, fb @ qa @ fb.T + qb, rtol=0.0, atol=2e-15)
    assert np.linalg.eigvalsh(q)[0] >= -1e-14


def test_bridge_conditional_moments_and_q_zero_limit():
    qc = np.eye(3) * 0.25
    a0, a1, bridge = integrated_wiener_bridge(0.2, 0.5, qc)
    assert np.linalg.eigvalsh(bridge)[0] >= -1e-14
    f, _ = integrated_wiener_transition(0.2, qc)
    whole, _ = integrated_wiener_transition(0.5, qc)
    np.testing.assert_allclose(a0 + a1 @ whole, f, rtol=0.0, atol=1e-13)
    zero_a0, zero_a1, zero_bridge = integrated_wiener_bridge(0.2, 0.5, np.zeros((3, 3)))
    np.testing.assert_allclose(zero_a0, f, rtol=0.0, atol=0.0)
    np.testing.assert_allclose(zero_a1, 0, rtol=0.0, atol=0.0)
    np.testing.assert_allclose(zero_bridge, 0, rtol=0.0, atol=0.0)


def test_singular_nonzero_spectral_density_is_explicitly_unsupported():
    with pytest.raises(ValueError, match="positive definite"):
        integrated_wiener_bridge(0.2, 0.5, np.diag([1.0, 0.0, 1.0]))


def test_even_small_negative_qc_eigenvalue_is_not_silently_regularized():
    with pytest.raises(ValueError, match="positive semidefinite"):
        acceleration_spectral_density(np.diag([0.25, 0.25, -1e-12]))
