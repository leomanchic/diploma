"""Integrated-Wiener acceleration transition for an ENU [q, v] state.

``Qc = L L.T`` has units m²/s³: ``dv=L dW`` has velocity units m/s.
This is filter uncertainty, not the deterministic acceleration prescribed by a
synthetic manoeuvre trajectory.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray


def acceleration_spectral_density(value: ArrayLike) -> NDArray[np.float64]:
    """Validate a symmetric positive-semidefinite 3x3 ``Qc`` in m²/s³."""

    matrix = np.asarray(value, dtype=float)
    if matrix.shape != (3, 3) or not np.all(np.isfinite(matrix)):
        raise ValueError("Qc must be a finite 3x3 matrix")
    if not np.allclose(matrix, matrix.T, rtol=0.0, atol=1e-14):
        raise ValueError("Qc must be symmetric")
    matrix = 0.5 * (matrix + matrix.T)
    eigenvalues = np.linalg.eigvalsh(matrix)
    if eigenvalues[0] < 0.0:
        raise ValueError("Qc must be positive semidefinite")
    result = matrix.copy()
    result.setflags(write=False)
    return result


def integrated_wiener_transition(
    dt_s: float, qc_m2_s3: ArrayLike
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Return exact ``F(dt), Qd(dt)`` for ``dq=v dt, dv=L dW``.

    ``Qd`` is in mixed SI units: position blocks m², cross blocks m²/s,
    and velocity blocks m²/s².  Negative intervals are not causal steps.
    """

    dt = float(dt_s)
    if not np.isfinite(dt) or dt < 0.0:
        raise ValueError("dt_s must be finite and non-negative")
    qc = acceleration_spectral_density(qc_m2_s3)
    f = np.eye(6)
    f[:3, 3:] = dt * np.eye(3)
    qd = np.zeros((6, 6))
    qd[:3, :3] = dt**3 / 3.0 * qc
    qd[:3, 3:] = dt**2 / 2.0 * qc
    qd[3:, :3] = qd[:3, 3:]
    qd[3:, 3:] = dt * qc
    return f, qd


def integrated_wiener_bridge(
    elapsed_s: float, interval_s: float, qc_m2_s3: ArrayLike
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Conditional distribution of an interior state given interval endpoints.

    ``E[x(t)|x0,x1] = A0*x0 + A1*x1`` and conditional covariance ``B``.
    For full-rank ``Qc`` the bridge uses a direct solve with ``Qd(h)``.
    ``Qc=0`` gives the deterministic CV limit with zero bridge covariance.
    A singular nonzero ``Qc`` is not yet supported by this history variant.
    """

    h = float(interval_s)
    s = float(elapsed_s)
    if not np.isfinite(h) or h <= 0.0 or not np.isfinite(s) or s < 0.0 or s > h:
        raise ValueError("bridge requires 0 <= elapsed_s <= interval_s and interval_s > 0")
    qc = acceleration_spectral_density(qc_m2_s3)
    fs, qs = integrated_wiener_transition(s, qc)
    fh, qh = integrated_wiener_transition(h, qc)
    if np.all(qc == 0.0):
        return fs, np.zeros((6, 6)), np.zeros((6, 6))
    if np.linalg.matrix_rank(qc) != 3:
        raise ValueError("nonzero Qc must be positive definite for Gaussian bridge")
    fremain, _ = integrated_wiener_transition(h - s, qc)
    cross = qs @ fremain.T
    a1 = np.linalg.solve(qh, cross.T).T
    a0 = fs - a1 @ fh
    b = 0.5 * ((qs - a1 @ cross.T) + (qs - a1 @ cross.T).T)
    if np.linalg.eigvalsh(b)[0] < -1e-11 * max(1.0, np.linalg.norm(qs)):
        raise ArithmeticError("Gaussian bridge covariance lost PSD")
    return a0, a1, b


__all__ = [
    "acceleration_spectral_density",
    "integrated_wiener_bridge",
    "integrated_wiener_transition",
]
