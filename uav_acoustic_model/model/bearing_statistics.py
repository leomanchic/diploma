"""Spherical bearing residuals and calibrated tangent-plane statistics.

Residual coordinates are angular arc lengths in radians in the orthonormal
azimuth/elevation tangent basis at the true direction.  Covariances therefore
have units radian squared.  No temporal filtering or tracking is implemented.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from model.geometry import direction_angles


class AntipodalDirectionError(ValueError):
    """Raised because the sphere log-map is not unique at the antipode."""


def _unit_vector(value: ArrayLike, *, name: str) -> NDArray[np.float64]:
    vector = np.asarray(value, dtype=float)
    if vector.shape != (3,) or not np.all(np.isfinite(vector)):
        raise ValueError(f"{name} must be a finite three-dimensional vector")
    norm = float(np.linalg.norm(vector))
    if norm == 0.0:
        raise ValueError(f"{name} must be non-zero")
    return vector / norm


def tangent_basis(phi: float, elevation: float) -> NDArray[np.float64]:
    """Return rows ``[e_az, e_el]`` forming an orthonormal tangent basis."""

    phi_value = float(phi)
    elevation_value = float(elevation)
    if not np.isfinite(phi_value) or not np.isfinite(elevation_value):
        raise ValueError("angles must be finite")
    e_az = np.asarray([-np.sin(phi_value), np.cos(phi_value), 0.0])
    e_el = np.asarray(
        [
            -np.sin(elevation_value) * np.cos(phi_value),
            -np.sin(elevation_value) * np.sin(phi_value),
            np.cos(elevation_value),
        ]
    )
    return np.vstack((e_az, e_el))


def sphere_log_map(
    true_direction: ArrayLike,
    estimated_direction: ArrayLike,
    *,
    antipodal_tolerance: float = 1e-10,
) -> NDArray[np.float64]:
    """Return ``Log_u(u_hat)`` as a three-dimensional tangent vector.

    Input norms are irrelevant.  At (or numerically too close to) the
    antipode, the shortest geodesic is not unique, so the function raises
    :class:`AntipodalDirectionError` instead of choosing an arbitrary axis.
    """

    u = _unit_vector(true_direction, name="true_direction")
    estimate = _unit_vector(estimated_direction, name="estimated_direction")
    dot = float(np.clip(u @ estimate, -1.0, 1.0))
    tolerance = float(antipodal_tolerance)
    if not np.isfinite(tolerance) or tolerance <= 0.0:
        raise ValueError("antipodal_tolerance must be finite and positive")
    if dot <= -1.0 + tolerance:
        raise AntipodalDirectionError(
            "sphere log-map is not unique for nearly antipodal directions"
        )
    tangent = estimate - dot * u
    tangent_norm = float(np.linalg.norm(tangent))
    if tangent_norm <= np.finfo(float).eps * 8.0:
        return np.zeros(3, dtype=float)
    # atan2 retains the first-order angle when ``dot`` has already rounded
    # close to one; arccos(dot) loses several digits in this regime.
    theta = float(np.arctan2(tangent_norm, dot))
    # theta / ||projection|| is theta/sin(theta), but the projection norm is
    # numerically more accurate than sin(arccos(dot)) for small angles.
    return tangent * (theta / tangent_norm)


def tangent_residual(
    true_direction: ArrayLike,
    estimated_direction: ArrayLike,
    *,
    antipodal_tolerance: float = 1e-10,
) -> NDArray[np.float64]:
    """Return ``[e_az^T Log, e_el^T Log]`` in angular-arc radians."""

    u = _unit_vector(true_direction, name="true_direction")
    phi, elevation = direction_angles(u)
    return tangent_basis(phi, elevation) @ sphere_log_map(
        u, estimated_direction, antipodal_tolerance=antipodal_tolerance
    )


def tangent_residual_jacobian_wrt_true_direction(
    true_direction: ArrayLike,
    estimated_direction: ArrayLike,
    *,
    antipodal_tolerance: float = 1e-10,
    pole_tolerance: float = 1e-10,
) -> NDArray[np.float64]:
    """Differentiate the 2-D spherical residual with respect to its base.

    The returned ``2x3`` matrix acts on tangent perturbations of the unit
    ``true_direction``.  It includes the connection term caused by motion of
    the azimuth/elevation tangent basis.  The local azimuth basis is undefined
    at a pole, and the sphere log-map is non-unique at the antipode; both are
    rejected explicitly.
    """

    u = _unit_vector(true_direction, name="true_direction")
    y = _unit_vector(estimated_direction, name="estimated_direction")
    tolerance = float(antipodal_tolerance)
    pole_limit = float(pole_tolerance)
    if not np.isfinite(tolerance) or tolerance <= 0.0:
        raise ValueError("antipodal_tolerance must be finite and positive")
    if not np.isfinite(pole_limit) or pole_limit <= 0.0:
        raise ValueError("pole_tolerance must be finite and positive")
    phi, elevation = direction_angles(u)
    horizontal = float(np.hypot(u[0], u[1]))
    if horizontal <= pole_limit:
        raise ValueError("azimuth/elevation tangent basis is undefined at a local pole")
    basis = tangent_basis(phi, elevation)
    dot = float(np.clip(u @ y, -1.0, 1.0))
    projection = y - dot * u
    sine = float(np.linalg.norm(projection))
    if dot <= -1.0 + tolerance:
        raise AntipodalDirectionError(
            "sphere residual Jacobian is not unique for antipodal bearings"
        )
    if sine <= 1e-7:
        log_map = sphere_log_map(u, y, antipodal_tolerance=tolerance)
        log_derivative = -(np.eye(3) - u[:, None] * u[None, :])
    else:
        theta = float(np.arctan2(sine, dot))
        scale = theta / sine
        alpha = -1.0 / sine**2 + theta * dot / sine**3
        log_map = scale * projection
        log_derivative = (
            -scale * (u[:, None] * y[None, :] + dot * np.eye(3))
            + alpha * projection[:, None] * projection[None, :]
        )
    residual_uncorrected = basis @ log_map
    phi_gradient = np.asarray([-u[1], u[0], 0.0]) / horizontal**2
    connection = np.asarray(
        [
            np.sin(elevation) * residual_uncorrected[1],
            -np.sin(elevation) * residual_uncorrected[0],
        ]
    )[:, None] * phi_gradient[None, :]
    return basis @ log_derivative + connection


@dataclass(frozen=True)
class BearingCovarianceCalibration:
    """Sample mean/covariance and numerical diagnostics for valid residuals."""

    sample_count: int
    mean_residual_rad: NDArray[np.float64]
    covariance_rad2: NDArray[np.float64]
    eigenvalues_rad2: NDArray[np.float64]
    condition_number: float
    correlation: float
    rank: int
    symmetric: bool
    positive_semidefinite: bool


def calibrate_bearing_covariance(residuals: ArrayLike) -> BearingCovarianceCalibration:
    """Fit an unbiased 2-D sample covariance without arbitrary regularization."""

    values = np.asarray(residuals, dtype=float)
    if values.ndim != 2 or values.shape[1] != 2 or values.shape[0] < 2:
        raise ValueError("residuals must have shape (N, 2) with N >= 2")
    if not np.all(np.isfinite(values)):
        raise ValueError("calibration residuals must be finite")
    mean = np.mean(values, axis=0)
    covariance = np.asarray(np.cov(values, rowvar=False, ddof=1), dtype=float)
    covariance = 0.5 * (covariance + covariance.T)
    eigenvalues = np.linalg.eigvalsh(covariance)
    scale = max(float(np.max(np.abs(eigenvalues))), np.finfo(float).tiny)
    tolerance = max(1e-14 * scale, np.finfo(float).eps * scale * 32.0)
    positive_semidefinite = bool(np.min(eigenvalues) >= -tolerance)
    if not positive_semidefinite:
        raise ValueError("sample bearing covariance is not positive semidefinite")
    rank = int(np.count_nonzero(eigenvalues > tolerance))
    condition = (
        float(np.max(eigenvalues) / np.min(eigenvalues))
        if rank == 2 and np.min(eigenvalues) > 0.0
        else float("inf")
    )
    denominator = float(np.sqrt(covariance[0, 0] * covariance[1, 1]))
    correlation = float(covariance[0, 1] / denominator) if denominator > 0.0 else float("nan")
    return BearingCovarianceCalibration(
        sample_count=int(values.shape[0]),
        mean_residual_rad=mean,
        covariance_rad2=covariance,
        eigenvalues_rad2=eigenvalues,
        condition_number=condition,
        correlation=correlation,
        rank=rank,
        symmetric=bool(np.allclose(covariance, covariance.T, rtol=0.0, atol=tolerance)),
        positive_semidefinite=positive_semidefinite,
    )


def normalized_innovation_squared(
    residuals: ArrayLike, covariance_rad2: ArrayLike
) -> NDArray[np.float64]:
    """Return ``r.T @ R^+ @ r`` for one or many tangent residuals."""

    values = np.asarray(residuals, dtype=float)
    one = values.ndim == 1
    if one:
        values = values[None, :]
    covariance = np.asarray(covariance_rad2, dtype=float)
    if values.ndim != 2 or values.shape[1] != 2 or not np.all(np.isfinite(values)):
        raise ValueError("residuals must have finite shape (2,) or (N, 2)")
    if covariance.shape != (2, 2) or not np.all(np.isfinite(covariance)):
        raise ValueError("covariance_rad2 must be a finite 2x2 matrix")
    if not np.allclose(covariance, covariance.T, rtol=1e-12, atol=1e-18):
        raise ValueError("covariance_rad2 must be symmetric")
    eigenvalues = np.linalg.eigvalsh(covariance)
    tolerance = max(float(np.max(np.abs(eigenvalues))) * 1e-12, 1e-24)
    if np.min(eigenvalues) < -tolerance:
        raise ValueError("covariance_rad2 must be positive semidefinite")
    inverse = np.linalg.pinv(covariance, rcond=1e-12, hermitian=True)
    result = np.einsum("ni,ij,nj->n", values, inverse, values)
    result = np.maximum(result, 0.0)
    return result[0] if one else result


__all__ = [
    "AntipodalDirectionError",
    "BearingCovarianceCalibration",
    "calibrate_bearing_covariance",
    "normalized_innovation_squared",
    "sphere_log_map",
    "tangent_basis",
    "tangent_residual",
    "tangent_residual_jacobian_wrt_true_direction",
]
