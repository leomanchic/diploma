"""Fisher information and conditional angular CRLB utilities."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

DEFAULT_SIGMA_TDOA = 50e-6


class DegenerateInformationError(np.linalg.LinAlgError):
    """Raised when a finite two-parameter CRLB does not exist."""


@dataclass(frozen=True)
class InformationDiagnostics:
    """Numerical observability diagnostics for a symmetric information matrix."""

    rank: int
    dimension: int
    condition_number: float
    eigenvalues: NDArray[np.float64]
    eigenvectors: NDArray[np.float64]

    @property
    def is_degenerate(self) -> bool:
        return self.rank < self.dimension


@dataclass(frozen=True)
class AngularCRLBResult:
    """Coordinate-aware angular CRLB or a rank-deficiency diagnosis."""

    full_rank: bool
    covariance: NDArray[np.float64] | None
    azimuth_std_rad: float | None
    elevation_std_rad: float | None
    angular_rms_rad: float | None
    azimuth_std_deg: float | None
    elevation_std_deg: float | None
    angular_rms_deg: float | None
    eigenvalues: NDArray[np.float64]
    unobservable_directions: NDArray[np.float64]
    unobservable_tangent_directions: NDArray[np.float64]


def _symmetric_matrix(matrix: ArrayLike, name: str) -> NDArray[np.float64]:
    result = np.asarray(matrix, dtype=float)
    if result.ndim != 2 or result.shape[0] != result.shape[1]:
        raise ValueError(f"{name} must be square")
    if not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must be finite")
    if not np.allclose(result, result.T, rtol=1e-10, atol=1e-14):
        raise ValueError(f"{name} must be symmetric")
    return (result + result.T) / 2.0


def _rank_tolerance(
    singular_values: NDArray[np.float64],
    shape: tuple[int, int],
    rtol: float | None,
) -> float:
    if singular_values.size == 0:
        return 0.0
    relative = max(shape) * np.finfo(float).eps if rtol is None else float(rtol)
    if not np.isfinite(relative) or relative < 0.0:
        raise ValueError("rtol must be finite and non-negative")
    return relative * float(singular_values[0])


def matrix_diagnostics(
    matrix: ArrayLike,
    rtol: float | None = 1e-10,
) -> InformationDiagnostics:
    """Return rank, condition number, and ordered symmetric eigensystem."""

    symmetric = _symmetric_matrix(matrix, "matrix")
    singular_values = np.linalg.svd(symmetric, compute_uv=False)
    tolerance = _rank_tolerance(singular_values, symmetric.shape, rtol)
    rank = int(np.count_nonzero(singular_values > tolerance))
    condition = (
        float(np.inf)
        if rank < symmetric.shape[0]
        else float(singular_values[0] / singular_values[-1])
    )
    eigenvalues, eigenvectors = np.linalg.eigh(symmetric)
    return InformationDiagnostics(
        rank=rank,
        dimension=symmetric.shape[0],
        condition_number=condition,
        eigenvalues=eigenvalues,
        eigenvectors=eigenvectors,
    )


def covariance_precision(
    covariance: ArrayLike,
    *,
    allow_singular: bool = False,
    rtol: float = 1e-10,
) -> NDArray[np.float64]:
    """Return covariance precision, optionally as a spectral pseudoinverse."""

    symmetric = _symmetric_matrix(covariance, "covariance")
    eigenvalues, eigenvectors = np.linalg.eigh(symmetric)
    scale = float(np.max(np.abs(eigenvalues)))
    tolerance = rtol * scale
    if np.min(eigenvalues) < -tolerance:
        raise ValueError("covariance must be positive semidefinite")
    positive = eigenvalues > tolerance
    if not np.all(positive) and not allow_singular:
        raise np.linalg.LinAlgError("covariance is singular")
    inverse_values = np.zeros_like(eigenvalues)
    inverse_values[positive] = 1.0 / eigenvalues[positive]
    return (eigenvectors * inverse_values) @ eigenvectors.T


def fisher_information(
    jacobian: ArrayLike,
    *,
    sigma_tdoa: float | None = None,
    tdoa_covariance: ArrayLike | None = None,
    allow_singular_covariance: bool = False,
) -> NDArray[np.float64]:
    """Return conditional Fisher information for one explicit noise model.

    With ``tdoa_covariance=None``, errors of the selected measured TDOAs are
    independent with standard deviation ``sigma_tdoa`` (50 microseconds by
    default). A supplied covariance represents a separate, possibly
    TOA-induced model; specifying both is rejected to prevent ambiguity.
    """

    matrix = np.asarray(jacobian, dtype=float)
    if matrix.ndim != 2 or matrix.shape[1] != 2 or not np.all(np.isfinite(matrix)):
        raise ValueError("jacobian must be a finite matrix with shape (N, 2)")
    if tdoa_covariance is None:
        standard_deviation = (
            DEFAULT_SIGMA_TDOA if sigma_tdoa is None else float(sigma_tdoa)
        )
        if not np.isfinite(standard_deviation) or standard_deviation <= 0.0:
            raise ValueError("sigma_tdoa must be finite and positive")
        information = matrix.T @ matrix / standard_deviation**2
    else:
        if sigma_tdoa is not None:
            raise ValueError("specify either sigma_tdoa or tdoa_covariance, not both")
        covariance_array = np.asarray(tdoa_covariance, dtype=float)
        if covariance_array.shape != (matrix.shape[0], matrix.shape[0]):
            raise ValueError("covariance shape does not match the Jacobian")
        precision = covariance_precision(
            covariance_array, allow_singular=allow_singular_covariance
        )
        information = matrix.T @ precision @ matrix
    return (information + information.T) / 2.0


def conditional_crlb(
    information: ArrayLike,
    *,
    rtol: float = 1e-10,
) -> NDArray[np.float64]:
    """Solve for a finite CRLB after rejecting rank deficiency."""

    matrix = _symmetric_matrix(information, "information")
    diagnostics = matrix_diagnostics(matrix, rtol=rtol)
    eigen_scale = float(np.max(np.abs(diagnostics.eigenvalues)))
    tolerance = rtol * eigen_scale
    if float(np.min(diagnostics.eigenvalues)) < -tolerance:
        raise ValueError("information must be positive semidefinite")
    if diagnostics.is_degenerate or float(np.min(diagnostics.eigenvalues)) <= tolerance:
        raise DegenerateInformationError(
            f"information matrix has rank {diagnostics.rank}/{diagnostics.dimension}"
        )
    bound = np.linalg.solve(matrix, np.eye(matrix.shape[0]))
    return (bound + bound.T) / 2.0


def angular_covariance_metrics(
    covariance: ArrayLike,
    elevation: float,
) -> dict[str, float]:
    """Return coordinate and local geodesic RMS values in radians/degrees."""

    matrix = _symmetric_matrix(covariance, "covariance")
    if matrix.shape != (2, 2):
        raise ValueError("angular covariance must have shape (2, 2)")
    eigenvalues = np.linalg.eigvalsh(matrix)
    tolerance = 1e-12 * float(np.max(np.abs(eigenvalues)))
    if float(np.min(eigenvalues)) < -tolerance:
        raise ValueError("angular covariance must be positive semidefinite")
    azimuth_std = float(np.sqrt(max(matrix[0, 0], 0.0)))
    elevation_std = float(np.sqrt(max(matrix[1, 1], 0.0)))
    angular_variance = np.cos(float(elevation)) ** 2 * matrix[0, 0] + matrix[1, 1]
    angular_rms = float(np.sqrt(max(angular_variance, 0.0)))
    degree_scale = 180.0 / np.pi
    return {
        "azimuth_std_rad": azimuth_std,
        "elevation_std_rad": elevation_std,
        "angular_rms_rad": angular_rms,
        "azimuth_std_deg": azimuth_std * degree_scale,
        "elevation_std_deg": elevation_std * degree_scale,
        "angular_rms_deg": angular_rms * degree_scale,
    }


def conditional_angular_crlb(
    information: ArrayLike,
    elevation: float,
    *,
    rtol: float = 1e-10,
) -> AngularCRLBResult:
    """Return coordinate-aware angular CRLB or explicit degeneracy details.

    The local geodesic metric is
    ``sqrt(cos(elevation)^2 C_phi_phi + C_elevation_elevation)``.
    No finite total is returned unless the information has full rank.
    """

    matrix = _symmetric_matrix(information, "information")
    if matrix.shape != (2, 2):
        raise ValueError("angular information must have shape (2, 2)")
    diagnostics = matrix_diagnostics(matrix, rtol=rtol)
    scale = float(np.max(np.abs(diagnostics.eigenvalues)))
    tolerance = rtol * scale
    if float(np.min(diagnostics.eigenvalues)) < -tolerance:
        raise ValueError("information must be positive semidefinite")
    unobservable = diagnostics.eigenvectors[:, np.abs(diagnostics.eigenvalues) <= tolerance].T
    tangent_unobservable = unobservable.copy()
    if tangent_unobservable.size:
        tangent_unobservable[:, 0] *= np.cos(float(elevation))
        tangent_norms = np.linalg.norm(tangent_unobservable, axis=1)
        nonzero = tangent_norms > 0.0
        tangent_unobservable[nonzero] /= tangent_norms[nonzero, None]
    if diagnostics.is_degenerate:
        return AngularCRLBResult(
            full_rank=False,
            covariance=None,
            azimuth_std_rad=None,
            elevation_std_rad=None,
            angular_rms_rad=None,
            azimuth_std_deg=None,
            elevation_std_deg=None,
            angular_rms_deg=None,
            eigenvalues=diagnostics.eigenvalues,
            unobservable_directions=unobservable,
            unobservable_tangent_directions=tangent_unobservable,
        )

    bound = conditional_crlb(matrix, rtol=rtol)
    metrics = angular_covariance_metrics(bound, elevation)
    return AngularCRLBResult(
        full_rank=True,
        covariance=bound,
        eigenvalues=diagnostics.eigenvalues,
        unobservable_directions=np.empty((0, 2), dtype=float),
        unobservable_tangent_directions=np.empty((0, 2), dtype=float),
        **metrics,
    )
