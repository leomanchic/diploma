"""Monte Carlo validation of the far-field WLS estimator against CRLB."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray

from estimators.wls_doa import estimate_doa_wls
from model.geometry import (
    incidence_matrix,
    microphone_positions,
    reference_pairs,
    validate_pairs,
)
from model.jacobian import far_field_tdoa_jacobian
from model.statistics import conditional_angular_crlb, fisher_information, matrix_diagnostics
from model.tdoa import (
    far_field_tdoa,
    independent_tdoa_covariance,
    tdoa_covariance_from_independent_toa,
)

NoiseModel = Literal["independent_tdoa", "independent_toa"]


@dataclass(frozen=True)
class MonteCarloResult:
    """Aggregate metrics from one array/direction/noise configuration."""

    array_name: str
    noise_model: NoiseModel
    phi_rad: float
    elevation_rad: float
    sigma_tdoa: float | None
    sigma_toa: float | None
    marginal_tdoa_std: float
    n_trials: int
    seed: int
    bias_rad: NDArray[np.float64]
    empirical_covariance: NDArray[np.float64]
    crlb_covariance: NDArray[np.float64]
    normalized_covariance: NDArray[np.float64]
    normalized_covariance_eigenvalues: NDArray[np.float64]
    normalized_covariance_frobenius_error: float
    whitened_bias_norm: float
    physical_bias_rad: float
    geodesic_rmse_rad: float
    median_geodesic_error_rad: float
    p95_geodesic_error_rad: float
    fraction_over_10deg: float
    angular_crlb_rms_rad: float
    rmse_to_crlb_ratio: float
    optimizer_success_fraction: float
    boundary_fraction: float
    fisher_rank: int
    fisher_condition_number: float
    fisher_eigenvalues: NDArray[np.float64]

    def to_record(self) -> dict[str, float | int | str | None]:
        """Flatten the result into a deterministic CSV-ready record."""

        degrees = 180.0 / np.pi
        return {
            "array": self.array_name,
            "noise_model": self.noise_model,
            "azimuth_deg": self.phi_rad * degrees,
            "elevation_deg": self.elevation_rad * degrees,
            "sigma_tdoa_us": None if self.sigma_tdoa is None else self.sigma_tdoa * 1e6,
            "sigma_toa_us": None if self.sigma_toa is None else self.sigma_toa * 1e6,
            "marginal_tdoa_std_us": self.marginal_tdoa_std * 1e6,
            "n_trials": self.n_trials,
            "seed": self.seed,
            "bias_azimuth_rad": self.bias_rad[0],
            "bias_elevation_rad": self.bias_rad[1],
            "bias_azimuth_deg": self.bias_rad[0] * degrees,
            "bias_elevation_deg": self.bias_rad[1] * degrees,
            "physical_bias_rad": self.physical_bias_rad,
            "physical_bias_deg": self.physical_bias_rad * degrees,
            "emp_cov_phi_phi_rad2": self.empirical_covariance[0, 0],
            "emp_cov_phi_elevation_rad2": self.empirical_covariance[0, 1],
            "emp_cov_elevation_elevation_rad2": self.empirical_covariance[1, 1],
            "crlb_phi_phi_rad2": self.crlb_covariance[0, 0],
            "crlb_phi_elevation_rad2": self.crlb_covariance[0, 1],
            "crlb_elevation_elevation_rad2": self.crlb_covariance[1, 1],
            "normalized_cov_eigenvalue_min": self.normalized_covariance_eigenvalues[0],
            "normalized_cov_eigenvalue_max": self.normalized_covariance_eigenvalues[-1],
            "normalized_cov_frobenius_error": self.normalized_covariance_frobenius_error,
            "whitened_bias_norm": self.whitened_bias_norm,
            "geodesic_rmse_rad": self.geodesic_rmse_rad,
            "geodesic_rmse_deg": self.geodesic_rmse_rad * degrees,
            "median_geodesic_error_rad": self.median_geodesic_error_rad,
            "median_geodesic_error_deg": self.median_geodesic_error_rad * degrees,
            "p95_geodesic_error_rad": self.p95_geodesic_error_rad,
            "p95_geodesic_error_deg": self.p95_geodesic_error_rad * degrees,
            "fraction_over_10deg": self.fraction_over_10deg,
            "angular_crlb_rms_rad": self.angular_crlb_rms_rad,
            "angular_crlb_rms_deg": self.angular_crlb_rms_rad * degrees,
            "rmse_to_crlb_ratio": self.rmse_to_crlb_ratio,
            "optimizer_success_fraction": self.optimizer_success_fraction,
            "boundary_fraction": self.boundary_fraction,
            "fisher_rank": self.fisher_rank,
            "fisher_condition_number": self.fisher_condition_number,
            "fisher_eigenvalue_min": self.fisher_eigenvalues[0],
            "fisher_eigenvalue_max": self.fisher_eigenvalues[-1],
        }


def _wrap_angle(angle: NDArray[np.float64] | float) -> NDArray[np.float64] | float:
    return np.arctan2(np.sin(angle), np.cos(angle))


def _inverse_square_root(matrix: NDArray[np.float64]) -> NDArray[np.float64]:
    eigenvalues, eigenvectors = np.linalg.eigh((matrix + matrix.T) / 2.0)
    if float(np.min(eigenvalues)) <= 0.0:
        raise np.linalg.LinAlgError("matrix must be positive definite")
    return (eigenvectors / np.sqrt(eigenvalues)) @ eigenvectors.T


def run_monte_carlo_wls(
    positions: ArrayLike,
    phi: float,
    elevation: float,
    *,
    array_name: str = "array",
    noise_model: NoiseModel = "independent_tdoa",
    sigma_tdoa: float | None = None,
    sigma_toa: float | None = None,
    pairs=None,
    n_trials: int = 1000,
    seed: int = 20260827,
) -> MonteCarloResult:
    """Run a deterministic WLS Monte Carlo experiment for one configuration.

    ``independent_tdoa`` generates errors directly in the selected TDOAs as
    ``N(0, sigma_tdoa**2 I)``. ``independent_toa`` generates one independent
    error per microphone and differences it with the selected incidence matrix.
    """

    coordinates = microphone_positions(positions)
    selected_pairs = (
        reference_pairs(len(coordinates))
        if pairs is None
        else validate_pairs(pairs, len(coordinates))
    )
    if int(n_trials) != n_trials or n_trials < 2:
        raise ValueError("n_trials must be an integer of at least 2")
    n_trials = int(n_trials)
    rng = np.random.default_rng(seed)
    truth = far_field_tdoa(phi, elevation, coordinates, selected_pairs)
    jacobian = far_field_tdoa_jacobian(phi, elevation, coordinates, selected_pairs)

    if noise_model == "independent_tdoa":
        if sigma_toa is not None:
            raise ValueError("sigma_toa is not part of the independent_tdoa model")
        if sigma_tdoa is None or not np.isfinite(sigma_tdoa) or sigma_tdoa <= 0.0:
            raise ValueError("positive sigma_tdoa is required")
        tdoa_covariance = independent_tdoa_covariance(len(selected_pairs), sigma_tdoa)
        noise = rng.normal(0.0, sigma_tdoa, size=(n_trials, len(selected_pairs)))
        information = fisher_information(jacobian, sigma_tdoa=sigma_tdoa)
        marginal_tdoa_std = float(sigma_tdoa)
    elif noise_model == "independent_toa":
        if sigma_tdoa is not None:
            raise ValueError("sigma_tdoa is not part of the independent_toa model")
        if sigma_toa is None or not np.isfinite(sigma_toa) or sigma_toa <= 0.0:
            raise ValueError("positive sigma_toa is required")
        incidence = incidence_matrix(selected_pairs, len(coordinates))
        toa_errors = rng.normal(0.0, sigma_toa, size=(n_trials, len(coordinates)))
        noise = toa_errors @ incidence.T
        tdoa_covariance = tdoa_covariance_from_independent_toa(
            len(coordinates), selected_pairs, sigma_toa
        )
        information = fisher_information(jacobian, tdoa_covariance=tdoa_covariance)
        marginal_tdoa_std = float(np.sqrt(np.mean(np.diag(tdoa_covariance))))
    else:
        raise ValueError(f"unsupported noise_model: {noise_model}")

    crlb = conditional_angular_crlb(information, elevation)
    if not crlb.full_rank or crlb.covariance is None or crlb.angular_rms_rad is None:
        raise ValueError(
            "Monte Carlo/finite-CRLB comparison requires full-rank angular information"
        )

    estimated_angles = np.empty((n_trials, 2), dtype=float)
    estimated_directions = np.empty((n_trials, 3), dtype=float)
    successes = np.empty(n_trials, dtype=bool)
    boundaries = np.empty(n_trials, dtype=bool)
    boundary_tolerance = 1e-9
    for index in range(n_trials):
        measured = truth + noise[index]
        if noise_model == "independent_tdoa":
            estimate = estimate_doa_wls(
                measured,
                coordinates,
                selected_pairs,
                sigma_tdoa=sigma_tdoa,
            )
        else:
            estimate = estimate_doa_wls(
                measured,
                coordinates,
                selected_pairs,
                tdoa_covariance=tdoa_covariance,
            )
        estimated_angles[index] = (estimate.phi, estimate.elevation)
        estimated_directions[index] = estimate.direction
        successes[index] = estimate.success
        boundaries[index] = (
            estimate.elevation <= boundary_tolerance
            or estimate.elevation >= np.pi / 2.0 - boundary_tolerance
        )

    errors = estimated_angles - np.asarray([phi, elevation])
    errors[:, 0] = _wrap_angle(errors[:, 0])
    bias = np.mean(errors, axis=0)
    empirical_covariance = np.cov(errors, rowvar=False, ddof=1)

    true_direction = np.asarray(
        [
            np.cos(elevation) * np.cos(phi),
            np.cos(elevation) * np.sin(phi),
            np.sin(elevation),
        ]
    )
    cross_norm = np.linalg.norm(np.cross(estimated_directions, true_direction), axis=1)
    dot_product = estimated_directions @ true_direction
    geodesic_errors = np.arctan2(cross_norm, dot_product)

    crlb_inverse_sqrt = _inverse_square_root(crlb.covariance)
    normalized_covariance = crlb_inverse_sqrt @ empirical_covariance @ crlb_inverse_sqrt
    normalized_covariance = (normalized_covariance + normalized_covariance.T) / 2.0
    normalized_eigenvalues = np.linalg.eigvalsh(normalized_covariance)
    physical_bias = float(
        np.sqrt(np.cos(elevation) ** 2 * bias[0] ** 2 + bias[1] ** 2)
    )
    geodesic_rmse = float(np.sqrt(np.mean(geodesic_errors**2)))
    diagnostics = matrix_diagnostics(information)
    return MonteCarloResult(
        array_name=array_name,
        noise_model=noise_model,
        phi_rad=float(phi),
        elevation_rad=float(elevation),
        sigma_tdoa=None if sigma_tdoa is None else float(sigma_tdoa),
        sigma_toa=None if sigma_toa is None else float(sigma_toa),
        marginal_tdoa_std=marginal_tdoa_std,
        n_trials=n_trials,
        seed=int(seed),
        bias_rad=bias,
        empirical_covariance=empirical_covariance,
        crlb_covariance=crlb.covariance,
        normalized_covariance=normalized_covariance,
        normalized_covariance_eigenvalues=normalized_eigenvalues,
        normalized_covariance_frobenius_error=float(
            np.linalg.norm(normalized_covariance - np.eye(2), ord="fro")
        ),
        whitened_bias_norm=float(np.linalg.norm(crlb_inverse_sqrt @ bias)),
        physical_bias_rad=physical_bias,
        geodesic_rmse_rad=geodesic_rmse,
        median_geodesic_error_rad=float(np.median(geodesic_errors)),
        p95_geodesic_error_rad=float(np.quantile(geodesic_errors, 0.95)),
        fraction_over_10deg=float(np.mean(geodesic_errors > np.deg2rad(10.0))),
        angular_crlb_rms_rad=crlb.angular_rms_rad,
        rmse_to_crlb_ratio=geodesic_rmse / crlb.angular_rms_rad,
        optimizer_success_fraction=float(np.mean(successes)),
        boundary_fraction=float(np.mean(boundaries)),
        fisher_rank=diagnostics.rank,
        fisher_condition_number=diagnostics.condition_number,
        fisher_eigenvalues=diagnostics.eigenvalues,
    )
