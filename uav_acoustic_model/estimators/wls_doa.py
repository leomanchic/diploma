"""Constrained weighted least-squares direction estimator."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.optimize import least_squares

from model.geometry import (
    DEFAULT_SOUND_SPEED,
    baselines,
    direction_vector,
    geometry_rank,
    microphone_positions,
    reference_pairs,
    validate_pairs,
)
from model.statistics import DEFAULT_SIGMA_TDOA
from model.tdoa import directional_spherical_tdoa


class UnobservableGeometryError(ValueError):
    """Raised when the array cannot determine two angular coordinates."""


@dataclass(frozen=True)
class DOAEstimate:
    """Result of constrained WLS on the upper hemisphere."""

    phi: float
    elevation: float
    direction: NDArray[np.float64]
    weighted_cost: float
    success: bool
    geometry_rank: int
    mirror_ambiguous: bool


def _residual_transform(
    count: int,
    sigma_tdoa: float | None,
    tdoa_covariance: ArrayLike | None,
    rtol: float = 1e-10,
) -> Callable[[NDArray[np.float64]], NDArray[np.float64]]:
    if tdoa_covariance is None:
        standard_deviation = (
            DEFAULT_SIGMA_TDOA if sigma_tdoa is None else float(sigma_tdoa)
        )
        if not np.isfinite(standard_deviation) or standard_deviation <= 0.0:
            raise ValueError("sigma_tdoa must be finite and positive")
        return lambda residual: residual / standard_deviation

    if sigma_tdoa is not None:
        raise ValueError("specify either sigma_tdoa or tdoa_covariance, not both")

    covariance_array = np.asarray(tdoa_covariance, dtype=float)
    if covariance_array.shape != (count, count):
        raise ValueError("covariance shape does not match the TDOA vector")
    if not np.all(np.isfinite(covariance_array)) or not np.allclose(
        covariance_array, covariance_array.T, rtol=1e-10, atol=1e-14
    ):
        raise ValueError("covariance must be finite and symmetric")
    eigenvalues, eigenvectors = np.linalg.eigh((covariance_array + covariance_array.T) / 2.0)
    scale = float(np.max(np.abs(eigenvalues)))
    tolerance = rtol * scale
    if float(np.min(eigenvalues)) < -tolerance:
        raise ValueError("covariance must be positive semidefinite")
    positive = eigenvalues > tolerance
    if not np.any(positive):
        raise ValueError("covariance has no positive-variance subspace")
    whitening = (eigenvectors[:, positive] / np.sqrt(eigenvalues[positive])).T
    return lambda residual: whitening @ residual


def _wrap_azimuth(phi: float) -> float:
    return float((phi + np.pi) % (2.0 * np.pi) - np.pi)


def _linear_wls_initial(
    design: NDArray[np.float64],
    observations: NDArray[np.float64],
    transform: Callable[[NDArray[np.float64]], NDArray[np.float64]],
    elevation_bounds: tuple[float, float],
) -> NDArray[np.float64]:
    """Construct a deterministic WLS initial point without the true angles."""

    weighted_design = transform(design)
    weighted_observations = transform(observations)
    direction, _, rank, _ = np.linalg.lstsq(
        weighted_design, weighted_observations, rcond=1e-10
    )
    if rank < 2:
        raise UnobservableGeometryError(
            f"selected TDOA design rank {rank} cannot identify two DOA angles"
        )

    norm = float(np.linalg.norm(direction))
    if rank == 3:
        if norm == 0.0:
            direction = np.asarray([1.0, 0.0, 0.0])
        else:
            direction = direction / norm
        candidates = [direction]
    else:
        _, _, right_vectors = np.linalg.svd(weighted_design, full_matrices=True)
        normal = right_vectors[-1]
        direction = direction - normal * float(normal @ direction)
        norm = float(np.linalg.norm(direction))
        if norm < 1.0:
            normal_component = np.sqrt(max(1.0 - norm**2, 0.0))
            candidates = [
                direction + normal_component * normal,
                direction - normal_component * normal,
            ]
        elif norm > 0.0:
            candidates = [direction / norm]
        else:
            candidates = [np.asarray([1.0, 0.0, 0.0])]

    lower_elevation, upper_elevation = elevation_bounds
    angle_candidates = []
    for candidate in candidates:
        x, y, z = candidate / np.linalg.norm(candidate)
        angles = np.asarray(
            [np.arctan2(y, x), np.arctan2(z, np.hypot(x, y))], dtype=float
        )
        if lower_elevation <= angles[1] <= upper_elevation:
            angle_candidates.append(angles)
    if not angle_candidates:
        candidate = max(candidates, key=lambda value: float(value[2]))
        x, y, z = candidate / np.linalg.norm(candidate)
        angle_candidates.append(
            np.asarray(
                [
                    np.arctan2(y, x),
                    np.clip(
                        np.arctan2(z, np.hypot(x, y)),
                        lower_elevation,
                        upper_elevation,
                    ),
                ]
            )
        )
    return min(
        angle_candidates,
        key=lambda angles: float(
            transform(design @ direction_vector(*angles) - observations)
            @ transform(design @ direction_vector(*angles) - observations)
        ),
    )


def estimate_doa_wls(
    measured_tdoa: ArrayLike,
    positions: ArrayLike,
    pairs: Iterable[Sequence[int]] | None = None,
    *,
    sound_speed: float = DEFAULT_SOUND_SPEED,
    sigma_tdoa: float | None = None,
    tdoa_covariance: ArrayLike | None = None,
    elevation_bounds: tuple[float, float] = (0.0, np.pi / 2.0),
    initial_angles: tuple[float, float] | None = None,
) -> DOAEstimate:
    """Estimate azimuth/elevation from TDOA with constrained WLS.

    The unit-vector constraint is enforced by the angular parameterisation.
    The default upper-hemisphere bound resolves the sign choice for planar
    arrays but does not remove their physical mirror ambiguity.
    """

    coordinates = microphone_positions(positions)
    checked_pairs = (
        reference_pairs(coordinates.shape[0])
        if pairs is None
        else validate_pairs(pairs, coordinates.shape[0])
    )
    observations = np.asarray(measured_tdoa, dtype=float)
    if observations.shape != (len(checked_pairs),) or not np.all(np.isfinite(observations)):
        raise ValueError("measured_tdoa must match the selected pairs and be finite")
    speed = float(sound_speed)
    if not np.isfinite(speed) or speed <= 0.0:
        raise ValueError("sound_speed must be finite and positive")
    lower_elevation, upper_elevation = map(float, elevation_bounds)
    if not (
        np.isfinite(lower_elevation)
        and np.isfinite(upper_elevation)
        and -np.pi / 2.0 <= lower_elevation < upper_elevation <= np.pi / 2.0
    ):
        raise ValueError("elevation_bounds must lie within [-pi/2, pi/2]")

    affine_rank = geometry_rank(coordinates)
    if affine_rank < 2:
        raise UnobservableGeometryError(
            f"array geometry rank {affine_rank} cannot identify two DOA angles"
        )

    design = baselines(coordinates, checked_pairs) / speed
    transform = _residual_transform(len(checked_pairs), sigma_tdoa, tdoa_covariance)

    def residual(angles: NDArray[np.float64]) -> NDArray[np.float64]:
        return transform(design @ direction_vector(angles[0], angles[1]) - observations)

    if initial_angles is None:
        initial = _linear_wls_initial(
            design,
            observations,
            transform,
            (lower_elevation, upper_elevation),
        )
    else:
        initial = np.asarray(initial_angles, dtype=float)
        if initial.shape != (2,) or not np.all(np.isfinite(initial)):
            raise ValueError("initial_angles must contain two finite values")
        initial[1] = np.clip(initial[1], lower_elevation, upper_elevation)

    result = least_squares(
        residual,
        initial,
        bounds=([-np.inf, lower_elevation], [np.inf, upper_elevation]),
        xtol=1e-13,
        ftol=1e-13,
        gtol=1e-13,
        max_nfev=2000,
    )
    phi = _wrap_azimuth(float(result.x[0]))
    elevation = float(result.x[1])
    whitened_residual = residual(np.asarray([phi, elevation]))
    return DOAEstimate(
        phi=phi,
        elevation=elevation,
        direction=direction_vector(phi, elevation),
        weighted_cost=float(whitened_residual @ whitened_residual),
        success=bool(result.success),
        geometry_rank=affine_rank,
        mirror_ambiguous=affine_rank == 2,
    )


def estimate_doa_spherical_wls(
    measured_tdoa: ArrayLike,
    positions: ArrayLike,
    distance_m: float,
    pairs: Iterable[Sequence[int]] | None = None,
    *,
    sound_speed: float = DEFAULT_SOUND_SPEED,
    sigma_tdoa: float | None = None,
    tdoa_covariance: ArrayLike | None = None,
    elevation_bounds: tuple[float, float] = (0.0, np.pi / 2.0),
    initial_angles: tuple[float, float] | None = None,
) -> DOAEstimate:
    """Estimate DOA with the exact spherical TDOA model at known distance.

    ``distance_m`` is measured from the array centroid. This estimator is for
    the controlled model-bias study; it does not estimate range jointly.
    """

    coordinates = microphone_positions(positions)
    checked_pairs = (
        reference_pairs(coordinates.shape[0])
        if pairs is None
        else validate_pairs(pairs, coordinates.shape[0])
    )
    observations = np.asarray(measured_tdoa, dtype=float)
    if observations.shape != (len(checked_pairs),) or not np.all(np.isfinite(observations)):
        raise ValueError("measured_tdoa must match the selected pairs and be finite")
    distance = float(distance_m)
    if not np.isfinite(distance) or distance <= 0.0:
        raise ValueError("distance_m must be finite and positive")
    speed = float(sound_speed)
    if not np.isfinite(speed) or speed <= 0.0:
        raise ValueError("sound_speed must be finite and positive")
    lower_elevation, upper_elevation = map(float, elevation_bounds)
    if not (
        np.isfinite(lower_elevation)
        and np.isfinite(upper_elevation)
        and -np.pi / 2.0 <= lower_elevation < upper_elevation <= np.pi / 2.0
    ):
        raise ValueError("elevation_bounds must lie within [-pi/2, pi/2]")
    affine_rank = geometry_rank(coordinates)
    if affine_rank < 2:
        raise UnobservableGeometryError(
            f"array geometry rank {affine_rank} cannot identify two DOA angles"
        )
    transform = _residual_transform(len(checked_pairs), sigma_tdoa, tdoa_covariance)

    def residual(angles: NDArray[np.float64]) -> NDArray[np.float64]:
        predicted = directional_spherical_tdoa(
            angles[0],
            angles[1],
            distance,
            coordinates,
            checked_pairs,
            speed,
        )
        return transform(predicted - observations)

    if initial_angles is None:
        plane_initial = estimate_doa_wls(
            observations,
            coordinates,
            checked_pairs,
            sound_speed=speed,
            sigma_tdoa=sigma_tdoa,
            tdoa_covariance=tdoa_covariance,
            elevation_bounds=(lower_elevation, upper_elevation),
        )
        initial = np.asarray([plane_initial.phi, plane_initial.elevation])
    else:
        initial = np.asarray(initial_angles, dtype=float)
        if initial.shape != (2,) or not np.all(np.isfinite(initial)):
            raise ValueError("initial_angles must contain two finite values")
        initial[1] = np.clip(initial[1], lower_elevation, upper_elevation)
    result = least_squares(
        residual,
        initial,
        bounds=([-np.inf, lower_elevation], [np.inf, upper_elevation]),
        xtol=1e-13,
        ftol=1e-13,
        gtol=1e-13,
        max_nfev=2000,
    )
    phi = _wrap_azimuth(float(result.x[0]))
    elevation = float(result.x[1])
    whitened_residual = residual(np.asarray([phi, elevation]))
    return DOAEstimate(
        phi=phi,
        elevation=elevation,
        direction=direction_vector(phi, elevation),
        weighted_cost=float(whitened_residual @ whitened_residual),
        success=bool(result.success),
        geometry_rank=affine_rank,
        mirror_ambiguous=affine_rank == 2,
    )
