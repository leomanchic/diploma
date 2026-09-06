"""Constrained weighted least-squares direction estimator."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.optimize import least_squares, minimize

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
    invalid_reason: str | None = None
    global_optimality: str = "not_evaluated"


@dataclass(frozen=True)
class _ResidualSubspaces:
    """Whitening plus exact support constraints for a PSD covariance."""

    whitening: NDArray[np.float64]
    zero_basis: NDArray[np.float64]
    support_tolerance: float

    def __call__(self, residual: NDArray[np.float64]) -> NDArray[np.float64]:
        return self.whitening @ residual

    def exact(self, residual: NDArray[np.float64]) -> NDArray[np.float64]:
        return self.zero_basis.T @ residual


def _residual_transform(
    count: int,
    sigma_tdoa: float | None,
    tdoa_covariance: ArrayLike | None,
    rtol: float = 1e-10,
) -> _ResidualSubspaces:
    if tdoa_covariance is None:
        standard_deviation = (
            DEFAULT_SIGMA_TDOA if sigma_tdoa is None else float(sigma_tdoa)
        )
        if not np.isfinite(standard_deviation) or standard_deviation <= 0.0:
            raise ValueError("sigma_tdoa must be finite and positive")
        return _ResidualSubspaces(
            np.eye(count) / standard_deviation,
            np.empty((count, 0), dtype=float),
            1e-12,
        )

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
    tolerance = max(rtol * scale, np.finfo(float).tiny)
    if float(np.min(eigenvalues)) < -tolerance:
        raise ValueError("covariance must be positive semidefinite")
    positive = eigenvalues > tolerance
    whitening = (
        (eigenvectors[:, positive] / np.sqrt(eigenvalues[positive])).T
        if np.any(positive)
        else np.empty((0, count), dtype=float)
    )
    zero_basis = eigenvectors[:, ~positive]
    return _ResidualSubspaces(whitening, zero_basis, 1e-10)


def _wrap_azimuth(phi: float) -> float:
    return float((phi + np.pi) % (2.0 * np.pi) - np.pi)


def _quadratic_sphere_candidates(
    hessian: NDArray[np.float64],
    linear: NDArray[np.float64],
    radius: float,
) -> list[NDArray[np.float64]]:
    """Enumerate stationary candidates of a quadratic on a small sphere."""

    dimension = linear.size
    if radius < 0.0 or dimension == 0:
        return []
    if radius <= 1e-14:
        return [np.zeros(dimension)]
    if dimension == 1:
        return [np.asarray([-radius]), np.asarray([radius])]
    scale = max(
        float(np.linalg.norm(hessian, ord=2)),
        float(np.linalg.norm(linear)) / max(radius, np.finfo(float).tiny),
        np.finfo(float).tiny,
    )
    eigenvalues, eigenvectors = np.linalg.eigh(hessian / scale)
    projected_linear = eigenvectors.T @ (linear / scale)
    factors = [np.polynomial.Polynomial([value, 1.0]) for value in eigenvalues]
    full_product = np.polynomial.Polynomial([1.0])
    for factor in factors:
        full_product *= factor * factor
    secular = -(radius**2) * full_product
    for index, coefficient in enumerate(projected_linear):
        product = np.polynomial.Polynomial([1.0])
        for other, factor in enumerate(factors):
            if other != index:
                product *= factor * factor
        secular += coefficient**2 * product
    candidates: list[NDArray[np.float64]] = []
    for root in secular.roots():
        if abs(float(np.imag(root))) > 1e-8 * max(1.0, abs(float(np.real(root)))):
            continue
        lagrange = float(np.real(root))
        denominator = eigenvalues + lagrange
        if np.min(np.abs(denominator)) <= 1e-10:
            continue
        coordinates = projected_linear / denominator
        norm = float(np.linalg.norm(coordinates))
        if norm > 0.0 and abs(norm - radius) <= 1e-6 * max(1.0, radius):
            candidates.append(eigenvectors @ (coordinates * radius / norm))
    # Cover hard cases at singular Lagrange multipliers.  This is essential
    # for planar arrays: the stochastic objective fixes horizontal components
    # while the zero-eigenvalue vertical component completes the unit norm.
    eigen_tolerance = 1e-10 * max(1.0, float(np.max(np.abs(eigenvalues))))
    for eigenvalue in eigenvalues:
        group = np.abs(eigenvalues - eigenvalue) <= eigen_tolerance
        if np.max(np.abs(projected_linear[group]), initial=0.0) > 1e-10:
            continue
        coordinates = np.zeros(dimension)
        other = ~group
        coordinates[other] = projected_linear[other] / (
            eigenvalues[other] - eigenvalue
        )
        remaining = radius**2 - float(coordinates @ coordinates)
        if remaining >= -1e-9 * max(1.0, radius**2):
            group_index = int(np.flatnonzero(group)[0])
            component = float(np.sqrt(max(remaining, 0.0)))
            for sign in (-1.0, 1.0):
                hard = coordinates.copy()
                hard[group_index] = sign * component
                candidates.append(eigenvectors @ hard)
    # Harmless deterministic fallbacks also cover fully flat objectives.
    for index in range(dimension):
        axis = eigenvectors[:, index]
        candidates.extend((-radius * axis, radius * axis))
    return candidates


def _affine_unit_sphere_candidates(
    hessian: NDArray[np.float64],
    linear: NDArray[np.float64],
    exact_design: NDArray[np.float64],
    exact_observations: NDArray[np.float64],
) -> list[NDArray[np.float64]]:
    """Enumerate quadratic candidates satisfying ``A u=b, ||u||=1``."""

    if exact_design.shape[0] == 0:
        particular = np.zeros(3)
        nullspace = np.eye(3)
    else:
        left, singular, right = np.linalg.svd(exact_design, full_matrices=True)
        tolerance = 1e-11 * max(
            float(singular[0]) if singular.size else 0.0, 1.0
        )
        rank = int(np.count_nonzero(singular > tolerance))
        particular = np.linalg.lstsq(exact_design, exact_observations, rcond=1e-11)[0]
        if np.max(
            np.abs(exact_design @ particular - exact_observations), initial=0.0
        ) > 1e-10:
            return []
        nullspace = right[rank:].T
    radius_squared = 1.0 - float(particular @ particular)
    if radius_squared < -1e-10:
        return []
    radius = float(np.sqrt(max(radius_squared, 0.0)))
    if nullspace.shape[1] == 0:
        return [particular] if abs(radius_squared) <= 1e-10 else []
    reduced_hessian = nullspace.T @ hessian @ nullspace
    reduced_linear = nullspace.T @ (linear - hessian @ particular)
    return [
        particular + nullspace @ reduced
        for reduced in _quadratic_sphere_candidates(
            reduced_hessian, reduced_linear, radius
        )
    ]


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
    """Estimate far-field DOA from the quadratic problem on the unit sphere.

    The positive covariance eigenspace is whitened, while its nullspace is
    imposed as exact linear equality constraints.  Interior stationary points,
    hard cases and both elevation boundaries are enumerated algebraically and
    the best feasible candidate is returned.  The enumeration is deterministic
    and independent of ``initial_angles``; that argument remains a validated
    compatibility input.  ``global_optimality`` describes the numerical
    candidate-enumeration contract rather than claiming a symbolic proof.
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
    transform = _residual_transform(
        len(checked_pairs), sigma_tdoa, tdoa_covariance
    )
    weighted_design = transform.whitening @ design
    weighted_observations = transform.whitening @ observations
    exact_design = transform.zero_basis.T @ design
    exact_observations = transform.zero_basis.T @ observations

    # Remove exact rows that are algebraically null (for example the cycle
    # nullspace of a consistent all-pair TOA covariance) and retain only an
    # independent row basis.  A null row with a non-zero right hand side is
    # immediately incompatible.
    if exact_design.shape[0]:
        left, singular, right = np.linalg.svd(exact_design, full_matrices=True)
        row_scale = max(float(np.linalg.norm(design, ord=2)), np.finfo(float).tiny)
        exact_rank = int(np.count_nonzero(singular > 1e-10 * row_scale))
        left_null = left[:, exact_rank:]
        compatibility = left_null.T @ exact_observations
        if np.max(np.abs(compatibility), initial=0.0) > transform.support_tolerance:
            return DOAEstimate(
                float("nan"), float("nan"), np.full(3, np.nan), float("inf"),
                False, affine_rank, affine_rank == 2,
                "incompatible_exact_constraints",
                "algebraic_quadratic_candidate_enumeration",
            )
        if exact_rank:
            exact_design = right[:exact_rank]
            exact_observations = (
                left[:, :exact_rank].T @ exact_observations
            ) / singular[:exact_rank]
        else:
            exact_design = np.empty((0, 3))
            exact_observations = np.empty(0)

    if initial_angles is not None:
        supplied = np.asarray(initial_angles, dtype=float)
        if supplied.shape != (2,) or not np.all(np.isfinite(supplied)):
            raise ValueError("initial_angles must contain two finite values")
    hessian = weighted_design.T @ weighted_design
    linear = weighted_design.T @ weighted_observations
    lower_z, upper_z = np.sin(lower_elevation), np.sin(upper_elevation)
    candidate_vectors = _affine_unit_sphere_candidates(
        hessian, linear, exact_design, exact_observations
    )
    for boundary_z in {float(lower_z), float(upper_z)}:
        boundary_design = np.vstack((exact_design, np.asarray([0.0, 0.0, 1.0])))
        boundary_observations = np.concatenate((exact_observations, [boundary_z]))
        candidate_vectors.extend(
            _affine_unit_sphere_candidates(
                hessian, linear, boundary_design, boundary_observations
            )
        )
    feasible_results = []
    for vector in candidate_vectors:
        vector = np.asarray(vector, dtype=float)
        exact_error = (
            float(np.max(np.abs(exact_design @ vector - exact_observations), initial=0.0))
            if exact_design.shape[0]
            else 0.0
        )
        sphere_error = abs(float(vector @ vector - 1.0))
        if (
            np.all(np.isfinite(vector))
            and exact_error <= transform.support_tolerance
            and sphere_error <= 1e-9
            and lower_z - 1e-10 <= vector[2] <= upper_z + 1e-10
        ):
            whitened = weighted_design @ vector - weighted_observations
            feasible_results.append((float(whitened @ whitened), vector))
    if not feasible_results:
        return DOAEstimate(
            float("nan"), float("nan"), np.full(3, np.nan), float("inf"),
            False, affine_rank, affine_rank == 2,
            "incompatible_exact_constraints" if exact_design.shape[0] else "optimizer_failed",
            "algebraic_quadratic_candidate_enumeration",
        )
    weighted_cost, vector = min(feasible_results, key=lambda item: item[0])
    vector = vector / np.linalg.norm(vector)
    phi = _wrap_azimuth(float(np.arctan2(vector[1], vector[0])))
    elevation = float(np.arctan2(vector[2], np.hypot(vector[0], vector[1])))
    return DOAEstimate(
        phi=phi,
        elevation=elevation,
        direction=vector,
        weighted_cost=float(weighted_cost),
        success=True,
        geometry_rank=affine_rank,
        mirror_ambiguous=affine_rank == 2,
        global_optimality="algebraic_quadratic_candidate_enumeration",
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
    the controlled model-bias study; it does not estimate range jointly.  Its
    deterministic multistart is an approximate global search, not a proof of
    the global optimum.
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

    def raw_residual(angles: NDArray[np.float64]) -> NDArray[np.float64]:
        predicted = directional_spherical_tdoa(
            angles[0],
            angles[1],
            distance,
            coordinates,
            checked_pairs,
            speed,
        )
        return predicted - observations

    def residual(angles: NDArray[np.float64]) -> NDArray[np.float64]:
        return transform(raw_residual(angles))

    def exact_constraints(angles: NDArray[np.float64]) -> NDArray[np.float64]:
        return transform.exact(raw_residual(angles))

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
    starts = [initial]
    interior_elevation = float(np.clip(1.0, lower_elevation, upper_elevation))
    starts.extend(
        np.asarray([phi_probe, interior_elevation])
        for phi_probe in np.linspace(-np.pi, np.pi, 4, endpoint=False)
    )
    for boundary in (lower_elevation, upper_elevation):
        starts.append(np.asarray([initial[0], boundary]))
        starts.append(np.asarray([_wrap_azimuth(initial[0] + np.pi), boundary]))
    candidates = []
    for start in starts:
        if transform.zero_basis.shape[1] == 0:
            optimized = least_squares(
                residual,
                start,
                bounds=([-2.0 * np.pi, lower_elevation], [2.0 * np.pi, upper_elevation]),
                xtol=1e-13,
                ftol=1e-13,
                gtol=1e-13,
                max_nfev=2000,
            )
        else:
            optimized = minimize(
                lambda angles: float(residual(angles) @ residual(angles)),
                start,
                method="SLSQP",
                bounds=[(-2.0 * np.pi, 2.0 * np.pi), (lower_elevation, upper_elevation)],
                constraints={"type": "eq", "fun": exact_constraints},
                options={"ftol": 1e-13, "maxiter": 1000},
            )
        angles = np.asarray(optimized.x, dtype=float)
        exact_error = float(np.max(np.abs(exact_constraints(angles)), initial=0.0))
        if np.all(np.isfinite(angles)) and exact_error <= transform.support_tolerance:
            white = residual(angles)
            candidates.append((float(white @ white), angles, optimized))
    if not candidates:
        return DOAEstimate(
            float("nan"), float("nan"), np.full(3, np.nan), float("inf"),
            False, affine_rank, affine_rank == 2,
            "incompatible_exact_constraints",
            "deterministic_multistart_approximation",
        )
    weighted_cost, angles, result = min(candidates, key=lambda value: value[0])
    phi = _wrap_azimuth(float(angles[0]))
    elevation = float(angles[1])
    return DOAEstimate(
        phi=phi,
        elevation=elevation,
        direction=direction_vector(phi, elevation),
        weighted_cost=weighted_cost,
        success=True,
        geometry_rank=affine_rank,
        mirror_ambiguous=affine_rank == 2,
        global_optimality="deterministic_multistart_approximation",
    )
