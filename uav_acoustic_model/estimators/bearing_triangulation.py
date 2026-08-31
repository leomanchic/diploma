"""Static multi-station 3-D triangulation from calibrated bearing vectors.

The main estimator minimizes spherical tangent-plane residuals.  It accepts
only measurements referring to one static source state/time; asynchronous
retarded-time fusion belongs to the later dynamic stage.
"""

from __future__ import annotations

import warnings
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.optimize import NonlinearConstraint, SR1, least_squares, minimize

from model.bearing_statistics import (
    AntipodalDirectionError,
    sphere_log_map,
    tangent_basis,
    tangent_residual,
)
from model.geometry import direction_angles
from model.measurements import BearingMeasurement
from model.station import StationPose


def _readonly(value: ArrayLike) -> NDArray[np.float64]:
    result = np.array(value, dtype=float, copy=True)
    result.setflags(write=False)
    return result


def _unit(value: ArrayLike, *, name: str) -> NDArray[np.float64]:
    result = np.asarray(value, dtype=float)
    if result.shape != (3,) or not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must be a finite three-vector")
    norm = float(np.linalg.norm(result))
    if norm == 0.0:
        raise ValueError(f"{name} must be non-zero")
    return result / norm


def _symmetric_rank(
    matrix: NDArray[np.float64], *, relative_tolerance: float = 1e-12
) -> tuple[NDArray[np.float64], NDArray[np.float64], int, float]:
    symmetric = 0.5 * (matrix + matrix.T)
    eigenvalues, eigenvectors = np.linalg.eigh(symmetric)
    scale = max(float(np.max(np.abs(eigenvalues))), np.finfo(float).tiny)
    tolerance = max(
        relative_tolerance * scale,
        256.0 * np.finfo(float).eps * matrix.shape[0] * scale,
    )
    rank = int(np.count_nonzero(eigenvalues > tolerance))
    positive = eigenvalues[eigenvalues > tolerance]
    condition = (
        float(np.max(positive) / np.min(positive))
        if rank == matrix.shape[0]
        else float("inf")
    )
    return eigenvalues, eigenvectors, rank, condition


@dataclass(frozen=True, slots=True)
class _CovarianceSubspaces:
    """Positive-variance and deterministic subspaces of one tangent covariance."""

    inverse: NDArray[np.float64]
    positive_basis: NDArray[np.float64]
    positive_eigenvalues: NDArray[np.float64]
    zero_basis: NDArray[np.float64]


def _covariance_subspaces(covariance: ArrayLike) -> _CovarianceSubspaces:
    """Split ``R`` into stochastic ``U+`` and exact-constraint ``U0`` parts.

    A numerical rank tolerance identifies eigenvalues that are zero at input
    precision.  No positive value is replaced by epsilon, and the resulting
    nullspace is returned explicitly instead of being discarded by ``R+``.
    """

    matrix = np.asarray(covariance, dtype=float)
    if matrix.shape != (2, 2) or not np.all(np.isfinite(matrix)):
        raise ValueError("bearing covariance must be a finite 2x2 matrix")
    if not np.allclose(matrix, matrix.T, rtol=0.0, atol=1e-15):
        raise ValueError("bearing covariance must be symmetric")
    eigenvalues, eigenvectors = np.linalg.eigh(0.5 * (matrix + matrix.T))
    scale = max(float(np.max(np.abs(eigenvalues))), np.finfo(float).tiny)
    tolerance = 256.0 * np.finfo(float).eps * 2.0 * scale
    if float(np.min(eigenvalues)) < -tolerance:
        raise ValueError("bearing covariance must be positive semidefinite")
    positive = eigenvalues > tolerance
    zero = ~positive
    inverse_values = np.zeros_like(eigenvalues)
    inverse_values[positive] = 1.0 / eigenvalues[positive]
    inverse = (eigenvectors * inverse_values) @ eigenvectors.T
    return _CovarianceSubspaces(
        inverse=inverse,
        positive_basis=eigenvectors[:, positive],
        positive_eigenvalues=eigenvalues[positive],
        zero_basis=eigenvectors[:, zero],
    )


@dataclass(frozen=True, slots=True)
class ClosestRaysResult:
    """Transparent closed-form closest-rays initial estimate."""

    position_world_m: NDArray[np.float64]
    valid: bool
    failure_reason: str | None
    rank: int
    condition_number: float
    eigenvalues: NDArray[np.float64]
    normal_matrix: NDArray[np.float64]
    contributing_station_ids: tuple[str, ...]
    signed_ray_ranges_m: NDArray[np.float64]
    perpendicular_ray_residuals_m: NDArray[np.float64]


@dataclass(frozen=True, slots=True)
class TriangulationResult:
    """Static constrained spherical-WLS and local observability diagnostics.

    ``information_matrix`` contains only finite positive-variance Gaussian
    information.  Exact zero-variance components are represented separately
    by ``constraint_jacobian``.  Position covariance is computed in the local
    nullspace of those constraints; it is zero in exactly constrained
    directions and finite only if stochastic information observes every
    remaining free direction.  Preliminary feasibility and final compatibility
    are reported separately.  The projected-KKT residual checks optimality on
    the final constraint manifold independently of the optimizer exit message.
    """

    position_world_m: NDArray[np.float64]
    covariance_position_m2: NDArray[np.float64]
    valid: bool
    failure_reason: str | None
    objective: float
    iterations: int
    optimizer_success: bool
    contributing_station_ids: tuple[str, ...]
    residuals_tangent_rad: NDArray[np.float64]
    ranges_m: NDArray[np.float64]
    jacobian: NDArray[np.float64]
    information_matrix: NDArray[np.float64]
    information_eigenvalues: NDArray[np.float64]
    information_rank: int
    information_condition_number: float
    positive_variance_residual_dimension: int
    exact_constraint_dimension: int
    exact_constraint_residuals: NDArray[np.float64]
    constraint_jacobian: NDArray[np.float64]
    constraint_rank: int
    constraints_satisfied: bool
    constraint_max_abs_rad: float
    preliminary_constraint_max_abs_rad: float
    constrained_optimization_performed: bool
    free_parameter_dimension: int
    reduced_information_eigenvalues: NDArray[np.float64]
    reduced_information_rank: int
    reduced_information_condition_number: float
    local_observability_rank: int
    raw_projected_gradient_norm: float
    scaled_projected_kkt_residual: float
    projected_kkt_tolerance: float
    projected_kkt_satisfied: bool
    unobservable_directions_world: NDArray[np.float64]
    gdop_like_sqrt_trace_m: float
    horizontal_std_rss_m: float
    vertical_std_m: float
    closest_rays: ClosestRaysResult
    optimizer_message: str


def _pose_map(stations: Sequence[StationPose]) -> dict[str, StationPose]:
    result: dict[str, StationPose] = {}
    for station in stations:
        if not isinstance(station, StationPose):
            raise TypeError("stations must contain StationPose instances")
        if station.station_id in result:
            raise ValueError(f"duplicate station_id: {station.station_id}")
        result[station.station_id] = station
    if len(result) < 2:
        raise ValueError("at least two station poses are required")
    return result


def _matched_inputs(
    stations: Sequence[StationPose],
    measurements: Sequence[BearingMeasurement],
    *,
    timestamp_tolerance_s: float | None = None,
) -> tuple[tuple[StationPose, ...], tuple[BearingMeasurement, ...]]:
    poses = _pose_map(stations)
    if len(measurements) < 2:
        raise ValueError("at least two bearing measurements are required")
    ordered_poses: list[StationPose] = []
    ordered_measurements: list[BearingMeasurement] = []
    seen: set[str] = set()
    for measurement in measurements:
        if not isinstance(measurement, BearingMeasurement):
            raise TypeError("measurements must contain BearingMeasurement instances")
        if measurement.station_id in seen:
            raise ValueError(f"duplicate bearing for station {measurement.station_id}")
        if measurement.station_id not in poses:
            raise ValueError(f"missing StationPose for {measurement.station_id}")
        seen.add(measurement.station_id)
        ordered_poses.append(poses[measurement.station_id])
        ordered_measurements.append(measurement)
    sequence_ids = {item.sequence_id for item in ordered_measurements}
    frame_indices = {item.frame_index for item in ordered_measurements}
    if len(sequence_ids) != 1 or len(frame_indices) != 1:
        raise ValueError("static triangulation requires one sequence_id and frame_index")
    if timestamp_tolerance_s is not None:
        timestamps = np.asarray(
            [item.reception_center_timestamp_s for item in ordered_measurements]
        )
        tolerance = float(timestamp_tolerance_s)
        if not np.isfinite(tolerance) or tolerance < 0.0:
            raise ValueError("timestamp_tolerance_s must be finite and non-negative")
        if float(np.ptp(timestamps)) > tolerance:
            raise ValueError("reception timestamps exceed the requested tolerance")
    return tuple(ordered_poses), tuple(ordered_measurements)


def closest_rays_triangulation(
    stations: Sequence[StationPose],
    measurements: Sequence[BearingMeasurement],
    *,
    weights: ArrayLike | Mapping[str, float] | None = None,
    timestamp_tolerance_s: float | None = None,
) -> ClosestRaysResult:
    """Return ``pinv(sum w A) @ sum w A p`` for measured world rays."""

    poses, bearings = _matched_inputs(
        stations, measurements, timestamp_tolerance_s=timestamp_tolerance_s
    )
    invalid_ids = [item.station_id for item in bearings if not item.valid]
    if invalid_ids:
        return _invalid_closest(poses, f"invalid_measurement:{','.join(invalid_ids)}")
    if weights is None:
        values = np.ones(len(bearings), dtype=float)
    elif isinstance(weights, Mapping):
        values = np.asarray([weights[item.station_id] for item in bearings], dtype=float)
    else:
        values = np.asarray(weights, dtype=float)
    if values.shape != (len(bearings),) or not np.all(np.isfinite(values)):
        raise ValueError("weights must be finite with one value per measurement")
    if np.any(values < 0.0) or not np.any(values > 0.0):
        raise ValueError("weights must be non-negative with at least one positive value")

    identity = np.eye(3)
    directions = np.asarray(
        [
            _unit(pose.local_to_world_direction(item.direction_local), name="world bearing")
            for pose, item in zip(poses, bearings, strict=True)
        ]
    )
    projectors = identity - directions[:, :, None] * directions[:, None, :]
    normal = np.einsum("n,nij->ij", values, projectors)
    right_hand_side = np.einsum(
        "n,nij,nj->i",
        values,
        projectors,
        np.asarray([pose.position_world_m for pose in poses]),
    )
    eigenvalues, _, rank, condition = _symmetric_rank(normal)
    position = np.linalg.pinv(normal, rcond=1e-12, hermitian=True) @ right_hand_side
    offsets = position - np.asarray([pose.position_world_m for pose in poses])
    signed_ranges = np.einsum("ni,ni->n", offsets, directions)
    perpendicular = np.linalg.norm(np.einsum("nij,nj->ni", projectors, offsets), axis=1)
    positive_ids = tuple(
        item.station_id
        for item, weight in zip(bearings, values, strict=True)
        if weight > 0.0
    )
    coordinate_scale = max(1.0, float(np.linalg.norm(position)))
    forward_tolerance = 256.0 * np.finfo(float).eps * coordinate_scale
    if rank < 3:
        valid = False
        reason = "degenerate_ray_geometry"
    elif np.any(signed_ranges[values > 0.0] <= forward_tolerance):
        valid = False
        reason = "estimated_source_not_forward_of_all_rays"
    else:
        valid = True
        reason = None
    return ClosestRaysResult(
        position_world_m=_readonly(position),
        valid=valid,
        failure_reason=reason,
        rank=rank,
        condition_number=condition,
        eigenvalues=_readonly(eigenvalues),
        normal_matrix=_readonly(normal),
        contributing_station_ids=positive_ids,
        signed_ray_ranges_m=_readonly(signed_ranges),
        perpendicular_ray_residuals_m=_readonly(perpendicular),
    )


def _invalid_closest(
    stations: Sequence[StationPose], reason: str
) -> ClosestRaysResult:
    count = len(stations)
    return ClosestRaysResult(
        position_world_m=_readonly(np.full(3, np.nan)),
        valid=False,
        failure_reason=reason,
        rank=0,
        condition_number=float("inf"),
        eigenvalues=_readonly(np.zeros(3)),
        normal_matrix=_readonly(np.zeros((3, 3))),
        contributing_station_ids=tuple(station.station_id for station in stations),
        signed_ray_ranges_m=_readonly(np.full(count, np.nan)),
        perpendicular_ray_residuals_m=_readonly(np.full(count, np.nan)),
    )


def predicted_local_direction(position_world_m: ArrayLike, station: StationPose) -> NDArray[np.float64]:
    """Return the candidate source direction in one station's local frame."""

    position = np.asarray(position_world_m, dtype=float)
    if position.shape != (3,) or not np.all(np.isfinite(position)):
        raise ValueError("position_world_m must be a finite three-vector")
    offset = position - station.position_world_m
    distance = float(np.linalg.norm(offset))
    if distance == 0.0:
        raise ValueError("candidate source cannot coincide with a station")
    return station.world_to_local_direction(offset / distance)


def bearing_residual(
    position_world_m: ArrayLike,
    station: StationPose,
    measurement: BearingMeasurement,
) -> NDArray[np.float64]:
    """Return bias-corrected spherical residual in angular-arc radians."""

    if station.station_id != measurement.station_id:
        raise ValueError("station and measurement ids do not match")
    if not measurement.valid:
        raise ValueError("invalid bearing has no spherical residual")
    predicted = predicted_local_direction(position_world_m, station)
    return tangent_residual(predicted, measurement.direction_local) - measurement.calibration_bias_tangent_rad


def bearing_residual_jacobian(
    position_world_m: ArrayLike,
    station: StationPose,
    measurement: BearingMeasurement,
) -> NDArray[np.float64]:
    """Analytic ``d r / d q`` for the bias-corrected spherical residual.

    With ``a=u.T y``, ``v=y-a*u``, ``s=||v||``, ``theta=atan2(s,a)`` and
    ``L=(theta/s)v``, differentiation gives

    ``dL/du = -k(u y.T + a I) + (-1/s^2 + theta*a/s^3) v v.T``.

    The two additional connection terms differentiate the moving
    azimuth/elevation tangent basis.  Calibration bias is constant and has
    zero derivative.  The azimuth basis is undefined at a local pole, which
    is rejected explicitly instead of choosing a hidden coordinate axis.
    """

    if station.station_id != measurement.station_id or not measurement.valid:
        raise ValueError("a matching valid station measurement is required")
    position = np.asarray(position_world_m, dtype=float)
    if position.shape != (3,) or not np.all(np.isfinite(position)):
        raise ValueError("position_world_m must be a finite three-vector")
    world_offset = position - station.position_world_m
    distance = float(np.linalg.norm(world_offset))
    if distance == 0.0:
        raise ValueError("candidate source cannot coincide with a station")
    world_direction = world_offset / distance
    u = _unit(station.world_to_local_direction(world_direction), name="predicted direction")
    y = _unit(measurement.direction_local, name="measured direction")
    phi, elevation = direction_angles(u)
    horizontal = float(np.hypot(u[0], u[1]))
    if horizontal <= 1e-10:
        raise ValueError("azimuth/elevation tangent basis is undefined at a local pole")
    basis = tangent_basis(phi, elevation)
    dot = float(np.clip(u @ y, -1.0, 1.0))
    projection = y - dot * u
    sine = float(np.linalg.norm(projection))
    if dot <= -1.0 + 1e-10:
        raise AntipodalDirectionError(
            "sphere residual Jacobian is not unique for antipodal bearings"
        )
    if sine <= 1e-7:
        log_map = sphere_log_map(u, y)
        log_derivative = -(np.eye(3) - u[:, None] * u[None, :])
    else:
        theta = float(np.arctan2(sine, dot))
        k = theta / sine
        alpha = -1.0 / sine**2 + theta * dot / sine**3
        log_map = k * projection
        log_derivative = (
            -k * (u[:, None] * y[None, :] + dot * np.eye(3))
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
    residual_wrt_local_direction = basis @ log_derivative + connection
    direction_wrt_position = (
        station.rotation_local_to_world.T
        @ (np.eye(3) - world_direction[:, None] * world_direction[None, :])
        / distance
    )
    return residual_wrt_local_direction @ direction_wrt_position


def numerical_bearing_residual_jacobian(
    position_world_m: ArrayLike,
    station: StationPose,
    measurement: BearingMeasurement,
    *,
    step_m: float | None = None,
) -> NDArray[np.float64]:
    """Central finite-difference cross-check of ``bearing_residual_jacobian``."""

    position = np.asarray(position_world_m, dtype=float)
    if position.shape != (3,) or not np.all(np.isfinite(position)):
        raise ValueError("position_world_m must be a finite three-vector")
    distance = float(np.linalg.norm(position - station.position_world_m))
    step = max(1e-6, distance * 1e-7) if step_m is None else float(step_m)
    if not np.isfinite(step) or step <= 0.0:
        raise ValueError("step_m must be finite and positive")
    result = np.empty((2, 3), dtype=float)
    for axis in range(3):
        delta = np.zeros(3)
        delta[axis] = step
        result[:, axis] = (
            bearing_residual(position + delta, station, measurement)
            - bearing_residual(position - delta, station, measurement)
        ) / (2.0 * step)
    return result


def _constraint_nullspace(
    constraint_jacobian: NDArray[np.float64],
) -> tuple[int, NDArray[np.float64]]:
    """Return local equality-constraint rank and its parameter nullspace."""

    if constraint_jacobian.ndim != 2 or constraint_jacobian.shape[1] != 3:
        raise ValueError("constraint_jacobian must have shape (C, 3)")
    if constraint_jacobian.shape[0] == 0:
        return 0, np.eye(3)
    _, singular_values, right_vectors = np.linalg.svd(
        constraint_jacobian, full_matrices=True
    )
    scale = max(float(singular_values[0]), np.finfo(float).tiny)
    tolerance = max(
        1e-12 * scale,
        256.0 * np.finfo(float).eps * max(constraint_jacobian.shape) * scale,
    )
    rank = int(np.count_nonzero(singular_values > tolerance))
    return rank, right_vectors[rank:].T


def triangulate_bearings_spherical_wls(
    stations: Sequence[StationPose],
    measurements: Sequence[BearingMeasurement],
    *,
    timestamp_tolerance_s: float | None = None,
    max_nfev: int = 300,
    constraint_tolerance_rad: float = 1e-10,
    projected_kkt_tolerance: float = 1e-6,
) -> TriangulationResult:
    """Estimate one static 3-D source state by constrained spherical WLS.

    For every covariance eigensystem, positive eigenvalues produce ordinary
    whitened stochastic residuals.  Every numerical zero eigenvalue produces
    an explicit nonlinear equality constraint.  The preliminary feasibility
    solve supplies only an initial point whenever a stochastic constrained
    optimization is possible.  Compatibility is decided from the final
    constraint residual.  ``constraint_tolerance_rad`` is only a numerical
    feasibility test; it does not regularize ``R``.

    ``raw_projected_gradient_norm`` is ``||g_Z||`` with
    ``g_Z=Z.T @ grad(J)``.  For full-rank reduced information
    ``I_Z=Z.T @ I @ Z``, the dimensionless KKT diagnostic is
    ``0.5*sqrt(g_Z.T @ solve(I_Z, g_Z))``.  It estimates the remaining Newton
    correction in local-standard-deviation units.  Optimizer exit status is
    reported but is not itself an acceptance condition.
    """

    poses, bearings = _matched_inputs(
        stations, measurements, timestamp_tolerance_s=timestamp_tolerance_s
    )
    closest = closest_rays_triangulation(
        poses, bearings, timestamp_tolerance_s=timestamp_tolerance_s
    )
    invalid_ids = [item.station_id for item in bearings if not item.valid]
    if invalid_ids:
        return _invalid_triangulation(
            poses,
            closest,
            f"invalid_measurement:{','.join(invalid_ids)}",
            "optimization not started",
        )
    if not closest.valid:
        return _invalid_triangulation(
            poses,
            closest,
            closest.failure_reason or "invalid_closest_rays_initialization",
            "optimization not started",
        )
    constraint_tolerance = float(constraint_tolerance_rad)
    if not np.isfinite(constraint_tolerance) or constraint_tolerance <= 0.0:
        raise ValueError("constraint_tolerance_rad must be finite and positive")
    kkt_tolerance = float(projected_kkt_tolerance)
    if not np.isfinite(kkt_tolerance) or kkt_tolerance <= 0.0:
        raise ValueError("projected_kkt_tolerance must be finite and positive")
    subspaces = [
        _covariance_subspaces(item.covariance_tangent_rad2) for item in bearings
    ]
    positive_dimension = int(
        sum(value.positive_eigenvalues.size for value in subspaces)
    )
    exact_dimension = int(sum(value.zero_basis.shape[1] for value in subspaces))

    def raw_residual(position: NDArray[np.float64]) -> NDArray[np.float64]:
        return np.concatenate(
            [
                bearing_residual(position, pose, item)
                for pose, item in zip(poses, bearings, strict=True)
            ]
        )

    def stochastic_residual(position: NDArray[np.float64]) -> NDArray[np.float64]:
        raw = raw_residual(position).reshape(-1, 2)
        return np.concatenate(
            [
                (subspace.positive_basis.T @ value)
                / np.sqrt(subspace.positive_eigenvalues)
                for subspace, value in zip(subspaces, raw, strict=True)
            ]
        )

    def raw_jacobian_at(position: NDArray[np.float64]) -> NDArray[np.float64]:
        return np.vstack(
            [
                bearing_residual_jacobian(position, pose, item)
                for pose, item in zip(poses, bearings, strict=True)
            ]
        )

    def stochastic_jacobian(position: NDArray[np.float64]) -> NDArray[np.float64]:
        raw = [
            bearing_residual_jacobian(position, pose, item)
            for pose, item in zip(poses, bearings, strict=True)
        ]
        return np.vstack(
            [
                (subspace.positive_basis.T @ value)
                / np.sqrt(subspace.positive_eigenvalues)[:, None]
                for subspace, value in zip(subspaces, raw, strict=True)
                if subspace.positive_eigenvalues.size > 0
            ]
        )

    def exact_constraints(position: NDArray[np.float64]) -> NDArray[np.float64]:
        if exact_dimension == 0:
            return np.empty(0, dtype=float)
        raw = raw_residual(position).reshape(-1, 2)
        return np.concatenate(
            [
                subspace.zero_basis.T @ value
                for subspace, value in zip(subspaces, raw, strict=True)
                if subspace.zero_basis.shape[1] > 0
            ]
        )

    def exact_constraint_jacobian(
        position: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        if exact_dimension == 0:
            return np.empty((0, 3), dtype=float)
        raw = raw_jacobian_at(position).reshape(-1, 2, 3)
        return np.vstack(
            [
                subspace.zero_basis.T @ value
                for subspace, value in zip(subspaces, raw, strict=True)
                if subspace.zero_basis.shape[1] > 0
            ]
        )

    try:
        if exact_dimension == 0:
            optimizer = least_squares(
                stochastic_residual,
                np.asarray(closest.position_world_m),
                jac=stochastic_jacobian,
                xtol=1e-13,
                ftol=1e-13,
                gtol=1e-13,
                max_nfev=int(max_nfev),
            )
            position = np.asarray(optimizer.x, dtype=float)
            optimizer_success = bool(optimizer.success)
            optimizer_message = str(optimizer.message)
            iterations = int(optimizer.nfev)
            preliminary_constraint_max_abs = float("nan")
            constrained_optimization_performed = False
        else:
            feasibility = least_squares(
                exact_constraints,
                np.asarray(closest.position_world_m),
                jac=exact_constraint_jacobian,
                xtol=1e-13,
                ftol=1e-13,
                gtol=1e-13,
                max_nfev=int(max_nfev),
            )
            position = np.asarray(feasibility.x, dtype=float)
            feasibility_residual = exact_constraints(position)
            preliminary_constraint_max_abs = float(
                np.max(np.abs(feasibility_residual), initial=0.0)
            )
            iterations = int(feasibility.nfev)
            local_constraint_rank, _ = _constraint_nullspace(
                exact_constraint_jacobian(position)
            )
            constrained_optimization_performed = bool(
                positive_dimension > 0 and local_constraint_rank < 3
            )
            if constrained_optimization_performed:
                nonlinear_constraint = NonlinearConstraint(
                    exact_constraints,
                    np.zeros(exact_dimension),
                    np.zeros(exact_dimension),
                    jac=exact_constraint_jacobian,
                    # Constraint curvature matters along the feasible
                    # manifold.  SR1 avoids the premature xtol termination
                    # observed with a zero constraint Hessian while leaving
                    # covariance eigenvalues untouched.
                    hess=SR1(),
                )

                def objective(candidate: NDArray[np.float64]) -> float:
                    weighted = stochastic_residual(candidate)
                    return float(weighted @ weighted)

                def objective_gradient(
                    candidate: NDArray[np.float64],
                ) -> NDArray[np.float64]:
                    weighted = stochastic_residual(candidate)
                    return 2.0 * stochastic_jacobian(candidate).T @ weighted

                def objective_hessian(
                    candidate: NDArray[np.float64],
                ) -> NDArray[np.float64]:
                    jacobian = stochastic_jacobian(candidate)
                    return 2.0 * jacobian.T @ jacobian

                with warnings.catch_warnings():
                    # A zero SR1 gradient update is a legitimate locally
                    # linear constraint step.  Keep this filter as narrow as
                    # SciPy's own constraint-conversion filter.
                    warnings.filterwarnings(
                        "ignore", message="delta_grad == 0.0", category=UserWarning
                    )
                    constrained = minimize(
                        objective,
                        position,
                        jac=objective_gradient,
                        hess=objective_hessian,
                        constraints=(nonlinear_constraint,),
                        method="trust-constr",
                        options={
                            "gtol": 1e-12,
                            "xtol": 1e-13,
                            "barrier_tol": 1e-13,
                            "maxiter": int(max_nfev),
                            "factorization_method": "SVDFactorization",
                        },
                    )
                position = np.asarray(constrained.x, dtype=float)
                iterations += int(constrained.niter)
                optimizer_success = bool(constrained.success)
                optimizer_message = str(constrained.message)
            else:
                optimizer_success = bool(feasibility.success)
                optimizer_message = (
                    "exact constraints determine the feasible solution: "
                    + str(feasibility.message)
                )
        corrected_residuals = raw_residual(position).reshape(-1, 2)
        raw_jacobian = raw_jacobian_at(position)
    except (ValueError, AntipodalDirectionError) as error:
        return _invalid_triangulation(
            poses, closest, f"optimization_error:{error}", str(error)
        )

    information = np.zeros((3, 3), dtype=float)
    objective_gradient_final = np.zeros(3, dtype=float)
    objective = 0.0
    for index, (subspace, residual_value) in enumerate(
        zip(subspaces, corrected_residuals, strict=True)
    ):
        block = raw_jacobian[2 * index : 2 * index + 2]
        information += block.T @ subspace.inverse @ block
        objective_gradient_final += (
            2.0 * block.T @ subspace.inverse @ residual_value
        )
        objective += float(residual_value @ subspace.inverse @ residual_value)
    eigenvalues, _, rank, condition = _symmetric_rank(information)
    constraint_residuals = exact_constraints(position)
    constraint_jacobian = exact_constraint_jacobian(position)
    constraint_rank, free_basis = _constraint_nullspace(constraint_jacobian)
    constraints_satisfied = bool(
        np.max(np.abs(constraint_residuals), initial=0.0) <= constraint_tolerance
    )
    constraint_max_abs = float(
        np.max(np.abs(constraint_residuals), initial=0.0)
    )
    free_dimension = int(free_basis.shape[1])
    if free_dimension == 0:
        reduced_information = np.empty((0, 0), dtype=float)
        reduced_eigenvalues = np.empty(0, dtype=float)
        reduced_rank = 0
        reduced_condition = 1.0
        unobservable = np.empty((0, 3), dtype=float)
    else:
        reduced_information = free_basis.T @ information @ free_basis
        (
            reduced_eigenvalues,
            reduced_eigenvectors,
            reduced_rank,
            reduced_condition,
        ) = _symmetric_rank(reduced_information)
        reduced_scale = max(
            float(np.max(np.abs(reduced_eigenvalues))), np.finfo(float).tiny
        )
        reduced_tolerance = max(
            1e-12 * reduced_scale,
            256.0
            * np.finfo(float).eps
            * free_dimension
            * reduced_scale,
        )
        unobservable = (
            free_basis
            @ reduced_eigenvectors[:, reduced_eigenvalues <= reduced_tolerance]
        ).T
    local_observability_rank = constraint_rank + reduced_rank
    projected_gradient = free_basis.T @ objective_gradient_final
    raw_projected_gradient_norm = float(np.linalg.norm(projected_gradient))
    if free_dimension == 0:
        scaled_projected_kkt_residual = 0.0
    elif reduced_rank == free_dimension:
        reduced_information_symmetric = 0.5 * (
            reduced_information + reduced_information.T
        )
        newton_quadratic = float(
            projected_gradient
            @ np.linalg.solve(reduced_information_symmetric, projected_gradient)
        )
        scaled_projected_kkt_residual = 0.5 * float(
            np.sqrt(max(newton_quadratic, 0.0))
        )
    else:
        scaled_projected_kkt_residual = float("inf")
    projected_kkt_required = bool(local_observability_rank == 3)
    projected_kkt_satisfied = bool(
        not projected_kkt_required
        or scaled_projected_kkt_residual <= kkt_tolerance
    )
    offsets = position - np.asarray([pose.position_world_m for pose in poses])
    ranges = np.linalg.norm(offsets, axis=1)
    measured_world = np.asarray(
        [
            _unit(pose.local_to_world_direction(item.direction_local), name="world bearing")
            for pose, item in zip(poses, bearings, strict=True)
        ]
    )
    signed_ranges = np.einsum("ni,ni->n", offsets, measured_world)
    forward_tolerance = 256.0 * np.finfo(float).eps * max(
        1.0, float(np.linalg.norm(position))
    )
    position_finite = bool(np.all(np.isfinite(position)))
    if not position_finite:
        valid = False
        failure_reason = "nonfinite_position_estimate"
    elif not constraints_satisfied:
        valid = False
        failure_reason = "incompatible_exact_constraints"
    elif np.any(signed_ranges <= forward_tolerance):
        valid = False
        failure_reason = "estimated_source_not_forward_of_all_rays"
    elif local_observability_rank < 3:
        valid = False
        failure_reason = "degenerate_constrained_position_information"
    elif not projected_kkt_satisfied:
        valid = False
        failure_reason = "projected_kkt_not_satisfied"
    else:
        valid = True
        failure_reason = None
    if constraints_satisfied and local_observability_rank == 3:
        if free_dimension == 0:
            covariance = np.zeros((3, 3), dtype=float)
        else:
            reduced_covariance = np.linalg.solve(
                0.5 * (reduced_information + reduced_information.T),
                np.eye(free_dimension),
            )
            covariance = free_basis @ reduced_covariance @ free_basis.T
        covariance = 0.5 * (covariance + covariance.T)
        gdop_like = float(np.sqrt(max(np.trace(covariance), 0.0)))
        horizontal_std = float(np.sqrt(max(covariance[0, 0] + covariance[1, 1], 0.0)))
        vertical_std = float(np.sqrt(max(covariance[2, 2], 0.0)))
    else:
        covariance = np.full((3, 3), np.nan)
        gdop_like = horizontal_std = vertical_std = float("nan")
    return TriangulationResult(
        position_world_m=_readonly(position),
        covariance_position_m2=_readonly(covariance),
        valid=valid,
        failure_reason=failure_reason,
        objective=objective,
        iterations=iterations,
        optimizer_success=optimizer_success,
        contributing_station_ids=tuple(item.station_id for item in bearings),
        residuals_tangent_rad=_readonly(corrected_residuals),
        ranges_m=_readonly(ranges),
        jacobian=_readonly(raw_jacobian),
        information_matrix=_readonly(information),
        information_eigenvalues=_readonly(eigenvalues),
        information_rank=rank,
        information_condition_number=condition,
        positive_variance_residual_dimension=positive_dimension,
        exact_constraint_dimension=exact_dimension,
        exact_constraint_residuals=_readonly(constraint_residuals),
        constraint_jacobian=_readonly(constraint_jacobian),
        constraint_rank=constraint_rank,
        constraints_satisfied=constraints_satisfied,
        constraint_max_abs_rad=constraint_max_abs,
        preliminary_constraint_max_abs_rad=preliminary_constraint_max_abs,
        constrained_optimization_performed=constrained_optimization_performed,
        free_parameter_dimension=free_dimension,
        reduced_information_eigenvalues=_readonly(reduced_eigenvalues),
        reduced_information_rank=reduced_rank,
        reduced_information_condition_number=reduced_condition,
        local_observability_rank=local_observability_rank,
        raw_projected_gradient_norm=raw_projected_gradient_norm,
        scaled_projected_kkt_residual=scaled_projected_kkt_residual,
        projected_kkt_tolerance=kkt_tolerance,
        projected_kkt_satisfied=projected_kkt_satisfied,
        unobservable_directions_world=_readonly(unobservable),
        gdop_like_sqrt_trace_m=gdop_like,
        horizontal_std_rss_m=horizontal_std,
        vertical_std_m=vertical_std,
        closest_rays=closest,
        optimizer_message=optimizer_message,
    )


def _invalid_triangulation(
    stations: Sequence[StationPose],
    closest: ClosestRaysResult,
    reason: str,
    message: str,
) -> TriangulationResult:
    count = len(stations)
    position = np.asarray(closest.position_world_m)
    return TriangulationResult(
        position_world_m=_readonly(position),
        covariance_position_m2=_readonly(np.full((3, 3), np.nan)),
        valid=False,
        failure_reason=reason,
        objective=float("nan"),
        iterations=0,
        optimizer_success=False,
        contributing_station_ids=tuple(station.station_id for station in stations),
        residuals_tangent_rad=_readonly(np.full((count, 2), np.nan)),
        ranges_m=_readonly(np.full(count, np.nan)),
        jacobian=_readonly(np.full((2 * count, 3), np.nan)),
        information_matrix=_readonly(np.zeros((3, 3))),
        information_eigenvalues=_readonly(np.zeros(3)),
        information_rank=0,
        information_condition_number=float("inf"),
        positive_variance_residual_dimension=0,
        exact_constraint_dimension=0,
        exact_constraint_residuals=_readonly(np.empty(0)),
        constraint_jacobian=_readonly(np.empty((0, 3))),
        constraint_rank=0,
        constraints_satisfied=False,
        constraint_max_abs_rad=float("nan"),
        preliminary_constraint_max_abs_rad=float("nan"),
        constrained_optimization_performed=False,
        free_parameter_dimension=3,
        reduced_information_eigenvalues=_readonly(np.zeros(3)),
        reduced_information_rank=0,
        reduced_information_condition_number=float("inf"),
        local_observability_rank=0,
        raw_projected_gradient_norm=float("nan"),
        scaled_projected_kkt_residual=float("nan"),
        projected_kkt_tolerance=float("nan"),
        projected_kkt_satisfied=False,
        unobservable_directions_world=_readonly(np.eye(3)),
        gdop_like_sqrt_trace_m=float("nan"),
        horizontal_std_rss_m=float("nan"),
        vertical_std_m=float("nan"),
        closest_rays=closest,
        optimizer_message=message,
    )


__all__ = [
    "ClosestRaysResult",
    "TriangulationResult",
    "bearing_residual",
    "bearing_residual_jacobian",
    "closest_rays_triangulation",
    "numerical_bearing_residual_jacobian",
    "predicted_local_direction",
    "triangulate_bearings_spherical_wls",
]
