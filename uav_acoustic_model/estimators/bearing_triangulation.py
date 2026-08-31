"""Static multi-station 3-D triangulation from calibrated bearing vectors.

The main estimator minimizes spherical tangent-plane residuals.  It accepts
only measurements referring to one static source state/time; asynchronous
retarded-time fusion belongs to the later dynamic stage.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.optimize import least_squares

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


def _covariance_pseudoinverse(
    covariance: ArrayLike,
) -> tuple[NDArray[np.float64], NDArray[np.float64], int]:
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
    inverse_values = np.zeros_like(eigenvalues)
    inverse_values[positive] = 1.0 / eigenvalues[positive]
    square_root_values = np.zeros_like(eigenvalues)
    square_root_values[positive] = 1.0 / np.sqrt(eigenvalues[positive])
    inverse = (eigenvectors * inverse_values) @ eigenvectors.T
    square_root = (eigenvectors * square_root_values) @ eigenvectors.T
    return inverse, square_root, int(np.count_nonzero(positive))


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
    """Static spherical-WLS estimate and local observability diagnostics."""

    position_world_m: NDArray[np.float64]
    covariance_position_m2: NDArray[np.float64]
    valid: bool
    failure_reason: str | None
    objective: float
    iterations: int
    contributing_station_ids: tuple[str, ...]
    residuals_tangent_rad: NDArray[np.float64]
    ranges_m: NDArray[np.float64]
    jacobian: NDArray[np.float64]
    information_matrix: NDArray[np.float64]
    information_eigenvalues: NDArray[np.float64]
    information_rank: int
    information_condition_number: float
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


def triangulate_bearings_spherical_wls(
    stations: Sequence[StationPose],
    measurements: Sequence[BearingMeasurement],
    *,
    timestamp_tolerance_s: float | None = None,
    max_nfev: int = 300,
) -> TriangulationResult:
    """Estimate one static 3-D source state from simultaneous bearings."""

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
    covariance_inverses: list[NDArray[np.float64]] = []
    square_roots: list[NDArray[np.float64]] = []
    for item in bearings:
        inverse, square_root, _ = _covariance_pseudoinverse(
            item.covariance_tangent_rad2
        )
        covariance_inverses.append(inverse)
        square_roots.append(square_root)

    def raw_residual(position: NDArray[np.float64]) -> NDArray[np.float64]:
        return np.concatenate(
            [
                bearing_residual(position, pose, item)
                for pose, item in zip(poses, bearings, strict=True)
            ]
        )

    def residual(position: NDArray[np.float64]) -> NDArray[np.float64]:
        raw = raw_residual(position).reshape(-1, 2)
        return np.concatenate(
            [root @ value for root, value in zip(square_roots, raw, strict=True)]
        )

    def jacobian(position: NDArray[np.float64]) -> NDArray[np.float64]:
        raw = [
            bearing_residual_jacobian(position, pose, item)
            for pose, item in zip(poses, bearings, strict=True)
        ]
        return np.vstack(
            [root @ value for root, value in zip(square_roots, raw, strict=True)]
        )

    try:
        result = least_squares(
            residual,
            np.asarray(closest.position_world_m),
            jac=jacobian,
            xtol=1e-13,
            ftol=1e-13,
            gtol=1e-13,
            max_nfev=int(max_nfev),
        )
        position = np.asarray(result.x, dtype=float)
        corrected_residuals = raw_residual(position).reshape(-1, 2)
        raw_jacobian = np.vstack(
            [
                bearing_residual_jacobian(position, pose, item)
                for pose, item in zip(poses, bearings, strict=True)
            ]
        )
    except (ValueError, AntipodalDirectionError) as error:
        return _invalid_triangulation(
            poses, closest, f"optimization_error:{error}", str(error)
        )

    information = np.zeros((3, 3), dtype=float)
    objective = 0.0
    for index, (inverse, residual_value) in enumerate(
        zip(covariance_inverses, corrected_residuals, strict=True)
    ):
        block = raw_jacobian[2 * index : 2 * index + 2]
        information += block.T @ inverse @ block
        objective += float(residual_value @ inverse @ residual_value)
    eigenvalues, eigenvectors, rank, condition = _symmetric_rank(information)
    unobservable = eigenvectors[:, eigenvalues <= max(
        1e-12 * max(float(np.max(np.abs(eigenvalues))), np.finfo(float).tiny),
        256.0 * np.finfo(float).eps * 3.0 * max(
            float(np.max(np.abs(eigenvalues))), np.finfo(float).tiny
        ),
    )].T
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
    optimizer_success = bool(result.success) and np.all(np.isfinite(position))
    if not optimizer_success:
        valid = False
        failure_reason = "optimizer_failed"
    elif np.any(signed_ranges <= forward_tolerance):
        valid = False
        failure_reason = "estimated_source_not_forward_of_all_rays"
    elif rank < 3:
        valid = False
        failure_reason = "degenerate_position_information"
    else:
        valid = True
        failure_reason = None
    if rank == 3:
        covariance = np.linalg.solve(0.5 * (information + information.T), np.eye(3))
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
        iterations=int(result.nfev),
        contributing_station_ids=tuple(item.station_id for item in bearings),
        residuals_tangent_rad=_readonly(corrected_residuals),
        ranges_m=_readonly(ranges),
        jacobian=_readonly(raw_jacobian),
        information_matrix=_readonly(information),
        information_eigenvalues=_readonly(eigenvalues),
        information_rank=rank,
        information_condition_number=condition,
        unobservable_directions_world=_readonly(unobservable),
        gdop_like_sqrt_trace_m=gdop_like,
        horizontal_std_rss_m=horizontal_std,
        vertical_std_m=vertical_std,
        closest_rays=closest,
        optimizer_message=str(result.message),
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
        contributing_station_ids=tuple(station.station_id for station in stations),
        residuals_tangent_rad=_readonly(np.full((count, 2), np.nan)),
        ranges_m=_readonly(np.full(count, np.nan)),
        jacobian=_readonly(np.full((2 * count, 3), np.nan)),
        information_matrix=_readonly(np.zeros((3, 3))),
        information_eigenvalues=_readonly(np.zeros(3)),
        information_rank=0,
        information_condition_number=float("inf"),
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
