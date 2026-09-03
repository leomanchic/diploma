"""Batch constant-velocity estimation from asynchronous bearing events.

Both offline and causal-prefix modes call the same exact retarded-time
objective.  They differ only in which truth-free events are available.  This
is a batch state estimate, not a recursive tracker.
"""

from __future__ import annotations

import time
import warnings
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.optimize import NonlinearConstraint, SR1, least_squares, minimize

from model.bearing_events import (
    BearingEventPrefix,
    CausalBearingEventStream,
    ScheduledBearingEvent,
    bearing_event_id,
)
from model.bearing_statistics import AntipodalDirectionError
from model.dynamic_state import ConstantVelocityState, rebase_constant_velocity_state
from model.geometry import DEFAULT_SOUND_SPEED
from model.measurements import BearingMeasurement
from model.retarded_bearing import (
    predict_retarded_bearing_measurement,
    retarded_bearing_residual,
    retarded_bearing_residual_jacobian,
)
from model.station import StationPose


def _readonly(value: ArrayLike) -> NDArray[np.float64]:
    result = np.array(value, dtype=float, copy=True)
    result.setflags(write=False)
    return result


@dataclass(frozen=True, slots=True)
class _CovarianceSubspaces:
    positive_basis: NDArray[np.float64]
    positive_eigenvalues: NDArray[np.float64]
    zero_basis: NDArray[np.float64]


def _covariance_subspaces(covariance: ArrayLike) -> _CovarianceSubspaces:
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
    return _CovarianceSubspaces(
        positive_basis=eigenvectors[:, positive],
        positive_eigenvalues=eigenvalues[positive],
        zero_basis=eigenvectors[:, ~positive],
    )


def _rank_and_nullspace(
    matrix: NDArray[np.float64], *, relative_tolerance: float = 1e-10
) -> tuple[int, NDArray[np.float64], NDArray[np.float64]]:
    values = np.asarray(matrix, dtype=float)
    if values.ndim != 2:
        raise ValueError("matrix must be two-dimensional")
    _, singular, right = np.linalg.svd(values, full_matrices=True)
    scale = max(float(singular[0]) if singular.size else 0.0, np.finfo(float).tiny)
    tolerance = max(
        relative_tolerance * scale,
        256.0 * np.finfo(float).eps * max(values.shape, default=1) * scale,
    )
    rank = int(np.count_nonzero(singular > tolerance))
    return rank, singular, right[rank:].T


def _information_diagnostics(
    information: NDArray[np.float64], *, relative_tolerance: float = 1e-10
) -> tuple[NDArray[np.float64], int, float]:
    symmetric = 0.5 * (information + information.T)
    eigenvalues = np.linalg.eigvalsh(symmetric)
    scale = max(float(np.max(np.abs(eigenvalues))), np.finfo(float).tiny)
    tolerance = max(
        relative_tolerance * scale,
        256.0 * np.finfo(float).eps * information.shape[0] * scale,
    )
    rank = int(np.count_nonzero(eigenvalues > tolerance))
    positive = eigenvalues[eigenvalues > tolerance]
    condition = (
        float(np.max(positive) / np.min(positive))
        if rank == information.shape[0]
        else float("inf")
    )
    return eigenvalues, rank, condition


def _station_map(stations: Sequence[StationPose]) -> dict[str, StationPose]:
    result: dict[str, StationPose] = {}
    for station in stations:
        if not isinstance(station, StationPose):
            raise TypeError("stations must contain StationPose instances")
        if station.station_id in result:
            raise ValueError(f"duplicate station_id: {station.station_id}")
        result[station.station_id] = station
    if not result:
        raise ValueError("at least one station is required")
    return result


def _validate_measurements(
    station_map: dict[str, StationPose],
    measurements: Sequence[BearingMeasurement],
) -> tuple[BearingMeasurement, ...]:
    result = tuple(measurements)
    if not result:
        return result
    identities: set[str] = set()
    variants: set[str] = set()
    sequences: set[str] = set()
    for measurement in result:
        if not isinstance(measurement, BearingMeasurement):
            raise TypeError("measurements must contain BearingMeasurement instances")
        if not measurement.valid:
            raise ValueError("invalid measurements must be filtered by the event stream")
        if measurement.station_id not in station_map:
            raise ValueError(f"missing StationPose for {measurement.station_id}")
        identity = bearing_event_id(measurement)
        if identity in identities:
            raise ValueError(f"duplicate event identity: {identity}")
        identities.add(identity)
        variants.add(measurement.estimator_variant)
        sequences.add(measurement.sequence_id)
    if len(variants) != 1:
        raise ValueError("one estimator variant must be selected per batch")
    if len(sequences) != 1:
        raise ValueError("one batch must contain one sequence_id")
    return tuple(
        sorted(
            result,
            key=lambda item: (
                item.reception_center_timestamp_s,
                item.station_id,
                item.frame_index,
            ),
        )
    )


def geometric_constant_velocity_initial_state(
    stations: Sequence[StationPose],
    measurements: Sequence[BearingMeasurement],
    *,
    reference_time_s: float,
) -> tuple[ConstantVelocityState | None, int, float]:
    """Truth-free linearized ray initialization, never the final estimate.

    It approximates ``q(t_r)`` as lying on the measured ray and solves
    ``(I-dd.T)(q0 + v*(t_r-t0) - p)=0``.  The exact optimizer recomputes every
    emission time for every candidate state.
    """

    station_map = _station_map(stations)
    bearings = _validate_measurements(station_map, measurements)
    reference = float(reference_time_s)
    if not np.isfinite(reference):
        raise ValueError("reference_time_s must be finite")
    if not bearings:
        return None, 0, float("inf")
    rows: list[NDArray[np.float64]] = []
    targets: list[NDArray[np.float64]] = []
    for measurement in bearings:
        station = station_map[measurement.station_id]
        direction = station.local_to_world_direction(measurement.direction_local)
        direction = direction / np.linalg.norm(direction)
        projector = np.eye(3) - np.outer(direction, direction)
        delta = measurement.reception_center_timestamp_s - reference
        rows.append(np.hstack((projector, delta * projector)))
        targets.append(projector @ station.position_world_m)
    design = np.vstack(rows)
    target = np.concatenate(targets)
    solution, _, _, _ = np.linalg.lstsq(design, target, rcond=1e-12)
    singular = np.linalg.svd(design, compute_uv=False)
    scale = max(float(singular[0]), np.finfo(float).tiny)
    tolerance = max(design.shape) * np.finfo(float).eps * scale
    rank = int(np.count_nonzero(singular > tolerance))
    condition = (
        float(singular[0] / singular[-1])
        if rank == 6 and singular.size >= 6
        else float("inf")
    )
    # A rank-deficient one-station ray system can choose the minimum-norm
    # solution at the station centroid, where the physical model is undefined.
    # Keep the rank diagnostic, but use a fixed truth-free 100 m ray point as
    # a safe optimizer seed.  This is initialization only.
    if min(
        np.linalg.norm(solution[:3] - station.position_world_m)
        for station in station_map.values()
    ) < 1.0:
        first = bearings[0]
        first_station = station_map[first.station_id]
        first_direction = first_station.local_to_world_direction(first.direction_local)
        solution[:3] = first_station.position_world_m + 100.0 * first_direction
        solution[3:] = 0.0
    return ConstantVelocityState(solution[:3], solution[3:], reference), rank, condition


def _velocity_from_unconstrained(
    value: NDArray[np.float64], sound_speed: float
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    z = np.asarray(value, dtype=float)
    denominator = float(np.sqrt(1.0 + z @ z))
    velocity = sound_speed * z / denominator
    derivative = sound_speed * (
        np.eye(3) / denominator - np.outer(z, z) / denominator**3
    )
    return velocity, derivative


def _unconstrained_from_velocity(
    velocity: NDArray[np.float64], sound_speed: float
) -> NDArray[np.float64]:
    squared = float(velocity @ velocity)
    if squared >= sound_speed**2:
        raise ValueError("initial velocity must satisfy |v| < sound_speed")
    return velocity / np.sqrt(sound_speed**2 - squared)


@dataclass(frozen=True, slots=True)
class RetardedBatchResult:
    """Result and independent acceptance diagnostics for one batch estimate."""

    state: ConstantVelocityState | None
    valid: bool
    failure_reason: str | None
    objective: float
    iterations: int
    optimizer_success: bool
    optimizer_message: str
    runtime_s: float
    used_event_ids: tuple[str, ...]
    station_ids: tuple[str, ...]
    measurement_count: int
    estimated_parameter_dimension: int
    residuals_tangent_rad: NDArray[np.float64]
    whitened_residuals: NDArray[np.float64]
    exact_constraint_residuals: NDArray[np.float64]
    residual_jacobian_state: NDArray[np.float64]
    information_matrix_state: NDArray[np.float64]
    covariance_state_linearization: NDArray[np.float64]
    scaled_information_eigenvalues: NDArray[np.float64]
    stochastic_information_rank: int
    constraint_rank: int
    local_observability_rank: int
    scaled_information_condition_number: float
    constraint_max_abs_rad: float
    preliminary_constraint_max_abs_rad: float
    constraints_satisfied: bool
    raw_projected_gradient_norm: float
    scaled_projected_kkt_residual: float
    projected_kkt_satisfied: bool
    maximum_angular_residual_rad: float
    maximum_speed_ratio: float
    forward_ray_conditions_satisfied: bool
    initialization_rank: int
    initialization_condition_number: float
    parameter_scale_position_m: float
    parameter_scale_velocity_mps: float
    unobservable_directions_state: NDArray[np.float64]


@dataclass(frozen=True, slots=True)
class RetardedBatchSystem:
    """Assembled physical, whitened and exact-constraint residual system."""

    residuals_tangent_rad: NDArray[np.float64]
    residual_jacobian_state: NDArray[np.float64]
    whitened_residuals: NDArray[np.float64]
    whitened_jacobian_state: NDArray[np.float64]
    exact_constraint_residuals: NDArray[np.float64]
    exact_constraint_jacobian_state: NDArray[np.float64]


@dataclass(frozen=True, slots=True)
class CausalPrefixBatchResult:
    """One immutable published causal-prefix batch estimate."""

    processing_time_s: float
    prefix: BearingEventPrefix
    estimate: RetardedBatchResult


def assemble_retarded_batch_system(
    stations: Sequence[StationPose],
    measurements: Sequence[BearingMeasurement],
    state: ConstantVelocityState,
    *,
    sound_speed: float = DEFAULT_SOUND_SPEED,
) -> RetardedBatchSystem:
    """Assemble residual/Jacobian blocks without estimating or using truth.

    Positive covariance eigenvectors are whitened by ``lambda**-1/2``;
    zero-eigenvalue vectors are returned as explicit equality constraints.
    """

    station_map = _station_map(stations)
    bearings = _validate_measurements(station_map, measurements)
    if not isinstance(state, ConstantVelocityState):
        raise TypeError("state must be ConstantVelocityState")
    speed = float(sound_speed)
    if not np.isfinite(speed) or speed <= 0.0:
        raise ValueError("sound_speed must be finite and positive")
    raw_residual = np.asarray(
        [
            retarded_bearing_residual(
                state, station_map[item.station_id], item, speed
            )
            for item in bearings
        ]
    )
    raw_jacobian = np.vstack(
        [
            retarded_bearing_residual_jacobian(
                state, station_map[item.station_id], item, speed
            )
            for item in bearings
        ]
    )
    subspaces = [
        _covariance_subspaces(item.covariance_tangent_rad2) for item in bearings
    ]
    jacobian_blocks = raw_jacobian.reshape(-1, 2, 6)
    whitened_residual_parts = [
        subspace.positive_basis.T @ residual
        / np.sqrt(subspace.positive_eigenvalues)
        for subspace, residual in zip(subspaces, raw_residual, strict=True)
        if subspace.positive_eigenvalues.size
    ]
    whitened_jacobian_parts = [
        subspace.positive_basis.T @ block
        / np.sqrt(subspace.positive_eigenvalues)[:, None]
        for subspace, block in zip(subspaces, jacobian_blocks, strict=True)
        if subspace.positive_eigenvalues.size
    ]
    exact_residual_parts = [
        subspace.zero_basis.T @ residual
        for subspace, residual in zip(subspaces, raw_residual, strict=True)
        if subspace.zero_basis.shape[1]
    ]
    exact_jacobian_parts = [
        subspace.zero_basis.T @ block
        for subspace, block in zip(subspaces, jacobian_blocks, strict=True)
        if subspace.zero_basis.shape[1]
    ]
    return RetardedBatchSystem(
        residuals_tangent_rad=_readonly(raw_residual),
        residual_jacobian_state=_readonly(raw_jacobian),
        whitened_residuals=_readonly(
            np.concatenate(whitened_residual_parts)
            if whitened_residual_parts
            else np.empty(0)
        ),
        whitened_jacobian_state=_readonly(
            np.vstack(whitened_jacobian_parts)
            if whitened_jacobian_parts
            else np.empty((0, 6))
        ),
        exact_constraint_residuals=_readonly(
            np.concatenate(exact_residual_parts)
            if exact_residual_parts
            else np.empty(0)
        ),
        exact_constraint_jacobian_state=_readonly(
            np.vstack(exact_jacobian_parts)
            if exact_jacobian_parts
            else np.empty((0, 6))
        ),
    )


def _invalid_result(
    *,
    start_time: float,
    measurements: Sequence[BearingMeasurement],
    reason: str,
    message: str,
    state: ConstantVelocityState | None = None,
    initialization_rank: int = 0,
    initialization_condition_number: float = float("inf"),
    estimated_dimension: int = 6,
    position_scale_m: float = 100.0,
    velocity_scale_mps: float = 10.0,
) -> RetardedBatchResult:
    count = len(measurements)
    return RetardedBatchResult(
        state=state,
        valid=False,
        failure_reason=reason,
        objective=float("nan"),
        iterations=0,
        optimizer_success=False,
        optimizer_message=message,
        runtime_s=time.perf_counter() - start_time,
        used_event_ids=tuple(bearing_event_id(item) for item in measurements),
        station_ids=tuple(item.station_id for item in measurements),
        measurement_count=count,
        estimated_parameter_dimension=estimated_dimension,
        residuals_tangent_rad=_readonly(np.full((count, 2), np.nan)),
        whitened_residuals=_readonly(np.empty(0)),
        exact_constraint_residuals=_readonly(np.empty(0)),
        residual_jacobian_state=_readonly(np.full((2 * count, 6), np.nan)),
        information_matrix_state=_readonly(np.zeros((6, 6))),
        covariance_state_linearization=_readonly(np.full((6, 6), np.nan)),
        scaled_information_eigenvalues=_readonly(np.zeros(estimated_dimension)),
        stochastic_information_rank=0,
        constraint_rank=0,
        local_observability_rank=0,
        scaled_information_condition_number=float("inf"),
        constraint_max_abs_rad=float("nan"),
        preliminary_constraint_max_abs_rad=float("nan"),
        constraints_satisfied=False,
        raw_projected_gradient_norm=float("nan"),
        scaled_projected_kkt_residual=float("nan"),
        projected_kkt_satisfied=False,
        maximum_angular_residual_rad=float("nan"),
        maximum_speed_ratio=float("nan"),
        forward_ray_conditions_satisfied=False,
        initialization_rank=initialization_rank,
        initialization_condition_number=initialization_condition_number,
        parameter_scale_position_m=position_scale_m,
        parameter_scale_velocity_mps=velocity_scale_mps,
        unobservable_directions_state=_readonly(np.eye(6)),
    )


def estimate_retarded_constant_velocity_batch(
    stations: Sequence[StationPose],
    measurements: Sequence[BearingMeasurement],
    *,
    reference_time_s: float,
    sound_speed: float = DEFAULT_SOUND_SPEED,
    initial_state: ConstantVelocityState | None = None,
    fixed_velocity_world_mps: ArrayLike | None = None,
    position_scale_m: float = 100.0,
    velocity_scale_mps: float = 10.0,
    constraint_tolerance_rad: float = 1e-9,
    projected_kkt_tolerance: float = 1e-6,
    max_nfev: int = 2000,
) -> RetardedBatchResult:
    """Estimate one constant-velocity state by exact retarded-time batch WLS.

    Zero covariance eigenvalues remain nonlinear equality constraints.  The
    Gaussian covariance output is only a local linearization benchmark and is
    finite only when all estimated parameters are locally observable.
    """

    started = time.perf_counter()
    station_map = _station_map(stations)
    bearings = _validate_measurements(station_map, measurements)
    reference = float(reference_time_s)
    speed = float(sound_speed)
    p_scale = float(position_scale_m)
    v_scale = float(velocity_scale_mps)
    constraint_tolerance = float(constraint_tolerance_rad)
    kkt_tolerance = float(projected_kkt_tolerance)
    if not np.isfinite(reference):
        raise ValueError("reference_time_s must be finite")
    if not np.isfinite(speed) or speed <= 0.0:
        raise ValueError("sound_speed must be finite and positive")
    if not np.isfinite(p_scale) or p_scale <= 0.0:
        raise ValueError("position_scale_m must be finite and positive")
    if not np.isfinite(v_scale) or v_scale <= 0.0:
        raise ValueError("velocity_scale_mps must be finite and positive")
    if not np.isfinite(constraint_tolerance) or constraint_tolerance <= 0.0:
        raise ValueError("constraint_tolerance_rad must be finite and positive")
    if not np.isfinite(kkt_tolerance) or kkt_tolerance <= 0.0:
        raise ValueError("projected_kkt_tolerance must be finite and positive")
    if len(bearings) < 3:
        return _invalid_result(
            start_time=started,
            measurements=bearings,
            reason="insufficient_measurements",
            message="at least three valid unique bearings are required",
            position_scale_m=p_scale,
            velocity_scale_mps=v_scale,
        )

    geometric, initialization_rank, initialization_condition = (
        geometric_constant_velocity_initial_state(
            stations, bearings, reference_time_s=reference
        )
    )
    if initial_state is None:
        if geometric is None:
            return _invalid_result(
                start_time=started,
                measurements=bearings,
                reason="initialization_failed",
                message="geometric initialization produced no state",
                position_scale_m=p_scale,
                velocity_scale_mps=v_scale,
            )
        initial = geometric
    else:
        if not isinstance(initial_state, ConstantVelocityState):
            raise TypeError("initial_state must be ConstantVelocityState")
        initial = rebase_constant_velocity_state(initial_state, reference)

    fixed_velocity: NDArray[np.float64] | None
    if fixed_velocity_world_mps is None:
        fixed_velocity = None
        estimated_dimension = 6
        parameter_basis = np.eye(6)
        parameter_scales = np.asarray([p_scale] * 3 + [v_scale] * 3)
    else:
        fixed_velocity = np.asarray(fixed_velocity_world_mps, dtype=float)
        if fixed_velocity.shape != (3,) or not np.all(np.isfinite(fixed_velocity)):
            raise ValueError("fixed_velocity_world_mps must be a finite three-vector")
        if float(np.linalg.norm(fixed_velocity)) >= speed:
            raise ValueError("fixed velocity must satisfy |v| < sound_speed")
        estimated_dimension = 3
        parameter_basis = np.vstack((np.eye(3), np.zeros((3, 3))))
        parameter_scales = np.asarray([p_scale] * 3)

    if fixed_velocity is None and float(np.linalg.norm(initial.velocity_world_mps)) >= speed:
        initial = ConstantVelocityState(
            initial.position_at_reference_world_m,
            initial.velocity_world_mps
            * (0.5 * speed / np.linalg.norm(initial.velocity_world_mps)),
            reference,
        )

    def state_from_parameter(
        parameter: NDArray[np.float64],
    ) -> tuple[ConstantVelocityState, NDArray[np.float64]]:
        value = np.asarray(parameter, dtype=float)
        if fixed_velocity is None:
            velocity, velocity_jacobian = _velocity_from_unconstrained(
                value[3:], speed
            )
            state = ConstantVelocityState(value[:3], velocity, reference)
            derivative = np.zeros((6, 6), dtype=float)
            derivative[:3, :3] = np.eye(3)
            derivative[3:, 3:] = velocity_jacobian
        else:
            state = ConstantVelocityState(value[:3], fixed_velocity, reference)
            derivative = parameter_basis
        return state, derivative

    if fixed_velocity is None:
        initial_parameter = np.concatenate(
            (
                initial.position_at_reference_world_m,
                _unconstrained_from_velocity(initial.velocity_world_mps, speed),
            )
        )
        optimizer_scale = np.asarray([p_scale] * 3 + [v_scale / speed] * 3)
    else:
        initial_parameter = np.asarray(initial.position_at_reference_world_m)
        optimizer_scale = np.asarray([p_scale] * 3)

    subspaces = [
        _covariance_subspaces(item.covariance_tangent_rad2) for item in bearings
    ]
    positive_dimension = int(
        sum(item.positive_eigenvalues.size for item in subspaces)
    )
    exact_dimension = int(sum(item.zero_basis.shape[1] for item in subspaces))

    def raw_components(
        parameter: NDArray[np.float64],
    ) -> tuple[NDArray[np.float64], NDArray[np.float64], ConstantVelocityState]:
        state, state_derivative = state_from_parameter(parameter)
        system = assemble_retarded_batch_system(
            stations, bearings, state, sound_speed=speed
        )
        return (
            np.asarray(system.residuals_tangent_rad),
            np.asarray(system.residual_jacobian_state) @ state_derivative,
            state,
        )

    def stochastic_residual(parameter: NDArray[np.float64]) -> NDArray[np.float64]:
        raw, _, _ = raw_components(parameter)
        values = [
            subspace.positive_basis.T @ residual
            / np.sqrt(subspace.positive_eigenvalues)
            for subspace, residual in zip(subspaces, raw, strict=True)
            if subspace.positive_eigenvalues.size
        ]
        return np.concatenate(values) if values else np.empty(0)

    def stochastic_jacobian(parameter: NDArray[np.float64]) -> NDArray[np.float64]:
        _, raw, _ = raw_components(parameter)
        blocks = raw.reshape(-1, 2, estimated_dimension)
        values = [
            subspace.positive_basis.T @ block
            / np.sqrt(subspace.positive_eigenvalues)[:, None]
            for subspace, block in zip(subspaces, blocks, strict=True)
            if subspace.positive_eigenvalues.size
        ]
        return np.vstack(values) if values else np.empty((0, estimated_dimension))

    def exact_residual(parameter: NDArray[np.float64]) -> NDArray[np.float64]:
        if not exact_dimension:
            return np.empty(0)
        raw, _, _ = raw_components(parameter)
        return np.concatenate(
            [
                subspace.zero_basis.T @ residual
                for subspace, residual in zip(subspaces, raw, strict=True)
                if subspace.zero_basis.shape[1]
            ]
        )

    def exact_jacobian(parameter: NDArray[np.float64]) -> NDArray[np.float64]:
        if not exact_dimension:
            return np.empty((0, estimated_dimension))
        _, raw, _ = raw_components(parameter)
        blocks = raw.reshape(-1, 2, estimated_dimension)
        return np.vstack(
            [
                subspace.zero_basis.T @ block
                for subspace, block in zip(subspaces, blocks, strict=True)
                if subspace.zero_basis.shape[1]
            ]
        )

    parameter = initial_parameter.copy()
    preliminary_constraint = float("nan")
    iterations = 0
    optimizer_success = False
    optimizer_message = "optimization not started"
    try:
        if exact_dimension:
            feasibility = least_squares(
                exact_residual,
                parameter,
                jac=exact_jacobian,
                x_scale=optimizer_scale,
                xtol=1e-12,
                ftol=1e-12,
                gtol=1e-12,
                max_nfev=int(max_nfev),
            )
            parameter = np.asarray(feasibility.x)
            iterations += int(feasibility.nfev)
            preliminary_constraint = float(
                np.max(np.abs(exact_residual(parameter)), initial=0.0)
            )
            constraint_rank_preliminary, _, _ = _rank_and_nullspace(
                exact_jacobian(parameter) * parameter_scales[None, :]
            )
            if positive_dimension and constraint_rank_preliminary < estimated_dimension:
                constraint = NonlinearConstraint(
                    exact_residual,
                    np.zeros(exact_dimension),
                    np.zeros(exact_dimension),
                    jac=exact_jacobian,
                    hess=SR1(),
                )

                def objective(value: NDArray[np.float64]) -> float:
                    residual = stochastic_residual(value)
                    return float(residual @ residual)

                def gradient(value: NDArray[np.float64]) -> NDArray[np.float64]:
                    return 2.0 * stochastic_jacobian(value).T @ stochastic_residual(value)

                def hessian(value: NDArray[np.float64]) -> NDArray[np.float64]:
                    jacobian = stochastic_jacobian(value)
                    return 2.0 * jacobian.T @ jacobian

                with warnings.catch_warnings():
                    warnings.filterwarnings(
                        "ignore", message="delta_grad == 0.0", category=UserWarning
                    )
                    warnings.filterwarnings(
                        "ignore",
                        message="overflow encountered in scalar divide",
                        category=RuntimeWarning,
                        module=r"scipy\.optimize\._hessian_update_strategy",
                    )
                    optimized = minimize(
                        objective,
                        parameter,
                        jac=gradient,
                        hess=hessian,
                        constraints=(constraint,),
                        method="trust-constr",
                        options={
                            "gtol": 1e-12,
                            "xtol": 1e-13,
                            "barrier_tol": 1e-13,
                            "maxiter": int(max_nfev),
                            "factorization_method": "SVDFactorization",
                        },
                    )
                parameter = np.asarray(optimized.x)
                iterations += int(optimized.niter)
                optimizer_success = bool(optimized.success)
                optimizer_message = str(optimized.message)
            else:
                optimizer_success = bool(feasibility.success)
                optimizer_message = str(feasibility.message)
        else:
            optimized = least_squares(
                stochastic_residual,
                parameter,
                jac=stochastic_jacobian,
                x_scale=optimizer_scale,
                xtol=1e-12,
                ftol=1e-12,
                gtol=1e-12,
                max_nfev=int(max_nfev),
            )
            parameter = np.asarray(optimized.x)
            iterations = int(optimized.nfev)
            optimizer_success = bool(optimized.success)
            optimizer_message = str(optimized.message)
        raw_residual, raw_jacobian_parameter, state = raw_components(parameter)
    except (ValueError, AntipodalDirectionError, np.linalg.LinAlgError) as error:
        return _invalid_result(
            start_time=started,
            measurements=bearings,
            reason=f"optimization_error:{type(error).__name__}",
            message=str(error),
            state=initial,
            initialization_rank=initialization_rank,
            initialization_condition_number=initialization_condition,
            estimated_dimension=estimated_dimension,
            position_scale_m=p_scale,
            velocity_scale_mps=v_scale,
        )

    whitened = stochastic_residual(parameter)
    whitened_jacobian = stochastic_jacobian(parameter)
    exact = exact_residual(parameter)
    exact_jacobian_parameter = exact_jacobian(parameter)
    _, state_derivative = state_from_parameter(parameter)
    raw_jacobian_state = np.vstack(
        [
            retarded_bearing_residual_jacobian(
                state, station_map[item.station_id], item, speed
            )
            for item in bearings
        ]
    )
    information_parameter = whitened_jacobian.T @ whitened_jacobian
    information_state = np.zeros((6, 6), dtype=float)
    for index, subspace in enumerate(subspaces):
        if not subspace.positive_eigenvalues.size:
            continue
        block = raw_jacobian_state[2 * index : 2 * index + 2]
        whiten = (
            subspace.positive_basis.T @ block
            / np.sqrt(subspace.positive_eigenvalues)[:, None]
        )
        information_state += whiten.T @ whiten

    scaled_constraint = exact_jacobian_parameter * parameter_scales[None, :]
    constraint_rank, _, free_scaled_basis = _rank_and_nullspace(scaled_constraint)
    scaled_information = (
        parameter_scales[:, None]
        * information_parameter
        * parameter_scales[None, :]
    )
    if free_scaled_basis.shape[1]:
        reduced_information = free_scaled_basis.T @ scaled_information @ free_scaled_basis
        reduced_eigenvalues, reduced_rank, reduced_condition = _information_diagnostics(
            reduced_information
        )
    else:
        reduced_information = np.empty((0, 0))
        reduced_eigenvalues = np.empty(0)
        reduced_rank = 0
        reduced_condition = 1.0
    local_rank = constraint_rank + reduced_rank
    stochastic_rank = _information_diagnostics(scaled_information)[1]
    gradient_twice_objective = 2.0 * whitened_jacobian.T @ whitened
    projected_gradient = free_scaled_basis.T @ (
        gradient_twice_objective * parameter_scales
    )
    raw_projected_gradient_norm = float(np.linalg.norm(projected_gradient))
    if free_scaled_basis.shape[1] == 0:
        scaled_kkt = 0.0
    elif reduced_rank == free_scaled_basis.shape[1]:
        quadratic = float(
            projected_gradient
            @ np.linalg.solve(
                0.5 * (reduced_information + reduced_information.T),
                projected_gradient,
            )
        )
        scaled_kkt = 0.5 * float(np.sqrt(max(quadratic, 0.0)))
    else:
        scaled_kkt = float("inf")
    kkt_satisfied = bool(local_rank < estimated_dimension or scaled_kkt <= kkt_tolerance)
    constraint_max = float(np.max(np.abs(exact), initial=0.0))
    constraints_satisfied = bool(constraint_max <= constraint_tolerance)

    forward = True
    for measurement in bearings:
        prediction = predict_retarded_bearing_measurement(
            state, station_map[measurement.station_id], measurement, speed
        )
        measured_world = station_map[measurement.station_id].local_to_world_direction(
            measurement.direction_local
        )
        if float(prediction.direction_world @ measured_world) <= 0.0:
            forward = False
            break
    speed_ratio = float(np.linalg.norm(state.velocity_world_mps) / speed)
    state_finite = bool(np.all(np.isfinite(state.vector)))
    if not state_finite:
        valid = False
        failure = "nonfinite_state_estimate"
    elif speed_ratio >= 1.0:
        valid = False
        failure = "non_subsonic_state_estimate"
    elif not constraints_satisfied:
        valid = False
        failure = "incompatible_exact_constraints"
    elif not forward:
        valid = False
        failure = "estimated_source_not_forward_of_all_bearings"
    elif local_rank < estimated_dimension:
        valid = False
        failure = "insufficient_local_observability"
    elif not kkt_satisfied:
        valid = False
        failure = "projected_kkt_not_satisfied"
    else:
        valid = True
        failure = None

    covariance_state = np.full((6, 6), np.nan)
    if constraints_satisfied and local_rank == estimated_dimension:
        if free_scaled_basis.shape[1] == 0:
            covariance_parameter = np.zeros((estimated_dimension, estimated_dimension))
        else:
            covariance_scaled = free_scaled_basis @ np.linalg.solve(
                0.5 * (reduced_information + reduced_information.T),
                free_scaled_basis.T,
            )
            covariance_parameter = (
                parameter_scales[:, None]
                * covariance_scaled
                * parameter_scales[None, :]
            )
        covariance_state = state_derivative @ covariance_parameter @ state_derivative.T
        covariance_state = 0.5 * (covariance_state + covariance_state.T)

    if free_scaled_basis.shape[1] and reduced_rank < free_scaled_basis.shape[1]:
        _, vectors = np.linalg.eigh(0.5 * (reduced_information + reduced_information.T))
        unobservable_scaled = free_scaled_basis @ vectors[:, : free_scaled_basis.shape[1] - reduced_rank]
        unobservable_parameter = parameter_scales[:, None] * unobservable_scaled
        unobservable_state = (state_derivative @ unobservable_parameter).T
    else:
        unobservable_state = np.empty((0, 6))
    maximum_residual = float(
        np.max(np.linalg.norm(raw_residual, axis=1), initial=0.0)
    )
    return RetardedBatchResult(
        state=state,
        valid=valid,
        failure_reason=failure,
        objective=0.5 * float(whitened @ whitened),
        iterations=iterations,
        optimizer_success=optimizer_success,
        optimizer_message=optimizer_message,
        runtime_s=time.perf_counter() - started,
        used_event_ids=tuple(bearing_event_id(item) for item in bearings),
        station_ids=tuple(item.station_id for item in bearings),
        measurement_count=len(bearings),
        estimated_parameter_dimension=estimated_dimension,
        residuals_tangent_rad=_readonly(raw_residual),
        whitened_residuals=_readonly(whitened),
        exact_constraint_residuals=_readonly(exact),
        residual_jacobian_state=_readonly(raw_jacobian_state),
        information_matrix_state=_readonly(information_state),
        covariance_state_linearization=_readonly(covariance_state),
        scaled_information_eigenvalues=_readonly(reduced_eigenvalues),
        stochastic_information_rank=stochastic_rank,
        constraint_rank=constraint_rank,
        local_observability_rank=local_rank,
        scaled_information_condition_number=reduced_condition,
        constraint_max_abs_rad=constraint_max,
        preliminary_constraint_max_abs_rad=preliminary_constraint,
        constraints_satisfied=constraints_satisfied,
        raw_projected_gradient_norm=raw_projected_gradient_norm,
        scaled_projected_kkt_residual=scaled_kkt,
        projected_kkt_satisfied=kkt_satisfied,
        maximum_angular_residual_rad=maximum_residual,
        maximum_speed_ratio=speed_ratio,
        forward_ray_conditions_satisfied=forward,
        initialization_rank=initialization_rank,
        initialization_condition_number=initialization_condition,
        parameter_scale_position_m=p_scale,
        parameter_scale_velocity_mps=v_scale,
        unobservable_directions_state=_readonly(unobservable_state),
    )


def estimate_offline_retarded_batch(
    stations: Sequence[StationPose],
    events: Sequence[ScheduledBearingEvent | BearingMeasurement],
    *,
    estimator_variant: str,
    reference_time_s: float,
    **estimator_options: object,
) -> tuple[BearingEventPrefix, RetardedBatchResult]:
    """Run the non-causal full-record reference after deterministic replay."""

    stream = CausalBearingEventStream(events, estimator_variant=estimator_variant)
    final_time = max(
        (
            (item.measurement if isinstance(item, ScheduledBearingEvent) else item).available_timestamp_s
            for item in events
        ),
        default=reference_time_s,
    )
    prefix = stream.advance_to(final_time)
    result = estimate_retarded_constant_velocity_batch(
        stations,
        prefix.measurements,
        reference_time_s=reference_time_s,
        **estimator_options,
    )
    return prefix, result


def estimate_causal_prefix_batches(
    stations: Sequence[StationPose],
    events: Sequence[ScheduledBearingEvent | BearingMeasurement],
    processing_times_s: Sequence[float],
    *,
    estimator_variant: str,
    reference_time_s: float,
    **estimator_options: object,
) -> tuple[CausalPrefixBatchResult, ...]:
    """Publish cumulative batch estimates without future-event access."""

    times = np.asarray(processing_times_s, dtype=float)
    if times.ndim != 1 or not np.all(np.isfinite(times)):
        raise ValueError("processing_times_s must be a finite vector")
    if np.any(np.diff(times) < 0.0):
        raise ValueError("processing_times_s must be nondecreasing")
    stream = CausalBearingEventStream(events, estimator_variant=estimator_variant)
    results: list[CausalPrefixBatchResult] = []
    for processing_time in times:
        prefix = stream.advance_to(float(processing_time))
        estimate = estimate_retarded_constant_velocity_batch(
            stations,
            prefix.measurements,
            reference_time_s=reference_time_s,
            **estimator_options,
        )
        results.append(
            CausalPrefixBatchResult(float(processing_time), prefix, estimate)
        )
    return tuple(results)


__all__ = [
    "CausalPrefixBatchResult",
    "RetardedBatchResult",
    "RetardedBatchSystem",
    "assemble_retarded_batch_system",
    "estimate_causal_prefix_batches",
    "estimate_offline_retarded_batch",
    "estimate_retarded_constant_velocity_batch",
    "geometric_constant_velocity_initial_state",
]
