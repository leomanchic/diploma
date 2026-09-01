"""Retarded-time bearing model for a constant-velocity 6-D source state.

For synchronized reception time ``t_r`` at station centroid ``p`` the model
solves ``t_r=t_e+||q(t_e)-p||/c``.  Station clock metadata is deliberately not
applied: timestamps entering :class:`~model.measurements.BearingMeasurement`
are already in the common time coordinate.  ``available_timestamp_s`` is only
a causal scheduling constraint and never enters the propagation equation.

This module predicts measurements and diagnoses stacked local observability.
It does not estimate a trajectory and is not a tracker.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from model.bearing_statistics import (
    tangent_residual,
    tangent_residual_jacobian_wrt_true_direction,
)
from model.dynamic_state import ConstantVelocityState
from model.geometry import DEFAULT_SOUND_SPEED
from model.measurements import BearingMeasurement
from model.station import StationPose
from simulation.moving_source import (
    constant_velocity_emission_time,
    emission_time_residual,
    solve_emission_time,
)
from simulation.trajectory import ConstantVelocityTrajectory


def _readonly(value: ArrayLike) -> NDArray[np.float64]:
    result = np.array(value, dtype=float, copy=True)
    result.setflags(write=False)
    return result


def _sound_speed(value: float) -> float:
    result = float(value)
    if not np.isfinite(result) or result <= 0.0:
        raise ValueError("sound_speed must be finite and positive")
    return result


def _trajectory(
    state: ConstantVelocityState, sound_speed: float
) -> ConstantVelocityTrajectory:
    if not isinstance(state, ConstantVelocityState):
        raise TypeError("state must be ConstantVelocityState")
    speed = _sound_speed(sound_speed)
    if float(np.linalg.norm(state.velocity_world_mps)) >= speed:
        raise ValueError("source speed must satisfy |v| < sound_speed")
    return ConstantVelocityTrajectory(
        state.position_at_reference_world_m,
        state.velocity_world_mps,
        state.reference_time_s,
        speed,
    )


@dataclass(frozen=True, slots=True)
class RetardedBearingPrediction:
    """Physical bearing prediction and retarded-time diagnostics in SI units."""

    reception_time_s: float
    emission_time_s: float
    propagation_delay_s: float
    source_position_at_emission_world_m: NDArray[np.float64]
    range_m: float
    direction_world: NDArray[np.float64]
    direction_local: NDArray[np.float64]
    radial_velocity_mps: float
    retarded_denominator: float


@dataclass(frozen=True, slots=True)
class DynamicObservabilityResult:
    """Local rank diagnostics for one candidate 6-D state, not an estimate."""

    residual_tangent_rad: NDArray[np.float64]
    jacobian: NDArray[np.float64]
    singular_values: NDArray[np.float64]
    rank: int
    condition_number: float
    station_ids: tuple[str, ...]
    sequence_ids: tuple[str, ...]
    frame_indices: tuple[int, ...]
    reception_timestamps_s: NDArray[np.float64]
    available_timestamps_s: NDArray[np.float64]
    emission_timestamps_s: NDArray[np.float64]


def predict_retarded_bearing(
    state: ConstantVelocityState,
    station: StationPose,
    reception_time_s: float,
    sound_speed: float = DEFAULT_SOUND_SPEED,
    *,
    emission_solver: str = "analytic",
) -> RetardedBearingPrediction:
    """Predict the local bearing observed at one synchronized reception time.

    ``emission_solver='analytic'`` uses the independent constant-velocity
    closed form.  ``'numerical'`` uses the general Newton/Brent solver from the
    moving-source signal model and is retained as a cross-check.
    """

    if not isinstance(station, StationPose):
        raise TypeError("station must be StationPose")
    reception = float(reception_time_s)
    if not np.isfinite(reception):
        raise ValueError("reception_time_s must be finite")
    speed = _sound_speed(sound_speed)
    trajectory = _trajectory(state, speed)
    if emission_solver == "analytic":
        emission = float(
            constant_velocity_emission_time(
                reception,
                station.position_world_m,
                trajectory,
                speed,
            )
        )
    elif emission_solver == "numerical":
        emission = float(
            solve_emission_time(
                reception,
                station.position_world_m,
                trajectory,
                speed,
            )
        )
    else:
        raise ValueError("emission_solver must be 'analytic' or 'numerical'")
    if not np.isfinite(emission) or emission >= reception:
        raise ValueError("retarded emission time must be finite and precede reception")
    source_position = state.position_at(emission)
    displacement = source_position - station.position_world_m
    range_m = float(np.linalg.norm(displacement))
    if not np.isfinite(range_m) or range_m <= 0.0:
        raise ValueError("source at emission must not coincide with station centroid")
    direction_world = displacement / range_m
    direction_local = station.world_to_local_direction(direction_world)
    radial_velocity = float(direction_world @ state.velocity_world_mps)
    denominator = 1.0 + radial_velocity / speed
    if denominator <= 0.0:
        raise ValueError("retarded-time map must be monotone")
    return RetardedBearingPrediction(
        reception_time_s=reception,
        emission_time_s=emission,
        propagation_delay_s=reception - emission,
        source_position_at_emission_world_m=_readonly(source_position),
        range_m=range_m,
        direction_world=_readonly(direction_world),
        direction_local=_readonly(direction_local),
        radial_velocity_mps=radial_velocity,
        retarded_denominator=denominator,
    )


def predict_retarded_bearing_measurement(
    state: ConstantVelocityState,
    station: StationPose,
    measurement: BearingMeasurement,
    sound_speed: float = DEFAULT_SOUND_SPEED,
    *,
    emission_solver: str = "analytic",
) -> RetardedBearingPrediction:
    """Predict using the measurement reception timestamp, never availability."""

    if station.station_id != measurement.station_id:
        raise ValueError("station and measurement ids do not match")
    return predict_retarded_bearing(
        state,
        station,
        measurement.reception_center_timestamp_s,
        sound_speed,
        emission_solver=emission_solver,
    )


def emission_time_jacobian_wrt_state(
    state: ConstantVelocityState,
    station: StationPose,
    reception_time_s: float,
    sound_speed: float = DEFAULT_SOUND_SPEED,
    *,
    prediction: RetardedBearingPrediction | None = None,
) -> NDArray[np.float64]:
    """Return analytic ``dt_e/d[q0,v]`` with shape ``(6,)``."""

    speed = _sound_speed(sound_speed)
    _trajectory(state, speed)
    predicted = prediction or predict_retarded_bearing(
        state, station, reception_time_s, speed
    )
    delta = predicted.emission_time_s - state.reference_time_s
    result = -np.concatenate(
        (predicted.direction_world, delta * predicted.direction_world)
    ) / (speed * predicted.retarded_denominator)
    return _readonly(result)


def predicted_local_direction_jacobian(
    state: ConstantVelocityState,
    station: StationPose,
    reception_time_s: float,
    sound_speed: float = DEFAULT_SOUND_SPEED,
    *,
    prediction: RetardedBearingPrediction | None = None,
) -> NDArray[np.float64]:
    """Return analytic ``du_local/d[q0,v]`` with shape ``(3,6)``."""

    speed = _sound_speed(sound_speed)
    predicted = prediction or predict_retarded_bearing(
        state, station, reception_time_s, speed
    )
    emission_jacobian = emission_time_jacobian_wrt_state(
        state,
        station,
        reception_time_s,
        speed,
        prediction=predicted,
    )
    delta = predicted.emission_time_s - state.reference_time_s
    direct = np.hstack((np.eye(3), delta * np.eye(3)))
    emission_position_jacobian = direct + np.outer(
        state.velocity_world_mps, emission_jacobian
    )
    projector = np.eye(3) - np.outer(
        predicted.direction_world, predicted.direction_world
    )
    world_direction_jacobian = (
        projector @ emission_position_jacobian / predicted.range_m
    )
    return _readonly(station.rotation_local_to_world.T @ world_direction_jacobian)


def retarded_bearing_residual(
    state: ConstantVelocityState,
    station: StationPose,
    measurement: BearingMeasurement,
    sound_speed: float = DEFAULT_SOUND_SPEED,
) -> NDArray[np.float64]:
    """Return calibrated spherical tangent residual in angular-arc radians."""

    if not measurement.valid:
        raise ValueError("invalid bearing has no spherical residual")
    predicted = predict_retarded_bearing_measurement(
        state, station, measurement, sound_speed
    )
    return (
        tangent_residual(predicted.direction_local, measurement.direction_local)
        - measurement.calibration_bias_tangent_rad
    )


def retarded_bearing_residual_jacobian(
    state: ConstantVelocityState,
    station: StationPose,
    measurement: BearingMeasurement,
    sound_speed: float = DEFAULT_SOUND_SPEED,
) -> NDArray[np.float64]:
    """Return analytic ``d residual/d[q0,v]`` with shape ``(2,6)``."""

    if not measurement.valid or station.station_id != measurement.station_id:
        raise ValueError("a matching valid station measurement is required")
    predicted = predict_retarded_bearing_measurement(
        state, station, measurement, sound_speed
    )
    residual_wrt_direction = tangent_residual_jacobian_wrt_true_direction(
        predicted.direction_local,
        measurement.direction_local,
    )
    direction_wrt_state = predicted_local_direction_jacobian(
        state,
        station,
        measurement.reception_center_timestamp_s,
        sound_speed,
        prediction=predicted,
    )
    return _readonly(residual_wrt_direction @ direction_wrt_state)


def available_bearing_measurements(
    measurements: Sequence[BearingMeasurement], processing_time_s: float
) -> tuple[BearingMeasurement, ...]:
    """Return records causally available by ``processing_time_s``.

    This helper affects event ordering only; it does not alter their physical
    reception timestamps or predicted retarded times.
    """

    processing_time = float(processing_time_s)
    if not np.isfinite(processing_time):
        raise ValueError("processing_time_s must be finite")
    return tuple(
        item
        for item in measurements
        if item.available_timestamp_s <= processing_time
    )


def stack_retarded_bearing_observability(
    state: ConstantVelocityState,
    stations: Sequence[StationPose],
    measurements: Sequence[BearingMeasurement],
    sound_speed: float = DEFAULT_SOUND_SPEED,
) -> DynamicObservabilityResult:
    """Stack residual/Jacobian rows and report local 6-D numerical rank.

    The candidate ``state`` is supplied by the caller.  No optimization,
    covariance claim, state update, future measurement access or tracking is
    performed.  Position and velocity columns retain their SI units, so the
    reported condition number is a parameterization diagnostic.
    """

    station_map = {station.station_id: station for station in stations}
    if len(station_map) != len(stations):
        raise ValueError("station ids must be unique")
    if not measurements:
        raise ValueError("at least one measurement is required")
    residuals: list[NDArray[np.float64]] = []
    jacobians: list[NDArray[np.float64]] = []
    emissions: list[float] = []
    for measurement in measurements:
        if measurement.station_id not in station_map:
            raise ValueError(f"missing station {measurement.station_id}")
        if not measurement.valid:
            raise ValueError("invalid measurements cannot enter observability stack")
        station = station_map[measurement.station_id]
        prediction = predict_retarded_bearing_measurement(
            state, station, measurement, sound_speed
        )
        emissions.append(prediction.emission_time_s)
        residuals.append(
            tangent_residual(prediction.direction_local, measurement.direction_local)
            - measurement.calibration_bias_tangent_rad
        )
        residual_wrt_direction = tangent_residual_jacobian_wrt_true_direction(
            prediction.direction_local, measurement.direction_local
        )
        jacobians.append(
            residual_wrt_direction
            @ predicted_local_direction_jacobian(
                state,
                station,
                measurement.reception_center_timestamp_s,
                sound_speed,
                prediction=prediction,
            )
        )
    residual = np.concatenate(residuals)
    jacobian = np.vstack(jacobians)
    singular_values = np.linalg.svd(jacobian, compute_uv=False)
    scale = max(float(singular_values[0]), np.finfo(float).tiny)
    tolerance = max(jacobian.shape) * np.finfo(float).eps * scale
    rank = int(np.count_nonzero(singular_values > tolerance))
    condition = (
        float(singular_values[0] / singular_values[-1])
        if rank == 6 and singular_values.size >= 6
        else float("inf")
    )
    return DynamicObservabilityResult(
        residual_tangent_rad=_readonly(residual),
        jacobian=_readonly(jacobian),
        singular_values=_readonly(singular_values),
        rank=rank,
        condition_number=condition,
        station_ids=tuple(item.station_id for item in measurements),
        sequence_ids=tuple(item.sequence_id for item in measurements),
        frame_indices=tuple(item.frame_index for item in measurements),
        reception_timestamps_s=_readonly(
            [item.reception_center_timestamp_s for item in measurements]
        ),
        available_timestamps_s=_readonly(
            [item.available_timestamp_s for item in measurements]
        ),
        emission_timestamps_s=_readonly(emissions),
    )


def retarded_equation_residual_s(
    state: ConstantVelocityState,
    station: StationPose,
    prediction: RetardedBearingPrediction,
    sound_speed: float = DEFAULT_SOUND_SPEED,
) -> float:
    """Evaluate the physical retarded equation using the shared reference solver."""

    trajectory = _trajectory(state, sound_speed)
    return float(
        emission_time_residual(
            prediction.emission_time_s,
            prediction.reception_time_s,
            station.position_world_m,
            trajectory,
            sound_speed,
        )
    )


__all__ = [
    "DynamicObservabilityResult",
    "RetardedBearingPrediction",
    "available_bearing_measurements",
    "emission_time_jacobian_wrt_state",
    "predict_retarded_bearing",
    "predict_retarded_bearing_measurement",
    "predicted_local_direction_jacobian",
    "retarded_bearing_residual",
    "retarded_bearing_residual_jacobian",
    "retarded_equation_residual_s",
    "stack_retarded_bearing_observability",
]
