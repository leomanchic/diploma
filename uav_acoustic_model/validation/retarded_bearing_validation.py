"""Deterministic validation for the S7C-A retarded bearing model.

The study audits a supplied constant-velocity state; it performs no state
estimation, filtering, process-noise modelling or tracking Monte Carlo.
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from model.bearing_statistics import tangent_basis
from model.dynamic_state import ConstantVelocityState
from model.geometry import direction_angles, tetrahedral_array
from model.measurements import BearingMeasurement
from model.retarded_bearing import (
    emission_time_jacobian_wrt_state,
    predict_retarded_bearing,
    predicted_local_direction_jacobian,
    retarded_bearing_residual,
    retarded_bearing_residual_jacobian,
    retarded_equation_residual_s,
    stack_retarded_bearing_observability,
)
from model.station import StationPose


DEFAULT_AUDIT_SEED = 20260901
DEFAULT_SCENE_COUNT = 1000


def validation_stations() -> tuple[StationPose, ...]:
    """Return the three-station ENU geometry used by the validation notebook."""

    microphones = tetrahedral_array(0.2)
    return (
        StationPose("A", [0.0, 0.0, 0.0], np.eye(3), microphones),
        StationPose("B", [120.0, 0.0, 3.0], np.eye(3), microphones),
        StationPose("C", [20.0, 100.0, -2.0], np.eye(3), microphones),
    )


def _state_from_vector(vector: np.ndarray, reference: float) -> ConstantVelocityState:
    return ConstantVelocityState(vector[:3], vector[3:], reference)


def _exp_map(direction: np.ndarray, coordinates: np.ndarray) -> np.ndarray:
    phi, elevation = direction_angles(direction)
    tangent = tangent_basis(phi, elevation).T @ coordinates
    theta = float(np.linalg.norm(tangent))
    if theta == 0.0:
        return direction.copy()
    return np.cos(theta) * direction + np.sin(theta) * tangent / theta


def _measurement(
    station: StationPose,
    state: ConstantVelocityState,
    reception: float,
    frame: int,
    offset: np.ndarray | None = None,
    sound_speed: float = 343.0,
) -> BearingMeasurement:
    prediction = predict_retarded_bearing(
        state, station, reception, sound_speed=sound_speed
    )
    measured = prediction.direction_local
    if offset is not None:
        measured = _exp_map(measured, offset)
    return BearingMeasurement(
        station.station_id,
        "s7c-a-validation",
        frame,
        reception,
        reception + 0.02 + frame * 1e-4,
        measured,
        np.diag([1e-5, 2e-5]),
        np.zeros(2),
        "direct_bearing",
    )


def _central(function, state: ConstantVelocityState, step: float = 2e-4) -> np.ndarray:
    columns = []
    for axis in range(6):
        delta = np.zeros(6)
        delta[axis] = step
        plus = function(_state_from_vector(state.vector + delta, state.reference_time_s))
        minus = function(_state_from_vector(state.vector - delta, state.reference_time_s))
        columns.append((np.asarray(plus) - np.asarray(minus)) / (2.0 * step))
    return np.stack(columns, axis=-1)


def _svd_diagnostics(jacobian: np.ndarray) -> tuple[np.ndarray, int, float]:
    """Return singular values, numerical rank and full-rank condition number.

    The tolerance is the standard backward-error scale
    ``max(shape)*eps*s_max``.  It is not adjusted to obtain a desired rank.
    Position and velocity columns retain their native SI units, so the
    condition number depends on that parameter scaling.
    """

    matrix = np.asarray(jacobian, dtype=float)
    if matrix.ndim != 2 or matrix.shape[1] != 6 or not np.all(np.isfinite(matrix)):
        raise ValueError("jacobian must have finite shape (N, 6)")
    singular_values = np.linalg.svd(matrix, compute_uv=False)
    scale = max(float(singular_values[0]), np.finfo(float).tiny)
    tolerance = max(matrix.shape) * np.finfo(float).eps * scale
    rank = int(np.count_nonzero(singular_values > tolerance))
    condition = (
        float(singular_values[0] / singular_values[-1])
        if rank == 6 and singular_values.size >= 6
        else float("inf")
    )
    return singular_values, rank, condition


def instantaneous_world_direction_jacobian(
    state: ConstantVelocityState,
    station: StationPose,
    reception_time_s: float,
) -> np.ndarray:
    """Independent ``c -> infinity`` Jacobian, without calling a solver.

    The model is evaluated directly at reception time:
    ``q=q0+v*(t_r-t0)`` and
    ``du/dx=(I-u*u.T)/range @ [I,(t_r-t0)I]``.
    """

    reception = float(reception_time_s)
    if not np.isfinite(reception):
        raise ValueError("reception_time_s must be finite")
    displacement = state.position_at(reception) - station.position_world_m
    source_range = float(np.linalg.norm(displacement))
    if source_range <= 0.0:
        raise ValueError("source cannot coincide with station")
    direction = displacement / source_range
    delta = reception - state.reference_time_s
    direct_state_jacobian = np.hstack((np.eye(3), delta * np.eye(3)))
    return (
        (np.eye(3) - np.outer(direction, direction))
        @ direct_state_jacobian
        / source_range
    )


def stacked_instantaneous_world_direction_jacobian(
    state: ConstantVelocityState,
    station: StationPose,
    reception_times_s: tuple[float, ...] | list[float] | np.ndarray,
) -> np.ndarray:
    """Stack the independent instantaneous direction Jacobian over time."""

    times = np.asarray(reception_times_s, dtype=float)
    if times.ndim != 1 or times.size == 0 or not np.all(np.isfinite(times)):
        raise ValueError("reception_times_s must be a non-empty finite vector")
    return np.vstack(
        [
            instantaneous_world_direction_jacobian(state, station, reception)
            for reception in times
        ]
    )


def _stacked_retarded_direction_jacobian(
    state: ConstantVelocityState,
    station: StationPose,
    reception_times_s: tuple[float, ...],
    sound_speed: float,
) -> np.ndarray:
    return np.vstack(
        [
            predicted_local_direction_jacobian(
                state, station, reception, sound_speed=sound_speed
            )
            for reception in reception_times_s
        ]
    )


def finite_difference_audit(
    *, scene_count: int = DEFAULT_SCENE_COUNT, seed: int = DEFAULT_AUDIT_SEED
) -> dict[str, float | int]:
    """Audit 6-D analytic derivatives on ordinary and near-sonic scenes."""

    count = int(scene_count)
    if count < 1:
        raise ValueError("scene_count must be positive")
    rng = np.random.default_rng(seed)
    maxima = {
        "max_emission_equation_residual_s": 0.0,
        "max_analytic_numeric_emission_time_difference_s": 0.0,
        "max_emission_time_jacobian_abs_mismatch": 0.0,
        "max_local_direction_jacobian_abs_mismatch": 0.0,
        "max_tangent_residual_jacobian_abs_mismatch": 0.0,
        "max_jacobian_relative_mismatch": 0.0,
    }
    near_sonic_count = max(1, count // 20)
    ordinary_count = count - near_sonic_count
    for trial in range(count):
        rotation = Rotation.random(random_state=rng).as_matrix()
        station_position = rng.uniform(-100.0, 100.0, 3)
        station = StationPose(
            "audit", station_position, rotation, tetrahedral_array(0.2)
        )
        world_direction = rng.normal(size=3)
        world_direction /= np.linalg.norm(world_direction)
        local_direction = rotation.T @ world_direction
        if np.hypot(local_direction[0], local_direction[1]) < 0.05:
            world_direction = np.cross(world_direction, rotation[:, 2])
            world_direction /= np.linalg.norm(world_direction)
        range_m = rng.uniform(10.0, 500.0)
        velocity_direction = rng.normal(size=3)
        velocity_direction /= np.linalg.norm(velocity_direction)
        source_speed = (
            rng.uniform(0.0, 60.0)
            if trial < ordinary_count
            else rng.uniform(0.75, 0.9) * 343.0
        )
        velocity = source_speed * velocity_direction
        reference = rng.uniform(-2.0, 2.0)
        reception = reference + rng.uniform(0.2, 3.0)
        position_at_reception = station_position + range_m * world_direction
        state = ConstantVelocityState(
            position_at_reception - velocity * (reception - reference),
            velocity,
            reference,
        )
        analytic_prediction = predict_retarded_bearing(state, station, reception)
        numeric_prediction = predict_retarded_bearing(
            state, station, reception, emission_solver="numerical"
        )
        maxima["max_emission_equation_residual_s"] = max(
            maxima["max_emission_equation_residual_s"],
            abs(retarded_equation_residual_s(state, station, analytic_prediction)),
        )
        maxima["max_analytic_numeric_emission_time_difference_s"] = max(
            maxima["max_analytic_numeric_emission_time_difference_s"],
            abs(analytic_prediction.emission_time_s - numeric_prediction.emission_time_s),
        )
        measurement = _measurement(
            station, state, reception, trial, rng.normal(0.0, 0.004, 2)
        )
        comparisons = (
            (
                "max_emission_time_jacobian_abs_mismatch",
                emission_time_jacobian_wrt_state(state, station, reception),
                _central(
                    lambda candidate: predict_retarded_bearing(
                        candidate, station, reception
                    ).emission_time_s,
                    state,
                ),
            ),
            (
                "max_local_direction_jacobian_abs_mismatch",
                predicted_local_direction_jacobian(state, station, reception),
                _central(
                    lambda candidate: predict_retarded_bearing(
                        candidate, station, reception
                    ).direction_local,
                    state,
                ),
            ),
            (
                "max_tangent_residual_jacobian_abs_mismatch",
                retarded_bearing_residual_jacobian(state, station, measurement),
                _central(
                    lambda candidate: retarded_bearing_residual(
                        candidate, station, measurement
                    ),
                    state,
                ),
            ),
        )
        for name, analytic, numerical in comparisons:
            mismatch = np.abs(analytic - numerical)
            maxima[name] = max(maxima[name], float(np.max(mismatch)))
            informative = np.abs(numerical) > 1e-7
            if np.any(informative):
                maxima["max_jacobian_relative_mismatch"] = max(
                    maxima["max_jacobian_relative_mismatch"],
                    float(np.max(mismatch[informative] / np.abs(numerical[informative]))),
                )
    return {
        "scene_count": count,
        "ordinary_speed_scene_count": ordinary_count,
        "near_sonic_scene_count": near_sonic_count,
        "seed": int(seed),
        **maxima,
    }


def observability_examples() -> dict[str, float | int]:
    """Return one-station, good multi-station and poor-window diagnostics."""

    state = ConstantVelocityState([80.0, 50.0, 40.0], [8.0, -3.0, 1.0], 0.0)
    stations = validation_stations()
    radial_state = ConstantVelocityState(
        [80.0, 50.0, 40.0], [8.0, 5.0, 4.0], 0.0
    )
    one = [
        _measurement(stations[0], radial_state, reception, frame)
        for frame, reception in enumerate((1.0, 2.0, 3.0, 4.0))
    ]
    temporal = [
        _measurement(station, state, reception, frame)
        for frame, reception in enumerate((1.0, 2.5, 4.0))
        for station in stations
    ]
    poor_stations = tuple(
        StationPose(
            f"P{index}",
            [10.0 * index, 0.01 * index, 0.0],
            np.eye(3),
            tetrahedral_array(0.2),
        )
        for index in range(3)
    )
    poor = [
        _measurement(station, state, reception, frame)
        for frame, reception in enumerate((1.0, 1.001, 1.002))
        for station in poor_stations
    ]
    one_result = stack_retarded_bearing_observability(
        radial_state, stations[:1], one
    )
    one_station = stations[0]
    one_station_times = (1.0, 2.0, 3.0, 4.0)
    nonradial = [
        _measurement(one_station, state, reception, frame)
        for frame, reception in enumerate(one_station_times)
    ]
    nonradial_result = stack_retarded_bearing_observability(
        state, stations[:1], nonradial
    )
    numerical_nonradial_jacobian = _central(
        lambda candidate: np.concatenate(
            [
                retarded_bearing_residual(candidate, one_station, measurement)
                for measurement in nonradial
            ]
        ),
        state,
    )
    fd_difference = nonradial_result.jacobian - numerical_nonradial_jacobian
    fd_max_abs = float(np.max(np.abs(fd_difference)))
    fd_spectral_norm = float(np.linalg.norm(fd_difference, ord=2))

    instantaneous_jacobian = stacked_instantaneous_world_direction_jacobian(
        state, one_station, one_station_times
    )
    instantaneous_singular, instantaneous_rank, instantaneous_condition = (
        _svd_diagnostics(instantaneous_jacobian)
    )
    scale_null_direction = np.concatenate(
        (
            state.position_at_reference_world_m - one_station.position_world_m,
            state.velocity_world_mps,
        )
    )
    instantaneous_null_residual = instantaneous_jacobian @ scale_null_direction

    finite_speed_metrics: dict[str, float | int] = {}
    for sound_speed in (343.0, 3430.0, 34300.0):
        retarded_direction_jacobian = _stacked_retarded_direction_jacobian(
            state, one_station, one_station_times, sound_speed
        )
        singular_values, rank, condition = _svd_diagnostics(
            retarded_direction_jacobian
        )
        label = f"c_{int(sound_speed)}"
        finite_speed_metrics[f"one_station_nonradial_{label}_rank"] = rank
        finite_speed_metrics[
            f"one_station_nonradial_{label}_smallest_singular_value"
        ] = float(singular_values[-1])
        finite_speed_metrics[
            f"one_station_nonradial_{label}_condition_number"
        ] = condition
        finite_speed_metrics[
            f"one_station_nonradial_{label}_to_instantaneous_spectral_norm"
        ] = float(
            np.linalg.norm(
                retarded_direction_jacobian - instantaneous_jacobian, ord=2
            )
        )
    temporal_result = stack_retarded_bearing_observability(
        state, stations, temporal
    )
    poor_result = stack_retarded_bearing_observability(
        state, poor_stations, poor
    )
    return {
        "one_station_temporal_radial_rank": one_result.rank,
        "one_station_temporal_radial_smallest_singular_value": float(
            one_result.singular_values[-1]
        ),
        "one_station_temporal_nonradial_rank": nonradial_result.rank,
        "one_station_temporal_nonradial_smallest_singular_value": float(
            nonradial_result.singular_values[-1]
        ),
        "one_station_temporal_nonradial_condition_number": (
            nonradial_result.condition_number
        ),
        "one_station_temporal_nonradial_fd_max_abs_mismatch": fd_max_abs,
        "one_station_temporal_nonradial_fd_spectral_norm_mismatch": (
            fd_spectral_norm
        ),
        "one_station_temporal_nonradial_fd_max_to_smin_ratio": (
            fd_max_abs / float(nonradial_result.singular_values[-1])
        ),
        "one_station_instantaneous_rank": instantaneous_rank,
        "one_station_instantaneous_smallest_singular_value": float(
            instantaneous_singular[-1]
        ),
        "one_station_instantaneous_condition_number": instantaneous_condition,
        "one_station_instantaneous_scale_null_max_abs": float(
            np.max(np.abs(instantaneous_null_residual))
        ),
        "three_station_temporal_rank": temporal_result.rank,
        "three_station_temporal_condition_number": temporal_result.condition_number,
        "poor_geometry_short_window_rank": poor_result.rank,
        "poor_geometry_short_window_condition_number": poor_result.condition_number,
        "three_station_smallest_singular_value": float(temporal_result.singular_values[-1]),
        "poor_geometry_smallest_singular_value": float(poor_result.singular_values[-1]),
        **finite_speed_metrics,
    }


def run_retarded_bearing_validation(
    output_path: str | Path = "results/retarded_bearing_model_summary.csv",
    *,
    scene_count: int = DEFAULT_SCENE_COUNT,
    seed: int = DEFAULT_AUDIT_SEED,
) -> dict[str, float | int]:
    """Run the S7C-A numerical audit and save a compact metric CSV."""

    metrics = {**finite_difference_audit(scene_count=scene_count, seed=seed), **observability_examples()}
    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["metric", "value"])
        writer.writeheader()
        for key, value in metrics.items():
            writer.writerow({"metric": key, "value": value})
    return metrics


__all__ = [
    "DEFAULT_AUDIT_SEED",
    "DEFAULT_SCENE_COUNT",
    "finite_difference_audit",
    "instantaneous_world_direction_jacobian",
    "observability_examples",
    "run_retarded_bearing_validation",
    "stacked_instantaneous_world_direction_jacobian",
    "validation_stations",
]
