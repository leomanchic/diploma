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
) -> BearingMeasurement:
    prediction = predict_retarded_bearing(state, station, reception)
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
    one = [_measurement(stations[0], state, 1.0, 0)]
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
    one_result = stack_retarded_bearing_observability(state, stations[:1], one)
    temporal_result = stack_retarded_bearing_observability(
        state, stations, temporal
    )
    poor_result = stack_retarded_bearing_observability(
        state, poor_stations, poor
    )
    return {
        "one_station_single_bearing_rank": one_result.rank,
        "three_station_temporal_rank": temporal_result.rank,
        "three_station_temporal_condition_number": temporal_result.condition_number,
        "poor_geometry_short_window_rank": poor_result.rank,
        "poor_geometry_short_window_condition_number": poor_result.condition_number,
        "three_station_smallest_singular_value": float(temporal_result.singular_values[-1]),
        "poor_geometry_smallest_singular_value": float(poor_result.singular_values[-1]),
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
    "observability_examples",
    "run_retarded_bearing_validation",
    "validation_stations",
]
