"""Bearing-level Monte Carlo for static multi-station 3-D triangulation.

Noise is injected directly in each spherical tangent plane.  This isolates
fusion geometry from GCC/SRP signal errors.  Calibration/evaluation random
streams are disjoint, and their independent realization counts are never
replaced by the number of two-component residuals.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

import numpy as np
from numpy.typing import NDArray
from scipy.spatial.transform import Rotation

from estimators.bearing_triangulation import (
    closest_rays_triangulation,
    triangulate_bearings_spherical_wls,
)
from model.bearing_statistics import calibrate_bearing_covariance, tangent_basis
from model.geometry import direction_angles, tetrahedral_array
from model.measurements import BearingMeasurement
from model.station import StationPose
from validation.study import write_summary_csv


MULTISTATION_BASE_SEED = 20260831
DEFAULT_CALIBRATION_REALIZATIONS = 128
DEFAULT_EVALUATION_REALIZATIONS = 256
CHI_SQUARE_3_P95 = 7.814727903251179
POSITION_MISMATCH_OFFSETS_M = np.asarray(
    [[0.08, -0.04, 0.03], [-0.05, 0.06, -0.02], [0.04, 0.03, 0.07]]
)
ORIENTATION_MISMATCH_ROTVECS_DEG = np.asarray(
    [[0.18, -0.10, 0.12], [-0.14, 0.20, -0.08], [0.10, 0.07, -0.22]]
)


@dataclass(frozen=True, slots=True)
class StaticStudyConfig:
    """Dimensionless station/target geometry for one study block."""

    name: str
    station_geometry: str
    baseline_m: float
    target_horizontal_range_over_baseline: float
    target_altitude_over_baseline: float
    orientation_mode: str = "aligned"


@dataclass(frozen=True, slots=True)
class StaticScenario:
    name: str
    station_indices: tuple[int, ...]
    pose_mismatch_kind: str
    bearing_covariance_mismatch: bool = False
    outlier_station_index: int | None = None


def default_static_study_configurations() -> tuple[StaticStudyConfig, ...]:
    """Return baseline/range/altitude/geometry/orientation sweep."""

    return (
        StaticStudyConfig("equilateral_near", "equilateral", 20.0, 1.0, 0.75),
        StaticStudyConfig("equilateral_mid", "equilateral", 20.0, 3.0, 1.0),
        StaticStudyConfig("equilateral_far", "equilateral", 20.0, 10.0, 2.0),
        StaticStudyConfig("equilateral_scale_small", "equilateral", 10.0, 3.0, 1.0),
        StaticStudyConfig("equilateral_scale_large", "equilateral", 50.0, 3.0, 1.0),
        StaticStudyConfig("elongated_mid", "elongated", 20.0, 3.0, 1.0),
        StaticStudyConfig("near_collinear_far", "near_collinear", 20.0, 10.0, 2.0),
        StaticStudyConfig(
            "equilateral_oriented", "equilateral", 20.0, 3.0, 1.0, "varied"
        ),
        StaticStudyConfig(
            "elongated_oriented", "elongated", 40.0, 6.0, 0.5, "varied"
        ),
    )


def default_static_scenarios() -> tuple[StaticScenario, ...]:
    """Return station-count, dropout, mismatch, and outlier comparisons."""

    return (
        StaticScenario("ideal_three", (0, 1, 2), "none"),
        StaticScenario("ideal_two", (0, 1), "none"),
        StaticScenario("one_station_failure", (0, 2), "none"),
        StaticScenario("position_calibration_mismatch", (0, 1, 2), "position"),
        StaticScenario("orientation_calibration_mismatch", (0, 1, 2), "orientation"),
        StaticScenario(
            "bearing_covariance_mismatch", (0, 1, 2), "none", True
        ),
        StaticScenario("one_erroneous_bearing", (0, 1, 2), "none", False, 2),
    )


def _normalized_station_positions(kind: str) -> NDArray[np.float64]:
    if kind == "equilateral":
        points = np.asarray(
            [[-0.5, -np.sqrt(3.0) / 6.0, 0.0],
             [0.5, -np.sqrt(3.0) / 6.0, 0.0],
             [0.0, np.sqrt(3.0) / 3.0, 0.0]]
        )
    elif kind == "elongated":
        points = np.asarray([[-0.5, -0.08, 0.0], [0.5, -0.08, 0.0], [0.28, 0.16, 0.0]])
    elif kind == "near_collinear":
        points = np.asarray([[-0.5, 0.0, 0.0], [0.0, 0.002, 0.0], [0.5, 0.0, 0.0]])
    else:
        raise ValueError(f"unknown station geometry: {kind}")
    points -= np.mean(points, axis=0, keepdims=True)
    diameter = float(
        np.max(np.linalg.norm(points[:, None, :] - points[None, :, :], axis=-1))
    )
    return points / diameter


def _rotations(mode: str) -> tuple[NDArray[np.float64], ...]:
    if mode == "aligned":
        return (np.eye(3), np.eye(3), np.eye(3))
    if mode == "varied":
        return tuple(
            Rotation.from_euler("zyx", angles).as_matrix()
            for angles in ((0.25, -0.08, 0.03), (-0.7, 0.16, -0.1), (1.1, -0.2, 0.12))
        )
    raise ValueError(f"unknown orientation mode: {mode}")


def build_static_scene(
    config: StaticStudyConfig,
) -> tuple[tuple[StationPose, ...], NDArray[np.float64]]:
    """Build true station poses and target in ENU metres."""

    positions = config.baseline_m * _normalized_station_positions(config.station_geometry)
    rotations = _rotations(config.orientation_mode)
    stations = tuple(
        StationPose(
            f"station_{index}",
            position,
            rotation,
            tetrahedral_array(),
        )
        for index, (position, rotation) in enumerate(zip(positions, rotations, strict=True))
    )
    azimuth = np.deg2rad(35.0)
    horizontal = config.target_horizontal_range_over_baseline * config.baseline_m
    target = np.asarray(
        [
            horizontal * np.cos(azimuth),
            horizontal * np.sin(azimuth),
            config.target_altitude_over_baseline * config.baseline_m,
        ]
    )
    return stations, target


def _true_noise_model(station_index: int) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    signs = np.asarray([[1.0, -1.0], [-0.6, 0.8], [0.4, 0.5]])
    mean = np.deg2rad([0.08, 0.05]) * signs[station_index]
    sigma = np.deg2rad([0.35 + 0.04 * station_index, 0.55 - 0.03 * station_index])
    correlation = (-0.2, 0.25, 0.1)[station_index]
    covariance = np.asarray(
        [
            [sigma[0] ** 2, correlation * sigma[0] * sigma[1]],
            [correlation * sigma[0] * sigma[1], sigma[1] ** 2],
        ]
    )
    return mean, covariance


def _exp_map(direction: NDArray[np.float64], tangent_coordinates: NDArray[np.float64]) -> NDArray[np.float64]:
    phi, elevation = direction_angles(direction)
    tangent = tangent_basis(phi, elevation).T @ tangent_coordinates
    theta = float(np.linalg.norm(tangent))
    if theta == 0.0:
        return direction.copy()
    return np.cos(theta) * direction + np.sin(theta) * tangent / theta


def _seed(config_index: int, role: str) -> int:
    role_code = {"calibration": 1701, "evaluation": 2903}[role]
    return int(
        np.random.SeedSequence(
            [MULTISTATION_BASE_SEED, config_index, role_code]
        ).generate_state(1)[0]
    )


def _draw_residuals(
    config_index: int,
    count: int,
    role: str,
) -> tuple[int, NDArray[np.float64]]:
    seed = _seed(config_index, role)
    generator = np.random.default_rng(seed)
    residuals = np.empty((count, 3, 2), dtype=float)
    for station_index in range(3):
        mean, covariance = _true_noise_model(station_index)
        residuals[:, station_index, :] = generator.multivariate_normal(
            mean, covariance, size=count
        )
    return seed, residuals


def _estimated_stations(
    true_stations: tuple[StationPose, ...], scenario: StaticScenario
) -> tuple[StationPose, ...]:
    rotation_offsets = tuple(
        Rotation.from_rotvec(np.deg2rad(value)).as_matrix()
        for value in ORIENTATION_MISMATCH_ROTVECS_DEG
    )
    estimated: list[StationPose] = []
    for index in scenario.station_indices:
        true = true_stations[index]
        position = true.position_world_m.copy()
        rotation = true.rotation_local_to_world.copy()
        if scenario.pose_mismatch_kind == "position":
            position += POSITION_MISMATCH_OFFSETS_M[index]
        elif scenario.pose_mismatch_kind == "orientation":
            rotation = rotation @ rotation_offsets[index]
        elif scenario.pose_mismatch_kind != "none":
            raise ValueError(f"unknown mismatch kind: {scenario.pose_mismatch_kind}")
        estimated.append(
            StationPose(
                true.station_id,
                position,
                rotation,
                true.microphone_positions_local_m,
            )
        )
    return tuple(estimated)


def _calibrations(
    calibration_residuals: NDArray[np.float64],
):
    return tuple(
        calibrate_bearing_covariance(calibration_residuals[:, index, :])
        for index in range(3)
    )


def _measurement(
    true_station: StationPose,
    target: NDArray[np.float64],
    residual: NDArray[np.float64],
    calibration,
    *,
    split: str,
    realization_index: int,
    covariance_mismatch: bool,
) -> BearingMeasurement:
    world_direction = target - true_station.position_world_m
    world_direction /= np.linalg.norm(world_direction)
    true_local = true_station.world_to_local_direction(world_direction)
    measured = _exp_map(true_local, residual)
    covariance = calibration.covariance_rad2
    if covariance_mismatch:
        distortion = np.diag([0.55, 1.8])
        covariance = distortion @ covariance @ distortion
    return BearingMeasurement(
        station_id=true_station.station_id,
        sequence_id=f"static-{split}",
        frame_index=realization_index,
        reception_center_timestamp_s=float(realization_index),
        available_timestamp_s=float(realization_index) + 0.01,
        direction_local=measured,
        covariance_tangent_rad2=covariance,
        calibration_bias_tangent_rad=calibration.mean_residual_rad,
        estimator_variant="direct_synthetic_bearing",
        quality_metadata={"synthetic_tangent_measurement": True},
    )


def _run_split(
    true_stations: tuple[StationPose, ...],
    estimated_stations: tuple[StationPose, ...],
    target: NDArray[np.float64],
    residuals: NDArray[np.float64],
    calibrations,
    scenario: StaticScenario,
    split: str,
):
    results = []
    started = perf_counter()
    for realization_index in range(residuals.shape[0]):
        measurements = []
        for station_index in scenario.station_indices:
            residual = residuals[realization_index, station_index].copy()
            if (
                scenario.outlier_station_index == station_index
                and split == "evaluation"
            ):
                residual += np.deg2rad([5.0, -2.0])
            measurements.append(
                _measurement(
                    true_stations[station_index],
                    target,
                    residual,
                    calibrations[station_index],
                    split=split,
                    realization_index=realization_index,
                    covariance_mismatch=scenario.bearing_covariance_mismatch,
                )
            )
        results.append(
            triangulate_bearings_spherical_wls(estimated_stations, measurements)
        )
    runtime = perf_counter() - started
    return results, runtime


def _json_array(value: NDArray[np.float64]) -> str:
    return json.dumps(np.asarray(value, dtype=float).tolist(), separators=(",", ":"))


def _summary_metrics(
    results,
    target: NDArray[np.float64],
    *,
    calibration_position_bias: NDArray[np.float64] | None,
) -> tuple[dict[str, object], NDArray[np.float64]]:
    valid = np.asarray([result.valid for result in results], dtype=bool)
    count = len(results)
    valid_results = [result for result in results if result.valid]
    if not valid_results:
        return {
            "coverage": 0.0,
            "failure_fraction": 1.0,
            "position_rmse_m": None,
            "position_error_p50_m": None,
            "position_error_p95_m": None,
            "position_error_p99_m": None,
            "horizontal_rmse_m": None,
            "horizontal_error_p95_m": None,
            "vertical_rmse_m": None,
            "vertical_error_p95_m": None,
            "bias_x_m": None,
            "bias_y_m": None,
            "bias_z_m": None,
            "empirical_position_covariance_m2_json": None,
            "mean_predicted_position_covariance_m2_json": None,
            "predicted_empirical_covariance_relative_frobenius": None,
            "predicted_empirical_covariance_trace_ratio": None,
            "normalized_position_error_p50": None,
            "normalized_position_error_p95": None,
            "position_ellipsoid_p95_coverage": None,
            "information_rank_min": 0,
            "information_condition_median": None,
            "information_condition_max": None,
            "information_eigenvalue_median_json": None,
            "mean_runtime_per_estimate_s": None,
        }, np.full(3, np.nan)
    errors = np.asarray([result.position_world_m - target for result in valid_results])
    distances = np.linalg.norm(errors, axis=1)
    horizontal = np.linalg.norm(errors[:, :2], axis=1)
    vertical = np.abs(errors[:, 2])
    bias = np.mean(errors, axis=0)
    empirical_covariance = (
        np.cov(errors, rowvar=False, ddof=1)
        if errors.shape[0] >= 2
        else np.full((3, 3), np.nan)
    )
    predicted_covariances = np.asarray(
        [result.covariance_position_m2 for result in valid_results]
    )
    mean_predicted = np.mean(predicted_covariances, axis=0)
    denominator = max(float(np.linalg.norm(empirical_covariance, ord="fro")), np.finfo(float).tiny)
    covariance_relative = float(
        np.linalg.norm(mean_predicted - empirical_covariance, ord="fro") / denominator
    )
    trace_ratio = float(np.trace(empirical_covariance) / np.trace(mean_predicted))
    center = bias if calibration_position_bias is None else calibration_position_bias
    normalized = np.asarray(
        [
            float(
                (error - center)
                @ np.linalg.pinv(result.covariance_position_m2, rcond=1e-12, hermitian=True)
                @ (error - center)
            )
            for error, result in zip(errors, valid_results, strict=True)
        ]
    )
    conditions = np.asarray(
        [result.information_condition_number for result in valid_results]
    )
    eigenvalues = np.asarray(
        [result.information_eigenvalues for result in valid_results]
    )
    metrics: dict[str, object] = {
        "coverage": float(np.mean(valid)),
        "failure_fraction": float(1.0 - np.mean(valid)),
        "position_rmse_m": float(np.sqrt(np.mean(distances**2))),
        "position_error_p50_m": float(np.percentile(distances, 50.0)),
        "position_error_p95_m": float(np.percentile(distances, 95.0)),
        "position_error_p99_m": float(np.percentile(distances, 99.0)),
        "horizontal_rmse_m": float(np.sqrt(np.mean(horizontal**2))),
        "horizontal_error_p95_m": float(np.percentile(horizontal, 95.0)),
        "vertical_rmse_m": float(np.sqrt(np.mean(vertical**2))),
        "vertical_error_p95_m": float(np.percentile(vertical, 95.0)),
        "bias_x_m": float(bias[0]),
        "bias_y_m": float(bias[1]),
        "bias_z_m": float(bias[2]),
        "empirical_position_covariance_m2_json": _json_array(empirical_covariance),
        "mean_predicted_position_covariance_m2_json": _json_array(mean_predicted),
        "predicted_empirical_covariance_relative_frobenius": covariance_relative,
        "predicted_empirical_covariance_trace_ratio": trace_ratio,
        "normalized_position_error_p50": float(np.percentile(normalized, 50.0)),
        "normalized_position_error_p95": float(np.percentile(normalized, 95.0)),
        "position_ellipsoid_p95_coverage": float(np.mean(normalized <= CHI_SQUARE_3_P95)),
        "information_rank_min": int(min(result.information_rank for result in valid_results)),
        "information_condition_median": float(np.median(conditions)),
        "information_condition_max": float(np.max(conditions)),
        "information_eigenvalue_median_json": _json_array(np.median(eigenvalues, axis=0)),
    }
    return metrics, bias


def _ray_angles(stations: tuple[StationPose, ...], target: NDArray[np.float64]):
    directions = []
    for station in stations:
        direction = target - station.position_world_m
        directions.append(direction / np.linalg.norm(direction))
    angles = []
    for first in range(len(directions)):
        for second in range(first + 1, len(directions)):
            angles.append(
                np.rad2deg(
                    np.arccos(np.clip(directions[first] @ directions[second], -1.0, 1.0))
                )
            )
    return float(np.min(angles)), float(np.max(angles))


def _geometry_record(config: StaticStudyConfig, stations, target):
    positions = np.asarray([station.position_world_m for station in stations])
    baseline = float(
        np.max(np.linalg.norm(positions[:, None] - positions[None, :], axis=-1))
    )
    min_angle, max_angle = _ray_angles(stations, target)
    ideal_measurements = []
    for station in stations:
        direction = target - station.position_world_m
        direction /= np.linalg.norm(direction)
        local = station.world_to_local_direction(direction)
        ideal_measurements.append(
            BearingMeasurement(
                station.station_id,
                "geometry",
                0,
                0.0,
                0.01,
                local,
                np.eye(2),
                np.zeros(2),
                "geometry_only",
            )
        )
    closest = closest_rays_triangulation(stations, ideal_measurements)
    triangle_area = 0.5 * float(
        np.linalg.norm(np.cross(positions[1] - positions[0], positions[2] - positions[0]))
    )
    return {
        "configuration": config.name,
        "station_geometry": config.station_geometry,
        "station_count": 3,
        "orientation_mode": config.orientation_mode,
        "baseline_m": baseline,
        "station_positions_world_m_json": _json_array(positions),
        "target_coordinates_world_m_json": _json_array(target),
        "target_x_m": float(target[0]),
        "target_y_m": float(target[1]),
        "target_z_m": float(target[2]),
        "target_range_over_baseline": float(np.linalg.norm(target) / baseline),
        "target_horizontal_range_over_baseline": config.target_horizontal_range_over_baseline,
        "target_altitude_over_baseline": config.target_altitude_over_baseline,
        "minimum_ray_intersection_angle_deg": min_angle,
        "maximum_ray_intersection_angle_deg": max_angle,
        "station_triangle_area_m2": triangle_area,
        "closest_rays_rank": closest.rank,
        "closest_rays_condition_number": closest.condition_number,
        "closest_rays_eigenvalues_json": _json_array(closest.eigenvalues),
        "truth_used_online": False,
        "dynamic_tracking_implemented": False,
        "retarded_time_fusion_implemented": False,
    }


def run_static_configuration(
    config: StaticStudyConfig,
    config_index: int,
    *,
    calibration_realizations: int = DEFAULT_CALIBRATION_REALIZATIONS,
    evaluation_realizations: int = DEFAULT_EVALUATION_REALIZATIONS,
):
    """Run all scenarios for one physical geometry with common random numbers."""

    if calibration_realizations < 4 or evaluation_realizations < 4:
        raise ValueError("each split requires at least four independent realizations")
    true_stations, target = build_static_scene(config)
    calibration_seed, calibration_residuals = _draw_residuals(
        config_index, calibration_realizations, "calibration"
    )
    evaluation_seed, evaluation_residuals = _draw_residuals(
        config_index, evaluation_realizations, "evaluation"
    )
    if calibration_seed == evaluation_seed:
        raise RuntimeError("calibration/evaluation bearing-noise seeds overlap")
    calibrations = _calibrations(calibration_residuals)
    min_angle, max_angle = _ray_angles(true_stations, target)
    records: list[dict[str, object]] = []
    for scenario in default_static_scenarios():
        estimated_stations = _estimated_stations(true_stations, scenario)
        calibration_results, calibration_runtime = _run_split(
            true_stations,
            estimated_stations,
            target,
            calibration_residuals,
            calibrations,
            scenario,
            "calibration",
        )
        calibration_metrics, calibration_position_bias = _summary_metrics(
            calibration_results, target, calibration_position_bias=None
        )
        evaluation_results, evaluation_runtime = _run_split(
            true_stations,
            estimated_stations,
            target,
            evaluation_residuals,
            calibrations,
            scenario,
            "evaluation",
        )
        evaluation_metrics, _ = _summary_metrics(
            evaluation_results,
            target,
            calibration_position_bias=calibration_position_bias,
        )
        for split, count, seed, metrics, runtime in (
            (
                "calibration",
                calibration_realizations,
                calibration_seed,
                calibration_metrics,
                calibration_runtime,
            ),
            (
                "evaluation",
                evaluation_realizations,
                evaluation_seed,
                evaluation_metrics,
                evaluation_runtime,
            ),
        ):
            calibration_json = [
                {
                    "station_id": true_stations[index].station_id,
                    "mu_cal_rad": calibrations[index].mean_residual_rad.tolist(),
                    "R_cal_rad2": calibrations[index].covariance_rad2.tolist(),
                    "R_rank": calibrations[index].rank,
                    "R_condition": calibrations[index].condition_number,
                }
                for index in scenario.station_indices
            ]
            row: dict[str, object] = {
                "configuration": f"{config.name}__{scenario.name}",
                "physical_configuration": config.name,
                "scenario": scenario.name,
                "split": split,
                "station_count": len(scenario.station_indices),
                "contributing_station_ids_json": json.dumps(
                    [true_stations[index].station_id for index in scenario.station_indices]
                ),
                "station_geometry": config.station_geometry,
                "orientation_mode": config.orientation_mode,
                "baseline_m": config.baseline_m,
                "target_coordinates_world_m_json": _json_array(target),
                "target_x_m": float(target[0]),
                "target_y_m": float(target[1]),
                "target_z_m": float(target[2]),
                "target_range_over_baseline": float(np.linalg.norm(target) / config.baseline_m),
                "target_horizontal_range_over_baseline": config.target_horizontal_range_over_baseline,
                "target_altitude_over_baseline": config.target_altitude_over_baseline,
                "minimum_ray_intersection_angle_deg": min_angle,
                "maximum_ray_intersection_angle_deg": max_angle,
                "true_angular_bias_and_covariance_json": json.dumps(
                    [
                        {
                            "station_id": true_stations[index].station_id,
                            "mu_rad": _true_noise_model(index)[0].tolist(),
                            "R_rad2": _true_noise_model(index)[1].tolist(),
                        }
                        for index in scenario.station_indices
                    ],
                    separators=(",", ":"),
                ),
                "calibrated_bearing_statistics_json": json.dumps(
                    calibration_json, separators=(",", ":")
                ),
                "bearing_covariance_source_split": "calibration",
                "evaluation_residual_used_to_fit_bearing_covariance": False,
                "pose_mismatch_kind": scenario.pose_mismatch_kind,
                "position_pose_mismatch_max_m": (
                    float(np.max(np.linalg.norm(POSITION_MISMATCH_OFFSETS_M, axis=1)))
                    if scenario.pose_mismatch_kind == "position"
                    else 0.0
                ),
                "orientation_pose_mismatch_max_deg": (
                    float(np.max(np.linalg.norm(ORIENTATION_MISMATCH_ROTVECS_DEG, axis=1)))
                    if scenario.pose_mismatch_kind == "orientation"
                    else 0.0
                ),
                "bearing_covariance_mismatch": scenario.bearing_covariance_mismatch,
                "erroneous_bearing_station_id": (
                    None
                    if scenario.outlier_station_index is None
                    else true_stations[scenario.outlier_station_index].station_id
                ),
                "erroneous_bearing_offset_az_el_deg_json": (
                    None if scenario.outlier_station_index is None else "[5.0,-2.0]"
                ),
                "independent_realization_count": count,
                "dependent_residual_component_count": count * len(scenario.station_indices) * 2,
                "residual_components_are_independent_trials": False,
                "base_seed": MULTISTATION_BASE_SEED,
                "split_bearing_noise_seed": seed,
                "calibration_bearing_noise_seed": calibration_seed,
                "evaluation_bearing_noise_seed": evaluation_seed,
                "calibration_evaluation_seed_overlap_count": 0,
                "seed_scope": "independent_bearing_level_realizations",
                "position_bias_centering_source_split": "calibration",
                "evaluation_mean_used_for_position_error_centering": False,
                "local_gaussian_covariance_benchmark": True,
                "exact_position_crlb_claimed": False,
                "truth_used_online": False,
                "dynamic_tracking_implemented": False,
                "retarded_time_fusion_implemented": False,
                "mean_runtime_per_estimate_s": runtime / count,
            }
            row.update(metrics)
            records.append(row)
    return records, _geometry_record(config, true_stations, target)


def run_multistation_static_study(
    *,
    calibration_realizations: int = DEFAULT_CALIBRATION_REALIZATIONS,
    evaluation_realizations: int = DEFAULT_EVALUATION_REALIZATIONS,
    summary_csv: str | Path = "results/multistation_static_summary.csv",
    geometry_csv: str | Path = "results/multistation_geometry_summary.csv",
):
    """Run the full bearing-level static study and write both result tables."""

    records: list[dict[str, object]] = []
    geometries: list[dict[str, object]] = []
    calibration_seeds: set[int] = set()
    evaluation_seeds: set[int] = set()
    for config_index, config in enumerate(default_static_study_configurations()):
        configuration_records, geometry = run_static_configuration(
            config,
            config_index,
            calibration_realizations=calibration_realizations,
            evaluation_realizations=evaluation_realizations,
        )
        records.extend(configuration_records)
        geometries.append(geometry)
        calibration_seeds.add(_seed(config_index, "calibration"))
        evaluation_seeds.add(_seed(config_index, "evaluation"))
    overlap = calibration_seeds & evaluation_seeds
    if overlap:
        raise RuntimeError(f"calibration/evaluation seeds overlap: {sorted(overlap)}")
    write_summary_csv(records, summary_csv)
    write_summary_csv(geometries, geometry_csv)
    return records, geometries


__all__ = [
    "DEFAULT_CALIBRATION_REALIZATIONS",
    "DEFAULT_EVALUATION_REALIZATIONS",
    "MULTISTATION_BASE_SEED",
    "StaticScenario",
    "StaticStudyConfig",
    "build_static_scene",
    "default_static_scenarios",
    "default_static_study_configurations",
    "run_multistation_static_study",
    "run_static_configuration",
]
