"""S7A calibration/evaluation benchmark for frame-wise bearing uncertainty.

The independent statistical unit is a complete continuous sequence. Overlap
frames may contribute residual samples but are explicitly dependent. No truth
or angular error is passed into an online estimator or quality metric.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np
from numpy.typing import NDArray
from scipy.stats import chi2, spearmanr

from model.bearing_statistics import (
    AntipodalDirectionError,
    BearingCovarianceCalibration,
    calibrate_bearing_covariance,
    normalized_innovation_squared,
    tangent_residual,
)
from model.geometry import comparison_arrays
from simulation.continuous_stream import synthesize_continuous_stream
from validation.gcc_statistical import _write_records
from validation.moving_source_study import MOVING_STUDY_METHODS, estimate_independent_frame
from validation.sequential_doa_study import (
    SequentialStudyConfig,
    estimate_stream_frames_sequentially,
    trajectory_for_sequence,
)


BEARING_UNCERTAINTY_BASE_SEED = 20260901
CALIBRATION_SEQUENCE_COUNT = 3
EVALUATION_SEQUENCE_COUNT = 3
BEARING_STUDY_DURATION_S = 0.125
BEARING_STUDY_SNR_DB = (-6.0, 5.0, 20.0)
BEARING_STUDY_TRAJECTORIES = ("stationary", "transverse", "piecewise")
BEARING_STUDY_SIGNALS = ("random_broadband", "deterministic_multisine")
BEARING_STUDY_GEOMETRIES = ("tetrahedral", "square")
SPLIT_CODES = {"calibration": 7101, "evaluation": 7201}


@dataclass(frozen=True)
class BearingUncertaintyConfig:
    geometry: str
    snr_db: float
    signal_model: str
    trajectory_kind: str


def default_bearing_uncertainty_configurations() -> tuple[BearingUncertaintyConfig, ...]:
    return tuple(
        BearingUncertaintyConfig(geometry, snr, signal, trajectory)
        for geometry in BEARING_STUDY_GEOMETRIES
        for snr in BEARING_STUDY_SNR_DB
        for signal in BEARING_STUDY_SIGNALS
        for trajectory in BEARING_STUDY_TRAJECTORIES
    )


def sequence_seed(split: str, configuration_index: int, sequence_index: int) -> int:
    """Return a deterministic split-scoped seed with disjoint role codes."""

    if split not in SPLIT_CODES:
        raise ValueError("split must be calibration or evaluation")
    sequence = np.random.SeedSequence(
        [
            BEARING_UNCERTAINTY_BASE_SEED,
            SPLIT_CODES[split],
            int(configuration_index),
            int(sequence_index),
        ]
    )
    return int(sequence.generate_state(1, dtype=np.uint32)[0])


def split_sequence_seeds(
    split: str, configuration_index: int, count: int
) -> tuple[int, ...]:
    if int(count) < 1:
        raise ValueError("sequence count must be positive")
    return tuple(sequence_seed(split, configuration_index, index) for index in range(int(count)))


def _sequence_rows(
    config: BearingUncertaintyConfig,
    configuration_index: int,
    split: str,
    count: int,
    *,
    duration_s: float,
) -> tuple[list[dict[str, object]], tuple[int, ...], tuple[int, ...], tuple[int, ...]]:
    positions = comparison_arrays()[config.geometry]
    rows: list[dict[str, object]] = []
    seeds = split_sequence_seeds(split, configuration_index, count)
    source_seeds: list[int] = []
    noise_seeds: list[int] = []
    for sequence_index, seed in enumerate(seeds):
        sequence_config = SequentialStudyConfig(
            name=f"s7a_{split}_{configuration_index}_{sequence_index}",
            trajectory_kind=config.trajectory_kind,
            snr_db=config.snr_db,
            geometry=config.geometry,
            signal_model=config.signal_model,
            duration_s=duration_s,
        )
        trajectory = trajectory_for_sequence(sequence_config, positions)
        stream = synthesize_continuous_stream(
            positions,
            trajectory,
            duration_s=duration_s,
            reception_start_time_s=0.1,
            signal_model=config.signal_model,
            snr_db=config.snr_db,
            seed=seed,
        )
        source_seeds.append(stream.source_seed)
        noise_seeds.append(stream.noise_seed)
        sequence_rows = estimate_stream_frames_sequentially(
            stream,
            positions,
            trajectory,
            sequence_name=sequence_config.name,
            frame_estimator=estimate_independent_frame,
        )
        for row in sequence_rows:
            row["split"] = split
            row["sequence_index_within_group"] = sequence_index
            row["sequence_seed"] = seed
            row["geometry"] = config.geometry
            row["snr_db"] = config.snr_db
            row["signal_model"] = config.signal_model
            row["trajectory_kind"] = config.trajectory_kind
            row["overlapping_frames_are_statistically_dependent"] = True
            row["frame_count_is_independent_trial_count"] = False
        rows.extend(sequence_rows)
    return rows, seeds, tuple(source_seeds), tuple(noise_seeds)


def _method_rows(rows: list[dict[str, object]], method: str) -> list[dict[str, object]]:
    return [row for row in rows if row["estimator_variant"] == method]


def _residual_data(rows: list[dict[str, object]]) -> dict[str, object]:
    residuals: list[NDArray[np.float64]] = []
    errors_rad: list[float] = []
    antipodal_count = 0
    valid_count = 0
    for row in rows:
        if not bool(row["valid"]):
            continue
        valid_count += 1
        truth = np.asarray([row["truth_x"], row["truth_y"], row["truth_z"]], dtype=float)
        estimate = np.asarray([row["estimate_x"], row["estimate_y"], row["estimate_z"]], dtype=float)
        errors_rad.append(float(np.deg2rad(row["geodesic_angular_error_deg"])))
        try:
            residuals.append(tangent_residual(truth, estimate))
        except AntipodalDirectionError:
            antipodal_count += 1
    return {
        "residuals": np.asarray(residuals, dtype=float).reshape((-1, 2)),
        "errors_rad": np.asarray(errors_rad, dtype=float),
        "valid_count": valid_count,
        "antipodal_count": antipodal_count,
    }


def _error_metrics(errors_rad: NDArray[np.float64]) -> dict[str, float | None]:
    if errors_rad.size == 0:
        return {
            "geodesic_rmse_rad": None,
            "geodesic_rmse_deg": None,
            "geodesic_p50_deg": None,
            "geodesic_p95_deg": None,
            "geodesic_p99_deg": None,
            "fraction_error_gt_5deg": None,
            "fraction_error_gt_10deg": None,
            "fraction_error_gt_30deg": None,
        }
    degrees = np.rad2deg(errors_rad)
    rmse = float(np.sqrt(np.mean(errors_rad**2)))
    return {
        "geodesic_rmse_rad": rmse,
        "geodesic_rmse_deg": float(np.rad2deg(rmse)),
        "geodesic_p50_deg": float(np.percentile(degrees, 50.0)),
        "geodesic_p95_deg": float(np.percentile(degrees, 95.0)),
        "geodesic_p99_deg": float(np.percentile(degrees, 99.0)),
        "fraction_error_gt_5deg": float(np.mean(degrees > 5.0)),
        "fraction_error_gt_10deg": float(np.mean(degrees > 10.0)),
        "fraction_error_gt_30deg": float(np.mean(degrees > 30.0)),
    }


def _matrix_json(matrix: NDArray[np.float64] | None) -> str | None:
    return None if matrix is None else json.dumps(np.asarray(matrix, dtype=float).tolist())


def _covariance_fields(calibration: BearingCovarianceCalibration | None) -> dict[str, object]:
    if calibration is None:
        return {
            "covariance_available": False,
            "covariance_rad2_json": None,
            "covariance_00_rad2": None,
            "covariance_01_rad2": None,
            "covariance_11_rad2": None,
            "covariance_eigenvalue_min_rad2": None,
            "covariance_eigenvalue_max_rad2": None,
            "covariance_condition_number": None,
            "covariance_correlation": None,
            "covariance_rank": None,
            "covariance_symmetric": None,
            "covariance_positive_semidefinite": None,
        }
    covariance = calibration.covariance_rad2
    return {
        "covariance_available": True,
        "covariance_rad2_json": _matrix_json(covariance),
        "covariance_00_rad2": float(covariance[0, 0]),
        "covariance_01_rad2": float(covariance[0, 1]),
        "covariance_11_rad2": float(covariance[1, 1]),
        "covariance_eigenvalue_min_rad2": float(calibration.eigenvalues_rad2[0]),
        "covariance_eigenvalue_max_rad2": float(calibration.eigenvalues_rad2[1]),
        "covariance_condition_number": calibration.condition_number,
        "covariance_correlation": calibration.correlation,
        "covariance_rank": calibration.rank,
        "covariance_symmetric": calibration.symmetric,
        "covariance_positive_semidefinite": calibration.positive_semidefinite,
    }


def _nis_fields(
    residuals: NDArray[np.float64], calibration: BearingCovarianceCalibration | None
) -> dict[str, object]:
    benchmark = {quantile: float(chi2.ppf(quantile / 100.0, df=2)) for quantile in (50, 95, 99)}
    if calibration is None or residuals.size == 0:
        return {
            "nis_sample_count": 0,
            "nis_p50": None,
            "nis_p95": None,
            "nis_p99": None,
            "chi_square_2_p50": benchmark[50],
            "chi_square_2_p95": benchmark[95],
            "chi_square_2_p99": benchmark[99],
            "nis_fraction_gt_chi_square_2_p95": None,
        }
    nis = np.asarray(
        normalized_innovation_squared(residuals, calibration.covariance_rad2), dtype=float
    )
    return {
        "nis_sample_count": int(nis.size),
        "nis_p50": float(np.percentile(nis, 50.0)),
        "nis_p95": float(np.percentile(nis, 95.0)),
        "nis_p99": float(np.percentile(nis, 99.0)),
        "chi_square_2_p50": benchmark[50],
        "chi_square_2_p95": benchmark[95],
        "chi_square_2_p99": benchmark[99],
        "nis_fraction_gt_chi_square_2_p95": float(np.mean(nis > benchmark[95])),
    }


def _finite_correlation(metric: NDArray[np.float64], error: NDArray[np.float64]) -> tuple[int, float | None, float | None]:
    usable = np.isfinite(metric) & np.isfinite(error)
    x, y = metric[usable], error[usable]
    if x.size < 3 or np.ptp(x) <= 0.0 or np.ptp(y) <= 0.0:
        return int(x.size), None, None
    return int(x.size), float(spearmanr(x, y).statistic), float(np.corrcoef(x, y)[0, 1])


def _quality_metrics(method: str) -> tuple[str, ...]:
    if method.startswith("reference") or method.startswith("all_6"):
        return (
            "gcc_mean_peak_ratio",
            "gcc_minimum_peak_ratio",
            "gcc_mean_peak_curvature",
            "gcc_total_spectral_energy",
            "gcc_boundary_count",
            "gcc_valid_pair_count",
        )
    return (
        "srp_peak_score",
        "srp_score_margin",
        "srp_local_curvature_eigenvalue_min",
        "srp_local_curvature_eigenvalue_max",
        "srp_used_spectral_energy",
        "srp_mean_spectral_energy_fraction",
        "boundary_hit",
    )


def _quality_value(row: dict[str, object], metric: str) -> float:
    if metric.startswith("srp_local_curvature_eigenvalue"):
        values = row.get("srp_local_curvature_eigenvalues")
        if values is None:
            return float("nan")
        eigenvalues = np.asarray(values, dtype=float)
        return float(eigenvalues[0 if metric.endswith("min") else 1])
    value = row.get(metric)
    return float(value) if value is not None else float("nan")


def _quality_records(
    config: BearingUncertaintyConfig,
    split: str,
    method: str,
    rows: list[dict[str, object]],
    seeds: tuple[int, ...],
) -> list[dict[str, object]]:
    valid_rows = [row for row in rows if bool(row["valid"])]
    errors = np.asarray([row["geodesic_angular_error_deg"] for row in valid_rows], dtype=float)
    records: list[dict[str, object]] = []
    for metric_name in _quality_metrics(method):
        values = np.asarray([_quality_value(row, metric_name) for row in valid_rows])
        count, spearman, pearson = _finite_correlation(values, np.abs(errors))
        finite = values[np.isfinite(values)]
        records.append(
            {
                "split": split,
                "sequence_count": len(seeds),
                "dependent_frame_count": len(rows),
                "estimator_variant": method,
                "geometry": config.geometry,
                "snr_db": config.snr_db,
                "signal_model": config.signal_model,
                "trajectory_kind": config.trajectory_kind,
                "quality_metric": metric_name,
                "quality_sample_count": count,
                "quality_mean": float(np.mean(finite)) if finite.size else None,
                "quality_p50": float(np.percentile(finite, 50.0)) if finite.size else None,
                "spearman_correlation_with_absolute_angular_error": spearman,
                "pearson_correlation_with_absolute_angular_error": pearson,
                "sequence_seeds_json": json.dumps(seeds),
                "seed_scope": f"s7a_{split}_independent_continuous_sequences",
                "quality_score_probability_claimed": False,
                "quality_metric_uses_truth_online": False,
                "offline_error_correlation_uses_truth": True,
                "overlapping_frames_are_statistically_dependent": True,
                "dependent_frames_are_independent_trials": False,
            }
        )
    return records


def run_bearing_uncertainty_configuration(
    config: BearingUncertaintyConfig,
    configuration_index: int,
    *,
    calibration_sequence_count: int = CALIBRATION_SEQUENCE_COUNT,
    evaluation_sequence_count: int = EVALUATION_SEQUENCE_COUNT,
    duration_s: float = BEARING_STUDY_DURATION_S,
) -> tuple[list[dict[str, object]], list[dict[str, object]], dict[str, list[dict[str, object]]]]:
    split_rows: dict[str, list[dict[str, object]]] = {}
    split_metadata: dict[str, tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]]] = {}
    for split, count in (
        ("calibration", calibration_sequence_count),
        ("evaluation", evaluation_sequence_count),
    ):
        rows, seeds, source_seeds, noise_seeds = _sequence_rows(
            config, configuration_index, split, count, duration_s=duration_s
        )
        split_rows[split] = rows
        split_metadata[split] = seeds, source_seeds, noise_seeds
    calibration_seeds = set(split_metadata["calibration"][0])
    evaluation_seeds = set(split_metadata["evaluation"][0])
    if calibration_seeds & evaluation_seeds:
        raise RuntimeError("calibration and evaluation sequence seeds overlap")
    covariance_records: list[dict[str, object]] = []
    quality_records: list[dict[str, object]] = []
    for method in MOVING_STUDY_METHODS:
        calibration_rows = _method_rows(split_rows["calibration"], method)
        calibration_data = _residual_data(calibration_rows)
        calibration_residuals = calibration_data["residuals"]
        calibration = (
            calibrate_bearing_covariance(calibration_residuals)
            if calibration_residuals.shape[0] >= 2
            else None
        )
        for split in ("calibration", "evaluation"):
            rows = _method_rows(split_rows[split], method)
            data = _residual_data(rows)
            residuals = data["residuals"]
            errors = data["errors_rad"]
            seeds, source_seeds, noise_seeds = split_metadata[split]
            bias = np.mean(residuals, axis=0) if residuals.size else np.full(2, np.nan)
            covariance_records.append(
                {
                    "split": split,
                    "sequence_count": len(seeds),
                    "dependent_frame_count": len(rows),
                    "successful_frame_count": int(data["valid_count"]),
                    "invalid_frame_count": len(rows) - int(data["valid_count"]),
                    "coverage": int(data["valid_count"]) / len(rows),
                    "residual_sample_count": int(residuals.shape[0]),
                    "antipodal_residual_count": int(data["antipodal_count"]),
                    "estimator_variant": method,
                    "geometry": config.geometry,
                    "snr_db": config.snr_db,
                    "signal_model": config.signal_model,
                    "trajectory_kind": config.trajectory_kind,
                    "bias_az_arc_rad": float(bias[0]) if np.isfinite(bias[0]) else None,
                    "bias_el_arc_rad": float(bias[1]) if np.isfinite(bias[1]) else None,
                    "bias_norm_deg": float(np.rad2deg(np.linalg.norm(bias)))
                    if np.all(np.isfinite(bias))
                    else None,
                    "radial_tangent_rmse_rad": float(
                        np.sqrt(np.mean(np.sum(residuals**2, axis=1)))
                    )
                    if residuals.size
                    else None,
                    **_error_metrics(errors),
                    **_covariance_fields(calibration),
                    **_nis_fields(residuals, calibration),
                    "covariance_source_split": "calibration",
                    "covariance_fit_used_this_split": split == "calibration",
                    "evaluation_residual_used_to_fit_covariance": False,
                    "conditioning_metadata": "sample_covariance_no_regularization_pseudoinverse_for_nis",
                    "sequence_seeds_json": json.dumps(seeds),
                    "source_seeds_json": json.dumps(source_seeds),
                    "noise_seeds_json": json.dumps(noise_seeds),
                    "calibration_sequence_seeds_json": json.dumps(
                        split_metadata["calibration"][0]
                    ),
                    "evaluation_sequence_seeds_json": json.dumps(
                        split_metadata["evaluation"][0]
                    ),
                    "seed_scope": f"s7a_{split}_independent_continuous_sequences",
                    "calibration_evaluation_seed_overlap": False,
                    "source_noise_realizations_disjoint_between_splits": True,
                    "common_signal_and_noise_across_methods_within_sequence": True,
                    "overlapping_frames_are_statistically_dependent": True,
                    "dependent_frames_are_independent_trials": False,
                    "truth_used_by_online_estimator": False,
                    "quality_score_probability_claimed": False,
                    "chi_square_2_is_diagnostic_gaussian_benchmark": True,
                    "gaussian_distribution_claimed": False,
                    "signal_level_crlb_claimed": False,
                    "tracking_implemented": False,
                }
            )
            quality_records.extend(_quality_records(config, split, method, rows, seeds))
    return covariance_records, quality_records, split_rows


def run_s7a_gates() -> dict[str, object]:
    """Run deterministic API and small independent-sequence smoke gates."""

    config = BearingUncertaintyConfig("tetrahedral", 20.0, "random_broadband", "stationary")
    covariance, quality, rows = run_bearing_uncertainty_configuration(
        config,
        999_001,
        calibration_sequence_count=2,
        evaluation_sequence_count=2,
        duration_s=0.05,
    )
    calibration = [row for row in covariance if row["split"] == "calibration"]
    passed = (
        all(row["covariance_symmetric"] and row["covariance_positive_semidefinite"] for row in calibration)
        and all(not row["evaluation_residual_used_to_fit_covariance"] for row in covariance)
        and not set(json.loads(covariance[0]["calibration_sequence_seeds_json"]))
        & set(json.loads(covariance[0]["evaluation_sequence_seeds_json"]))
        and all(not row["truth_used_by_estimator"] for split in rows.values() for row in split)
    )
    if not passed:
        raise RuntimeError("S7A deterministic/smoke gate failed")
    return {
        "passed": True,
        "calibration_sequence_count": 2,
        "evaluation_sequence_count": 2,
        "covariance_record_count": len(covariance),
        "quality_record_count": len(quality),
    }


def run_bearing_uncertainty_study(
    *,
    configurations: tuple[BearingUncertaintyConfig, ...] | None = None,
    calibration_sequence_count: int = CALIBRATION_SEQUENCE_COUNT,
    evaluation_sequence_count: int = EVALUATION_SEQUENCE_COUNT,
    duration_s: float = BEARING_STUDY_DURATION_S,
    covariance_output_csv: str | Path = "results/bearing_covariance_summary.csv",
    quality_output_csv: str | Path = "results/bearing_quality_summary.csv",
    run_gates: bool = True,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    if run_gates:
        run_s7a_gates()
    selected = (
        default_bearing_uncertainty_configurations()
        if configurations is None
        else configurations
    )
    covariance_records: list[dict[str, object]] = []
    quality_records: list[dict[str, object]] = []
    for index, config in enumerate(selected):
        covariance, quality, _ = run_bearing_uncertainty_configuration(
            config,
            index,
            calibration_sequence_count=calibration_sequence_count,
            evaluation_sequence_count=evaluation_sequence_count,
            duration_s=duration_s,
        )
        covariance_records.extend(covariance)
        quality_records.extend(quality)
    _write_records(covariance_records, covariance_output_csv)
    _write_records(quality_records, quality_output_csv)
    return covariance_records, quality_records


__all__ = [
    "BEARING_UNCERTAINTY_BASE_SEED",
    "BEARING_STUDY_DURATION_S",
    "CALIBRATION_SEQUENCE_COUNT",
    "EVALUATION_SEQUENCE_COUNT",
    "BearingUncertaintyConfig",
    "default_bearing_uncertainty_configurations",
    "run_bearing_uncertainty_configuration",
    "run_bearing_uncertainty_study",
    "run_s7a_gates",
    "sequence_seed",
    "split_sequence_seeds",
]
