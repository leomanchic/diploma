"""Sequential independent DOA estimates from one continuous channel stream.

Frames are processed chronologically and independently.  The estimator sees
only the current frame samples; it receives neither truth nor future bearings.
This module deliberately implements no tracking or temporal filter.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from model.geometry import array_centroid, comparison_arrays, direction_angles, direction_vector
from simulation.continuous_stream import (
    DEFAULT_STREAM_CHUNK_SIZE,
    DEFAULT_STREAM_DURATION_S,
    DEFAULT_STREAM_FRAME_LENGTH,
    DEFAULT_STREAM_HOP_LENGTH,
    DEFAULT_STREAM_SAMPLING_RATE_HZ,
    ContinuousStreamResult,
    extract_overlapping_frames,
    synthesize_continuous_stream,
)
from simulation.moving_source import centroid_emission_time
from simulation.trajectory import (
    CircularTrajectory,
    ConstantVelocityTrajectory,
    PiecewiseLinearTrajectory,
    StationaryTrajectory,
    Trajectory,
)
from validation.gcc_statistical import _write_records
from validation.moving_source_study import MOVING_STUDY_METHODS, estimate_independent_frame


SEQUENTIAL_STUDY_BASE_SEED = 20260831


@dataclass(frozen=True)
class SequentialStudyConfig:
    name: str
    trajectory_kind: str
    snr_db: float
    geometry: str = "tetrahedral"
    signal_model: str = "random_broadband"
    duration_s: float = DEFAULT_STREAM_DURATION_S
    dropout_intervals_samples: tuple[tuple[int, int], ...] = ()


def default_sequential_configurations() -> tuple[SequentialStudyConfig, ...]:
    """Seven deterministic diagnostic sequence definitions."""

    return (
        SequentialStudyConfig("stationary", "stationary", 20.0),
        SequentialStudyConfig("constant_velocity_transverse", "transverse", 15.0),
        SequentialStudyConfig("constant_velocity_receding", "receding", 15.0),
        SequentialStudyConfig("circular", "circular", 15.0),
        SequentialStudyConfig("piecewise_linear_maneuver", "piecewise", 15.0),
        SequentialStudyConfig("azimuth_wrap_359_to_0", "azimuth_wrap", 20.0),
        SequentialStudyConfig(
            "low_snr_with_invalid_dropout",
            "stationary",
            -12.0,
            dropout_intervals_samples=((4608, 6656),),
        ),
    )


def trajectory_for_sequence(
    config: SequentialStudyConfig, positions: NDArray[np.float64]
) -> Trajectory:
    centroid = array_centroid(positions)
    reference_time = 0.15
    base_direction = direction_vector(*np.deg2rad([45.0, 30.0]))
    base_position = centroid + 25.0 * base_direction
    kind = config.trajectory_kind
    if kind == "stationary":
        return StationaryTrajectory(base_position)
    if kind == "transverse":
        tangent = np.cross(np.array([0.0, 0.0, 1.0]), base_direction)
        tangent /= np.linalg.norm(tangent)
        return ConstantVelocityTrajectory(base_position, 20.0 * tangent, reference_time)
    if kind == "receding":
        return ConstantVelocityTrajectory(base_position, 20.0 * base_direction, reference_time)
    if kind == "circular":
        return CircularTrajectory(
            centroid + np.array([0.0, 0.0, 8.0]),
            radius_m=25.0,
            angular_speed_rad_s=0.6,
            plane_normal=[0.0, 0.0, 1.0],
            phase_at_reference_rad=np.deg2rad(35.0),
            reference_time_s=reference_time,
        )
    if kind == "piecewise":
        return PiecewiseLinearTrajectory(
            [-0.2, 0.05, 0.16, 0.30, 0.50],
            centroid
            + np.asarray(
                [
                    [26.0, -2.0, 8.0],
                    [25.0, 0.0, 8.0],
                    [24.0, 2.0, 9.0],
                    [22.0, 0.0, 11.0],
                    [24.0, -2.0, 9.0],
                ]
            ),
        )
    if kind == "azimuth_wrap":
        return ConstantVelocityTrajectory(
            centroid + np.array([25.0, 0.0, 8.0]),
            [0.0, 12.0, 0.0],
            reference_time,
        )
    raise ValueError(f"unknown trajectory_kind: {kind}")


def _truth_at_frame_center(
    center_reception_time_s: float,
    positions: NDArray[np.float64],
    trajectory: Trajectory,
) -> dict[str, object]:
    emission = float(centroid_emission_time(center_reception_time_s, positions, trajectory))
    source_position = np.asarray(trajectory.q(emission), dtype=float)
    displacement = source_position - array_centroid(positions)
    distance = float(np.linalg.norm(displacement))
    if distance <= 0.0:
        raise ValueError("trajectory intersects the array centroid")
    direction = displacement / distance
    azimuth, elevation = direction_angles(direction)
    return {
        "emission_time_s": emission,
        "source_position_m": source_position,
        "direction": direction,
        "azimuth_rad": float(azimuth),
        "elevation_rad": float(elevation),
        "distance_m": distance,
    }


def _geodesic_error_deg(estimate: NDArray[np.float64], truth: NDArray[np.float64]) -> float:
    return float(np.rad2deg(np.arccos(np.clip(float(estimate @ truth), -1.0, 1.0))))


def _frame_estimator_result_template() -> dict[str, dict[str, object]]:
    return {
        method: {
            "direction": np.full(3, np.nan),
            "valid": False,
            "boundary": False,
            "shared_gcc_frontend_runtime_s": 0.0,
            "estimator_backend_runtime_s": 0.0,
            "total_runtime_s": 0.0,
            "gcc_frontend_pair_count": 6,
            "estimator_backend_pair_count": 3 if method == "reference_3_gcc_wls" else 6,
            "quality_score_probability_claimed": False,
        }
        for method in MOVING_STUDY_METHODS
    }


def estimate_stream_frames_sequentially(
    stream: ContinuousStreamResult,
    positions: NDArray[np.float64],
    trajectory: Trajectory,
    *,
    sequence_name: str,
    frame_length: int = DEFAULT_STREAM_FRAME_LENGTH,
    hop_length: int = DEFAULT_STREAM_HOP_LENGTH,
    frame_estimator: Callable[
        [NDArray[np.float64], NDArray[np.float64], float],
        dict[str, dict[str, object]],
    ] = estimate_independent_frame,
) -> list[dict[str, object]]:
    """Process frames in increasing time order without future samples or DOAs."""

    frames = extract_overlapping_frames(
        stream.channels,
        stream.reception_times_s,
        frame_length=frame_length,
        hop_length=hop_length,
    )
    rows: list[dict[str, object]] = []
    previous_end = -1
    previous_emission = -np.inf
    for frame_index in range(frames.frame_count):
        start_sample = int(frames.start_sample_indices[frame_index])
        end_sample = start_sample + frames.frame_length - 1
        if end_sample <= previous_end:
            raise RuntimeError("frame processing order is not strictly chronological")
        previous_end = end_sample
        frame = frames.frames[frame_index]
        # Estimation is deliberately completed before truth is computed.  The
        # estimator receives only current samples, geometry, and sampling rate.
        estimates = frame_estimator(frame, positions, stream.propagation.sampling_rate_hz)
        center_reception = float(frames.center_reception_times_s[frame_index])
        truth = _truth_at_frame_center(center_reception, positions, trajectory)
        if float(truth["emission_time_s"]) <= previous_emission:
            raise RuntimeError("centroid emission timestamps must be strictly increasing")
        previous_emission = float(truth["emission_time_s"])
        frame_hash = sha256(np.ascontiguousarray(frame).view(np.uint8)).hexdigest()
        for method in MOVING_STUDY_METHODS:
            result = estimates.get(method, _frame_estimator_result_template()[method])
            valid = bool(result["valid"])
            direction = np.asarray(result["direction"], dtype=float)
            if valid and (direction.shape != (3,) or not np.all(np.isfinite(direction))):
                raise RuntimeError("valid estimator returned a non-finite direction")
            if valid:
                direction /= np.linalg.norm(direction)
                estimate_azimuth, estimate_elevation = direction_angles(direction)
                error = _geodesic_error_deg(direction, np.asarray(truth["direction"]))
                estimate_x, estimate_y, estimate_z = map(float, direction)
                estimate_azimuth_deg = float(np.rad2deg(estimate_azimuth) % 360.0)
                estimate_elevation_deg = float(np.rad2deg(estimate_elevation))
            else:
                error = None
                estimate_x = estimate_y = estimate_z = None
                estimate_azimuth_deg = estimate_elevation_deg = None
            frontend_runtime = float(result["shared_gcc_frontend_runtime_s"])
            backend_runtime = float(result["estimator_backend_runtime_s"])
            algorithm_runtime = float(result["total_runtime_s"])
            end_reception = float(frames.end_reception_times_s[frame_index])
            emission_time = float(truth["emission_time_s"])
            acquisition_latency = end_reception - center_reception
            physical_delay = center_reception - emission_time
            estimate_available_time = end_reception + algorithm_runtime
            rows.append(
                {
                    "sequence_name": sequence_name,
                    "frame_index": frame_index,
                    "estimator_variant": method,
                    "frame_start_sample": start_sample,
                    "frame_end_sample_inclusive": end_sample,
                    "maximum_reception_sample_used": end_sample,
                    "frame_length": frames.frame_length,
                    "hop_length": frames.hop_length,
                    "overlap_samples": frames.overlap_samples,
                    "overlap_fraction": frames.overlap_samples / frames.frame_length,
                    "frame_content_sha256": frame_hash,
                    "frame_start_reception_time_s": float(
                        frames.start_reception_times_s[frame_index]
                    ),
                    "frame_center_reception_time_s": center_reception,
                    "frame_end_reception_time_s": end_reception,
                    "frame_reception_span_s": end_reception
                    - float(frames.start_reception_times_s[frame_index]),
                    "nominal_frame_duration_s": frames.frame_length
                    / stream.propagation.sampling_rate_hz,
                    "centroid_emission_time_s": emission_time,
                    "estimate_reference_timestamp_s": emission_time,
                    "estimate_available_reception_time_s": estimate_available_time,
                    "physical_propagation_delay_s": physical_delay,
                    "frame_acquisition_latency_s": acquisition_latency,
                    "frame_acquisition_latency_definition": "frame_center_reception_to_frame_end_reception",
                    "shared_gcc_frontend_runtime_s": frontend_runtime,
                    "estimator_backend_runtime_s": backend_runtime,
                    "algorithm_runtime_s": algorithm_runtime,
                    "total_emission_to_available_latency_s": (
                        estimate_available_time - emission_time
                    ),
                    "truth_x": float(np.asarray(truth["direction"])[0]),
                    "truth_y": float(np.asarray(truth["direction"])[1]),
                    "truth_z": float(np.asarray(truth["direction"])[2]),
                    "truth_azimuth_deg": float(np.rad2deg(float(truth["azimuth_rad"])) % 360.0),
                    "truth_elevation_deg": float(np.rad2deg(float(truth["elevation_rad"]))),
                    "truth_distance_m": float(truth["distance_m"]),
                    "estimate_x": estimate_x,
                    "estimate_y": estimate_y,
                    "estimate_z": estimate_z,
                    "estimate_azimuth_deg": estimate_azimuth_deg,
                    "estimate_elevation_deg": estimate_elevation_deg,
                    "valid": valid,
                    "boundary_hit": bool(result["boundary"]),
                    "quality_score_probability_claimed": False,
                    "gcc_peak_ratios_used_pairs": result.get("gcc_peak_ratios_used_pairs"),
                    "gcc_peak_curvatures_used_pairs": result.get(
                        "gcc_peak_curvatures_used_pairs"
                    ),
                    "gcc_spectral_energies_used_pairs": result.get(
                        "gcc_spectral_energies_used_pairs"
                    ),
                    "gcc_mean_peak_ratio": result.get("gcc_mean_peak_ratio"),
                    "gcc_minimum_peak_ratio": result.get("gcc_minimum_peak_ratio"),
                    "gcc_mean_peak_curvature": result.get("gcc_mean_peak_curvature"),
                    "gcc_total_spectral_energy": result.get("gcc_total_spectral_energy"),
                    "gcc_boundary_count": result.get("gcc_boundary_count"),
                    "gcc_valid_pair_count": result.get("gcc_valid_pair_count"),
                    "srp_peak_score": result.get("srp_peak_score"),
                    "srp_score_margin": result.get("srp_score_margin"),
                    "srp_local_negative_score_hessian": None
                    if result.get("srp_local_negative_score_hessian") is None
                    else tuple(
                        np.asarray(result["srp_local_negative_score_hessian"], dtype=float)
                        .ravel()
                        .tolist()
                    ),
                    "srp_local_curvature_eigenvalues": None
                    if result.get("srp_local_curvature_eigenvalues") is None
                    else tuple(
                        np.asarray(result["srp_local_curvature_eigenvalues"], dtype=float)
                        .ravel()
                        .tolist()
                    ),
                    "srp_used_spectral_energy": result.get("srp_used_spectral_energy"),
                    "srp_mean_spectral_energy_fraction": result.get(
                        "srp_mean_spectral_energy_fraction"
                    ),
                    "geodesic_angular_error_deg": error,
                    "nominal_stream_snr_db": stream.nominal_snr_db,
                    "effective_stream_snr_db": stream.effective_snr_db,
                    "base_seed": stream.base_seed,
                    "source_seed": stream.source_seed,
                    "noise_seed": stream.noise_seed,
                    "chunk_size_samples": stream.propagation.chunk_size_samples,
                    "fir_length": stream.propagation.fir_length,
                    "maximum_interpolation_working_set_elements": (
                        stream.propagation.maximum_interpolation_working_set_elements
                    ),
                    "truth_definition": "array_centroid_emission_time_at_geometric_frame_center",
                    "truth_used_by_estimator": False,
                    "future_samples_used": False,
                    "future_doa_estimates_used": False,
                    "frame_samples_from_single_continuous_stream": True,
                    "sequential_independent_bearings_not_tracking": True,
                    "frame_count_is_independent_trial_count": False,
                }
            )
    return rows


def _conditional_metrics(errors: NDArray[np.float64]) -> dict[str, float | None]:
    if errors.size == 0:
        return {
            "conditional_rmse_deg": None,
            "conditional_p95_deg": None,
            "conditional_p99_deg": None,
        }
    return {
        "conditional_rmse_deg": float(np.sqrt(np.mean(errors**2))),
        "conditional_p95_deg": float(np.percentile(errors, 95.0)),
        "conditional_p99_deg": float(np.percentile(errors, 99.0)),
    }


def summarize_sequence_rows(
    rows: list[dict[str, object]], stream: ContinuousStreamResult
) -> list[dict[str, object]]:
    """Create method-level metrics; frame counts are never called trials."""

    summaries: list[dict[str, object]] = []
    for method in MOVING_STUDY_METHODS:
        selected = [row for row in rows if row["estimator_variant"] == method]
        valid = [row for row in selected if bool(row["valid"])]
        errors = np.asarray([float(row["geodesic_angular_error_deg"]) for row in valid])
        metrics = _conditional_metrics(errors)
        truth_azimuth = np.asarray([float(row["truth_azimuth_deg"]) for row in selected])
        truth_directions = np.asarray(
            [[float(row["truth_x"]), float(row["truth_y"]), float(row["truth_z"])] for row in selected]
        )
        truth_change = np.rad2deg(
            np.arccos(np.clip(truth_directions @ truth_directions[0], -1.0, 1.0))
        )
        summaries.append(
            {
                "sequence_name": selected[0]["sequence_name"],
                "estimator_variant": method,
                "duration_s": stream.reception_times_s.size
                / stream.propagation.sampling_rate_hz,
                "sampling_rate_hz": stream.propagation.sampling_rate_hz,
                "frame_length": selected[0]["frame_length"],
                "hop_length": selected[0]["hop_length"],
                "overlap_samples": selected[0]["overlap_samples"],
                "overlap_fraction": selected[0]["overlap_fraction"],
                "frame_reception_span_s": selected[0]["frame_reception_span_s"],
                "nominal_frame_duration_s": selected[0]["nominal_frame_duration_s"],
                "frame_count": len(selected),
                "frame_count_is_independent_trial_count": False,
                "successful_frame_count": len(valid),
                "invalid_frame_count": len(selected) - len(valid),
                "coverage": len(valid) / len(selected),
                "boundary_hit_fraction": float(
                    np.mean([bool(row["boundary_hit"]) for row in selected])
                ),
                **metrics,
                "mean_physical_propagation_delay_s": float(
                    np.mean([float(row["physical_propagation_delay_s"]) for row in selected])
                ),
                "frame_acquisition_latency_s": float(
                    selected[0]["frame_acquisition_latency_s"]
                ),
                "frame_acquisition_latency_definition": "frame_center_reception_to_frame_end_reception",
                "mean_shared_gcc_frontend_runtime_s": float(
                    np.mean([float(row["shared_gcc_frontend_runtime_s"]) for row in selected])
                ),
                "mean_estimator_backend_runtime_s": float(
                    np.mean([float(row["estimator_backend_runtime_s"]) for row in selected])
                ),
                "mean_algorithm_runtime_s": float(
                    np.mean([float(row["algorithm_runtime_s"]) for row in selected])
                ),
                "mean_total_emission_to_available_latency_s": float(
                    np.mean(
                        [float(row["total_emission_to_available_latency_s"]) for row in selected]
                    )
                ),
                "maximum_truth_doa_change_from_first_deg": float(np.max(truth_change)),
                "azimuth_wrap_359_to_0_present": bool(
                    np.any(truth_azimuth > 350.0) and np.any(truth_azimuth < 10.0)
                ),
                "nominal_stream_snr_db": stream.nominal_snr_db,
                "effective_stream_snr_db": stream.effective_snr_db,
                "base_seed": stream.base_seed,
                "source_seed": stream.source_seed,
                "noise_seed": stream.noise_seed,
                "chunk_size_samples": stream.propagation.chunk_size_samples,
                "fir_length": stream.propagation.fir_length,
                "maximum_interpolation_working_set_elements": (
                    stream.propagation.maximum_interpolation_working_set_elements
                ),
                "noise_generated_once_for_stream": True,
                "overlapping_frames_are_statistically_dependent": True,
                "sequential_independent_bearings_not_tracking": True,
            }
        )
    return summaries


def run_sequential_configuration(
    config: SequentialStudyConfig,
    configuration_index: int,
    *,
    frame_length: int = DEFAULT_STREAM_FRAME_LENGTH,
    hop_length: int = DEFAULT_STREAM_HOP_LENGTH,
    chunk_size_samples: int = DEFAULT_STREAM_CHUNK_SIZE,
    frame_estimator=estimate_independent_frame,
) -> tuple[list[dict[str, object]], list[dict[str, object]], ContinuousStreamResult]:
    positions = comparison_arrays()[config.geometry]
    trajectory = trajectory_for_sequence(config, positions)
    stream = synthesize_continuous_stream(
        positions,
        trajectory,
        duration_s=config.duration_s,
        reception_start_time_s=0.1,
        signal_model=config.signal_model,
        snr_db=config.snr_db,
        seed=SEQUENTIAL_STUDY_BASE_SEED + int(configuration_index),
        chunk_size_samples=chunk_size_samples,
        dropout_intervals_samples=config.dropout_intervals_samples,
    )
    frame_rows = estimate_stream_frames_sequentially(
        stream,
        positions,
        trajectory,
        sequence_name=config.name,
        frame_length=frame_length,
        hop_length=hop_length,
        frame_estimator=frame_estimator,
    )
    summaries = summarize_sequence_rows(frame_rows, stream)
    return frame_rows, summaries, stream


def run_sequential_doa_study(
    *,
    configurations: tuple[SequentialStudyConfig, ...] | None = None,
    frame_output_csv: str | Path = "results/sequential_doa_frame_results.csv",
    summary_output_csv: str | Path = "results/sequential_doa_summary.csv",
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    selected = default_sequential_configurations() if configurations is None else configurations
    frame_records: list[dict[str, object]] = []
    summary_records: list[dict[str, object]] = []
    for index, config in enumerate(selected):
        frames, summaries, _ = run_sequential_configuration(config, index)
        frame_records.extend(frames)
        summary_records.extend(summaries)
    _write_records(frame_records, frame_output_csv)
    _write_records(summary_records, summary_output_csv)
    return frame_records, summary_records


__all__ = [
    "SEQUENTIAL_STUDY_BASE_SEED",
    "SequentialStudyConfig",
    "default_sequential_configurations",
    "estimate_stream_frames_sequentially",
    "run_sequential_configuration",
    "run_sequential_doa_study",
    "summarize_sequence_rows",
    "trajectory_for_sequence",
]
