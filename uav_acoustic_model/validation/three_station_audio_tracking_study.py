"""Reproducible S7C pilot from continuous three-station audio to 3-D tracking.

The calibration/evaluation split is by complete continuous sequences.  Truth
is confined to waveform generation and offline scoring.  Online estimators
receive channel frames or truth-free :class:`BearingMeasurement` objects.
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.stats import chi2

from estimators.retarded_ekf_manoeuvre import (
    CausalManoeuvreRetardedTimeEKF,
    ManoeuvreHistoryConfig,
)
from estimators.retarded_ekf_recovery import InitializationRecoveryConfig
from model.bearing_events import bearing_event_id
from model.bearing_statistics import calibrate_bearing_covariance, tangent_residual
from model.geometry import tetrahedral_array
from model.measurements import BearingMeasurement
from model.station import StationPose
from simulation.continuous_stream import extract_overlapping_frames
from simulation.manoeuvre_trajectory import BenchmarkManoeuvreTrajectory
from simulation.moving_source import solve_emission_time
from simulation.multistation_audio import MultistationAudioStream, synthesize_multistation_audio
from validation.moving_source_study import estimate_independent_frame


CALIBRATION_BASE_SEED = 20260918
EVALUATION_BASE_SEED = 20260919
SMOKE_BASE_SEED = 20260917
CALIBRATION_SEQUENCE_COUNT = 1
EVALUATION_SEQUENCE_COUNT = 1
DURATION_S = 4.5
RECEPTION_START_TIME_S = 0.5
SAMPLING_RATE_HZ = 48_000.0
FRAME_LENGTH = 1024
HOP_LENGTH = 512
TRACKER_FRAME_STRIDE = 32
SIGNAL_MODEL = "random_broadband"
SOURCE_MAXIMUM_FREQUENCY_HZ = 10_000.0
SNR_LEVELS_DB = (-6.0, 10.0)
TRAJECTORY_KINDS = (
    "constant_velocity",
    "constant_acceleration_segment",
    "smooth_turn",
)
ESTIMATOR_VARIANTS = ("all_6_equal_gcc_wls", "equal_weight_srp_phat")
MODELED_PROCESSING_DELAY_S = 0.010
STATION_DELIVERY_DELAY_S = {"S0": 0.000, "S1": 0.015, "S2": 0.030}
QC_ALPHA_M2_S3 = 1.0
MANOEUVRE_START_S = 2.5
MANOEUVRE_END_S = 3.5
HISTORY_WINDOW_S = 0.85
MAXIMUM_TRACKER_RANGE_M = 150.0
MAXIMUM_TRANSPORT_DELAY_S = 0.10
POSITION_COVERAGE_THRESHOLD = float(chi2.ppf(0.95, 6))
RESULTS = Path(__file__).resolve().parents[1] / "results"
CALIBRATION_POOLING_RULE = (
    "unweighted_valid_frames_from_all_predeclared_calibration_scenarios"
)
MOTION_PHASES = ("before_manoeuvre", "during_manoeuvre", "after_manoeuvre")


@dataclass(frozen=True, slots=True)
class AudioPilotConfig:
    trajectory_kind: str
    snr_db: float


@dataclass(frozen=True, slots=True)
class AudioBearingCalibration:
    station_id: str
    estimator_variant: str
    pooling_rule: str
    physical_configuration_count: int
    sequence_count: int
    dependent_frame_count: int
    successful_frame_count: int
    mean_residual_rad: np.ndarray
    covariance_rad2: np.ndarray
    eigenvalues_rad2: np.ndarray
    condition_number: float
    temporal_lag1_az: float
    temporal_lag1_el: float
    maximum_absolute_interstation_correlation: float
    included_trajectory_kinds: tuple[str, ...]
    included_snr_db: tuple[float, ...]


def pilot_stations() -> tuple[StationPose, ...]:
    positions = ([0.0, 0.0, 0.0], [100.0, 0.0, 5.0], [10.0, 90.0, -2.0])
    return tuple(
        StationPose(f"S{index}", position, np.eye(3), tetrahedral_array())
        for index, position in enumerate(positions)
    )


def pilot_configurations() -> tuple[AudioPilotConfig, ...]:
    return tuple(
        AudioPilotConfig(kind, snr)
        for kind in TRAJECTORY_KINDS
        for snr in SNR_LEVELS_DB
    )


def audio_sequence_seed(split: str, configuration_index: int, sequence_index: int) -> int:
    split_code = {"calibration": 7301, "evaluation": 7401, "smoke": 7501}
    if split not in split_code:
        raise ValueError("split must be calibration, evaluation or smoke")
    base = {
        "calibration": CALIBRATION_BASE_SEED,
        "evaluation": EVALUATION_BASE_SEED,
        "smoke": SMOKE_BASE_SEED,
    }[split]
    return int(
        np.random.SeedSequence(
            [base, split_code[split], int(configuration_index), int(sequence_index)]
        ).generate_state(1, dtype=np.uint64)[0]
    )


def trajectory_for_audio_pilot(kind: str, sequence_index: int) -> BenchmarkManoeuvreTrajectory:
    if kind not in TRAJECTORY_KINDS:
        raise ValueError("unknown trajectory kind")
    # Small deterministic offsets make whole sequences independent physical
    # realizations without changing the predeclared geometry class.
    offset = np.asarray([0.7, -0.4, 0.25]) * int(sequence_index)
    return BenchmarkManoeuvreTrajectory(
        np.asarray([70.0, 55.0, 40.0]) + offset,
        [7.0, -3.0, 1.5],
        kind,
        manoeuvre_start_s=MANOEUVRE_START_S,
        manoeuvre_end_s=MANOEUVRE_END_S,
        acceleration_mps2=[0.0, 3.0, 0.8],
        turn_angle_rad=0.45,
    )


def _unit_direction_at_reception(station, trajectory, reception_time_s):
    emission = float(
        solve_emission_time(reception_time_s, station.position_world_m, trajectory)
    )
    displacement = np.asarray(trajectory.q(emission)) - station.position_world_m
    direction_world = displacement / np.linalg.norm(displacement)
    direction_local = station.world_to_local_direction(direction_world)
    return emission, direction_world, direction_local


def _scalar_quality(result: dict[str, object]) -> dict[str, float | int | bool | str | None]:
    allowed = {}
    for name in (
        "gcc_mean_peak_ratio",
        "gcc_minimum_peak_ratio",
        "gcc_mean_peak_curvature",
        "gcc_total_spectral_energy",
        "gcc_boundary_count",
        "gcc_valid_pair_count",
        "srp_peak_score",
        "srp_score_margin",
        "srp_used_spectral_energy",
        "srp_mean_spectral_energy_fraction",
        "boundary",
    ):
        value = result.get(name)
        if value is not None and np.isscalar(value) and (
            not isinstance(value, (float, np.floating)) or np.isfinite(value)
        ):
            allowed[name] = value.item() if isinstance(value, np.generic) else value
    return allowed


def extract_audio_bearing_records(
    stream: MultistationAudioStream,
    stations: tuple[StationPose, ...],
    trajectory: BenchmarkManoeuvreTrajectory,
    *,
    split: str,
    configuration_index: int,
    sequence_index: int,
) -> tuple[list[dict[str, object]], float]:
    """Estimate GCC/SRP bearings from views of continuous station streams."""

    station_map = {station.station_id: station for station in stations}
    records: list[dict[str, object]] = []
    wall_started = time.perf_counter()
    for station_stream in stream.stations:
        station = station_map[station_stream.station_id]
        frames = extract_overlapping_frames(
            station_stream.channels,
            stream.reception_times_s,
            frame_length=FRAME_LENGTH,
            hop_length=HOP_LENGTH,
        )
        for frame_index in range(frames.frame_count):
            frame = frames.frames[frame_index]
            estimates = estimate_independent_frame(
                frame, station.microphone_positions_local_m, stream.sampling_rate_hz
            )
            center = float(frames.center_reception_times_s[frame_index])
            end = float(frames.end_reception_times_s[frame_index])
            emission, truth_world, truth_local = _unit_direction_at_reception(
                station, trajectory, center
            )
            available = (
                end + MODELED_PROCESSING_DELAY_S
                + STATION_DELIVERY_DELAY_S[station.station_id]
            )
            for method in ESTIMATOR_VARIANTS:
                result = estimates[method]
                valid = bool(result["valid"])
                estimate_local = np.asarray(result["direction"], dtype=float)
                if valid:
                    estimate_local = estimate_local / np.linalg.norm(estimate_local)
                    estimate_world = station.local_to_world_direction(estimate_local)
                    residual = tangent_residual(truth_local, estimate_local)
                    error_deg = float(np.rad2deg(np.linalg.norm(residual)))
                else:
                    estimate_local = np.full(3, np.nan)
                    estimate_world = np.full(3, np.nan)
                    residual = np.full(2, np.nan)
                    error_deg = float("nan")
                records.append(
                    {
                        "split": split,
                        "configuration_index": configuration_index,
                        "sequence_index": sequence_index,
                        "sequence_seed": stream.base_seed,
                        "sequence_id": (
                            f"s7c-audio-{split}-{configuration_index}-{sequence_index}"
                        ),
                        "trajectory_kind": trajectory.kind,
                        "snr_db": station_stream.nominal_snr_db,
                        "signal_model": stream.signal_model,
                        "source_maximum_frequency_hz": (
                            stream.maximum_emitted_frequency_hz
                        ),
                        "doppler_bandlimit_checked": (
                            stream.doppler_bandlimit_checked
                        ),
                        "station_id": station.station_id,
                        "estimator_variant": method,
                        "frame_index": frame_index,
                        "frame_start_reception_time_s": float(
                            frames.start_reception_times_s[frame_index]
                        ),
                        "frame_center_reception_time_s": center,
                        "frame_end_reception_time_s": end,
                        "true_emission_time_s_evaluator_only": emission,
                        "available_timestamp_s": available,
                        "modeled_processing_delay_s": MODELED_PROCESSING_DELAY_S,
                        "modeled_delivery_delay_s": STATION_DELIVERY_DELAY_S[station.station_id],
                        "measured_algorithm_runtime_s": float(result["total_runtime_s"]),
                        "effective_station_snr_db": station_stream.effective_snr_db,
                        "valid": valid,
                        "invalid_reason": "" if valid else "audio_bearing_invalid",
                        "boundary_hit": bool(result["boundary"]),
                        "truth_local": truth_local,
                        "truth_world": truth_world,
                        "estimate_local": estimate_local,
                        "estimate_world": estimate_world,
                        "residual_rad": residual,
                        "geodesic_error_deg": error_deg,
                        "quality_metadata": _scalar_quality(result),
                        "source_seed": stream.source_seed,
                        "noise_seed": station_stream.noise_seed,
                        "truth_used_by_audio_estimator": False,
                        "frame_from_continuous_stream": True,
                    }
                )
    return records, time.perf_counter() - wall_started


def generate_audio_sequence(
    config: AudioPilotConfig,
    configuration_index: int,
    split: str,
    sequence_index: int,
    *,
    duration_s: float = DURATION_S,
) -> tuple[tuple[StationPose, ...], BenchmarkManoeuvreTrajectory, MultistationAudioStream, list[dict[str, object]], dict[str, float]]:
    pipeline_started = time.perf_counter()
    stations = pilot_stations()
    trajectory = trajectory_for_audio_pilot(config.trajectory_kind, sequence_index)
    seed = audio_sequence_seed(split, configuration_index, sequence_index)
    synthesis_started = time.perf_counter()
    stream = synthesize_multistation_audio(
        stations,
        trajectory,
        duration_s=duration_s,
        reception_start_time_s=RECEPTION_START_TIME_S,
        sampling_rate_hz=SAMPLING_RATE_HZ,
        signal_model=SIGNAL_MODEL,
        snr_db=config.snr_db,
        seed=seed,
        maximum_emitted_frequency_hz=SOURCE_MAXIMUM_FREQUENCY_HZ,
    )
    synthesis_runtime = time.perf_counter() - synthesis_started
    records, bearing_runtime = extract_audio_bearing_records(
        stream,
        stations,
        trajectory,
        split=split,
        configuration_index=configuration_index,
        sequence_index=sequence_index,
    )
    runtimes = {
        "audio_synthesis_wall_runtime_s": synthesis_runtime,
        "bearing_frontend_wall_runtime_s": bearing_runtime,
        "audio_pipeline_wall_runtime_s": time.perf_counter() - pipeline_started,
    }
    return stations, trajectory, stream, records, runtimes


def _correlation(first, second) -> float:
    x = np.asarray(first, dtype=float)
    y = np.asarray(second, dtype=float)
    finite = np.isfinite(x) & np.isfinite(y)
    if np.count_nonzero(finite) < 3 or np.ptp(x[finite]) == 0 or np.ptp(y[finite]) == 0:
        return float("nan")
    return float(np.corrcoef(x[finite], y[finite])[0, 1])


def _temporal_correlations(rows: list[dict[str, object]]) -> tuple[float, float]:
    values = [[], []]
    by_sequence: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        if row["valid"]:
            by_sequence[str(row["sequence_id"])].append(row)
    for sequence_rows in by_sequence.values():
        sequence_rows.sort(key=lambda row: int(row["frame_index"]))
        residuals = np.asarray([row["residual_rad"] for row in sequence_rows])
        for axis in range(2):
            values[axis].append(_correlation(residuals[:-1, axis], residuals[1:, axis]))
    result = []
    for axis in values:
        finite = np.asarray(axis, dtype=float)
        finite = finite[np.isfinite(finite)]
        result.append(float(np.mean(finite)) if finite.size else float("nan"))
    return tuple(result)


def _interstation_maximum(rows: list[dict[str, object]]) -> float:
    correlations = []
    station_ids = sorted({str(row["station_id"]) for row in rows})
    lookup = {
        (str(row["sequence_id"]), int(row["frame_index"]), str(row["station_id"])): row
        for row in rows if row["valid"]
    }
    for left_index, left in enumerate(station_ids):
        for right in station_ids[left_index + 1 :]:
            common = sorted(
                (sequence, frame)
                for sequence, frame, station in lookup
                if station == left and (sequence, frame, right) in lookup
            )
            if not common:
                continue
            for axis in range(2):
                correlations.append(
                    _correlation(
                        [lookup[(s, f, left)]["residual_rad"][axis] for s, f in common],
                        [lookup[(s, f, right)]["residual_rad"][axis] for s, f in common],
                    )
                )
    finite = np.abs(np.asarray(correlations)[np.isfinite(correlations)])
    return float(np.max(finite)) if finite.size else float("nan")


def calibrate_audio_bearings(
    rows: list[dict[str, object]], sequence_count: int
) -> dict[tuple[str, str], AudioBearingCalibration]:
    """Fit one predeclared pooled bias/R per station and estimator.

    All valid frames from all supplied calibration scenarios enter with equal
    frame weight.  Scenario truth labels are used only to document the pool;
    they are never keys when an evaluation measurement is constructed.
    """

    declared_per_configuration = int(sequence_count)
    if declared_per_configuration < 1:
        raise ValueError("sequence_count must be positive")
    calibrations = {}
    groups: dict[tuple[str, str], list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        if row["split"] != "calibration":
            raise ValueError("calibration must not consume evaluation rows")
        key = (str(row["station_id"]), str(row["estimator_variant"]))
        groups[key].append(row)
    for key, subset in groups.items():
        valid = [row for row in subset if row["valid"]]
        residuals = np.asarray([row["residual_rad"] for row in valid], dtype=float)
        fitted = calibrate_bearing_covariance(residuals)
        if fitted.rank != 2 or np.min(fitted.eigenvalues_rad2) <= 0.0:
            raise RuntimeError(f"audio bearing calibration is not positive definite: {key}")
        method_rows = [row for row in rows if row["estimator_variant"] == key[1]]
        sequence_ids = {str(row["sequence_id"]) for row in subset}
        configuration_ids = {int(row["configuration_index"]) for row in subset}
        expected_sequence_count = declared_per_configuration * len(configuration_ids)
        if len(sequence_ids) != expected_sequence_count:
            raise RuntimeError(
                "calibration pool does not contain the declared number of whole sequences"
            )
        temporal = _temporal_correlations(valid)
        calibrations[key] = AudioBearingCalibration(
            key[0], key[1], CALIBRATION_POOLING_RULE, len(configuration_ids),
            len(sequence_ids), len(subset), len(valid),
            fitted.mean_residual_rad.copy(), fitted.covariance_rad2.copy(),
            fitted.eigenvalues_rad2.copy(), fitted.condition_number,
            temporal[0], temporal[1], _interstation_maximum(method_rows),
            tuple(sorted({str(row["trajectory_kind"]) for row in subset})),
            tuple(sorted({float(row["snr_db"]) for row in subset})),
        )
    return calibrations


def bearing_measurements_from_records(
    rows: list[dict[str, object]],
    calibrations: dict[tuple[str, str], AudioBearingCalibration],
    method: str,
    *,
    frame_stride: int = 1,
) -> tuple[BearingMeasurement, ...]:
    if frame_stride <= 0:
        raise ValueError("frame_stride must be positive")
    measurements = []
    for row in rows:
        if (
            row["estimator_variant"] != method
            or int(row["frame_index"]) % frame_stride != 0
        ):
            continue
        key = (str(row["station_id"]), method)
        calibration = calibrations[key]
        common = dict(
            station_id=str(row["station_id"]),
            sequence_id=str(row["sequence_id"]),
            frame_index=int(row["frame_index"]),
            reception_center_timestamp_s=float(row["frame_center_reception_time_s"]),
            available_timestamp_s=float(row["available_timestamp_s"]),
            estimator_variant=method,
            quality_metadata=row["quality_metadata"],
        )
        if row["valid"]:
            measurement = BearingMeasurement(
                **common,
                direction_local=row["estimate_local"],
                covariance_tangent_rad2=calibration.covariance_rad2,
                calibration_bias_tangent_rad=calibration.mean_residual_rad,
                tangent_frame="prediction",
            )
        else:
            measurement = BearingMeasurement.invalid(
                **common, invalid_reason=str(row["invalid_reason"])
            )
        measurements.append(measurement)
    return tuple(measurements)


def _motion_phase(time_s: float) -> str:
    """Return the predeclared offline manoeuvre phase for one truth epoch."""

    epoch = float(time_s)
    if epoch < MANOEUVRE_START_S:
        return "before_manoeuvre"
    if epoch < MANOEUVRE_END_S:
        return "during_manoeuvre"
    return "after_manoeuvre"


def run_tracker(
    stations: tuple[StationPose, ...],
    trajectory: BenchmarkManoeuvreTrajectory,
    measurements: tuple[BearingMeasurement, ...],
    method: str,
) -> tuple[list[dict[str, object]], dict[str, object], list[dict[str, object]]]:
    config = ManoeuvreHistoryConfig(
        np.eye(3) * QC_ALPHA_M2_S3,
        history_step_s=0.25,
        history_window_s=HISTORY_WINDOW_S,
        maximum_range_m=MAXIMUM_TRACKER_RANGE_M,
        maximum_transport_delay_s=MAXIMUM_TRANSPORT_DELAY_S,
    )
    estimator = CausalManoeuvreRetardedTimeEKF(
        stations,
        measurements,
        estimator_variant=method,
        history_config=config,
        recovery_config=InitializationRecoveryConfig(
            confirmation_reception_span_s=0.05,
            maximum_confirmation_failures=20,
            maximum_confirmation_events=180,
            maximum_initialization_buffer_events=360,
        ),
    )
    publication_times = sorted({float(item.available_timestamp_s) for item in measurements})
    started = time.perf_counter()
    publications = [estimator.advance_to(epoch) for epoch in publication_times]
    runtime = time.perf_counter() - started
    station_map = {station.station_id: station for station in stations}
    measurement_map = {bearing_event_id(item): item for item in measurements}
    rows = []
    update_rows = []
    last_accepted_update_time = float("nan")
    for publication in publications:
        epoch = float(publication.processing_time_s)
        for diagnostic in publication.update_diagnostics:
            measurement = measurement_map[diagnostic.event_id]
            station = station_map[measurement.station_id]
            true_emission = float(
                solve_emission_time(
                    measurement.reception_center_timestamp_s,
                    station.position_world_m,
                    trajectory,
                )
            )
            diagnostic_processing_time = float(
                getattr(diagnostic, "processing_time_s", epoch)
            )
            applied = bool(diagnostic.update_applied)
            if applied:
                last_accepted_update_time = diagnostic_processing_time
            update_rows.append(
                {
                    "event_id": diagnostic.event_id,
                    "station_id": measurement.station_id,
                    "frame_index": measurement.frame_index,
                    "estimator_variant": method,
                    "processing_time_s": diagnostic_processing_time,
                    "reception_center_timestamp_s": (
                        measurement.reception_center_timestamp_s
                    ),
                    "true_emission_time_s_evaluator_only": true_emission,
                    "update_motion_phase_evaluator_only": _motion_phase(true_emission),
                    "update_applied": applied,
                    "failure_reason": diagnostic.failure_reason or "",
                    "pre_update_nis": float(
                        getattr(
                            diagnostic,
                            "pre_update_nis",
                            getattr(diagnostic, "normalized_innovation_squared", np.nan),
                        )
                    ),
                    "truth_used_by_tracker": False,
                }
            )
        truth = np.concatenate((trajectory.q(epoch), trajectory.v(epoch)))
        valid = bool(publication.valid and publication.state is not None)
        position_error = velocity_error = state_nees = float("nan")
        covered = False
        estimate = np.full(6, np.nan)
        if valid:
            estimate = publication.state.vector
            residual = estimate - truth
            position_error = float(np.linalg.norm(residual[:3]))
            velocity_error = float(np.linalg.norm(residual[3:]))
            covariance = 0.5 * (
                publication.covariance_state + publication.covariance_state.T
            )
            try:
                np.linalg.cholesky(covariance)
                state_nees = float(residual @ np.linalg.solve(covariance, residual))
                covered = state_nees <= POSITION_COVERAGE_THRESHOLD
            except np.linalg.LinAlgError:
                valid = False
        rows.append(
            {
                "processing_time_s": epoch,
                "estimator_variant": method,
                "valid": valid,
                "confirmed": bool(publication.confirmed),
                "status": publication.status,
                "failure_reason": publication.failure_reason or "",
                "truth_position_x_m": float(truth[0]),
                "truth_position_y_m": float(truth[1]),
                "truth_position_z_m": float(truth[2]),
                "estimate_position_x_m": float(estimate[0]),
                "estimate_position_y_m": float(estimate[1]),
                "estimate_position_z_m": float(estimate[2]),
                "position_error_m": position_error,
                "velocity_error_mps": velocity_error,
                "state_nees": state_nees,
                "valid_and_covered": bool(valid and covered),
                "reset_count": int(publication.reset_count),
                "publication_motion_phase_evaluator_only": _motion_phase(epoch),
                "last_accepted_update_processing_time_s": (
                    last_accepted_update_time
                ),
                "time_since_last_accepted_update_s": (
                    epoch - last_accepted_update_time
                    if np.isfinite(last_accepted_update_time)
                    else float("nan")
                ),
                "correction_status": (
                    "measurement_correction_available"
                    if np.isfinite(last_accepted_update_time)
                    else "prediction_without_correction"
                ),
                "truth_used_by_tracker": False,
            }
        )
    final = publications[-1]
    time_since_values = np.asarray(
        [row["time_since_last_accepted_update_s"] for row in rows], dtype=float
    )
    finite_time_since = time_since_values[np.isfinite(time_since_values)]
    accepted_update_count = sum(row["update_applied"] for row in update_rows)
    rejected_update_count = len(update_rows) - accepted_update_count
    sequence = {
        "estimator_variant": method,
        "event_count": len(measurements),
        "valid_measurement_count": sum(item.valid for item in measurements),
        "initialization_event_count": len(final.initialization_event_ids),
        "accepted_update_count": accepted_update_count,
        "rejected_update_count": rejected_update_count,
        "rejected_event_count": len(final.rejected_event_ids),
        "first_confirmation_time_s": final.first_confirmation_time_s,
        "confirmed_before_manoeuvre": bool(
            np.isfinite(final.first_confirmation_time_s)
            and final.first_confirmation_time_s < MANOEUVRE_START_S
        ),
        "final_time_since_last_accepted_update_s": rows[-1][
            "time_since_last_accepted_update_s"
        ],
        "maximum_time_since_last_accepted_update_s": (
            float(np.max(finite_time_since))
            if finite_time_since.size
            else float("nan")
        ),
        "correction_status": (
            "prediction_with_measurement_corrections"
            if accepted_update_count
            else "prediction_without_correction"
        ),
        "final_valid": bool(final.valid),
        "final_confirmed": bool(final.confirmed),
        "reset_count": int(final.reset_count),
        "failure_reason": final.failure_reason or "",
        "rejection_reasons_json": json.dumps(Counter(dict(final.rejection_reasons).values()), sort_keys=True),
        "tracker_runtime_s": runtime,
        "maximum_history_memory_bytes": estimator.maximum_history_memory_bytes,
        "maximum_history_nodes": estimator.maximum_history_node_count,
    }
    return rows, sequence, update_rows


def summarize_sequence_phases(
    tracking_rows: list[dict[str, object]],
    update_rows: list[dict[str, object]],
    sequence_row: dict[str, object],
) -> list[dict[str, object]]:
    """Report per-sequence phase metrics without feeding truth to the tracker."""

    result = []
    for phase in MOTION_PHASES:
        publications = [
            row for row in tracking_rows
            if row["publication_motion_phase_evaluator_only"] == phase
        ]
        valid = [row for row in publications if row["valid"]]
        updates = [
            row for row in update_rows
            if row["update_motion_phase_evaluator_only"] == phase
        ]
        accepted = [row for row in updates if row["update_applied"]]
        rejected = [row for row in updates if not row["update_applied"]]
        time_since = np.asarray(
            [row["time_since_last_accepted_update_s"] for row in publications],
            dtype=float,
        )
        time_since = time_since[np.isfinite(time_since)]
        result.append(
            {
                "estimator_variant": sequence_row["estimator_variant"],
                "motion_phase": phase,
                "publication_phase_basis": "state_processing_time_s",
                "update_phase_basis": "evaluator_only_true_emission_time_s",
                "confirmed_before_manoeuvre": sequence_row[
                    "confirmed_before_manoeuvre"
                ],
                "accepted_update_count": len(accepted),
                "rejected_update_count": len(rejected),
                "update_attempt_count": len(updates),
                "correction_status": (
                    "measurement_correction_in_phase"
                    if accepted
                    else "prediction_without_correction_in_phase"
                ),
                "dependent_publication_count": len(publications),
                "valid_publication_count": len(valid),
                "valid_publication_fraction": (
                    len(valid) / len(publications) if publications else float("nan")
                ),
                "position_rmse_m_conditional": (
                    float(
                        np.sqrt(
                            np.mean([row["position_error_m"] ** 2 for row in valid])
                        )
                    )
                    if valid else float("nan")
                ),
                "position_p95_m_conditional": _percentile(
                    [row["position_error_m"] for row in valid], 95
                ),
                "velocity_rmse_mps_conditional": (
                    float(
                        np.sqrt(
                            np.mean([row["velocity_error_mps"] ** 2 for row in valid])
                        )
                    )
                    if valid else float("nan")
                ),
                "coverage_conditional": (
                    float(np.mean([row["valid_and_covered"] for row in valid]))
                    if valid else float("nan")
                ),
                "valid_and_covered_fraction": (
                    float(np.mean([row["valid_and_covered"] for row in publications]))
                    if publications else float("nan")
                ),
                "final_time_since_last_accepted_update_s": (
                    float(time_since[-1]) if time_since.size else float("nan")
                ),
                "maximum_time_since_last_accepted_update_s": (
                    float(np.max(time_since)) if time_since.size else float("nan")
                ),
            }
        )
    return result


def _csv_bearing_row(row: dict[str, object]) -> dict[str, object]:
    result = dict(row)
    for name in ("truth_local", "truth_world", "estimate_local", "estimate_world", "residual_rad"):
        value = np.asarray(result.pop(name), dtype=float)
        for index, number in enumerate(value):
            result[f"{name}_{index}"] = float(number)
    result["quality_metadata_json"] = json.dumps(result.pop("quality_metadata"), sort_keys=True)
    return result


def _calibration_rows(calibrations) -> list[dict[str, object]]:
    rows = []
    for value in calibrations.values():
        rows.append(
            {
                "split": "calibration",
                "station_id": value.station_id,
                "estimator_variant": value.estimator_variant,
                "pooling_rule": value.pooling_rule,
                "physical_configuration_count": value.physical_configuration_count,
                "included_trajectory_kinds_json": json.dumps(
                    value.included_trajectory_kinds
                ),
                "included_snr_db_json": json.dumps(value.included_snr_db),
                "independent_sequence_count": value.sequence_count,
                "dependent_frame_count": value.dependent_frame_count,
                "successful_frame_count": value.successful_frame_count,
                "bias_az_arc_rad": float(value.mean_residual_rad[0]),
                "bias_el_arc_rad": float(value.mean_residual_rad[1]),
                "covariance_00_rad2": float(value.covariance_rad2[0, 0]),
                "covariance_01_rad2": float(value.covariance_rad2[0, 1]),
                "covariance_11_rad2": float(value.covariance_rad2[1, 1]),
                "eigenvalue_min_rad2": float(value.eigenvalues_rad2[0]),
                "eigenvalue_max_rad2": float(value.eigenvalues_rad2[1]),
                "condition_number": value.condition_number,
                "temporal_lag1_az": value.temporal_lag1_az,
                "temporal_lag1_el": value.temporal_lag1_el,
                "maximum_absolute_interstation_correlation": value.maximum_absolute_interstation_correlation,
                "evaluation_used_for_calibration": False,
                "filter_assumes_measurement_independence": True,
            }
        )
    return rows


def _percentile(values, q):
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    return float(np.percentile(finite, q)) if finite.size else float("nan")


def _finite_mean(values):
    finite = np.asarray(
        [value for value in values if value is not None], dtype=float
    )
    finite = finite[np.isfinite(finite)]
    return float(np.mean(finite)) if finite.size else float("nan")


def summarize_pilot(bearing_rows, tracking_rows, sequence_rows) -> list[dict[str, object]]:
    summaries = []
    for config in pilot_configurations():
        for method in ESTIMATOR_VARIANTS:
            bearings = [
                row for row in bearing_rows
                if row["trajectory_kind"] == config.trajectory_kind
                and float(row["snr_db"]) == config.snr_db
                and row["estimator_variant"] == method
            ]
            tracks = [
                row for row in tracking_rows
                if row["trajectory_kind"] == config.trajectory_kind
                and float(row["snr_db"]) == config.snr_db
                and row["estimator_variant"] == method
            ]
            sequences = [
                row for row in sequence_rows
                if row["trajectory_kind"] == config.trajectory_kind
                and float(row["snr_db"]) == config.snr_db
                and row["estimator_variant"] == method
            ]
            valid_bearings = [row for row in bearings if row["valid"]]
            valid_tracks = [row for row in tracks if row["valid"]]
            failures = Counter(row["failure_reason"] for row in sequences if row["failure_reason"])
            summaries.append(
                {
                    "split": "evaluation",
                    "trajectory_kind": config.trajectory_kind,
                    "snr_db": config.snr_db,
                    "estimator_variant": method,
                    "independent_sequence_count": len(sequences),
                    "dependent_bearing_count": len(bearings),
                    "bearing_valid_fraction": len(valid_bearings) / len(bearings),
                    "bearing_rmse_deg_conditional": float(np.sqrt(np.mean([
                        row["geodesic_error_deg"] ** 2 for row in valid_bearings
                    ]))) if valid_bearings else float("nan"),
                    "bearing_p95_deg_conditional": _percentile(
                        [row["geodesic_error_deg"] for row in valid_bearings], 95
                    ),
                    "dependent_publication_count": len(tracks),
                    "confirmed_publication_fraction": float(np.mean([
                        row["confirmed"] for row in tracks
                    ])),
                    "valid_publication_fraction": len(valid_tracks) / len(tracks),
                    "final_valid_sequence_fraction": float(np.mean([
                        row["final_valid"] for row in sequences
                    ])),
                    "position_rmse_m_conditional": float(np.sqrt(np.mean([
                        row["position_error_m"] ** 2 for row in valid_tracks
                    ]))) if valid_tracks else float("nan"),
                    "position_p95_m_conditional": _percentile(
                        [row["position_error_m"] for row in valid_tracks], 95
                    ),
                    "position_max_m_conditional": max(
                        [row["position_error_m"] for row in valid_tracks], default=float("nan")
                    ),
                    "velocity_rmse_mps_conditional": float(np.sqrt(np.mean([
                        row["velocity_error_mps"] ** 2 for row in valid_tracks
                    ]))) if valid_tracks else float("nan"),
                    "velocity_p95_mps_conditional": _percentile(
                        [row["velocity_error_mps"] for row in valid_tracks], 95
                    ),
                    "coverage_conditional": float(np.mean([
                        row["valid_and_covered"] for row in valid_tracks
                    ])) if valid_tracks else float("nan"),
                    "valid_and_covered_fraction": float(np.mean([
                        row["valid_and_covered"] for row in tracks
                    ])),
                    "mean_first_confirmation_time_s": _finite_mean([
                        row["first_confirmation_time_s"] for row in sequences
                    ]),
                    "confirmed_before_manoeuvre_sequence_fraction": float(np.mean([
                        row["confirmed_before_manoeuvre"] for row in sequences
                    ])),
                    "prediction_without_correction_sequence_fraction": float(np.mean([
                        row["accepted_update_count"] == 0 for row in sequences
                    ])),
                    "mean_reset_count": float(np.mean([row["reset_count"] for row in sequences])),
                    "failure_reasons_json": json.dumps(failures, sort_keys=True),
                    "mean_tracker_runtime_s_per_sequence": float(np.mean([
                        row["tracker_runtime_s"] for row in sequences
                    ])),
                    "mean_audio_synthesis_wall_runtime_s_per_sequence": float(np.mean([
                        row["audio_synthesis_wall_runtime_s"] for row in sequences
                    ])),
                    "mean_bearing_frontend_wall_runtime_s_per_sequence": float(np.mean([
                        row["bearing_frontend_wall_runtime_s"] for row in sequences
                    ])),
                    "mean_audio_pipeline_wall_runtime_s_per_sequence": float(np.mean([
                        row["audio_pipeline_wall_runtime_s"] for row in sequences
                    ])),
                    "maximum_history_memory_bytes": max(
                        row["maximum_history_memory_bytes"] for row in sequences
                    ),
                    "maximum_history_nodes": max(row["maximum_history_nodes"] for row in sequences),
                    "conditional_metrics_use_successful_results_only": True,
                    "overlapping_frames_are_independent_trials": False,
                }
            )
    return summaries


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError("cannot write an empty result table")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _audit_seed_rows(rows: list[dict[str, object]]) -> None:
    by_split = defaultdict(set)
    role_by_split = defaultdict(set)
    for row in rows:
        split = str(row["split"])
        seed = int(row["sequence_seed"])
        if seed in by_split[split]:
            raise RuntimeError("duplicate sequence seed inside split")
        by_split[split].add(seed)
        role_values = [int(row["source_seed"]), *json.loads(row["noise_seeds_json"])]
        if len(set(role_values)) != len(role_values):
            raise RuntimeError("source/noise seed collision inside sequence")
        if role_by_split[split] & set(role_values):
            raise RuntimeError("source/noise seed collision across sequences")
        role_by_split[split].update(role_values)
    if by_split["calibration"] & by_split["evaluation"]:
        raise RuntimeError("calibration/evaluation sequence seed overlap")
    if role_by_split["calibration"] & role_by_split["evaluation"]:
        raise RuntimeError("calibration/evaluation source/noise seed overlap")


def run_three_station_audio_tracking_pilot(
    *,
    calibration_sequence_count: int = CALIBRATION_SEQUENCE_COUNT,
    evaluation_sequence_count: int = EVALUATION_SEQUENCE_COUNT,
    duration_s: float = DURATION_S,
    output_directory: Path = RESULTS,
    progress: bool = False,
) -> tuple[list[dict[str, object]], list[dict[str, object]], list[dict[str, object]]]:
    calibration_raw = []
    seed_rows = []
    for configuration_index, config in enumerate(pilot_configurations()):
        for sequence_index in range(calibration_sequence_count):
            _, _, stream, rows, _ = generate_audio_sequence(
                config, configuration_index, "calibration", sequence_index,
                duration_s=duration_s,
            )
            calibration_raw.extend(rows)
            if progress:
                print(
                    f"calibration {configuration_index + 1}/6 sequence "
                    f"{sequence_index + 1}/{calibration_sequence_count}",
                    flush=True,
                )
            seed_rows.append({
                "split": "calibration", "configuration_index": configuration_index,
                "sequence_index": sequence_index, "sequence_seed": stream.base_seed,
                "source_seed": stream.source_seed,
                "noise_seeds_json": json.dumps([item.noise_seed for item in stream.stations]),
            })
    calibrations = calibrate_audio_bearings(calibration_raw, calibration_sequence_count)

    evaluation_bearings = []
    tracking_rows = []
    update_rows = []
    phase_rows = []
    sequence_rows = []
    for configuration_index, config in enumerate(pilot_configurations()):
        for sequence_index in range(evaluation_sequence_count):
            stations, trajectory, stream, rows, audio_runtimes = generate_audio_sequence(
                config, configuration_index, "evaluation", sequence_index,
                duration_s=duration_s,
            )
            evaluation_bearings.extend(rows)
            seed_rows.append({
                "split": "evaluation", "configuration_index": configuration_index,
                "sequence_index": sequence_index, "sequence_seed": stream.base_seed,
                "source_seed": stream.source_seed,
                "noise_seeds_json": json.dumps([item.noise_seed for item in stream.stations]),
            })
            for method in ESTIMATOR_VARIANTS:
                measurements = bearing_measurements_from_records(
                    rows, calibrations, method, frame_stride=TRACKER_FRAME_STRIDE
                )
                track, sequence, updates = run_tracker(
                    stations, trajectory, measurements, method
                )
                for row in track:
                    row.update(
                        configuration_index=configuration_index,
                        sequence_index=sequence_index,
                        sequence_seed=stream.base_seed,
                        trajectory_kind=config.trajectory_kind,
                        snr_db=config.snr_db,
                    )
                for row in updates:
                    row.update(
                        configuration_index=configuration_index,
                        sequence_index=sequence_index,
                        sequence_seed=stream.base_seed,
                        trajectory_kind=config.trajectory_kind,
                        snr_db=config.snr_db,
                    )
                sequence.update(
                    configuration_index=configuration_index,
                    sequence_index=sequence_index,
                    sequence_seed=stream.base_seed,
                    trajectory_kind=config.trajectory_kind,
                    snr_db=config.snr_db,
                    **audio_runtimes,
                )
                phases = summarize_sequence_phases(track, updates, sequence)
                for row in phases:
                    row.update(
                        configuration_index=configuration_index,
                        sequence_index=sequence_index,
                        sequence_seed=stream.base_seed,
                        trajectory_kind=config.trajectory_kind,
                        snr_db=config.snr_db,
                    )
                tracking_rows.extend(track)
                update_rows.extend(updates)
                phase_rows.extend(phases)
                sequence_rows.append(sequence)
            if progress:
                print(
                    f"evaluation {configuration_index + 1}/6 sequence "
                    f"{sequence_index + 1}/{evaluation_sequence_count}",
                    flush=True,
                )
    _audit_seed_rows(seed_rows)
    summaries = summarize_pilot(evaluation_bearings, tracking_rows, sequence_rows)
    _write_csv(output_directory / "three_station_audio_calibration.csv", _calibration_rows(calibrations))
    _write_csv(output_directory / "three_station_audio_bearing_results.csv", [
        _csv_bearing_row(row) for row in evaluation_bearings
    ])
    _write_csv(output_directory / "three_station_audio_tracking_results.csv", tracking_rows)
    _write_csv(output_directory / "three_station_audio_update_results.csv", update_rows)
    _write_csv(output_directory / "three_station_audio_phase_summary.csv", phase_rows)
    _write_csv(output_directory / "three_station_audio_sequence_results.csv", sequence_rows)
    _write_csv(output_directory / "three_station_audio_summary.csv", summaries)
    _write_csv(output_directory / "three_station_audio_seed_provenance.csv", seed_rows)
    return summaries, sequence_rows, tracking_rows


def smoke_test() -> dict[str, object]:
    config = AudioPilotConfig("constant_velocity", 10.0)
    stations, trajectory, stream, rows, audio_runtimes = generate_audio_sequence(
        config, 0, "smoke", 0, duration_s=1.20
    )
    # Smoke calibration uses its own truth only to exercise the contract; the
    # full pilot always uses independent calibration and evaluation sequences.
    calibration_rows = [dict(row, split="calibration") for row in rows]
    calibrations = calibrate_audio_bearings(calibration_rows, 1)
    outcomes = {}
    for method in ESTIMATOR_VARIANTS:
        measurements = bearing_measurements_from_records(
            rows, calibrations, method, frame_stride=TRACKER_FRAME_STRIDE
        )
        track, sequence, _ = run_tracker(stations, trajectory, measurements, method)
        outcomes[method] = {
            "bearing_count": len(measurements),
            "valid_bearing_count": sum(item.valid for item in measurements),
            "publication_count": len(track),
            "final_valid": sequence["final_valid"],
        }
    return {"audio_runtimes_s": audio_runtimes, "outcomes": outcomes}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true")
    arguments = parser.parse_args()
    if arguments.smoke:
        print(json.dumps(smoke_test(), indent=2, sort_keys=True))
    else:
        summary, sequences, _ = run_three_station_audio_tracking_pilot(progress=True)
        print("summary rows", len(summary), "sequence rows", len(sequences))


if __name__ == "__main__":
    main()


__all__ = [
    "AudioBearingCalibration",
    "AudioPilotConfig",
    "CALIBRATION_POOLING_RULE",
    "MANOEUVRE_END_S",
    "MANOEUVRE_START_S",
    "SOURCE_MAXIMUM_FREQUENCY_HZ",
    "audio_sequence_seed",
    "bearing_measurements_from_records",
    "calibrate_audio_bearings",
    "extract_audio_bearing_records",
    "generate_audio_sequence",
    "pilot_configurations",
    "pilot_stations",
    "run_three_station_audio_tracking_pilot",
    "run_tracker",
    "smoke_test",
    "summarize_sequence_phases",
    "trajectory_for_audio_pilot",
]
