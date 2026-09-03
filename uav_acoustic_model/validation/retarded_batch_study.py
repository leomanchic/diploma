"""Reproducible S7C-B bearing-event and retarded batch validation study.

Truth and scenario generation live here, outside the event processor and
estimator contracts.  The independent Monte Carlo unit is one complete
sequence; causal prefixes from the same sequence are dependent diagnostics.
"""

from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from estimators.retarded_state_batch import (
    CausalPrefixBatchResult,
    RetardedBatchResult,
    estimate_causal_prefix_batches,
    estimate_offline_retarded_batch,
)
from model.bearing_events import ScheduledBearingEvent
from model.bearing_statistics import tangent_basis
from model.dynamic_state import ConstantVelocityState
from model.geometry import direction_angles, tetrahedral_array
from model.measurements import BearingMeasurement
from model.retarded_bearing import predict_retarded_bearing
from model.station import StationPose


DEFAULT_STUDY_SEED = 20260903
DEFAULT_SEQUENCE_COUNT = 4
PROCESSING_TIMES_S = (1.5, 2.5, 3.5, 4.5, 7.0)


@dataclass(frozen=True, slots=True)
class RetardedBatchStudyConfig:
    geometry: str
    motion: str
    angular_noise_std_deg: float
    delivery_schedule: str
    sequence_count: int = DEFAULT_SEQUENCE_COUNT
    base_seed: int = DEFAULT_STUDY_SEED


@dataclass(frozen=True, slots=True)
class RetardedBatchScenario:
    """Validation-only container; never passed whole to an online component."""

    config: RetardedBatchStudyConfig
    sequence_index: int
    sequence_seed: int
    bearing_noise_seed: int
    delivery_seed: int
    stations: tuple[StationPose, ...]
    truth_state: ConstantVelocityState
    events: tuple[ScheduledBearingEvent, ...]


def default_retarded_batch_configurations(
    *, sequence_count: int = DEFAULT_SEQUENCE_COUNT
) -> tuple[RetardedBatchStudyConfig, ...]:
    """Return the fixed pre-evaluation S7C-B study matrix."""

    return tuple(
        RetardedBatchStudyConfig(
            geometry,
            motion,
            noise,
            schedule,
            sequence_count=sequence_count,
        )
        for geometry in ("wide", "compact")
        for motion in ("stationary", "oblique_slow", "oblique_fast")
        for noise in (0.05, 0.2)
        for schedule in ("ordered", "reordered_dropout")
    )


def _stations(geometry: str) -> tuple[StationPose, ...]:
    if geometry == "wide":
        positions = ([0.0, 0.0, 0.0], [100.0, 0.0, 5.0], [10.0, 90.0, -2.0])
    elif geometry == "compact":
        positions = ([0.0, 0.0, 0.0], [35.0, 0.0, 2.0], [8.0, 30.0, -1.0])
    else:
        raise ValueError(f"unknown geometry: {geometry}")
    return tuple(
        StationPose(f"S{index}", position, np.eye(3), tetrahedral_array())
        for index, position in enumerate(positions)
    )


def _velocity(motion: str) -> np.ndarray:
    values = {
        "stationary": np.asarray([0.0, 0.0, 0.0]),
        "oblique_slow": np.asarray([6.0, -3.0, 1.0]),
        "oblique_fast": np.asarray([-15.0, 8.0, 2.0]),
    }
    if motion not in values:
        raise ValueError(f"unknown motion: {motion}")
    return values[motion]


def _exp_map(direction: np.ndarray, offset: np.ndarray) -> np.ndarray:
    phi, elevation = direction_angles(direction)
    tangent = tangent_basis(phi, elevation).T @ offset
    angle = float(np.linalg.norm(tangent))
    if angle == 0.0:
        return direction.copy()
    return np.cos(angle) * direction + np.sin(angle) * tangent / angle


def generate_retarded_batch_scenario(
    config: RetardedBatchStudyConfig, sequence_index: int
) -> RetardedBatchScenario:
    """Generate one independent full sequence with separate noise/delivery RNGs."""

    index = int(sequence_index)
    if index < 0 or index >= config.sequence_count:
        raise ValueError("sequence_index outside configured sequence_count")
    configuration_index = default_retarded_batch_configurations(
        sequence_count=config.sequence_count
    ).index(config)
    sequence_seed = config.base_seed + 1000 * configuration_index + index
    seed_sequence = np.random.SeedSequence(sequence_seed)
    noise_seed, delivery_seed = [
        int(child.generate_state(1, dtype=np.uint32)[0])
        for child in seed_sequence.spawn(2)
    ]
    noise_rng = np.random.default_rng(noise_seed)
    delivery_rng = np.random.default_rng(delivery_seed)
    stations = _stations(config.geometry)
    truth = ConstantVelocityState(
        np.asarray([55.0, 45.0, 35.0])
        + noise_rng.normal(0.0, 1.0, size=3),
        _velocity(config.motion),
        0.0,
    )
    covariance = np.eye(2) * np.deg2rad(config.angular_noise_std_deg) ** 2
    station_times = (
        np.arange(0.8, 5.01, 0.6),
        np.arange(0.95, 5.01, 0.75),
        np.arange(1.1, 5.01, 0.9),
    )
    events: list[ScheduledBearingEvent] = []
    for station_index, (station, reception_times) in enumerate(
        zip(stations, station_times, strict=True)
    ):
        for frame_index, reception in enumerate(reception_times):
            prediction = predict_retarded_bearing(truth, station, float(reception))
            angular_noise = noise_rng.normal(
                0.0, np.deg2rad(config.angular_noise_std_deg), size=2
            )
            measured = _exp_map(prediction.direction_local, angular_noise)
            if config.delivery_schedule == "ordered":
                delay = 0.02 + 0.005 * station_index
            elif config.delivery_schedule == "reordered_dropout":
                delay = float(delivery_rng.uniform(0.01, 0.65))
            else:
                raise ValueError(
                    f"unknown delivery schedule: {config.delivery_schedule}"
                )
            measurement = BearingMeasurement(
                station.station_id,
                f"s7cb-{configuration_index}-{index}",
                frame_index,
                float(reception),
                float(reception + delay),
                measured,
                covariance,
                np.zeros(2),
                "direct_bearing",
                quality_metadata={"synthetic_bearing_energy": 1.0},
            )
            dropped = bool(
                config.delivery_schedule == "reordered_dropout"
                and station_index == 2
                and frame_index in (1, 2)
            )
            events.append(
                ScheduledBearingEvent(
                    measurement,
                    dropped=dropped,
                    drop_reason="temporary_station_outage" if dropped else None,
                )
            )
    if config.delivery_schedule == "reordered_dropout":
        # One exact transport duplicate exercises deduplication without adding
        # statistical weight.  It is the same immutable measurement object.
        events.append(events[2])
    return RetardedBatchScenario(
        config=config,
        sequence_index=index,
        sequence_seed=sequence_seed,
        bearing_noise_seed=noise_seed,
        delivery_seed=delivery_seed,
        stations=stations,
        truth_state=truth,
        events=tuple(events),
    )


def _result_row(
    scenario: RetardedBatchScenario,
    mode: str,
    prefix_index: int,
    processing_time_s: float,
    prefix,
    estimate: RetardedBatchResult,
) -> dict[str, object]:
    truth = scenario.truth_state
    if estimate.valid and estimate.state is not None:
        position_error_vector = (
            estimate.state.position_at_reference_world_m
            - truth.position_at_reference_world_m
        )
        velocity_error_vector = (
            estimate.state.velocity_world_mps - truth.velocity_world_mps
        )
        position_error = float(np.linalg.norm(position_error_vector))
        velocity_error = float(np.linalg.norm(velocity_error_vector))
        estimate_vector = estimate.state.vector
    else:
        position_error_vector = np.full(3, np.nan)
        velocity_error_vector = np.full(3, np.nan)
        position_error = velocity_error = float("nan")
        estimate_vector = np.full(6, np.nan)
    actions = Counter(item.action for item in prefix.journal)
    event_journal = [
        {
            "processing_time_s": item.processing_time_s,
            "event_id": item.event_id,
            "audio_frame_id": item.audio_frame_id,
            "station_id": item.station_id,
            "frame_index": item.frame_index,
            "reception_timestamp_s": item.reception_timestamp_s,
            "available_timestamp_s": item.available_timestamp_s,
            "action": item.action,
            "reason": item.reason,
        }
        for item in prefix.journal
    ]
    return {
        "row_type": "sequence_result",
        "geometry": scenario.config.geometry,
        "motion": scenario.config.motion,
        "angular_noise_std_deg": scenario.config.angular_noise_std_deg,
        "delivery_schedule": scenario.config.delivery_schedule,
        "mode": mode,
        "prefix_index": prefix_index,
        "processing_time_s": processing_time_s,
        "reference_time_s": truth.reference_time_s,
        "sequence_index": scenario.sequence_index,
        "sequence_seed": scenario.sequence_seed,
        "bearing_noise_seed": scenario.bearing_noise_seed,
        "delivery_seed": scenario.delivery_seed,
        "independent_unit": "whole_sequence",
        "prefixes_within_sequence_are_dependent": True,
        "valid": estimate.valid,
        "failure_reason": estimate.failure_reason or "",
        "scheduled_event_count": len(scenario.events),
        "available_event_log_count": len(prefix.journal),
        "event_journal_json": json.dumps(
            event_journal, sort_keys=True, separators=(",", ":")
        ),
        "accepted_event_ids_json": json.dumps(prefix.accepted_event_ids),
        "conflicted_event_ids_json": json.dumps(prefix.conflicted_event_ids),
        "used_measurement_count": estimate.measurement_count,
        "accepted_count": actions["accepted"],
        "duplicate_count": actions["duplicate_exact"],
        "dropped_count": actions["excluded_dropped"],
        "invalid_count": actions["excluded_invalid"],
        "conflict_count": actions["excluded_conflict"],
        "truth_q0_e_m": truth.position_at_reference_world_m[0],
        "truth_q0_n_m": truth.position_at_reference_world_m[1],
        "truth_q0_u_m": truth.position_at_reference_world_m[2],
        "truth_v_e_mps": truth.velocity_world_mps[0],
        "truth_v_n_mps": truth.velocity_world_mps[1],
        "truth_v_u_mps": truth.velocity_world_mps[2],
        "estimate_q0_e_m": estimate_vector[0],
        "estimate_q0_n_m": estimate_vector[1],
        "estimate_q0_u_m": estimate_vector[2],
        "estimate_v_e_mps": estimate_vector[3],
        "estimate_v_n_mps": estimate_vector[4],
        "estimate_v_u_mps": estimate_vector[5],
        "position_error_e_m": position_error_vector[0],
        "position_error_n_m": position_error_vector[1],
        "position_error_u_m": position_error_vector[2],
        "velocity_error_e_mps": velocity_error_vector[0],
        "velocity_error_n_mps": velocity_error_vector[1],
        "velocity_error_u_mps": velocity_error_vector[2],
        "position_error_m": position_error,
        "velocity_error_mps": velocity_error,
        "objective": estimate.objective,
        "maximum_angular_residual_rad": estimate.maximum_angular_residual_rad,
        "local_observability_rank": estimate.local_observability_rank,
        "scaled_information_condition_number": estimate.scaled_information_condition_number,
        "optimizer_success": estimate.optimizer_success,
        "optimizer_message": estimate.optimizer_message,
        "constraint_max_abs_rad": estimate.constraint_max_abs_rad,
        "scaled_projected_kkt_residual": estimate.scaled_projected_kkt_residual,
        "runtime_s": estimate.runtime_s,
    }


def run_retarded_batch_configuration(
    config: RetardedBatchStudyConfig,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for sequence_index in range(config.sequence_count):
        scenario = generate_retarded_batch_scenario(config, sequence_index)
        offline_prefix, offline = estimate_offline_retarded_batch(
            scenario.stations,
            scenario.events,
            estimator_variant="direct_bearing",
            reference_time_s=0.0,
        )
        rows.append(
            _result_row(
                scenario,
                "offline_full_record",
                -1,
                offline_prefix.processing_time_s,
                offline_prefix,
                offline,
            )
        )
        causal: tuple[CausalPrefixBatchResult, ...] = estimate_causal_prefix_batches(
            scenario.stations,
            scenario.events,
            PROCESSING_TIMES_S,
            estimator_variant="direct_bearing",
            reference_time_s=0.0,
        )
        for prefix_index, item in enumerate(causal):
            rows.append(
                _result_row(
                    scenario,
                    "causal_prefix",
                    prefix_index,
                    item.processing_time_s,
                    item.prefix,
                    item.estimate,
                )
            )
    return rows


def _summary_rows(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    def finite_max(values: list[object]) -> float:
        array = np.asarray(values, dtype=float)
        finite = array[np.isfinite(array)]
        return float(np.max(finite)) if finite.size else float("nan")

    grouped: dict[tuple[object, ...], list[dict[str, object]]] = defaultdict(list)
    keys = (
        "geometry",
        "motion",
        "angular_noise_std_deg",
        "delivery_schedule",
        "mode",
        "prefix_index",
    )
    for row in rows:
        grouped[tuple(row[key] for key in keys)].append(row)
    summaries: list[dict[str, object]] = []
    for key, group in grouped.items():
        valid = [row for row in group if row["valid"]]
        position = np.asarray([row["position_error_m"] for row in valid], dtype=float)
        velocity = np.asarray([row["velocity_error_mps"] for row in valid], dtype=float)
        position_vectors = np.asarray(
            [
                [row["position_error_e_m"], row["position_error_n_m"], row["position_error_u_m"]]
                for row in valid
            ],
            dtype=float,
        )
        velocity_vectors = np.asarray(
            [
                [row["velocity_error_e_mps"], row["velocity_error_n_mps"], row["velocity_error_u_mps"]]
                for row in valid
            ],
            dtype=float,
        )
        failure_counts = Counter(
            str(row["failure_reason"]) for row in group if not row["valid"]
        )
        summary = dict(zip(keys, key, strict=True))
        summary.update(
            {
                "row_type": "aggregate",
                "processing_time_s": float(
                    np.mean([row["processing_time_s"] for row in group])
                ),
                "independent_sequence_count": len(group),
                "dependent_prefix_count_per_sequence": len(PROCESSING_TIMES_S),
                "successful_sequence_count": len(valid),
                "successful_fraction": len(valid) / len(group),
                "failure_reason_counts": json.dumps(failure_counts, sort_keys=True),
                "position_rmse_m": (
                    float(np.sqrt(np.mean(position**2))) if len(valid) else float("nan")
                ),
                "position_median_m": (
                    float(np.median(position)) if len(valid) else float("nan")
                ),
                "position_p95_m": (
                    float(np.percentile(position, 95)) if len(valid) else float("nan")
                ),
                "velocity_rmse_mps": (
                    float(np.sqrt(np.mean(velocity**2))) if len(valid) else float("nan")
                ),
                "velocity_median_mps": (
                    float(np.median(velocity)) if len(valid) else float("nan")
                ),
                "velocity_p95_mps": (
                    float(np.percentile(velocity, 95)) if len(valid) else float("nan")
                ),
                "position_bias_e_m": (
                    float(np.mean(position_vectors[:, 0])) if len(valid) else float("nan")
                ),
                "position_bias_n_m": (
                    float(np.mean(position_vectors[:, 1])) if len(valid) else float("nan")
                ),
                "position_bias_u_m": (
                    float(np.mean(position_vectors[:, 2])) if len(valid) else float("nan")
                ),
                "velocity_bias_e_mps": (
                    float(np.mean(velocity_vectors[:, 0])) if len(valid) else float("nan")
                ),
                "velocity_bias_n_mps": (
                    float(np.mean(velocity_vectors[:, 1])) if len(valid) else float("nan")
                ),
                "velocity_bias_u_mps": (
                    float(np.mean(velocity_vectors[:, 2])) if len(valid) else float("nan")
                ),
                "mean_used_measurement_count": float(
                    np.mean([row["used_measurement_count"] for row in group])
                ),
                "median_condition_number": (
                    float(np.median([row["scaled_information_condition_number"] for row in valid]))
                    if len(valid)
                    else float("nan")
                ),
                "median_runtime_s": float(
                    np.median([row["runtime_s"] for row in group])
                ),
                "maximum_constraint_residual_rad": finite_max(
                    [row["constraint_max_abs_rad"] for row in group]
                ),
                "maximum_scaled_kkt_residual": finite_max(
                    [row["scaled_projected_kkt_residual"] for row in group]
                ),
            }
        )
        summaries.append(summary)
    return summaries


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0])
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def run_retarded_batch_study(
    output_directory: str | Path = "results",
    *,
    sequence_count: int = DEFAULT_SEQUENCE_COUNT,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    """Run the fixed study matrix and save sequence/aggregate CSV artifacts."""

    rows: list[dict[str, object]] = []
    for config in default_retarded_batch_configurations(
        sequence_count=sequence_count
    ):
        rows.extend(run_retarded_batch_configuration(config))
    summaries = _summary_rows(rows)
    output = Path(output_directory)
    _write_csv(output / "retarded_batch_sequence_results.csv", rows)
    _write_csv(output / "retarded_batch_summary.csv", summaries)
    return rows, summaries


__all__ = [
    "DEFAULT_SEQUENCE_COUNT",
    "DEFAULT_STUDY_SEED",
    "PROCESSING_TIMES_S",
    "RetardedBatchScenario",
    "RetardedBatchStudyConfig",
    "default_retarded_batch_configurations",
    "generate_retarded_batch_scenario",
    "run_retarded_batch_configuration",
    "run_retarded_batch_study",
]
