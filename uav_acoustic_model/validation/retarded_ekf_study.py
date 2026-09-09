"""Sequence-level validation for the strict-CV retarded-time EKF baseline.

The protocol is fixed before evaluation: 24 physical configurations and four
independent sequences per configuration.  Prefixes and updates within a
sequence are dependent observations; uncertainty intervals are therefore
formed from whole-sequence final outcomes, never from pooled time samples.
"""

from __future__ import annotations

import csv
import json
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import ArrayLike
from scipy.stats import chi2

from estimators.retarded_ekf import CausalRetardedTimeEKF
from estimators.retarded_state_batch import (
    estimate_offline_retarded_batch,
    estimate_retarded_constant_velocity_batch,
)
from model.bearing_statistics import tangent_basis
from model.dynamic_state import ConstantVelocityState
from model.geometry import direction_angles, tetrahedral_array
from model.measurements import BearingMeasurement
from model.retarded_bearing import predict_retarded_bearing
from model.station import StationPose


DEFAULT_STUDY_SEED = 20260908
DEFAULT_SEQUENCE_COUNT = 4
P95_METHOD = "linear"


def linear_percentile(values: ArrayLike, percentile: float) -> float:
    """Return an explicitly linearly interpolated percentile.

    C1 reports P95 at the grain stated by the caller.  The explicit method
    prevents accidental mixing with the discontinuous ``higher`` convention.
    """

    array = np.asarray(values, dtype=float)
    if array.size == 0:
        return float("nan")
    return float(np.percentile(array, percentile, method=P95_METHOD))
STATE_COVERAGE_PROBABILITY = 0.95
_STATE_COVERAGE_THRESHOLD = float(chi2.ppf(STATE_COVERAGE_PROBABILITY, 6))
_NIS_COVERAGE_THRESHOLD = float(chi2.ppf(STATE_COVERAGE_PROBABILITY, 2))
_PHYSICAL_IDENTITIES = tuple(
    (geometry, motion, azimuth_std, elevation_std, schedule)
    for geometry in ("informative", "poorly_conditioned")
    for motion in ("stationary", "uniform_oblique")
    for azimuth_std, elevation_std in ((0.03, 0.05), (0.10, 0.18), (0.30, 0.50))
    for schedule in ("ordered", "reordered")
)


@dataclass(frozen=True, slots=True)
class RetardedEKFStudyConfig:
    geometry: str
    motion: str
    azimuth_noise_std_deg: float
    elevation_noise_std_deg: float
    delivery_schedule: str
    sequence_count: int = DEFAULT_SEQUENCE_COUNT
    base_seed: int = DEFAULT_STUDY_SEED


@dataclass(frozen=True, slots=True)
class RetardedEKFScenario:
    config: RetardedEKFStudyConfig
    configuration_index: int
    sequence_index: int
    truth_seed: int
    bearing_noise_seed: int
    delivery_seed: int
    stations: tuple[StationPose, ...]
    truth_state: ConstantVelocityState
    events: tuple[BearingMeasurement, ...]


def retarded_ekf_seed_provenance(
    base_seed: int, configuration_index: int, sequence_index: int
) -> tuple[tuple[str, ...], tuple[int, ...]]:
    """Return named collision-auditable streams for one whole sequence."""

    if configuration_index < 0 or sequence_index < 0:
        raise ValueError("configuration_index and sequence_index must be non-negative")
    labels = ("truth", "bearing_noise", "delivery")
    seeds = tuple(
        int(
            np.random.SeedSequence(
                [int(base_seed), int(configuration_index), int(sequence_index), stream]
            ).generate_state(1, dtype=np.uint64)[0]
        )
        for stream in range(len(labels))
    )
    identifiers = tuple(
        f"s7cc1:{int(base_seed)}:{configuration_index}:{sequence_index}:{label}"
        for label in labels
    )
    return identifiers, seeds


def default_retarded_ekf_configurations(
    *,
    sequence_count: int = DEFAULT_SEQUENCE_COUNT,
    base_seed: int = DEFAULT_STUDY_SEED,
) -> tuple[RetardedEKFStudyConfig, ...]:
    return tuple(
        RetardedEKFStudyConfig(
            geometry,
            motion,
            azimuth_std,
            elevation_std,
            schedule,
            sequence_count=sequence_count,
            base_seed=base_seed,
        )
        for geometry, motion, azimuth_std, elevation_std, schedule in _PHYSICAL_IDENTITIES
    )


def _stations(geometry: str) -> tuple[StationPose, ...]:
    if geometry == "informative":
        positions = ([0.0, 0.0, 0.0], [100.0, 0.0, 5.0], [10.0, 90.0, -2.0])
    elif geometry == "poorly_conditioned":
        positions = ([0.0, 0.0, 0.0], [90.0, 0.0, 1.0], [180.0, 4.0, 2.0])
    else:
        raise ValueError(f"unknown geometry: {geometry}")
    return tuple(
        StationPose(f"S{index}", value, np.eye(3), tetrahedral_array())
        for index, value in enumerate(positions)
    )


def _velocity(motion: str) -> np.ndarray:
    if motion == "stationary":
        return np.zeros(3)
    if motion == "uniform_oblique":
        return np.asarray([7.0, -3.0, 1.5])
    raise ValueError(f"unknown motion: {motion}")


def _exp_map(direction: np.ndarray, tangent_offset: np.ndarray) -> np.ndarray:
    phi, elevation = direction_angles(direction)
    tangent = tangent_basis(phi, elevation).T @ tangent_offset
    angle = float(np.linalg.norm(tangent))
    if angle == 0.0:
        return direction.copy()
    return np.cos(angle) * direction + np.sin(angle) * tangent / angle


def generate_retarded_ekf_scenario(
    config: RetardedEKFStudyConfig, sequence_index: int
) -> RetardedEKFScenario:
    """Generate direct bearings with errors in the declared prediction frame."""

    identity = (
        config.geometry,
        config.motion,
        config.azimuth_noise_std_deg,
        config.elevation_noise_std_deg,
        config.delivery_schedule,
    )
    try:
        configuration_index = _PHYSICAL_IDENTITIES.index(identity)
    except ValueError as error:
        raise ValueError(f"unknown physical S7C-C1 configuration: {identity}") from error
    if sequence_index < 0 or sequence_index >= config.sequence_count:
        raise ValueError("sequence_index outside configured sequence_count")
    _, seeds = retarded_ekf_seed_provenance(
        config.base_seed, configuration_index, sequence_index
    )
    truth_rng, noise_rng, delivery_rng = (
        np.random.default_rng(seed) for seed in seeds
    )
    stations = _stations(config.geometry)
    truth = ConstantVelocityState(
        np.asarray([70.0, 55.0, 40.0]) + truth_rng.normal(0.0, 2.0, 3),
        _velocity(config.motion),
        0.0,
    )
    covariance = np.diag(
        np.deg2rad(
            [config.azimuth_noise_std_deg, config.elevation_noise_std_deg]
        )
        ** 2
    )
    reception_grids = (
        np.asarray([0.60, 1.40, 2.20, 3.00, 3.80]),
        np.asarray([0.75, 1.55, 2.35, 3.15, 3.95]),
        np.asarray([0.90, 1.70, 2.50, 3.30, 4.10]),
    )
    events: list[BearingMeasurement] = []
    sequence_id = f"s7cc1-{config.base_seed}-{configuration_index}-{sequence_index}"
    for station_index, (station, times) in enumerate(
        zip(stations, reception_grids, strict=True)
    ):
        for frame_index, reception_time in enumerate(times):
            prediction = predict_retarded_bearing(
                truth, station, float(reception_time)
            )
            tangent_error = noise_rng.multivariate_normal(np.zeros(2), covariance)
            measured = _exp_map(prediction.direction_local, tangent_error)
            if config.delivery_schedule == "ordered":
                delivery_delay = 0.020 + 0.004 * station_index
            elif config.delivery_schedule == "reordered":
                delivery_delay = float(delivery_rng.uniform(0.01, 0.42))
            else:
                raise ValueError(f"unknown delivery schedule: {config.delivery_schedule}")
            events.append(
                BearingMeasurement(
                    station.station_id,
                    sequence_id,
                    frame_index,
                    float(reception_time),
                    float(reception_time + delivery_delay),
                    measured,
                    covariance,
                    np.zeros(2),
                    "direct_bearing",
                    quality_metadata={"synthetic_bearing_energy": 1.0},
                    tangent_frame="prediction",
                )
            )
    return RetardedEKFScenario(
        config,
        configuration_index,
        sequence_index,
        seeds[0],
        seeds[1],
        seeds[2],
        stations,
        truth,
        tuple(events),
    )


def _state_metrics(
    state: ConstantVelocityState | None,
    covariance: np.ndarray,
    truth: ConstantVelocityState,
    time_s: float,
) -> dict[str, object]:
    if state is None or not np.all(np.isfinite(covariance)):
        return {
            "position_error_m": float("nan"),
            "velocity_error_mps": float("nan"),
            "state_nees": float("nan"),
            "state_95_coverage": False,
            "covariance_rank": 0,
            "covariance_condition_number": float("inf"),
            "covariance_symmetry_error": float("nan"),
            "covariance_minimum_eigenvalue": float("nan"),
        }
    truth_at_time = truth.position_at(time_s)
    estimate_at_time = state.position_at(time_s)
    error = np.concatenate(
        (estimate_at_time - truth_at_time, state.velocity_world_mps - truth.velocity_world_mps)
    )
    symmetric = 0.5 * (covariance + covariance.T)
    eigenvalues = np.linalg.eigvalsh(symmetric)
    rank = int(np.linalg.matrix_rank(symmetric))
    nees = (
        float(error @ np.linalg.solve(symmetric, error))
        if rank == 6 and np.min(eigenvalues) > 0.0
        else float("nan")
    )
    return {
        "position_error_m": float(np.linalg.norm(error[:3])),
        "velocity_error_mps": float(np.linalg.norm(error[3:])),
        "state_nees": nees,
        "state_95_coverage": bool(np.isfinite(nees) and nees <= _STATE_COVERAGE_THRESHOLD),
        "covariance_rank": rank,
        "covariance_condition_number": (
            float(np.max(eigenvalues) / np.min(eigenvalues))
            if rank == 6 and np.min(eigenvalues) > 0.0
            else float("inf")
        ),
        "covariance_symmetry_error": float(
            np.max(np.abs(covariance - covariance.T), initial=0.0)
        ),
        "covariance_minimum_eigenvalue": float(np.min(eigenvalues)),
    }


def _base_row(scenario: RetardedEKFScenario) -> dict[str, object]:
    identifiers, _ = retarded_ekf_seed_provenance(
        scenario.config.base_seed,
        scenario.configuration_index,
        scenario.sequence_index,
    )
    return {
        "geometry": scenario.config.geometry,
        "motion": scenario.config.motion,
        "azimuth_noise_std_deg": scenario.config.azimuth_noise_std_deg,
        "elevation_noise_std_deg": scenario.config.elevation_noise_std_deg,
        "delivery_schedule": scenario.config.delivery_schedule,
        "sequence_index": scenario.sequence_index,
        "base_seed": scenario.config.base_seed,
        "seed_scheme": "SeedSequence([base_seed,configuration_index,sequence_index,stream_id])",
        "truth_seed": scenario.truth_seed,
        "bearing_noise_seed": scenario.bearing_noise_seed,
        "delivery_seed": scenario.delivery_seed,
        "truth_provenance": identifiers[0],
        "bearing_noise_provenance": identifiers[1],
        "delivery_provenance": identifiers[2],
        "independent_unit": "whole_sequence",
        "time_samples_within_sequence_are_dependent": True,
        "tangent_frame": "prediction",
    }


def _publication_row(
    scenario: RetardedEKFScenario,
    method: str,
    publication_index: int,
    processing_time: float,
    prefix_measurement_count: int,
    valid: bool,
    failure_reason: str | None,
    state: ConstantVelocityState | None,
    covariance: np.ndarray,
    *,
    initialization_time_s: float = float("nan"),
    update_nis: tuple[float, ...] = (),
    measurement_update_runtime_s: float = 0.0,
    total_runtime_s: float = 0.0,
) -> dict[str, object]:
    row = _base_row(scenario)
    row.update(
        {
            "method": method,
            "publication_index": publication_index,
            "processing_time_s": processing_time,
            "prefix_measurement_count": prefix_measurement_count,
            "valid": valid,
            "failure_reason": failure_reason or "",
            "initialization_time_s": initialization_time_s,
            "update_nis_count": len(update_nis),
            "mean_update_nis": float(np.mean(update_nis)) if update_nis else float("nan"),
            "update_nis_95_coverage_fraction": (
                float(np.mean(np.asarray(update_nis) <= _NIS_COVERAGE_THRESHOLD))
                if update_nis
                else float("nan")
            ),
            "measurement_update_runtime_s": measurement_update_runtime_s,
            "total_processing_runtime_s": total_runtime_s,
        }
    )
    row.update(_state_metrics(state, covariance, scenario.truth_state, processing_time))
    truth_position = scenario.truth_state.position_at(processing_time)
    if state is None or not np.all(np.isfinite(covariance)):
        estimate_vector = np.full(6, np.nan)
        standard_deviation = np.full(6, np.nan)
    else:
        estimate_vector = np.concatenate(
            (state.position_at(processing_time), state.velocity_world_mps)
        )
        standard_deviation = np.sqrt(
            np.maximum(np.diag(0.5 * (covariance + covariance.T)), 0.0)
        )
    row.update(
        {
            "truth_q_e_m": truth_position[0],
            "truth_q_n_m": truth_position[1],
            "truth_q_u_m": truth_position[2],
            "truth_v_e_mps": scenario.truth_state.velocity_world_mps[0],
            "truth_v_n_mps": scenario.truth_state.velocity_world_mps[1],
            "truth_v_u_mps": scenario.truth_state.velocity_world_mps[2],
            "estimate_q_e_m": estimate_vector[0],
            "estimate_q_n_m": estimate_vector[1],
            "estimate_q_u_m": estimate_vector[2],
            "estimate_v_e_mps": estimate_vector[3],
            "estimate_v_n_mps": estimate_vector[4],
            "estimate_v_u_mps": estimate_vector[5],
            "std_q_e_m": standard_deviation[0],
            "std_q_n_m": standard_deviation[1],
            "std_q_u_m": standard_deviation[2],
            "std_v_e_mps": standard_deviation[3],
            "std_v_n_mps": standard_deviation[4],
            "std_v_u_mps": standard_deviation[5],
            "state_nees_dof": 6,
            "state_95_chi_square_threshold": _STATE_COVERAGE_THRESHOLD,
            "update_nis_dof": 2,
            "update_nis_95_chi_square_threshold": _NIS_COVERAGE_THRESHOLD,
        }
    )
    return row


def run_retarded_ekf_configuration(
    config: RetardedEKFStudyConfig,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    """Run matched EKF, causal batch, no-update and offline references."""

    publication_rows: list[dict[str, object]] = []
    sequence_rows: list[dict[str, object]] = []
    for sequence_index in range(config.sequence_count):
        scenario = generate_retarded_ekf_scenario(config, sequence_index)
        processor = CausalRetardedTimeEKF(
            scenario.stations,
            scenario.events,
            estimator_variant="direct_bearing",
        )
        processing_times = sorted(
            {item.available_timestamp_s for item in scenario.events}
        )
        no_update_state = None
        no_update_covariance = np.full((6, 6), np.nan)
        initialization_time = float("nan")
        method_rows: dict[str, list[dict[str, object]]] = defaultdict(list)
        all_ekf_nis: list[float] = []
        for publication_index, processing_time in enumerate(processing_times):
            ekf = processor.advance_to(processing_time)
            just_initialized = (
                ekf.valid
                and ekf.initialized
                and no_update_state is None
            )
            if ekf.valid and ekf.initialized and no_update_state is None:
                no_update_state = ekf.state
                no_update_covariance = np.asarray(ekf.covariance_state)
                initialization_time = processing_time
            nis = tuple(
                item.normalized_innovation_squared
                for item in ekf.update_diagnostics
                if item.update_applied
            )
            all_ekf_nis.extend(nis)
            row = _publication_row(
                scenario,
                "retarded_ekf",
                publication_index,
                processing_time,
                len(ekf.prefix.measurements),
                ekf.valid,
                ekf.failure_reason,
                ekf.state,
                np.asarray(ekf.covariance_state),
                initialization_time_s=initialization_time,
                update_nis=nis,
                measurement_update_runtime_s=ekf.measurement_update_runtime_s,
                total_runtime_s=ekf.total_runtime_s,
            )
            publication_rows.append(row)
            method_rows["retarded_ekf"].append(row)

            batch = estimate_retarded_constant_velocity_batch(
                scenario.stations,
                ekf.prefix.measurements,
                reference_time_s=processing_time,
            )
            batch_row = _publication_row(
                scenario,
                "causal_prefix_batch",
                publication_index,
                processing_time,
                len(ekf.prefix.measurements),
                batch.valid,
                batch.failure_reason,
                batch.state,
                np.asarray(batch.covariance_state_linearization),
                total_runtime_s=batch.runtime_s,
            )
            publication_rows.append(batch_row)
            method_rows["causal_prefix_batch"].append(batch_row)

            baseline_started = time.perf_counter()
            if no_update_state is None:
                baseline_state = None
                baseline_covariance = np.full((6, 6), np.nan)
                baseline_valid = False
                baseline_reason = "not_initialized"
            else:
                dt = processing_time - no_update_state.reference_time_s
                transition = np.eye(6)
                transition[:3, 3:] = dt * np.eye(3)
                baseline_state = ConstantVelocityState(
                    no_update_state.position_at(processing_time),
                    no_update_state.velocity_world_mps,
                    processing_time,
                )
                baseline_covariance = transition @ no_update_covariance @ transition.T
                baseline_valid = True
                baseline_reason = None
            baseline_runtime = time.perf_counter() - baseline_started
            if just_initialized:
                baseline_runtime += ekf.initialization_runtime_s
            baseline_row = _publication_row(
                scenario,
                "initial_batch_no_updates",
                publication_index,
                processing_time,
                len(ekf.prefix.measurements),
                baseline_valid,
                baseline_reason,
                baseline_state,
                baseline_covariance,
                initialization_time_s=initialization_time,
                total_runtime_s=baseline_runtime,
            )
            publication_rows.append(baseline_row)
            method_rows["initial_batch_no_updates"].append(baseline_row)

        offline_prefix, offline = estimate_offline_retarded_batch(
            scenario.stations,
            scenario.events,
            estimator_variant="direct_bearing",
            reference_time_s=processing_times[-1],
        )
        offline_row = _publication_row(
            scenario,
            "offline_full_record_noncausal",
            len(processing_times),
            processing_times[-1],
            len(offline_prefix.measurements),
            offline.valid,
            offline.failure_reason,
            offline.state,
            np.asarray(offline.covariance_state_linearization),
            total_runtime_s=offline.runtime_s,
        )
        publication_rows.append(offline_row)
        method_rows["offline_full_record_noncausal"].append(offline_row)

        for method, rows in method_rows.items():
            valid_rows = [row for row in rows if row["valid"]]
            final = rows[-1]
            failures = Counter(
                str(row["failure_reason"]) for row in rows if not row["valid"]
            )
            position = np.asarray(
                [row["position_error_m"] for row in valid_rows], dtype=float
            )
            velocity = np.asarray(
                [row["velocity_error_mps"] for row in valid_rows], dtype=float
            )
            sequence_row = _base_row(scenario)
            sequence_row.update(
                {
                    "method": method,
                    "dependent_publication_count": len(rows),
                    "valid_publication_count": len(valid_rows),
                    "initialization_succeeded": bool(valid_rows),
                    "time_to_first_estimate_s": (
                        float(valid_rows[0]["processing_time_s"] - processing_times[0])
                        if valid_rows
                        else float("nan")
                    ),
                    "final_valid": bool(final["valid"]),
                    "final_failure_reason": final["failure_reason"],
                    "failure_reason_counts": json.dumps(failures, sort_keys=True),
                    "dependent_time_position_rmse_m": (
                        float(np.sqrt(np.mean(position**2))) if position.size else float("nan")
                    ),
                    "dependent_time_position_p95_m": (
                        linear_percentile(position, 95)
                    ),
                    "dependent_time_velocity_rmse_mps": (
                        float(np.sqrt(np.mean(velocity**2))) if velocity.size else float("nan")
                    ),
                    "dependent_time_velocity_p95_mps": (
                        linear_percentile(velocity, 95)
                    ),
                    "dependent_time_p95_method": P95_METHOD,
                    "dependent_time_p95_grain": "dependent_publication_within_sequence",
                    "final_position_error_m": final["position_error_m"],
                    "final_velocity_error_mps": final["velocity_error_mps"],
                    "final_state_nees": final["state_nees"],
                    "final_state_95_coverage": final["state_95_coverage"],
                    "mean_measurement_nis": (
                        float(np.mean(all_ekf_nis))
                        if method == "retarded_ekf" and all_ekf_nis
                        else float("nan")
                    ),
                    "measurement_nis_95_coverage_fraction": (
                        float(np.mean(np.asarray(all_ekf_nis) <= _NIS_COVERAGE_THRESHOLD))
                        if method == "retarded_ekf" and all_ekf_nis
                        else float("nan")
                    ),
                    "total_runtime_s": float(
                        np.sum([row["total_processing_runtime_s"] for row in rows])
                    ),
                    "mean_update_runtime_s": (
                        float(
                            np.sum([row["measurement_update_runtime_s"] for row in rows])
                            / max(len(all_ekf_nis), 1)
                        )
                        if method == "retarded_ekf"
                        else float("nan")
                    ),
                    "maximum_covariance_symmetry_error": float(
                        np.nanmax([row["covariance_symmetry_error"] for row in valid_rows])
                    ) if valid_rows else float("nan"),
                    "minimum_covariance_eigenvalue": float(
                        np.nanmin([row["covariance_minimum_eigenvalue"] for row in valid_rows])
                    ) if valid_rows else float("nan"),
                }
            )
            sequence_rows.append(sequence_row)
    return publication_rows, sequence_rows


def _wilson_interval(successes: int, count: int, z: float = 1.959963984540054) -> tuple[float, float]:
    if count == 0:
        return float("nan"), float("nan")
    proportion = successes / count
    denominator = 1.0 + z**2 / count
    center = (proportion + z**2 / (2.0 * count)) / denominator
    half = z * np.sqrt(proportion * (1.0 - proportion) / count + z**2 / (4.0 * count**2)) / denominator
    return float(center - half), float(center + half)


def summarize_retarded_ekf_sequences(
    sequence_rows: list[dict[str, object]],
) -> list[dict[str, object]]:
    """Aggregate only whole-sequence final outcomes and Wilson intervals."""

    keys = (
        "geometry",
        "motion",
        "azimuth_noise_std_deg",
        "elevation_noise_std_deg",
        "delivery_schedule",
        "method",
    )
    grouped: dict[tuple[object, ...], list[dict[str, object]]] = defaultdict(list)
    for row in sequence_rows:
        grouped[tuple(row[key] for key in keys)].append(row)
    summaries: list[dict[str, object]] = []
    for key, rows in grouped.items():
        final_valid = [row for row in rows if row["final_valid"]]
        initialized = [row for row in rows if row["initialization_succeeded"]]
        coverage_count = sum(bool(row["final_state_95_coverage"]) for row in final_valid)
        coverage_low, coverage_high = _wilson_interval(coverage_count, len(final_valid))
        initialization_low, initialization_high = _wilson_interval(len(initialized), len(rows))
        position = np.asarray(
            [row["final_position_error_m"] for row in final_valid], dtype=float
        )
        velocity = np.asarray(
            [row["final_velocity_error_mps"] for row in final_valid], dtype=float
        )
        failure_counts = Counter(
            str(row["final_failure_reason"]) for row in rows if not row["final_valid"]
        )
        summary = dict(zip(keys, key, strict=True))
        summary.update(
            {
                "independent_sequence_count": len(rows),
                "successful_initialization_count": len(initialized),
                "successful_initialization_fraction": len(initialized) / len(rows),
                "successful_initialization_fraction_ci95_low": initialization_low,
                "successful_initialization_fraction_ci95_high": initialization_high,
                "final_valid_count": len(final_valid),
                "final_failure_fraction": 1.0 - len(final_valid) / len(rows),
                "final_failure_reason_counts": json.dumps(failure_counts, sort_keys=True),
                "mean_time_to_first_estimate_s": float(
                    np.mean([row["time_to_first_estimate_s"] for row in initialized])
                ) if initialized else float("nan"),
                "final_position_rmse_m": float(np.sqrt(np.mean(position**2))) if position.size else float("nan"),
                "final_position_p95_m": linear_percentile(position, 95),
                "final_velocity_rmse_mps": float(np.sqrt(np.mean(velocity**2))) if velocity.size else float("nan"),
                "final_velocity_p95_mps": linear_percentile(velocity, 95),
                "final_p95_method": P95_METHOD,
                "final_p95_grain": "one_final_error_per_independent_sequence",
                "final_state_95_coverage_fraction": coverage_count / len(final_valid) if final_valid else float("nan"),
                "final_state_95_coverage_ci95_low": coverage_low,
                "final_state_95_coverage_ci95_high": coverage_high,
                "mean_sequence_measurement_nis": float(
                    np.nanmean([row["mean_measurement_nis"] for row in rows])
                ) if any(np.isfinite(float(row["mean_measurement_nis"])) for row in rows) else float("nan"),
                "mean_sequence_nis_95_coverage_fraction": float(
                    np.nanmean([row["measurement_nis_95_coverage_fraction"] for row in rows])
                ) if any(np.isfinite(float(row["measurement_nis_95_coverage_fraction"])) for row in rows) else float("nan"),
                "median_total_runtime_s": float(np.median([row["total_runtime_s"] for row in rows])),
                "mean_update_runtime_s": float(
                    np.nanmean([row["mean_update_runtime_s"] for row in rows])
                ) if any(np.isfinite(float(row["mean_update_runtime_s"])) for row in rows) else float("nan"),
                "maximum_covariance_symmetry_error": float(
                    np.nanmax([row["maximum_covariance_symmetry_error"] for row in rows])
                ),
                "minimum_covariance_eigenvalue": float(
                    np.nanmin([row["minimum_covariance_eigenvalue"] for row in rows])
                ),
                "coverage_interval_unit": "whole_sequence_final_outcome",
                "temporal_nis_nees_are_dependent": True,
            }
        )
        summaries.append(summary)
    return summaries


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def run_retarded_ekf_study(
    output_directory: str | Path = "results",
    *,
    sequence_count: int = DEFAULT_SEQUENCE_COUNT,
    base_seed: int = DEFAULT_STUDY_SEED,
) -> tuple[list[dict[str, object]], list[dict[str, object]], list[dict[str, object]]]:
    """Execute the fixed C1 protocol and write event, sequence and summary CSV."""

    publications: list[dict[str, object]] = []
    sequences: list[dict[str, object]] = []
    for config in default_retarded_ekf_configurations(
        sequence_count=sequence_count, base_seed=base_seed
    ):
        config_publications, config_sequences = run_retarded_ekf_configuration(config)
        publications.extend(config_publications)
        sequences.extend(config_sequences)
    summaries = summarize_retarded_ekf_sequences(sequences)
    output = Path(output_directory)
    _write_csv(output / "retarded_ekf_event_results.csv", publications)
    _write_csv(output / "retarded_ekf_sequence_results.csv", sequences)
    _write_csv(output / "retarded_ekf_summary.csv", summaries)
    return publications, sequences, summaries


__all__ = [
    "DEFAULT_SEQUENCE_COUNT",
    "DEFAULT_STUDY_SEED",
    "RetardedEKFScenario",
    "RetardedEKFStudyConfig",
    "P95_METHOD",
    "default_retarded_ekf_configurations",
    "generate_retarded_ekf_scenario",
    "linear_percentile",
    "retarded_ekf_seed_provenance",
    "run_retarded_ekf_configuration",
    "run_retarded_ekf_study",
    "summarize_retarded_ekf_sequences",
]
