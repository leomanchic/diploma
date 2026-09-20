"""S7C-D1 paired stress benchmark for the accepted strict-CV C1 EKF.

The estimator is deliberately unchanged.  This module owns synthetic truth,
transport failures and outlier labels; none of those evaluator-only fields is
passed to the causal event stream or estimator.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import time
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.stats import chi2

from estimators.retarded_ekf import (
    CausalRetardedTimeEKF,
    propagate_constant_velocity_estimate,
)
from estimators.retarded_state_batch import estimate_retarded_constant_velocity_batch
from model.bearing_events import bearing_event_id
from model.bearing_statistics import tangent_basis
from model.dynamic_state import ConstantVelocityState
from model.geometry import direction_angles, tetrahedral_array
from model.measurements import BearingMeasurement
from model.retarded_bearing import predict_retarded_bearing
from model.station import StationPose
from validation.retarded_ekf_study import P95_METHOD, linear_percentile


DEFAULT_STRESS_SEED = 20260910
DEFAULT_SEQUENCE_COUNT = 100
BOOTSTRAP_RESAMPLE_COUNT = 2_000
EVALUATION_TIMES_S = tuple(np.arange(0.0, 12.5 + 0.25, 0.5)) + (14.5,)
RECEPTION_STARTS_S = (0.60, 0.75, 0.90)
RECEPTION_STEP_S = 0.80
RECEPTION_COUNT_PER_STATION = 15
NOMINAL_STD_DEG = (0.3, 0.5)
GAP_START_S = 3.0
GAP_END_S = 7.0
STATE_COVERAGE_PROBABILITY = 0.95
STATE_COVERAGE_THRESHOLD = float(chi2.ppf(STATE_COVERAGE_PROBABILITY, 6))
NIS_COVERAGE_THRESHOLD = float(chi2.ppf(STATE_COVERAGE_PROBABILITY, 2))
CAUSAL_METHODS = (
    "retarded_ekf",
    "causal_prefix_batch",
    "initial_batch_no_updates",
)
OFFLINE_METHOD = "offline_full_record_noncausal"
_GEOMETRIES = ("informative", "poorly_conditioned")
_MECHANISMS = (
    "truth",
    "nominal_bearing_noise",
    "transport_loss_uniform",
    "transport_delay_uniform",
    "outlier_mask_uniform",
    "outlier_direction",
)


def _readonly(value: ArrayLike) -> NDArray[np.float64]:
    result = np.array(value, dtype=float, copy=True)
    result.setflags(write=False)
    return result


@dataclass(frozen=True, slots=True)
class StressProfile:
    """One pre-declared transport/observation violation profile."""

    name: str
    dropout_probability: float = 0.0
    gap_scope: str = "none"
    delay_min_s: float = 0.01
    delay_max_s: float = 0.42
    outlier_probability: float = 0.0
    outlier_length_deg: float = 0.0

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("profile name is required")
        if not 0.0 <= self.dropout_probability <= 1.0:
            raise ValueError("dropout_probability must lie in [0,1]")
        if self.gap_scope not in {"none", "station_S1", "all_stations"}:
            raise ValueError("unsupported gap_scope")
        if not (
            np.isfinite(self.delay_min_s)
            and np.isfinite(self.delay_max_s)
            and 0.0 <= self.delay_min_s <= self.delay_max_s
        ):
            raise ValueError("invalid delivery delay interval")
        if not 0.0 <= self.outlier_probability <= 1.0:
            raise ValueError("outlier_probability must lie in [0,1]")
        if not np.isfinite(self.outlier_length_deg) or self.outlier_length_deg < 0.0:
            raise ValueError("outlier_length_deg must be finite and non-negative")


def default_stress_profiles() -> tuple[StressProfile, ...]:
    """Return the nine profiles frozen in :mod:`S7C_D1_PROTOCOL`."""

    return (
        StressProfile("nominal"),
        StressProfile("dropout_20", dropout_probability=0.20),
        StressProfile("dropout_50", dropout_probability=0.50),
        StressProfile("one_station_gap", gap_scope="station_S1"),
        StressProfile("all_station_gap", gap_scope="all_stations"),
        StressProfile("long_delay", delay_max_s=2.00),
        StressProfile(
            "outlier_mild", outlier_probability=0.05, outlier_length_deg=5.0
        ),
        StressProfile(
            "outlier_strong", outlier_probability=0.10, outlier_length_deg=20.0
        ),
        StressProfile(
            "mixed",
            dropout_probability=0.20,
            delay_max_s=2.00,
            outlier_probability=0.05,
            outlier_length_deg=20.0,
        ),
    )


@dataclass(frozen=True, slots=True)
class StressSeedProvenance:
    identifiers: tuple[str, ...]
    generated_seeds: tuple[int, ...]

    def as_dict(self) -> dict[str, int]:
        return dict(zip(_MECHANISMS, self.generated_seeds, strict=True))


def stress_seed_provenance(
    base_seed: int, geometry_index: int, sequence_index: int
) -> StressSeedProvenance:
    """Return collision-auditable mechanism streams for one base block."""

    if geometry_index < 0 or sequence_index < 0:
        raise ValueError("geometry_index and sequence_index must be non-negative")
    generated = tuple(
        int(
            np.random.SeedSequence(
                [int(base_seed), int(geometry_index), int(sequence_index), mechanism]
            ).generate_state(1, dtype=np.uint64)[0]
        )
        for mechanism in range(len(_MECHANISMS))
    )
    identifiers = tuple(
        f"s7cd1:{int(base_seed)}:{geometry_index}:{sequence_index}:{name}"
        for name in _MECHANISMS
    )
    return StressSeedProvenance(identifiers, generated)


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


def _event_grid() -> tuple[tuple[int, int, float], ...]:
    return tuple(
        (
            station_index,
            frame_index,
            round(start + frame_index * RECEPTION_STEP_S, 12),
        )
        for station_index, start in enumerate(RECEPTION_STARTS_S)
        for frame_index in range(RECEPTION_COUNT_PER_STATION)
    )


@dataclass(frozen=True, slots=True)
class StressBaseBlock:
    """All common random components generated before profile selection."""

    geometry: str
    geometry_index: int
    sequence_index: int
    base_seed: int
    provenance: StressSeedProvenance
    stations: tuple[StationPose, ...]
    truth_state: ConstantVelocityState
    nominal_tangent_errors_rad: NDArray[np.float64]
    loss_uniforms: NDArray[np.float64]
    delay_uniforms: NDArray[np.float64]
    outlier_mask_uniforms: NDArray[np.float64]
    outlier_directions_rad: NDArray[np.float64]


def generate_stress_base_block(
    geometry: str,
    sequence_index: int,
    *,
    base_seed: int = DEFAULT_STRESS_SEED,
) -> StressBaseBlock:
    """Generate a whole paired block without consulting any profile."""

    try:
        geometry_index = _GEOMETRIES.index(geometry)
    except ValueError as error:
        raise ValueError(f"unknown geometry: {geometry}") from error
    if sequence_index < 0:
        raise ValueError("sequence_index must be non-negative")
    provenance = stress_seed_provenance(base_seed, geometry_index, sequence_index)
    rng = {
        name: np.random.default_rng(seed)
        for name, seed in zip(_MECHANISMS, provenance.generated_seeds, strict=True)
    }
    covariance = np.diag(np.deg2rad(NOMINAL_STD_DEG) ** 2)
    truth = ConstantVelocityState(
        np.asarray([70.0, 55.0, 40.0]) + rng["truth"].normal(0.0, 2.0, 3),
        np.asarray([7.0, -3.0, 1.5]),
        0.0,
    )
    event_count = len(_event_grid())
    return StressBaseBlock(
        geometry=geometry,
        geometry_index=geometry_index,
        sequence_index=int(sequence_index),
        base_seed=int(base_seed),
        provenance=provenance,
        stations=_stations(geometry),
        truth_state=truth,
        nominal_tangent_errors_rad=_readonly(
            rng["nominal_bearing_noise"].multivariate_normal(
                np.zeros(2), covariance, size=event_count
            )
        ),
        loss_uniforms=_readonly(rng["transport_loss_uniform"].random(event_count)),
        delay_uniforms=_readonly(rng["transport_delay_uniform"].random(event_count)),
        outlier_mask_uniforms=_readonly(
            rng["outlier_mask_uniform"].random(event_count)
        ),
        outlier_directions_rad=_readonly(
            rng["outlier_direction"].uniform(0.0, 2.0 * np.pi, event_count)
        ),
    )


def spherical_exp_map_from_tangent(
    direction: ArrayLike, tangent_offset_rad: ArrayLike
) -> NDArray[np.float64]:
    """Apply one spherical exponential map in the accepted tangent basis."""

    unit = np.asarray(direction, dtype=float)
    unit = unit / np.linalg.norm(unit)
    offset = np.asarray(tangent_offset_rad, dtype=float)
    if offset.shape != (2,) or not np.all(np.isfinite(offset)):
        raise ValueError("tangent_offset_rad must be a finite two-vector")
    phi, elevation = direction_angles(unit)
    tangent_vector = tangent_basis(phi, elevation).T @ offset
    angle = float(np.linalg.norm(tangent_vector))
    if angle == 0.0:
        result = unit.copy()
    else:
        result = (
            np.cos(angle) * unit
            + np.sin(angle) * tangent_vector / angle
        )
    result /= np.linalg.norm(result)
    return _readonly(result)


@dataclass(frozen=True, slots=True)
class StressEventTruth:
    """Evaluator-only transport and contamination record."""

    event_id: str
    station_id: str
    frame_index: int
    reception_timestamp_s: float
    available_timestamp_s: float
    delivered: bool
    loss_reason: str
    is_outlier: bool
    nominal_tangent_error_rad: NDArray[np.float64]
    outlier_tangent_offset_rad: NDArray[np.float64]


@dataclass(frozen=True, slots=True)
class StressScenario:
    base_block: StressBaseBlock
    profile: StressProfile
    sequence_id: str
    events: tuple[BearingMeasurement, ...]
    event_truth: tuple[StressEventTruth, ...]


def _gap_drops(profile: StressProfile, station_id: str, reception_time: float) -> bool:
    in_gap = GAP_START_S <= reception_time <= GAP_END_S
    return bool(
        in_gap
        and (
            profile.gap_scope == "all_stations"
            or (profile.gap_scope == "station_S1" and station_id == "S1")
        )
    )


def generate_stress_scenario(
    base_block: StressBaseBlock, profile: StressProfile
) -> StressScenario:
    """Apply one profile to pre-generated common random components."""

    if not isinstance(base_block, StressBaseBlock):
        raise TypeError("base_block must be StressBaseBlock")
    if not isinstance(profile, StressProfile):
        raise TypeError("profile must be StressProfile")
    covariance = np.diag(np.deg2rad(NOMINAL_STD_DEG) ** 2)
    sequence_id = (
        f"s7cd1-{base_block.base_seed}-{base_block.geometry_index}-"
        f"{base_block.sequence_index}-{profile.name}"
    )
    events: list[BearingMeasurement] = []
    event_truth: list[StressEventTruth] = []
    for event_index, (station_index, frame_index, reception_time) in enumerate(
        _event_grid()
    ):
        station = base_block.stations[station_index]
        random_drop = bool(
            base_block.loss_uniforms[event_index] < profile.dropout_probability
        )
        gap_drop = _gap_drops(profile, station.station_id, reception_time)
        delivered = not (random_drop or gap_drop)
        reasons = []
        if random_drop:
            reasons.append("random_dropout")
        if gap_drop:
            reasons.append(profile.gap_scope)
        loss_reason = "+".join(reasons)
        delay = profile.delay_min_s + base_block.delay_uniforms[event_index] * (
            profile.delay_max_s - profile.delay_min_s
        )
        prediction = predict_retarded_bearing(
            base_block.truth_state, station, reception_time
        )
        is_outlier = bool(
            base_block.outlier_mask_uniforms[event_index]
            < profile.outlier_probability
        )
        outlier = np.zeros(2)
        if is_outlier:
            angle = base_block.outlier_directions_rad[event_index]
            outlier = np.deg2rad(profile.outlier_length_deg) * np.asarray(
                [np.cos(angle), np.sin(angle)]
            )
        measured = spherical_exp_map_from_tangent(
            prediction.direction_local,
            base_block.nominal_tangent_errors_rad[event_index] + outlier,
        )
        measurement = BearingMeasurement(
            station.station_id,
            sequence_id,
            frame_index,
            reception_time,
            reception_time + delay,
            measured,
            covariance,
            np.zeros(2),
            "direct_bearing",
            quality_metadata={"synthetic_bearing_energy": 1.0},
            tangent_frame="prediction",
        )
        identity = bearing_event_id(measurement)
        if delivered:
            events.append(measurement)
        event_truth.append(
            StressEventTruth(
                event_id=identity,
                station_id=station.station_id,
                frame_index=frame_index,
                reception_timestamp_s=reception_time,
                available_timestamp_s=reception_time + delay,
                delivered=delivered,
                loss_reason=loss_reason,
                is_outlier=is_outlier,
                nominal_tangent_error_rad=_readonly(
                    base_block.nominal_tangent_errors_rad[event_index]
                ),
                outlier_tangent_offset_rad=_readonly(outlier),
            )
        )
    return StressScenario(
        base_block=base_block,
        profile=profile,
        sequence_id=sequence_id,
        events=tuple(events),
        event_truth=tuple(event_truth),
    )


def _state_metrics(
    state: ConstantVelocityState | None,
    covariance: ArrayLike,
    truth: ConstantVelocityState,
    evaluation_time_s: float,
) -> dict[str, object]:
    matrix = np.asarray(covariance, dtype=float)
    if state is None or matrix.shape != (6, 6) or not np.all(np.isfinite(matrix)):
        return {
            "position_error_m": float("nan"),
            "velocity_error_mps": float("nan"),
            "state_nees": float("nan"),
            "state_95_covered": False,
            "covariance_rank": 0,
            "covariance_condition_number": float("inf"),
            "covariance_symmetry_error": float("nan"),
            "covariance_minimum_eigenvalue": float("nan"),
        }
    symmetric = 0.5 * (matrix + matrix.T)
    eigenvalues = np.linalg.eigvalsh(symmetric)
    error = np.concatenate(
        (
            state.position_at(evaluation_time_s) - truth.position_at(evaluation_time_s),
            state.velocity_world_mps - truth.velocity_world_mps,
        )
    )
    rank = int(np.linalg.matrix_rank(symmetric))
    positive_definite = bool(rank == 6 and np.min(eigenvalues) > 0.0)
    nees = (
        float(error @ np.linalg.solve(symmetric, error))
        if positive_definite
        else float("nan")
    )
    return {
        "position_error_m": float(np.linalg.norm(error[:3])),
        "velocity_error_mps": float(np.linalg.norm(error[3:])),
        "state_nees": nees,
        "state_95_covered": bool(
            np.isfinite(nees) and nees <= STATE_COVERAGE_THRESHOLD
        ),
        "covariance_rank": rank,
        "covariance_condition_number": (
            float(np.max(eigenvalues) / np.min(eigenvalues))
            if positive_definite
            else float("inf")
        ),
        "covariance_symmetry_error": float(
            np.max(np.abs(matrix - matrix.T), initial=0.0)
        ),
        "covariance_minimum_eigenvalue": float(np.min(eigenvalues)),
    }


def _epoch_record(
    scenario: StressScenario,
    method: str,
    evaluation_time_s: float,
    *,
    state: ConstantVelocityState | None,
    covariance: ArrayLike,
    valid: bool,
    initialized: bool,
    failure_reason: str | None,
    available_measurement_count: int,
    maximum_available_timestamp_used_s: float,
    total_runtime_s: float,
    update_runtime_s: float = 0.0,
    update_nis_values: Sequence[float] = (),
    update_attempt_count: int = 0,
    event_counts: dict[str, int] | None = None,
) -> dict[str, object]:
    record: dict[str, object] = {
        "geometry": scenario.base_block.geometry,
        "profile": scenario.profile.name,
        "sequence_index": scenario.base_block.sequence_index,
        "base_seed": scenario.base_block.base_seed,
        "method": method,
        "is_causal": method != OFFLINE_METHOD,
        "evaluation_time_s": float(evaluation_time_s),
        "valid": bool(valid),
        "initialized": bool(initialized),
        "failure_reason": failure_reason or "",
        "available_measurement_count": int(available_measurement_count),
        "maximum_available_timestamp_used_s": float(
            maximum_available_timestamp_used_s
        ),
        "total_runtime_s": float(total_runtime_s),
        "measurement_update_runtime_s": float(update_runtime_s),
        "pre_update_nis_defined_count": int(
            np.count_nonzero(np.isfinite(np.asarray(update_nis_values, dtype=float)))
        ),
        "pre_update_nis_undefined_count": int(
            update_attempt_count
            - np.count_nonzero(np.isfinite(np.asarray(update_nis_values, dtype=float)))
        ),
        "pre_update_nis_values": tuple(float(value) for value in update_nis_values),
        "independent_unit": "whole_base_sequence",
        "epochs_within_sequence_are_dependent": True,
        "profiles_within_base_block_are_paired": True,
    }
    if event_counts is not None:
        record.update({name: int(value) for name, value in event_counts.items()})
    record.update(
        _state_metrics(
            state if valid else None,
            covariance,
            scenario.base_block.truth_state,
            evaluation_time_s,
        )
    )
    return record


def _prefix_maximum_available(events: Sequence[BearingMeasurement], time_s: float) -> float:
    values = [
        event.available_timestamp_s
        for event in events
        if event.available_timestamp_s <= time_s
    ]
    return float(max(values, default=float("nan")))


def _propagate_no_update(
    state: ConstantVelocityState | None,
    covariance: NDArray[np.float64] | None,
    evaluation_time_s: float,
) -> tuple[ConstantVelocityState | None, NDArray[np.float64]]:
    if state is None or covariance is None:
        return None, np.full((6, 6), np.nan)
    propagated_state, propagated_covariance, _ = propagate_constant_velocity_estimate(
        state, covariance, evaluation_time_s
    )
    return propagated_state, np.asarray(propagated_covariance)


def _first_valid_time(records: Sequence[dict[str, object]]) -> float:
    for record in records:
        if bool(record["valid"]):
            return float(record["evaluation_time_s"])
    return float("nan")


def _record_at(
    records: Sequence[dict[str, object]], time_s: float
) -> dict[str, object] | None:
    for record in records:
        if float(record["evaluation_time_s"]) == float(time_s):
            return record
    return None


def _endpoint_fields(
    record: dict[str, object] | None, prefix: str
) -> dict[str, object]:
    if record is None:
        return {
            f"{prefix}_valid": False,
            f"{prefix}_failure_reason": "not_evaluated",
            f"{prefix}_position_error_m": float("nan"),
            f"{prefix}_velocity_error_mps": float("nan"),
            f"{prefix}_state_nees": float("nan"),
            f"{prefix}_state_95_covered": False,
            f"{prefix}_covariance_condition_number": float("inf"),
            f"{prefix}_covariance_symmetry_error": float("nan"),
            f"{prefix}_covariance_minimum_eigenvalue": float("nan"),
        }
    return {
        f"{prefix}_valid": bool(record["valid"]),
        f"{prefix}_failure_reason": str(record["failure_reason"]),
        f"{prefix}_position_error_m": float(record["position_error_m"]),
        f"{prefix}_velocity_error_mps": float(record["velocity_error_mps"]),
        f"{prefix}_state_nees": float(record["state_nees"]),
        f"{prefix}_state_95_covered": bool(record["state_95_covered"]),
        f"{prefix}_covariance_condition_number": float(
            record["covariance_condition_number"]
        ),
        f"{prefix}_covariance_symmetry_error": float(
            record["covariance_symmetry_error"]
        ),
        f"{prefix}_covariance_minimum_eigenvalue": float(
            record["covariance_minimum_eigenvalue"]
        ),
    }


def run_stress_sequence(
    scenario: StressScenario,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    """Run three matched causal methods and a separately labelled offline result."""

    processor = CausalRetardedTimeEKF(
        scenario.base_block.stations,
        scenario.events,
        estimator_variant="direct_bearing",
    )
    method_records: dict[str, list[dict[str, object]]] = defaultdict(list)
    no_update_state: ConstantVelocityState | None = None
    no_update_covariance: NDArray[np.float64] | None = None
    no_update_origin_ids: tuple[str, ...] = ()
    exact_initialization_time_s = float("nan")
    cumulative_nis: list[float] = []
    cumulative_update_attempt_count = 0
    total_ekf_runtime = 0.0
    total_ekf_update_runtime = 0.0
    total_batch_runtime = 0.0
    total_no_update_runtime = 0.0
    last_publication = None
    profile_delivered_ids = {
        item.event_id for item in scenario.event_truth if item.delivered
    }
    profile_lost_count = len(scenario.event_truth) - len(profile_delivered_ids)

    for evaluation_time in EVALUATION_TIMES_S:
        publication = processor.advance_to(evaluation_time)
        last_publication = publication
        total_ekf_runtime += publication.total_runtime_s
        total_ekf_update_runtime += publication.measurement_update_runtime_s
        nis_values = [
            diagnostic.normalized_innovation_squared
            for diagnostic in publication.update_diagnostics
        ]
        cumulative_nis.extend(nis_values)
        cumulative_update_attempt_count += len(publication.update_diagnostics)
        if (
            no_update_state is None
            and publication.initialization_batch is not None
            and publication.initialization_batch.valid
            and publication.initialization_batch.state is not None
        ):
            initialization = publication.initialization_batch
            no_update_state = initialization.state
            no_update_covariance = np.asarray(
                initialization.covariance_state_linearization
            )
            no_update_origin_ids = tuple(initialization.used_event_ids)
            initialized_lifecycle = [
                item
                for item in publication.new_lifecycle_diagnostics
                if item.action in {"initialized", "reinitialized_after_conflict"}
            ]
            exact_initialization_time_s = (
                initialized_lifecycle[0].processing_time_s
                if initialized_lifecycle
                else no_update_state.reference_time_s
            )

        maximum_available = _prefix_maximum_available(
            scenario.events, evaluation_time
        )
        available_ids = {
            bearing_event_id(measurement)
            for measurement in publication.prefix.measurements
        }
        initialization_ids_at_epoch = set(no_update_origin_ids)
        applied_ids_at_epoch = set(publication.applied_event_ids)
        rejected_ids_at_epoch = set(publication.rejected_event_ids)
        quarantined_ids_at_epoch = set(publication.prefix.conflicted_event_ids)
        ekf_partition_ids = (
            initialization_ids_at_epoch
            | applied_ids_at_epoch
            | rejected_ids_at_epoch
            | quarantined_ids_at_epoch
        )
        common_epoch_counts = {
            "potential_event_count": len(scenario.event_truth),
            "profile_delivered_event_count": len(profile_delivered_ids),
            "profile_lost_event_count": profile_lost_count,
            "causally_available_event_count": len(available_ids),
        }
        method_records["retarded_ekf"].append(
            _epoch_record(
                scenario,
                "retarded_ekf",
                evaluation_time,
                state=publication.state,
                covariance=publication.covariance_state,
                valid=publication.valid,
                initialized=publication.initialized,
                failure_reason=publication.failure_reason,
                available_measurement_count=len(publication.prefix.measurements),
                maximum_available_timestamp_used_s=maximum_available,
                total_runtime_s=publication.total_runtime_s,
                update_runtime_s=publication.measurement_update_runtime_s,
                update_nis_values=nis_values,
                update_attempt_count=len(publication.update_diagnostics),
                event_counts={
                    **common_epoch_counts,
                    "method_used_event_count": len(
                        initialization_ids_at_epoch | applied_ids_at_epoch
                    ),
                    "method_initialization_event_count": len(
                        initialization_ids_at_epoch
                    ),
                    "method_update_event_count": len(applied_ids_at_epoch),
                    "method_rejected_event_count": len(rejected_ids_at_epoch),
                    "stream_quarantined_event_count": len(
                        quarantined_ids_at_epoch
                    ),
                    "method_remaining_unprocessed_available_count": len(
                        available_ids - ekf_partition_ids
                    ),
                },
            )
        )

        batch_started = time.perf_counter()
        batch = estimate_retarded_constant_velocity_batch(
            scenario.base_block.stations,
            publication.prefix.measurements,
            reference_time_s=evaluation_time,
        )
        batch_runtime = time.perf_counter() - batch_started
        total_batch_runtime += batch_runtime
        method_records["causal_prefix_batch"].append(
            _epoch_record(
                scenario,
                "causal_prefix_batch",
                evaluation_time,
                state=batch.state,
                covariance=batch.covariance_state_linearization,
                valid=batch.valid,
                initialized=batch.valid,
                failure_reason=batch.failure_reason,
                available_measurement_count=len(publication.prefix.measurements),
                maximum_available_timestamp_used_s=maximum_available,
                total_runtime_s=batch_runtime,
                event_counts={
                    **common_epoch_counts,
                    "method_used_event_count": len(available_ids),
                    "method_initialization_event_count": 0,
                    "method_update_event_count": 0,
                    "method_rejected_event_count": 0,
                    "stream_quarantined_event_count": len(
                        quarantined_ids_at_epoch
                    ),
                    "method_remaining_unprocessed_available_count": 0,
                },
            )
        )

        no_update_started = time.perf_counter()
        baseline_state, baseline_covariance = _propagate_no_update(
            no_update_state, no_update_covariance, evaluation_time
        )
        no_update_runtime = time.perf_counter() - no_update_started
        total_no_update_runtime += no_update_runtime
        method_records["initial_batch_no_updates"].append(
            _epoch_record(
                scenario,
                "initial_batch_no_updates",
                evaluation_time,
                state=baseline_state,
                covariance=baseline_covariance,
                valid=baseline_state is not None,
                initialized=baseline_state is not None,
                failure_reason=(None if baseline_state is not None else "not_initialized"),
                available_measurement_count=len(publication.prefix.measurements),
                maximum_available_timestamp_used_s=maximum_available,
                total_runtime_s=no_update_runtime,
                event_counts={
                    **common_epoch_counts,
                    "method_used_event_count": len(initialization_ids_at_epoch),
                    "method_initialization_event_count": len(
                        initialization_ids_at_epoch
                    ),
                    "method_update_event_count": 0,
                    "method_rejected_event_count": 0,
                    "stream_quarantined_event_count": len(
                        quarantined_ids_at_epoch
                    ),
                    "method_remaining_unprocessed_available_count": len(
                        available_ids - initialization_ids_at_epoch
                    ),
                },
            )
        )

    assert last_publication is not None
    offline_started = time.perf_counter()
    offline = estimate_retarded_constant_velocity_batch(
        scenario.base_block.stations,
        scenario.events,
        reference_time_s=14.5,
    )
    offline_runtime = time.perf_counter() - offline_started
    offline_record = _epoch_record(
        scenario,
        OFFLINE_METHOD,
        14.5,
        state=offline.state,
        covariance=offline.covariance_state_linearization,
        valid=offline.valid,
        initialized=offline.valid,
        failure_reason=offline.failure_reason,
        available_measurement_count=len(scenario.events),
        maximum_available_timestamp_used_s=max(
            (event.available_timestamp_s for event in scenario.events),
            default=float("nan"),
        ),
        total_runtime_s=offline_runtime,
        event_counts={
            "potential_event_count": len(scenario.event_truth),
            "profile_delivered_event_count": len(profile_delivered_ids),
            "profile_lost_event_count": profile_lost_count,
            "causally_available_event_count": len(profile_delivered_ids),
            "method_used_event_count": len(profile_delivered_ids),
            "method_initialization_event_count": 0,
            "method_update_event_count": 0,
            "method_rejected_event_count": 0,
            "stream_quarantined_event_count": 0,
            "method_remaining_unprocessed_available_count": 0,
        },
    )
    method_records[OFFLINE_METHOD].append(offline_record)

    event_by_id = {item.event_id: item for item in scenario.event_truth}
    delivered_ids = {item.event_id for item in scenario.event_truth if item.delivered}
    # The no-update baseline captures the first accepted initialization batch.
    # A later invalid EKF publication may expose an empty current-state field,
    # so the sequence-level historical count must come from that immutable
    # origin rather than from the final publication snapshot.
    initialization_ids = set(no_update_origin_ids)
    applied_ids = set(last_publication.applied_event_ids)
    rejected_ids = set(last_publication.rejected_event_ids)
    quarantined_ids = set(last_publication.prefix.conflicted_event_ids)
    partition_ids = initialization_ids | applied_ids | rejected_ids | quarantined_ids
    loss_reasons = Counter(
        item.loss_reason for item in scenario.event_truth if not item.delivered
    )

    sequence_rows: list[dict[str, object]] = []
    for method, records in method_records.items():
        valid_records = [record for record in records if bool(record["valid"])]
        failures = Counter(
            str(record["failure_reason"])
            for record in records
            if not bool(record["valid"])
        )
        nis_values = (
            np.asarray(cumulative_nis, dtype=float)
            if method == "retarded_ekf"
            else np.empty(0)
        )
        finite_nis = nis_values[np.isfinite(nis_values)]
        if method == "retarded_ekf":
            used_ids = initialization_ids | applied_ids
            method_initialization_count = len(initialization_ids)
            method_update_count = len(applied_ids)
            method_rejected_count = len(rejected_ids)
            method_remaining_count: int | float = len(delivered_ids - partition_ids)
            method_total_runtime = total_ekf_runtime
            method_update_runtime = total_ekf_update_runtime
            initialization_time = exact_initialization_time_s
        elif method == "initial_batch_no_updates":
            used_ids = set(no_update_origin_ids)
            method_initialization_count = len(no_update_origin_ids)
            method_update_count = 0
            method_rejected_count = 0
            method_remaining_count = len(delivered_ids - used_ids)
            method_total_runtime = total_no_update_runtime
            method_update_runtime = 0.0
            initialization_time = exact_initialization_time_s
        elif method == "causal_prefix_batch":
            used_ids = delivered_ids
            method_initialization_count = 0
            method_update_count = 0
            method_rejected_count = 0
            method_remaining_count = 0
            method_total_runtime = total_batch_runtime
            method_update_runtime = 0.0
            initialization_time = _first_valid_time(records)
        else:
            used_ids = delivered_ids
            method_initialization_count = 0
            method_update_count = 0
            method_rejected_count = 0
            method_remaining_count = 0
            method_total_runtime = offline_runtime
            method_update_runtime = 0.0
            initialization_time = 14.5 if offline.valid else float("nan")

        gap_station_ids = (
            {"S1"}
            if scenario.profile.gap_scope == "station_S1"
            else ({"S0", "S1", "S2"} if scenario.profile.gap_scope == "all_stations" else set())
        )
        post_gap_candidates = [
            event_by_id[identity]
            for identity in used_ids
            if identity in event_by_id
            and event_by_id[identity].station_id in gap_station_ids
            and event_by_id[identity].reception_timestamp_s > GAP_END_S
        ]
        first_post_gap_use = min(
            (item.available_timestamp_s for item in post_gap_candidates),
            default=float("nan"),
        )
        post_gap_epoch = next(
            (
                value
                for value in EVALUATION_TIMES_S
                if np.isfinite(first_post_gap_use) and value >= first_post_gap_use
            ),
            float("nan"),
        )
        gap_end_record = _record_at(records, GAP_END_S)
        post_gap_record = (
            _record_at(records, post_gap_epoch)
            if np.isfinite(post_gap_epoch)
            else None
        )
        row: dict[str, object] = {
            "geometry": scenario.base_block.geometry,
            "profile": scenario.profile.name,
            "sequence_index": scenario.base_block.sequence_index,
            "base_seed": scenario.base_block.base_seed,
            "sequence_id": scenario.sequence_id,
            "method": method,
            "is_causal": method != OFFLINE_METHOD,
            "independent_unit": "whole_base_sequence",
            "profiles_within_base_block_are_paired": True,
            "epochs_within_sequence_are_dependent": True,
            "potential_event_count": len(scenario.event_truth),
            "delivered_event_count": len(delivered_ids),
            "lost_event_count": len(scenario.event_truth) - len(delivered_ids),
            "loss_reason_counts": json.dumps(loss_reasons, sort_keys=True),
            "potential_outlier_count": sum(item.is_outlier for item in scenario.event_truth),
            "delivered_outlier_count": sum(
                item.is_outlier and item.delivered for item in scenario.event_truth
            ),
            "method_used_event_count": len(used_ids),
            "method_initialization_event_count": method_initialization_count,
            "method_update_event_count": method_update_count,
            "method_rejected_event_count": method_rejected_count,
            "stream_quarantined_event_count": len(quarantined_ids),
            "method_remaining_unprocessed_delivered_count": method_remaining_count,
            "event_partition_applies": method == "retarded_ekf",
            "initialization_succeeded": bool(valid_records),
            "initialization_time_s": initialization_time,
            "failure_reason_counts_over_dependent_epochs": json.dumps(
                failures, sort_keys=True
            ),
            "pre_update_nis_defined_count": len(finite_nis),
            "pre_update_nis_undefined_count": (
                cumulative_update_attempt_count - len(finite_nis)
                if method == "retarded_ekf"
                else 0
            ),
            "mean_pre_update_nis": (
                float(np.mean(finite_nis)) if finite_nis.size else float("nan")
            ),
            "pre_update_nis_p95": (
                linear_percentile(finite_nis, 95)
                if finite_nis.size
                else float("nan")
            ),
            "pre_update_nis_p95_method": P95_METHOD,
            "pre_update_nis_dof": 2,
            "total_processing_runtime_s": method_total_runtime,
            "measurement_update_runtime_s": method_update_runtime,
            "no_update_origin_event_ids_json": json.dumps(no_update_origin_ids),
            "ekf_initialization_event_ids_json": json.dumps(
                sorted(initialization_ids)
            ),
            "ekf_applied_event_ids_json": json.dumps(sorted(applied_ids)),
            "gap_end_time_s": GAP_END_S if gap_station_ids else float("nan"),
            "first_post_gap_used_available_time_s": first_post_gap_use,
            "time_to_first_used_observation_after_gap_s": (
                first_post_gap_use - GAP_END_S
                if np.isfinite(first_post_gap_use)
                else float("nan")
            ),
            "post_gap_first_used_evaluation_time_s": post_gap_epoch,
            "gap_end_position_error_m": (
                float(gap_end_record["position_error_m"])
                if gap_end_record is not None
                else float("nan")
            ),
            "gap_end_velocity_error_mps": (
                float(gap_end_record["velocity_error_mps"])
                if gap_end_record is not None
                else float("nan")
            ),
            "post_gap_first_used_position_error_m": (
                float(post_gap_record["position_error_m"])
                if post_gap_record is not None
                else float("nan")
            ),
            "post_gap_first_used_velocity_error_mps": (
                float(post_gap_record["velocity_error_mps"])
                if post_gap_record is not None
                else float("nan")
            ),
            "post_gap_metric_is_accuracy_recovery_claim": False,
        }
        row.update(_endpoint_fields(_record_at(records, 12.5), "at_12_5_s"))
        row.update(_endpoint_fields(_record_at(records, 14.5), "at_14_5_s"))
        sequence_rows.append(row)

    epoch_records = [
        record
        for records in method_records.values()
        for record in records
    ]
    return epoch_records, sequence_rows


def _wilson_interval(
    successes: int, count: int, z: float = 1.959963984540054
) -> tuple[float, float]:
    if count == 0:
        return float("nan"), float("nan")
    proportion = successes / count
    denominator = 1.0 + z**2 / count
    center = (proportion + z**2 / (2.0 * count)) / denominator
    half = z * np.sqrt(
        proportion * (1.0 - proportion) / count
        + z**2 / (4.0 * count**2)
    ) / denominator
    # Floating-point round-off at the binomial endpoints can otherwise put a
    # nominal probability interval just outside its mathematical support.
    lower = 0.0 if successes == 0 else max(0.0, center - half)
    upper = 1.0 if successes == count else min(1.0, center + half)
    return float(lower), float(upper)


def summarize_independent_outcomes(
    rows: Sequence[dict[str, object]],
    *,
    valid_field: str,
    position_error_field: str,
    velocity_error_field: str,
    covered_field: str,
    failure_reason_field: str,
) -> dict[str, object]:
    """Aggregate one outcome per independent sequence without hidden removal."""

    total = len(rows)
    valid_rows = [row for row in rows if bool(row[valid_field])]
    valid_count = len(valid_rows)
    initialized_count = sum(bool(row.get("initialization_succeeded", False)) for row in rows)
    covered_count = sum(bool(row[covered_field]) for row in valid_rows)
    position = np.asarray(
        [float(row[position_error_field]) for row in valid_rows], dtype=float
    )
    velocity = np.asarray(
        [float(row[velocity_error_field]) for row in valid_rows], dtype=float
    )
    failures = Counter(
        str(row[failure_reason_field]) for row in rows if not bool(row[valid_field])
    )
    valid_ci = _wilson_interval(valid_count, total)
    initialized_ci = _wilson_interval(initialized_count, total)
    conditional_ci = _wilson_interval(covered_count, valid_count)
    unconditional_ci = _wilson_interval(covered_count, total)
    return {
        "independent_sequence_count": total,
        "valid_count": valid_count,
        "invalid_count": total - valid_count,
        "valid_fraction": valid_count / total if total else float("nan"),
        "valid_fraction_ci95_low": valid_ci[0],
        "valid_fraction_ci95_high": valid_ci[1],
        "successful_initialization_count": initialized_count,
        "successful_initialization_fraction": (
            initialized_count / total if total else float("nan")
        ),
        "successful_initialization_fraction_ci95_low": initialized_ci[0],
        "successful_initialization_fraction_ci95_high": initialized_ci[1],
        "failure_reason_counts": json.dumps(failures, sort_keys=True),
        "conditional_position_rmse_m": (
            float(np.sqrt(np.mean(position**2))) if position.size else float("nan")
        ),
        "conditional_position_p95_m": linear_percentile(position, 95),
        "conditional_velocity_rmse_mps": (
            float(np.sqrt(np.mean(velocity**2))) if velocity.size else float("nan")
        ),
        "conditional_velocity_p95_mps": linear_percentile(velocity, 95),
        "conditional_metric_valid_denominator": valid_count,
        "total_sequence_denominator": total,
        "conditional_state_95_coverage_fraction": (
            covered_count / valid_count if valid_count else float("nan")
        ),
        "conditional_state_95_coverage_ci95_low": conditional_ci[0],
        "conditional_state_95_coverage_ci95_high": conditional_ci[1],
        "unconditional_valid_and_covered_fraction": (
            covered_count / total if total else float("nan")
        ),
        "unconditional_valid_and_covered_ci95_low": unconditional_ci[0],
        "unconditional_valid_and_covered_ci95_high": unconditional_ci[1],
        "state_95_covered_count": covered_count,
        "p95_method": P95_METHOD,
        "independent_unit": "whole_base_sequence",
    }


def summarize_stress_epochs(
    epoch_records: Sequence[dict[str, object]],
) -> list[dict[str, object]]:
    """Aggregate dependent time slices separately at each fixed epoch."""

    keys = ("geometry", "profile", "method", "is_causal", "evaluation_time_s")
    grouped: dict[tuple[object, ...], list[dict[str, object]]] = defaultdict(list)
    for record in epoch_records:
        grouped[tuple(record[key] for key in keys)].append(record)
    summaries: list[dict[str, object]] = []
    for key in sorted(grouped, key=lambda item: tuple(str(value) for value in item)):
        rows = grouped[key]
        valid_rows = [row for row in rows if bool(row["valid"])]
        summary = dict(zip(keys, key, strict=True))
        for row in rows:
            row["initialization_succeeded"] = bool(row["initialized"])
        summary.update(
            summarize_independent_outcomes(
                rows,
                valid_field="valid",
                position_error_field="position_error_m",
                velocity_error_field="velocity_error_mps",
                covered_field="state_95_covered",
                failure_reason_field="failure_reason",
            )
        )
        finite_nees = np.asarray(
            [float(row["state_nees"]) for row in valid_rows], dtype=float
        )
        finite_nees = finite_nees[np.isfinite(finite_nees)]
        nis = np.asarray(
            [
                value
                for row in rows
                for value in row["pre_update_nis_values"]
                if np.isfinite(value)
            ],
            dtype=float,
        )
        undefined_nis = sum(int(row["pre_update_nis_undefined_count"]) for row in rows)
        conditions = np.asarray(
            [float(row["covariance_condition_number"]) for row in valid_rows],
            dtype=float,
        )
        summary.update(
            {
                "mean_state_nees": (
                    float(np.mean(finite_nees)) if finite_nees.size else float("nan")
                ),
                "state_nees_p95": (
                    linear_percentile(finite_nees, 95)
                    if finite_nees.size
                    else float("nan")
                ),
                "state_nees_dof": 6,
                "pre_update_nis_defined_count": len(nis),
                "pre_update_nis_undefined_count": undefined_nis,
                "mean_pre_update_nis": (
                    float(np.mean(nis)) if nis.size else float("nan")
                ),
                "pre_update_nis_p95": (
                    linear_percentile(nis, 95) if nis.size else float("nan")
                ),
                "pre_update_nis_95_coverage_fraction": (
                    float(np.mean(nis <= NIS_COVERAGE_THRESHOLD))
                    if nis.size
                    else float("nan")
                ),
                "pre_update_nis_dof": 2,
                "nis_samples_within_sequence_are_dependent": True,
                "mean_total_runtime_s": float(
                    np.mean([float(row["total_runtime_s"]) for row in rows])
                ),
                "mean_measurement_update_runtime_s": float(
                    np.mean(
                        [float(row["measurement_update_runtime_s"]) for row in rows]
                    )
                ),
                "maximum_covariance_symmetry_error": (
                    float(
                        np.max(
                            [float(row["covariance_symmetry_error"]) for row in valid_rows]
                        )
                    )
                    if valid_rows
                    else float("nan")
                ),
                "minimum_covariance_eigenvalue": (
                    float(
                        np.min(
                            [
                                float(row["covariance_minimum_eigenvalue"])
                                for row in valid_rows
                            ]
                        )
                    )
                    if valid_rows
                    else float("nan")
                ),
                "median_covariance_condition_number": (
                    float(np.median(conditions))
                    if conditions.size
                    else float("nan")
                ),
                "maximum_covariance_condition_number": (
                    float(np.max(conditions))
                    if conditions.size
                    else float("nan")
                ),
                "maximum_available_timestamp_used_s": float(
                    np.nanmax(
                        [float(row["maximum_available_timestamp_used_s"]) for row in rows]
                    )
                )
                if any(
                    np.isfinite(float(row["maximum_available_timestamp_used_s"]))
                    for row in rows
                )
                else float("nan"),
                "epochs_within_sequence_are_dependent": True,
                "profiles_within_base_block_are_paired": True,
                "observation_model_is_gaussian": key[1]
                not in {"outlier_mild", "outlier_strong", "mixed"},
            }
        )
        for count_name in (
            "potential_event_count",
            "profile_delivered_event_count",
            "profile_lost_event_count",
            "causally_available_event_count",
            "method_used_event_count",
            "method_initialization_event_count",
            "method_update_event_count",
            "method_rejected_event_count",
            "stream_quarantined_event_count",
            "method_remaining_unprocessed_available_count",
        ):
            values = np.asarray([int(row[count_name]) for row in rows], dtype=float)
            summary[f"mean_{count_name}"] = float(np.mean(values))
            summary[f"minimum_{count_name}"] = int(np.min(values))
            summary[f"maximum_{count_name}"] = int(np.max(values))
        summaries.append(summary)
    return summaries


def _paired_bootstrap_difference(
    profile_values: NDArray[np.float64],
    nominal_values: NDArray[np.float64],
    *,
    seed_coordinates: Sequence[int],
) -> tuple[float, float, float]:
    if profile_values.shape != nominal_values.shape or profile_values.ndim != 1:
        raise ValueError("paired bootstrap inputs must be equal one-dimensional arrays")
    difference = profile_values - nominal_values
    estimate = float(np.mean(difference))
    if difference.size == 0:
        return estimate, float("nan"), float("nan")
    rng = np.random.default_rng(
        np.random.SeedSequence([DEFAULT_STRESS_SEED, *map(int, seed_coordinates)])
    )
    indices = rng.integers(
        0, difference.size, size=(BOOTSTRAP_RESAMPLE_COUNT, difference.size)
    )
    bootstrap = np.mean(difference[indices], axis=1)
    low, high = np.percentile(bootstrap, [2.5, 97.5], method="linear")
    return estimate, float(low), float(high)


def summarize_stress_profiles(
    sequence_rows: Sequence[dict[str, object]],
) -> list[dict[str, object]]:
    """Summarize 12.5/14.5-s outcomes with paired profile comparisons."""

    grouped: dict[tuple[str, str, str], list[dict[str, object]]] = defaultdict(list)
    for row in sequence_rows:
        grouped[(str(row["geometry"]), str(row["profile"]), str(row["method"]))].append(row)
    profiles = default_stress_profiles()
    profile_index = {profile.name: index for index, profile in enumerate(profiles)}
    method_index = {
        method: index for index, method in enumerate((*CAUSAL_METHODS, OFFLINE_METHOD))
    }
    summaries: list[dict[str, object]] = []
    for (geometry, profile, method), rows in sorted(grouped.items()):
        nominal_rows = grouped.get((geometry, "nominal", method), [])
        nominal_by_sequence = {
            int(row["sequence_index"]): row for row in nominal_rows
        }
        ordered_rows = sorted(rows, key=lambda row: int(row["sequence_index"]))
        geometry_index = _GEOMETRIES.index(geometry)
        for endpoint_index, (endpoint, prefix) in enumerate(
            ((12.5, "at_12_5_s"), (14.5, "at_14_5_s"))
        ):
            if method == OFFLINE_METHOD and endpoint == 12.5:
                continue
            aggregate = summarize_independent_outcomes(
                ordered_rows,
                valid_field=f"{prefix}_valid",
                position_error_field=f"{prefix}_position_error_m",
                velocity_error_field=f"{prefix}_velocity_error_mps",
                covered_field=f"{prefix}_state_95_covered",
                failure_reason_field=f"{prefix}_failure_reason",
            )
            profile_valid = np.asarray(
                [bool(row[f"{prefix}_valid"]) for row in ordered_rows], dtype=float
            )
            profile_covered = np.asarray(
                [
                    bool(row[f"{prefix}_valid"])
                    and bool(row[f"{prefix}_state_95_covered"])
                    for row in ordered_rows
                ],
                dtype=float,
            )
            matching_nominal = [
                nominal_by_sequence[int(row["sequence_index"])] for row in ordered_rows
            ]
            nominal_valid = np.asarray(
                [bool(row[f"{prefix}_valid"]) for row in matching_nominal], dtype=float
            )
            nominal_covered = np.asarray(
                [
                    bool(row[f"{prefix}_valid"])
                    and bool(row[f"{prefix}_state_95_covered"])
                    for row in matching_nominal
                ],
                dtype=float,
            )
            valid_difference = _paired_bootstrap_difference(
                profile_valid,
                nominal_valid,
                seed_coordinates=(
                    geometry_index,
                    profile_index[profile],
                    method_index[method],
                    endpoint_index,
                    0,
                ),
            )
            coverage_difference = _paired_bootstrap_difference(
                profile_covered,
                nominal_covered,
                seed_coordinates=(
                    geometry_index,
                    profile_index[profile],
                    method_index[method],
                    endpoint_index,
                    1,
                ),
            )
            finite_nis = np.asarray(
                [float(row["mean_pre_update_nis"]) for row in ordered_rows],
                dtype=float,
            )
            finite_nis = finite_nis[np.isfinite(finite_nis)]
            result = {
                "geometry": geometry,
                "profile": profile,
                "method": method,
                "is_causal": method != OFFLINE_METHOD,
                "evaluation_time_s": endpoint,
                **aggregate,
                "mean_sequence_pre_update_nis": (
                    float(np.mean(finite_nis)) if finite_nis.size else float("nan")
                ),
                "sequence_pre_update_nis_count": len(finite_nis),
                "pre_update_nis_dof": 2,
                "paired_valid_fraction_difference_vs_nominal": valid_difference[0],
                "paired_valid_fraction_difference_ci95_low": valid_difference[1],
                "paired_valid_fraction_difference_ci95_high": valid_difference[2],
                "paired_unconditional_coverage_difference_vs_nominal": coverage_difference[0],
                "paired_unconditional_coverage_difference_ci95_low": coverage_difference[1],
                "paired_unconditional_coverage_difference_ci95_high": coverage_difference[2],
                "paired_bootstrap_resample_count": BOOTSTRAP_RESAMPLE_COUNT,
                "paired_bootstrap_unit": "whole_base_sequence",
                "profiles_within_base_block_are_paired": True,
                "observation_model_is_gaussian": profile
                not in {"outlier_mild", "outlier_strong", "mixed"},
            }
            summaries.append(result)
    return summaries


def _seed_row(block: StressBaseBlock) -> dict[str, object]:
    row: dict[str, object] = {
        "geometry": block.geometry,
        "geometry_index": block.geometry_index,
        "sequence_index": block.sequence_index,
        "base_seed": block.base_seed,
        "independent_unit": "whole_base_sequence",
        "profile_run_count_per_block": len(default_stress_profiles()),
        "profiles_are_paired": True,
        "seed_scheme": "SeedSequence([base_seed,geometry_index,sequence_index,mechanism_id])",
    }
    for name, identifier, seed in zip(
        _MECHANISMS,
        block.provenance.identifiers,
        block.provenance.generated_seeds,
        strict=True,
    ):
        row[f"{name}_provenance"] = identifier
        row[f"{name}_seed"] = seed
    return row


def _write_csv(path: Path, rows: Sequence[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError(f"cannot write empty CSV: {path}")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def audit_stress_results(
    epoch_summaries: Sequence[dict[str, object]],
    sequence_rows: Sequence[dict[str, object]],
    profile_summaries: Sequence[dict[str, object]],
    seed_rows: Sequence[dict[str, object]],
    *,
    sequence_count: int,
) -> dict[str, int]:
    """Raise on schema/grain/causality inconsistencies in saved D1 results."""

    expected_epoch_rows = len(_GEOMETRIES) * len(default_stress_profiles()) * (
        len(CAUSAL_METHODS) * len(EVALUATION_TIMES_S) + 1
    )
    expected_sequence_rows = (
        len(_GEOMETRIES)
        * len(default_stress_profiles())
        * (len(CAUSAL_METHODS) + 1)
        * sequence_count
    )
    expected_profile_rows = len(_GEOMETRIES) * len(default_stress_profiles()) * (
        len(CAUSAL_METHODS) * 2 + 1
    )
    expected_seed_rows = len(_GEOMETRIES) * sequence_count
    observed = (
        len(epoch_summaries),
        len(sequence_rows),
        len(profile_summaries),
        len(seed_rows),
    )
    expected = (
        expected_epoch_rows,
        expected_sequence_rows,
        expected_profile_rows,
        expected_seed_rows,
    )
    if observed != expected:
        raise AssertionError(f"D1 row counts {observed} != {expected}")
    provenance_values: list[str] = []
    generated_seeds: list[int] = []
    for row in seed_rows:
        for mechanism in _MECHANISMS:
            provenance_values.append(str(row[f"{mechanism}_provenance"]))
            generated_seeds.append(int(row[f"{mechanism}_seed"]))
    if len(set(provenance_values)) != len(provenance_values):
        raise AssertionError("duplicate D1 seed provenance identifier")
    if len(set(generated_seeds)) != len(generated_seeds):
        raise AssertionError("duplicate D1 generated mechanism seed")
    for row in sequence_rows:
        potential = int(row["potential_event_count"])
        delivered = int(row["delivered_event_count"])
        lost = int(row["lost_event_count"])
        if potential != delivered + lost:
            raise AssertionError("potential events do not partition into delivered+lost")
        if row["method"] == "retarded_ekf":
            partition = sum(
                int(row[name])
                for name in (
                    "method_initialization_event_count",
                    "method_update_event_count",
                    "method_rejected_event_count",
                    "stream_quarantined_event_count",
                    "method_remaining_unprocessed_delivered_count",
                )
            )
            if partition != delivered:
                raise AssertionError("drained EKF delivered-event partition is incomplete")
        if row["method"] == "initial_batch_no_updates":
            no_update_ids = sorted(
                json.loads(str(row["no_update_origin_event_ids_json"]))
            )
            ekf_initialization_ids = sorted(
                json.loads(str(row["ekf_initialization_event_ids_json"]))
            )
            if no_update_ids != ekf_initialization_ids:
                raise AssertionError("no-update origin differs from EKF initialization batch")
    for row in epoch_summaries:
        maximum_used = float(row["maximum_available_timestamp_used_s"])
        evaluation_time = float(row["evaluation_time_s"])
        if np.isfinite(maximum_used) and maximum_used > evaluation_time + 2e-15:
            raise AssertionError("future event entered a causal epoch summary")
        if str(row["p95_method"]) != P95_METHOD:
            raise AssertionError("epoch P95 method is not explicit linear interpolation")
        potential = float(row["mean_potential_event_count"])
        delivered = float(row["mean_profile_delivered_event_count"])
        lost = float(row["mean_profile_lost_event_count"])
        if not np.isclose(potential, delivered + lost, rtol=0.0, atol=1e-12):
            raise AssertionError("epoch profile event counts do not partition")
        available = float(row["mean_causally_available_event_count"])
        method = str(row["method"])
        if method == "retarded_ekf":
            partition = sum(
                float(row[f"mean_{name}"])
                for name in (
                    "method_initialization_event_count",
                    "method_update_event_count",
                    "method_rejected_event_count",
                    "stream_quarantined_event_count",
                    "method_remaining_unprocessed_available_count",
                )
            )
        elif method == "initial_batch_no_updates":
            partition = float(row["mean_method_used_event_count"]) + float(
                row["mean_method_remaining_unprocessed_available_count"]
            )
        else:
            partition = float(row["mean_method_used_event_count"]) + float(
                row["mean_stream_quarantined_event_count"]
            )
        if not np.isclose(available, partition, rtol=0.0, atol=1e-12):
            raise AssertionError("epoch available-event partition is incomplete")
    for row in profile_summaries:
        if str(row["p95_method"]) != P95_METHOD:
            raise AssertionError("profile P95 method is not explicit linear interpolation")
    return {
        "epoch_row_count": len(epoch_summaries),
        "sequence_row_count": len(sequence_rows),
        "profile_row_count": len(profile_summaries),
        "seed_row_count": len(seed_rows),
        "unique_mechanism_seed_count": len(set(generated_seeds)),
    }


def _run_base_block_task(
    arguments: tuple[str, int, int]
) -> tuple[list[dict[str, object]], list[dict[str, object]], dict[str, object]]:
    geometry, sequence_index, base_seed = arguments
    block = generate_stress_base_block(
        geometry, sequence_index, base_seed=base_seed
    )
    epoch_records: list[dict[str, object]] = []
    sequence_rows: list[dict[str, object]] = []
    for profile in default_stress_profiles():
        scenario = generate_stress_scenario(block, profile)
        current_epoch, current_sequences = run_stress_sequence(scenario)
        epoch_records.extend(current_epoch)
        sequence_rows.extend(current_sequences)
    return epoch_records, sequence_rows, _seed_row(block)


def run_retarded_ekf_stress_study(
    output_directory: str | Path = "results",
    *,
    sequence_count: int = DEFAULT_SEQUENCE_COUNT,
    base_seed: int = DEFAULT_STRESS_SEED,
    workers: int = 1,
    progress: bool = False,
) -> tuple[
    list[dict[str, object]],
    list[dict[str, object]],
    list[dict[str, object]],
    list[dict[str, object]],
]:
    """Run the frozen paired D1 study and write four reproducible CSV files."""

    if sequence_count <= 0:
        raise ValueError("sequence_count must be positive")
    if workers <= 0:
        raise ValueError("workers must be positive")
    epoch_records: list[dict[str, object]] = []
    sequence_rows: list[dict[str, object]] = []
    seed_rows: list[dict[str, object]] = []
    tasks = [
        (geometry, sequence_index, int(base_seed))
        for geometry in _GEOMETRIES
        for sequence_index in range(sequence_count)
    ]
    if workers == 1:
        iterator = map(_run_base_block_task, tasks)
        executor = None
    else:
        executor = ProcessPoolExecutor(max_workers=workers)
        iterator = executor.map(_run_base_block_task, tasks, chunksize=1)
    try:
        for completed, (current_epoch, current_sequences, seed_row) in enumerate(
            iterator, start=1
        ):
            epoch_records.extend(current_epoch)
            sequence_rows.extend(current_sequences)
            seed_rows.append(seed_row)
            if progress:
                print(
                    f"S7C-D1 completed base blocks: {completed}/{len(tasks)}",
                    flush=True,
                )
    finally:
        if executor is not None:
            executor.shutdown(wait=True, cancel_futures=False)
    epoch_summaries = summarize_stress_epochs(epoch_records)
    profile_summaries = summarize_stress_profiles(sequence_rows)
    audit_stress_results(
        epoch_summaries,
        sequence_rows,
        profile_summaries,
        seed_rows,
        sequence_count=sequence_count,
    )
    output = Path(output_directory)
    _write_csv(output / "retarded_ekf_stress_epoch_summary.csv", epoch_summaries)
    _write_csv(output / "retarded_ekf_stress_sequence_results.csv", sequence_rows)
    _write_csv(output / "retarded_ekf_stress_profile_summary.csv", profile_summaries)
    _write_csv(output / "retarded_ekf_stress_seed_provenance.csv", seed_rows)
    return epoch_summaries, sequence_rows, profile_summaries, seed_rows


def _main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-directory", default="results")
    parser.add_argument("--sequence-count", type=int, default=DEFAULT_SEQUENCE_COUNT)
    parser.add_argument("--base-seed", type=int, default=DEFAULT_STRESS_SEED)
    parser.add_argument(
        "--workers", type=int, default=max(1, min(os.cpu_count() or 1, 8))
    )
    arguments = parser.parse_args()
    run_retarded_ekf_stress_study(
        arguments.output_directory,
        sequence_count=arguments.sequence_count,
        base_seed=arguments.base_seed,
        workers=arguments.workers,
        progress=True,
    )


if __name__ == "__main__":
    _main()


__all__ = [
    "BOOTSTRAP_RESAMPLE_COUNT",
    "CAUSAL_METHODS",
    "DEFAULT_SEQUENCE_COUNT",
    "DEFAULT_STRESS_SEED",
    "EVALUATION_TIMES_S",
    "GAP_END_S",
    "GAP_START_S",
    "NOMINAL_STD_DEG",
    "OFFLINE_METHOD",
    "RECEPTION_COUNT_PER_STATION",
    "RECEPTION_STARTS_S",
    "RECEPTION_STEP_S",
    "StressBaseBlock",
    "StressEventTruth",
    "StressProfile",
    "StressScenario",
    "StressSeedProvenance",
    "audit_stress_results",
    "default_stress_profiles",
    "generate_stress_base_block",
    "generate_stress_scenario",
    "run_retarded_ekf_stress_study",
    "run_stress_sequence",
    "spherical_exp_map_from_tangent",
    "stress_seed_provenance",
    "summarize_independent_outcomes",
    "summarize_stress_epochs",
    "summarize_stress_profiles",
]
