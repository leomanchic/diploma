"""Causal confirmation and bounded recovery for the strict-CV bearing EKF.

This opt-in estimator preserves the C1/D2 model: exact retarded-time bearings,
known constant sound speed, strict constant velocity and ``Q=0``.  It adds a
truth-free lifecycle around the accepted EKF algebra.  A six-event hypothesis
is tentative until distinct later events confirm it.  Sustained contradictory
measurements invalidate the posterior and start a fresh causal generation;
the old posterior is never combined with the recovery batch.
"""

from __future__ import annotations

import hashlib
import time
from collections.abc import Sequence
from dataclasses import dataclass
from itertools import combinations

import numpy as np
from numpy.typing import ArrayLike, NDArray

from estimators.retarded_ekf import (
    RetardedEKFInitializationCriteria,
    RetardedEKFUpdateResult,
    propagate_constant_velocity_estimate,
    update_retarded_ekf,
)
from estimators.retarded_state_batch import (
    RetardedBatchResult,
    estimate_retarded_constant_velocity_batch,
    geometric_constant_velocity_initial_state,
)
from model.bearing_events import (
    BearingEventPrefix,
    CausalBearingEventStream,
    ScheduledBearingEvent,
    bearing_event_id,
)
from model.bearing_statistics import AntipodalDirectionError
from model.dynamic_state import ConstantVelocityState
from model.geometry import DEFAULT_SOUND_SPEED
from model.measurements import BearingMeasurement
from model.retarded_bearing import retarded_bearing_residual
from model.retarded_bearing import retarded_bearing_residual_jacobian
from model.station import StationPose


def _readonly(value: ArrayLike) -> NDArray[np.float64]:
    result = np.array(value, dtype=float, copy=True)
    result.setflags(write=False)
    return result


def _positive_definite(value: ArrayLike, shape: tuple[int, int], *, name: str) -> NDArray[np.float64]:
    matrix = np.asarray(value, dtype=float)
    if matrix.shape != shape or not np.all(np.isfinite(matrix)):
        raise ValueError(f"{name} must be a finite matrix with shape {shape}")
    if not np.allclose(matrix, matrix.T, rtol=0.0, atol=1e-14):
        raise ValueError(f"{name} must be symmetric")
    matrix = 0.5 * (matrix + matrix.T)
    try:
        np.linalg.cholesky(matrix)
    except np.linalg.LinAlgError as error:
        raise ValueError(f"{name} must be positive definite") from error
    return matrix


@dataclass(frozen=True, slots=True)
class InitializationRecoveryConfig:
    """Frozen observable-only rules for confirmation and recovery."""

    pre_update_nis_threshold: float = 9.210340371976184
    fit_nis_threshold: float = 10.596634733096073
    construction_measurement_count: int = 6
    confirmation_measurement_count: int = 3
    confirmation_station_count: int = 2
    confirmation_reception_span_s: float = 0.5
    maximum_confirmation_failures: int = 2
    maximum_confirmation_events: int = 8
    maximum_confirmation_age_s: float = 3.0
    maximum_hypotheses_per_generation: int = 8
    maximum_candidate_count: int = 16
    maximum_initialization_buffer_events: int = 18
    inconsistency_rejection_count: int = 4
    inconsistency_station_count: int = 2
    inconsistency_reception_span_s: float = 0.5

    def __post_init__(self) -> None:
        positive = (
            (self.pre_update_nis_threshold, "pre_update_nis_threshold"),
            (self.fit_nis_threshold, "fit_nis_threshold"),
            (self.confirmation_reception_span_s, "confirmation_reception_span_s"),
            (self.maximum_confirmation_age_s, "maximum_confirmation_age_s"),
            (self.inconsistency_reception_span_s, "inconsistency_reception_span_s"),
        )
        for value, name in positive:
            if not np.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive")
        integer_positive = (
            (self.construction_measurement_count, "construction_measurement_count"),
            (self.confirmation_measurement_count, "confirmation_measurement_count"),
            (self.confirmation_station_count, "confirmation_station_count"),
            (self.maximum_confirmation_failures, "maximum_confirmation_failures"),
            (self.maximum_confirmation_events, "maximum_confirmation_events"),
            (self.maximum_hypotheses_per_generation, "maximum_hypotheses_per_generation"),
            (self.maximum_candidate_count, "maximum_candidate_count"),
            (self.maximum_initialization_buffer_events, "maximum_initialization_buffer_events"),
            (self.inconsistency_rejection_count, "inconsistency_rejection_count"),
            (self.inconsistency_station_count, "inconsistency_station_count"),
        )
        for value, name in integer_positive:
            if value <= 0:
                raise ValueError(f"{name} must be positive")
        if self.maximum_initialization_buffer_events < (
            self.construction_measurement_count + self.confirmation_measurement_count
        ):
            raise ValueError("initialization buffer is too short for confirmation")


@dataclass(frozen=True, slots=True)
class RecoveryHypothesisDiagnostic:
    processing_time_s: float
    generation: int
    hypothesis_index: int
    action: str
    reason: str
    construction_event_ids: tuple[str, ...]
    confirmation_event_ids: tuple[str, ...]
    excluded_event_ids: tuple[str, ...]
    preliminary_nis_values: tuple[tuple[str, float], ...]
    confirmation_nis_values: tuple[tuple[str, float], ...]
    final_nis_values: tuple[tuple[str, float], ...]
    local_observability_rank: int
    scaled_condition_number: float
    optimizer_valid: bool


@dataclass(frozen=True, slots=True)
class RecoveryLifecycleDiagnostic:
    processing_time_s: float
    generation: int
    action: str
    reason: str
    event_ids: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class RecoveryEventUse:
    event_id: str
    generation: int
    role: str


@dataclass(frozen=True, slots=True)
class RecoveryPublication:
    processing_time_s: float
    status: str
    initialized: bool
    confirmed: bool
    valid: bool
    failure_reason: str | None
    state: ConstantVelocityState | None
    covariance_state: NDArray[np.float64]
    tentative_state: ConstantVelocityState | None
    tentative_covariance_state: NDArray[np.float64]
    generation: int
    prefix: BearingEventPrefix
    initialization_event_ids: tuple[str, ...]
    applied_event_ids: tuple[str, ...]
    rejected_event_ids: tuple[str, ...]
    rejection_reasons: tuple[tuple[str, str], ...]
    update_diagnostics: tuple[RetardedEKFUpdateResult, ...]
    hypothesis_diagnostics: tuple[RecoveryHypothesisDiagnostic, ...]
    new_hypothesis_diagnostics: tuple[RecoveryHypothesisDiagnostic, ...]
    lifecycle_diagnostics: tuple[RecoveryLifecycleDiagnostic, ...]
    new_lifecycle_diagnostics: tuple[RecoveryLifecycleDiagnostic, ...]
    event_uses: tuple[RecoveryEventUse, ...]
    reset_count: int
    first_confirmation_time_s: float
    last_recovery_duration_s: float
    total_runtime_s: float


@dataclass(slots=True)
class _TentativeHypothesis:
    batch: RetardedBatchResult
    creation_time_s: float
    generation: int
    index: int
    construction_ids: tuple[str, ...]
    creation_available_ids: frozenset[str]
    preliminary_scores: tuple[tuple[str, float], ...]


def _batch_passes(batch: RetardedBatchResult, criteria: RetardedEKFInitializationCriteria) -> bool:
    return bool(
        batch.valid
        and batch.state is not None
        and batch.local_observability_rank == criteria.required_local_rank
        and batch.scaled_information_condition_number <= criteria.maximum_scaled_condition_number
        and batch.maximum_angular_residual_rad <= criteria.maximum_angular_residual_rad
        and batch.scaled_projected_kkt_residual <= criteria.maximum_scaled_kkt_residual
    )


def _fit_nis(
    state: ConstantVelocityState,
    station_map: dict[str, StationPose],
    measurement: BearingMeasurement,
    sound_speed: float,
) -> float:
    try:
        residual = retarded_bearing_residual(
            state, station_map[measurement.station_id], measurement, sound_speed
        )
        covariance = _positive_definite(
            measurement.covariance_tangent_rad2,
            (2, 2),
            name="observation_covariance",
        )
        return float(residual @ np.linalg.solve(covariance, residual))
    except (ValueError, AntipodalDirectionError, np.linalg.LinAlgError):
        return float("inf")


def _scores(
    state: ConstantVelocityState,
    station_map: dict[str, StationPose],
    measurements: Sequence[BearingMeasurement],
    sound_speed: float,
) -> tuple[tuple[str, float], ...]:
    return tuple(
        (bearing_event_id(item), _fit_nis(state, station_map, item, sound_speed))
        for item in measurements
    )


def _predictive_scores(
    batch: RetardedBatchResult,
    station_map: dict[str, StationPose],
    measurements: Sequence[BearingMeasurement],
    sound_speed: float,
) -> tuple[tuple[str, float], ...]:
    if batch.state is None:
        return tuple((bearing_event_id(item), float("inf")) for item in measurements)
    covariance = _positive_definite(
        batch.covariance_state_linearization,
        (6, 6),
        name="hypothesis_covariance",
    )
    result = []
    for item in measurements:
        identity = bearing_event_id(item)
        try:
            residual = retarded_bearing_residual(
                batch.state, station_map[item.station_id], item, sound_speed
            )
            jacobian = retarded_bearing_residual_jacobian(
                batch.state, station_map[item.station_id], item, sound_speed
            )
            innovation = (
                jacobian @ covariance @ jacobian.T
                + np.asarray(item.covariance_tangent_rad2, dtype=float)
            )
            value = float(residual @ np.linalg.solve(innovation, residual))
        except (ValueError, AntipodalDirectionError, np.linalg.LinAlgError):
            value = float("inf")
        result.append((identity, value))
    return tuple(result)


def _construction_candidates(
    measurements: Sequence[BearingMeasurement],
    count: int,
    maximum_count: int,
) -> tuple[tuple[BearingMeasurement, ...], ...]:
    ordered = tuple(
        sorted(
            measurements,
            key=lambda item: (
                item.available_timestamp_s,
                bearing_event_id(item),
            ),
        )
    )
    if len(ordered) < count:
        return ()
    all_combinations = combinations(range(len(ordered)), count)
    if len(ordered) <= count + 2:
        indices = list(all_combinations)[:maximum_count]
    else:
        first = tuple(range(count))
        digest = hashlib.sha256(
            "\n".join(bearing_event_id(item) for item in ordered).encode("utf-8")
        ).digest()
        rng = np.random.default_rng(int.from_bytes(digest[:8], "little"))
        index_set = {first}
        attempts = 0
        while len(index_set) < maximum_count and attempts < 20 * maximum_count:
            candidate = tuple(sorted(rng.choice(len(ordered), count, replace=False)))
            index_set.add(candidate)
            attempts += 1
        indices = [first, *sorted(index_set - {first})]
    return tuple(tuple(ordered[index] for index in choice) for choice in indices)


def _refine_consensus(
    stations: Sequence[StationPose],
    station_map: dict[str, StationPose],
    measurements: Sequence[BearingMeasurement],
    *,
    reference_time_s: float,
    sound_speed: float,
    threshold: float,
    criteria: RetardedEKFInitializationCriteria,
) -> tuple[RetardedBatchResult | None, tuple[tuple[str, float], ...], tuple[BearingMeasurement, ...]]:
    current = tuple(measurements)
    last_ids: tuple[str, ...] | None = None
    for _ in range(6):
        if len(current) < criteria.minimum_measurement_count:
            return None, (), ()
        batch = estimate_retarded_constant_velocity_batch(
            stations,
            current,
            reference_time_s=reference_time_s,
            sound_speed=sound_speed,
        )
        if not _batch_passes(batch, criteria) or batch.state is None:
            return None, (), ()
        final_scores = _scores(batch.state, station_map, measurements, sound_speed)
        inlier_ids = tuple(
            identity
            for identity, value in final_scores
            if np.isfinite(value) and value <= threshold
        )
        if inlier_ids == last_ids or set(inlier_ids) == {
            bearing_event_id(item) for item in current
        }:
            selected = tuple(
                item for item in measurements if bearing_event_id(item) in set(inlier_ids)
            )
            return batch, final_scores, selected
        last_ids = inlier_ids
        selected_ids = set(inlier_ids)
        current = tuple(
            item for item in measurements if bearing_event_id(item) in selected_ids
        )
    return None, (), ()


class CausalConfirmedRetardedTimeEKF:
    """Strict-CV EKF with tentative initialization and bounded causal recovery."""

    def __init__(
        self,
        stations: Sequence[StationPose],
        events: Sequence[ScheduledBearingEvent | BearingMeasurement],
        *,
        estimator_variant: str,
        sound_speed: float = DEFAULT_SOUND_SPEED,
        initialization_criteria: RetardedEKFInitializationCriteria | None = None,
        recovery_config: InitializationRecoveryConfig | None = None,
    ) -> None:
        self._stations = tuple(stations)
        self._station_map = {item.station_id: item for item in self._stations}
        if not self._station_map or len(self._station_map) != len(self._stations):
            raise ValueError("station IDs must be non-empty and unique")
        self._stream = CausalBearingEventStream(
            events, estimator_variant=estimator_variant
        )
        self._sound_speed = float(sound_speed)
        if not np.isfinite(self._sound_speed) or self._sound_speed <= 0.0:
            raise ValueError("sound_speed must be finite and positive")
        self._criteria = initialization_criteria or RetardedEKFInitializationCriteria()
        self._config = recovery_config or InitializationRecoveryConfig()
        if self._criteria.minimum_measurement_count != self._config.construction_measurement_count:
            raise ValueError("construction count must match initialization criteria")
        self._state: ConstantVelocityState | None = None
        self._covariance: NDArray[np.float64] | None = None
        self._tentative: _TentativeHypothesis | None = None
        self._status = "uninitialized"
        self._failure_reason: str | None = "not_initialized"
        self._generation = 0
        self._generation_start_time = float("-inf")
        self._hypothesis_attempt_count = 0
        self._failed_signatures: set[tuple[str, ...]] = set()
        self._processed_ids: set[str] = set()
        self._initialization_ids: set[str] = set()
        self._applied_ids: set[str] = set()
        self._rejected_ids: set[str] = set()
        self._rejection_reasons: dict[str, str] = {}
        self._event_uses: list[RecoveryEventUse] = []
        self._hypothesis_history: list[RecoveryHypothesisDiagnostic] = []
        self._lifecycle_history: list[RecoveryLifecycleDiagnostic] = []
        self._inconsistency_streak: list[BearingMeasurement] = []
        self._reset_count = 0
        self._first_confirmation_time = float("nan")
        self._recovery_started_time = float("nan")
        self._last_recovery_duration = float("nan")
        self._last_processing_time = float("-inf")
        self._publications: list[RecoveryPublication] = []

    @property
    def publications(self) -> tuple[RecoveryPublication, ...]:
        return tuple(self._publications)

    def _lifecycle(self, time_s: float, action: str, reason: str, ids: Sequence[str] = ()) -> RecoveryLifecycleDiagnostic:
        item = RecoveryLifecycleDiagnostic(
            float(time_s), self._generation, action, reason, tuple(sorted(ids))
        )
        self._lifecycle_history.append(item)
        return item

    def _eligible(self, prefix: BearingEventPrefix) -> tuple[BearingMeasurement, ...]:
        result = []
        for measurement in prefix.measurements:
            identity = bearing_event_id(measurement)
            if identity in self._processed_ids or identity in self._rejected_ids:
                continue
            if measurement.station_id not in self._station_map:
                self._rejected_ids.add(identity)
                self._rejection_reasons[identity] = "unknown_station_id"
                continue
            try:
                _positive_definite(
                    measurement.covariance_tangent_rad2,
                    (2, 2),
                    name="observation_covariance",
                )
            except ValueError:
                self._rejected_ids.add(identity)
                self._rejection_reasons[identity] = "unsupported_singular_covariance"
                continue
            if measurement.available_timestamp_s <= self._generation_start_time:
                continue
            result.append(measurement)
        result.sort(key=lambda item: (item.available_timestamp_s, bearing_event_id(item)))
        return tuple(result[-self._config.maximum_initialization_buffer_events :])

    def _make_tentative(
        self,
        measurements: Sequence[BearingMeasurement],
        processing_time_s: float,
    ) -> RecoveryHypothesisDiagnostic | None:
        if self._hypothesis_attempt_count >= self._config.maximum_hypotheses_per_generation:
            self._status = "uninitialized" if self._generation == 0 else "recovering"
            self._failure_reason = "initialization_failed:hypothesis_limit"
            return None
        viable: list[
            tuple[
                tuple[object, ...],
                tuple[str, ...],
                RetardedBatchResult,
                tuple[tuple[str, float], ...],
            ]
        ] = []
        for subset in _construction_candidates(
            measurements,
            self._config.construction_measurement_count,
            self._config.maximum_candidate_count,
        ):
            signature = tuple(sorted(bearing_event_id(item) for item in subset))
            if signature in self._failed_signatures:
                continue
            _, rank, _ = geometric_constant_velocity_initial_state(
                self._stations, subset, reference_time_s=processing_time_s
            )
            if rank < self._criteria.required_local_rank:
                self._failed_signatures.add(signature)
                continue
            batch = estimate_retarded_constant_velocity_batch(
                self._stations,
                subset,
                reference_time_s=processing_time_s,
                sound_speed=self._sound_speed,
            )
            if not _batch_passes(batch, self._criteria) or batch.state is None:
                self._failed_signatures.add(signature)
                continue
            preliminary = _scores(
                batch.state, self._station_map, measurements, self._sound_speed
            )
            finite = [value for _, value in preliminary if np.isfinite(value)]
            inlier_count = sum(
                value <= self._config.fit_nis_threshold for value in finite
            )
            robust_sum = float(
                np.sum(
                    np.minimum(
                        finite,
                        4.0 * self._config.fit_nis_threshold,
                    )
                )
            )
            viable.append(
                (
                    (
                        -inlier_count,
                        robust_sum,
                        batch.scaled_information_condition_number,
                        signature,
                    ),
                    signature,
                    batch,
                    preliminary,
                )
            )
        if viable:
            _, signature, batch, preliminary = min(viable, key=lambda item: item[0])
            self._hypothesis_attempt_count += 1
            self._tentative = _TentativeHypothesis(
                batch=batch,
                creation_time_s=float(processing_time_s),
                generation=self._generation + 1,
                index=self._hypothesis_attempt_count,
                construction_ids=signature,
                creation_available_ids=frozenset(
                    bearing_event_id(item) for item in measurements
                ),
                preliminary_scores=preliminary,
            )
            self._status = "tentative"
            self._failure_reason = "tentative_initialization_unconfirmed"
            diagnostic = RecoveryHypothesisDiagnostic(
                processing_time_s=float(processing_time_s),
                generation=self._generation + 1,
                hypothesis_index=self._hypothesis_attempt_count,
                action="tentative_created",
                reason="six_event_hypothesis_requires_independent_confirmation",
                construction_event_ids=signature,
                confirmation_event_ids=(),
                excluded_event_ids=(),
                preliminary_nis_values=preliminary,
                confirmation_nis_values=(),
                final_nis_values=(),
                local_observability_rank=batch.local_observability_rank,
                scaled_condition_number=batch.scaled_information_condition_number,
                optimizer_valid=batch.valid,
            )
            self._hypothesis_history.append(diagnostic)
            return diagnostic
        self._failure_reason = "initialization_failed:no_observable_hypothesis"
        return None

    def _reject_tentative(
        self,
        processing_time_s: float,
        reason: str,
        confirmation_scores: tuple[tuple[str, float], ...],
    ) -> RecoveryHypothesisDiagnostic:
        assert self._tentative is not None
        tentative = self._tentative
        self._failed_signatures.add(tentative.construction_ids)
        diagnostic = RecoveryHypothesisDiagnostic(
            processing_time_s=float(processing_time_s),
            generation=tentative.generation,
            hypothesis_index=tentative.index,
            action="tentative_rejected",
            reason=reason,
            construction_event_ids=tentative.construction_ids,
            confirmation_event_ids=tuple(identity for identity, _ in confirmation_scores),
            excluded_event_ids=(),
            preliminary_nis_values=tentative.preliminary_scores,
            confirmation_nis_values=confirmation_scores,
            final_nis_values=(),
            local_observability_rank=tentative.batch.local_observability_rank,
            scaled_condition_number=tentative.batch.scaled_information_condition_number,
            optimizer_valid=tentative.batch.valid,
        )
        self._hypothesis_history.append(diagnostic)
        self._tentative = None
        self._status = "uninitialized" if self._generation == 0 else "recovering"
        self._failure_reason = f"initialization_failed:{reason}"
        return diagnostic

    def _evaluate_tentative(
        self,
        measurements: Sequence[BearingMeasurement],
        processing_time_s: float,
    ) -> tuple[RecoveryHypothesisDiagnostic | None, RecoveryLifecycleDiagnostic | None]:
        if self._tentative is None:
            return None, None
        tentative = self._tentative
        assert tentative.batch.state is not None
        confirmation = tuple(
            item
            for item in measurements
            if bearing_event_id(item) not in tentative.creation_available_ids
        )
        scores = _predictive_scores(
            tentative.batch,
            self._station_map,
            confirmation,
            self._sound_speed,
        )
        score_map = dict(scores)
        inliers = tuple(
            item
            for item in confirmation
            if np.isfinite(score_map[bearing_event_id(item)])
            and score_map[bearing_event_id(item)] <= self._config.fit_nis_threshold
        )
        failures = len(confirmation) - len(inliers)
        stations = {item.station_id for item in inliers}
        reception_span = (
            max(item.reception_center_timestamp_s for item in inliers)
            - min(item.reception_center_timestamp_s for item in inliers)
            if len(inliers) >= 2
            else 0.0
        )
        enough = bool(
            len(inliers) >= self._config.confirmation_measurement_count
            and len(stations) >= self._config.confirmation_station_count
            and reception_span >= self._config.confirmation_reception_span_s
        )
        if enough:
            preliminary_all = _scores(
                tentative.batch.state,
                self._station_map,
                measurements,
                self._sound_speed,
            )
            preliminary_map = dict(preliminary_all)
            preliminary_inliers = tuple(
                item
                for item in measurements
                if np.isfinite(preliminary_map[bearing_event_id(item)])
                and preliminary_map[bearing_event_id(item)] <= self._config.fit_nis_threshold
            )
            final_batch, final_scores, final_inliers = _refine_consensus(
                self._stations,
                self._station_map,
                preliminary_inliers,
                reference_time_s=processing_time_s,
                sound_speed=self._sound_speed,
                threshold=self._config.fit_nis_threshold,
                criteria=self._criteria,
            )
            final_ids = {bearing_event_id(item) for item in final_inliers}
            confirmation_ids = tuple(
                sorted(bearing_event_id(item) for item in inliers if bearing_event_id(item) in final_ids)
            )
            final_confirmation = tuple(
                item for item in inliers if bearing_event_id(item) in final_ids
            )
            final_span = (
                max(item.reception_center_timestamp_s for item in final_confirmation)
                - min(item.reception_center_timestamp_s for item in final_confirmation)
                if len(final_confirmation) >= 2
                else 0.0
            )
            final_confirmed = bool(
                final_batch is not None
                and len(final_confirmation) >= self._config.confirmation_measurement_count
                and len({item.station_id for item in final_confirmation})
                >= self._config.confirmation_station_count
                and final_span >= self._config.confirmation_reception_span_s
            )
            if final_confirmed and final_batch is not None and final_batch.state is not None:
                covariance = _positive_definite(
                    final_batch.covariance_state_linearization,
                    (6, 6),
                    name="initialization_covariance",
                )
                self._generation += 1
                self._state = final_batch.state
                self._covariance = _readonly(covariance)
                used_ids = tuple(final_batch.used_event_ids)
                used_set = set(used_ids)
                available_ids = {bearing_event_id(item) for item in measurements}
                excluded_ids = tuple(sorted(available_ids - used_set))
                for identity in used_ids:
                    self._processed_ids.add(identity)
                    self._initialization_ids.add(identity)
                    self._event_uses.append(
                        RecoveryEventUse(identity, self._generation, "initialization")
                    )
                self._rejected_ids.update(excluded_ids)
                self._rejection_reasons.update(
                    {
                        identity: "confirmed_initialization_consensus_outlier"
                        for identity in excluded_ids
                    }
                )
                self._processed_ids.update(excluded_ids)
                self._tentative = None
                self._status = "confirmed"
                self._failure_reason = None
                self._inconsistency_streak.clear()
                if not np.isfinite(self._first_confirmation_time):
                    self._first_confirmation_time = float(processing_time_s)
                if np.isfinite(self._recovery_started_time):
                    self._last_recovery_duration = (
                        float(processing_time_s) - self._recovery_started_time
                    )
                    self._recovery_started_time = float("nan")
                diagnostic = RecoveryHypothesisDiagnostic(
                    processing_time_s=float(processing_time_s),
                    generation=self._generation,
                    hypothesis_index=tentative.index,
                    action="hypothesis_confirmed",
                    reason="independent_bearings_and_final_refit_passed",
                    construction_event_ids=tentative.construction_ids,
                    confirmation_event_ids=confirmation_ids,
                    excluded_event_ids=excluded_ids,
                    preliminary_nis_values=preliminary_all,
                    confirmation_nis_values=scores,
                    final_nis_values=_scores(
                        final_batch.state,
                        self._station_map,
                        measurements,
                        self._sound_speed,
                    ),
                    local_observability_rank=final_batch.local_observability_rank,
                    scaled_condition_number=final_batch.scaled_information_condition_number,
                    optimizer_valid=final_batch.valid,
                )
                self._hypothesis_history.append(diagnostic)
                lifecycle = self._lifecycle(
                    processing_time_s,
                    "initialized" if self._generation == 1 else "reinitialized",
                    "tentative_hypothesis_confirmed",
                    used_ids,
                )
                return diagnostic, lifecycle
        expired = bool(
            failures >= self._config.maximum_confirmation_failures
            or len(confirmation) >= self._config.maximum_confirmation_events
            or processing_time_s - tentative.creation_time_s
            >= self._config.maximum_confirmation_age_s
        )
        if expired:
            reason = (
                "confirmation_inconsistent"
                if failures >= self._config.maximum_confirmation_failures
                else "confirmation_timeout"
            )
            return self._reject_tentative(processing_time_s, reason, scores), None
        return None, None

    def _triggered_inconsistency(self) -> bool:
        streak = self._inconsistency_streak
        if len(streak) < self._config.inconsistency_rejection_count:
            return False
        stations = {item.station_id for item in streak}
        span = (
            max(item.reception_center_timestamp_s for item in streak)
            - min(item.reception_center_timestamp_s for item in streak)
        )
        return bool(
            len(stations) >= self._config.inconsistency_station_count
            and span >= self._config.inconsistency_reception_span_s
        )

    def _reset_for_recovery(self, processing_time_s: float) -> RecoveryLifecycleDiagnostic:
        trigger_ids = tuple(bearing_event_id(item) for item in self._inconsistency_streak)
        self._state = None
        self._covariance = None
        self._tentative = None
        self._status = "questionable"
        self._failure_reason = "consistency_lost_reinitialization_required"
        self._generation_start_time = float(processing_time_s)
        self._hypothesis_attempt_count = 0
        self._failed_signatures.clear()
        self._inconsistency_streak.clear()
        self._reset_count += 1
        self._recovery_started_time = float(processing_time_s)
        return self._lifecycle(
            processing_time_s,
            "consistency_lost",
            "sustained_multi_station_nis_rejections",
            trigger_ids,
        )

    def _process_group(
        self,
        processing_time_s: float,
        prefix: BearingEventPrefix,
    ) -> tuple[list[RetardedEKFUpdateResult], list[RecoveryHypothesisDiagnostic], list[RecoveryLifecycleDiagnostic]]:
        updates: list[RetardedEKFUpdateResult] = []
        hypothesis_changes: list[RecoveryHypothesisDiagnostic] = []
        lifecycle_changes: list[RecoveryLifecycleDiagnostic] = []
        if self._state is None:
            measurements = self._eligible(prefix)
            change, lifecycle = self._evaluate_tentative(measurements, processing_time_s)
            if change is not None:
                hypothesis_changes.append(change)
            if lifecycle is not None:
                lifecycle_changes.append(lifecycle)
                return updates, hypothesis_changes, lifecycle_changes
            if self._tentative is None:
                created = self._make_tentative(measurements, processing_time_s)
                if created is not None:
                    hypothesis_changes.append(created)
            return updates, hypothesis_changes, lifecycle_changes

        assert self._covariance is not None
        if processing_time_s > self._state.reference_time_s:
            self._state, self._covariance, _ = propagate_constant_velocity_estimate(
                self._state, self._covariance, processing_time_s
            )
        pending = self._eligible(prefix)
        for measurement in pending:
            identity = bearing_event_id(measurement)
            update = update_retarded_ekf(
                self._state,
                self._covariance,
                self._station_map[measurement.station_id],
                measurement,
                sound_speed=self._sound_speed,
                maximum_pre_update_nis=self._config.pre_update_nis_threshold,
            )
            updates.append(update)
            self._processed_ids.add(identity)
            if update.update_applied:
                self._state = update.posterior_state
                self._covariance = update.posterior_covariance
                self._applied_ids.add(identity)
                self._event_uses.append(
                    RecoveryEventUse(identity, self._generation, "update")
                )
                self._inconsistency_streak.clear()
            else:
                self._rejected_ids.add(identity)
                self._rejection_reasons[identity] = (
                    update.failure_reason or "measurement_update_rejected"
                )
                if update.failure_reason == "pre_update_nis_gate":
                    self._inconsistency_streak.append(measurement)
                else:
                    self._inconsistency_streak.clear()
        if self._triggered_inconsistency():
            lifecycle_changes.append(self._reset_for_recovery(processing_time_s))
        return updates, hypothesis_changes, lifecycle_changes

    def advance_to(self, processing_time_s: float) -> RecoveryPublication:
        started = time.perf_counter()
        processing_time = float(processing_time_s)
        if not np.isfinite(processing_time):
            raise ValueError("processing_time_s must be finite")
        if processing_time < self._last_processing_time:
            raise ValueError("processing time cannot move backwards")
        updates: list[RetardedEKFUpdateResult] = []
        hypotheses: list[RecoveryHypothesisDiagnostic] = []
        lifecycle: list[RecoveryLifecycleDiagnostic] = []
        last_prefix: BearingEventPrefix | None = None
        while True:
            next_time = self._stream.next_available_timestamp_s
            if next_time is None or next_time > processing_time:
                break
            last_prefix = self._stream.advance_to(next_time)
            group_updates, group_hypotheses, group_lifecycle = self._process_group(
                next_time, last_prefix
            )
            updates.extend(group_updates)
            hypotheses.extend(group_hypotheses)
            lifecycle.extend(group_lifecycle)
            if self._status == "questionable":
                self._status = "recovering"
        prefix = self._stream.advance_to(processing_time)
        state = None
        covariance = _readonly(np.full((6, 6), np.nan))
        tentative_state = None
        tentative_covariance = _readonly(np.full((6, 6), np.nan))
        if self._state is not None:
            state, covariance, _ = propagate_constant_velocity_estimate(
                self._state, self._covariance, processing_time
            )
        elif self._tentative is not None and self._tentative.batch.state is not None:
            tentative_state, tentative_covariance, _ = propagate_constant_velocity_estimate(
                self._tentative.batch.state,
                self._tentative.batch.covariance_state_linearization,
                processing_time,
            )
        immediate_loss = any(item.action == "consistency_lost" for item in lifecycle)
        published_status = (
            self._status
            if state is not None
            else ("questionable" if immediate_loss else self._status)
        )
        publication = RecoveryPublication(
            processing_time_s=processing_time,
            status=published_status,
            initialized=self._generation > 0,
            confirmed=state is not None,
            valid=state is not None,
            failure_reason=None if state is not None else self._failure_reason,
            state=state,
            covariance_state=_readonly(covariance),
            tentative_state=tentative_state,
            tentative_covariance_state=_readonly(tentative_covariance),
            generation=self._generation,
            prefix=prefix,
            initialization_event_ids=tuple(sorted(self._initialization_ids)),
            applied_event_ids=tuple(sorted(self._applied_ids)),
            rejected_event_ids=tuple(sorted(self._rejected_ids)),
            rejection_reasons=tuple(sorted(self._rejection_reasons.items())),
            update_diagnostics=tuple(updates),
            hypothesis_diagnostics=tuple(self._hypothesis_history),
            new_hypothesis_diagnostics=tuple(hypotheses),
            lifecycle_diagnostics=tuple(self._lifecycle_history),
            new_lifecycle_diagnostics=tuple(lifecycle),
            event_uses=tuple(self._event_uses),
            reset_count=self._reset_count,
            first_confirmation_time_s=self._first_confirmation_time,
            last_recovery_duration_s=self._last_recovery_duration,
            total_runtime_s=time.perf_counter() - started,
        )
        self._publications.append(publication)
        self._last_processing_time = processing_time
        return publication


__all__ = [
    "CausalConfirmedRetardedTimeEKF",
    "InitializationRecoveryConfig",
    "RecoveryEventUse",
    "RecoveryHypothesisDiagnostic",
    "RecoveryLifecycleDiagnostic",
    "RecoveryPublication",
]
