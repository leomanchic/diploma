"""Opt-in causal retarded-bearing EKF with correlated stochastic history.

The C1, D2 and confirmed-recovery implementations are imported unchanged.
This subclass reuses recovery's truth-free confirmation and event lifecycle,
then replaces *only* its confirmed-state measurement update with a bounded
augmented-history Joseph EKF.  See ``MANOEUVRE_TRACKING_MODEL.md``.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Sequence

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.optimize import brentq

from estimators.retarded_ekf_recovery import (
    CausalConfirmedRetardedTimeEKF,
    InitializationRecoveryConfig,
    RecoveryEventUse,
    RecoveryHypothesisDiagnostic,
    RecoveryLifecycleDiagnostic,
    RecoveryPublication,
)
from model.bearing_events import BearingEventPrefix, bearing_event_id
from model.bearing_statistics import (
    AntipodalDirectionError,
    measurement_anchored_tangent_residual,
    measurement_anchored_tangent_residual_jacobian,
    tangent_residual,
    tangent_residual_jacobian_wrt_true_direction,
)
from model.dynamic_state import ConstantVelocityState
from model.geometry import DEFAULT_SOUND_SPEED
from model.measurements import BearingMeasurement
from model.station import StationPose
from model.stochastic_motion import (
    acceleration_spectral_density,
    integrated_wiener_bridge,
    integrated_wiener_transition,
)


class HistoryObservationError(ValueError):
    """One observation cannot be evaluated inside the declared history."""

    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


@dataclass(frozen=True, slots=True)
class ManoeuvreHistoryConfig:
    """Physical bounds and explicitly selected acceleration spectral density."""

    qc_m2_s3: ArrayLike
    history_step_s: float = 0.25
    history_window_s: float = 3.5
    maximum_range_m: float = 250.0
    maximum_transport_delay_s: float = 2.0

    def __post_init__(self) -> None:
        qc = acceleration_spectral_density(self.qc_m2_s3)
        if np.any(qc) and np.linalg.matrix_rank(qc) != 3:
            raise ValueError("nonzero Qc must be positive definite for the bridge")
        for name in (
            "history_step_s", "history_window_s", "maximum_range_m",
            "maximum_transport_delay_s",
        ):
            value = float(getattr(self, name))
            if not np.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive")
            object.__setattr__(self, name, value)
        object.__setattr__(self, "qc_m2_s3", qc)

    def validate_for_sound_speed(self, sound_speed: float) -> None:
        required = (
            self.maximum_range_m / sound_speed
            + self.maximum_transport_delay_s
            + self.history_step_s
        )
        if self.history_window_s < required:
            raise ValueError(
                "history_window_s must cover max range/c + max transport delay + step"
            )


@dataclass(frozen=True, slots=True)
class HistoryBearingPrediction:
    emission_time_s: float
    range_m: float
    direction_local: NDArray[np.float64]
    residual_tangent_rad: NDArray[np.float64]
    jacobian_augmented: NDArray[np.float64]


@dataclass(frozen=True, slots=True)
class ManoeuvreUpdateDiagnostic:
    event_id: str
    processing_time_s: float
    reception_time_s: float
    emission_time_s: float
    pre_update_nis: float
    update_applied: bool
    failure_reason: str | None
    runtime_s: float
    history_node_count: int
    history_memory_bytes: int


@dataclass(frozen=True, slots=True)
class ManoeuvrePublication:
    """Recovery lifecycle plus a Qd-aware output-only state prediction."""

    base: RecoveryPublication
    state: ConstantVelocityState | None
    covariance_state: NDArray[np.float64]
    history_node_count: int
    history_oldest_time_s: float
    history_memory_bytes: int
    qc_m2_s3: NDArray[np.float64]

    def __getattr__(self, name: str):
        return getattr(self.base, name)


class AugmentedMotionHistory:
    """Joint Gaussian [q,v] nodes, including every retained cross covariance."""

    def __init__(
        self, state: ConstantVelocityState, covariance: ArrayLike,
        config: ManoeuvreHistoryConfig, sound_speed: float = DEFAULT_SOUND_SPEED,
    ) -> None:
        matrix = np.asarray(covariance, dtype=float)
        if matrix.shape != (6, 6) or not np.all(np.isfinite(matrix)):
            raise ValueError("initial covariance must be finite 6x6")
        np.linalg.cholesky(0.5 * (matrix + matrix.T))
        config.validate_for_sound_speed(sound_speed)
        self.config = config
        self.sound_speed = float(sound_speed)
        self.times_s = [float(state.reference_time_s)]
        self.mean = np.array(state.vector, copy=True)
        self.covariance = 0.5 * (matrix + matrix.T)
        self.maximum_node_count = 1
        self.maximum_memory_bytes = self.memory_bytes

    @property
    def node_count(self) -> int:
        return len(self.times_s)

    @property
    def memory_bytes(self) -> int:
        """Owned numerical history payload, not process RSS or temporaries.

        The value is ``mean.nbytes + covariance.nbytes + 8 bytes/epoch``.
        It deliberately excludes Python container overhead and temporary
        arrays allocated by linear algebra operations.
        """

        return self.mean.nbytes + self.covariance.nbytes + 8 * len(self.times_s)

    def _record_peak_storage(self) -> None:
        """Record owned history storage before any following marginalization."""

        self.maximum_node_count = max(self.maximum_node_count, self.node_count)
        self.maximum_memory_bytes = max(self.maximum_memory_bytes, self.memory_bytes)

    @property
    def oldest_time_s(self) -> float:
        return self.times_s[0]

    def latest_state(self) -> tuple[ConstantVelocityState, NDArray[np.float64]]:
        vector = self.mean[-6:]
        return (
            ConstantVelocityState(vector[:3], vector[3:], self.times_s[-1]),
            np.array(self.covariance[-6:, -6:], copy=True),
        )

    def predict_latest_to(
        self, time_s: float
    ) -> tuple[ConstantVelocityState, NDArray[np.float64]]:
        state, p = self.latest_state()
        dt = float(time_s) - state.reference_time_s
        if dt < 0.0:
            raise ValueError("publication time cannot precede history posterior")
        f, qd = integrated_wiener_transition(dt, self.config.qc_m2_s3)
        vector = f @ state.vector
        return ConstantVelocityState(vector[:3], vector[3:], float(time_s)), f @ p @ f.T + qd

    def _append_one(self, time_s: float) -> None:
        dt = time_s - self.times_s[-1]
        f, qd = integrated_wiener_transition(dt, self.config.qc_m2_s3)
        old_dimension = self.mean.size
        latest = self.mean[-6:]
        cross = f @ self.covariance[-6:, :]
        self.mean = np.concatenate((self.mean, f @ latest))
        enlarged = np.empty((old_dimension + 6, old_dimension + 6))
        enlarged[:old_dimension, :old_dimension] = self.covariance
        enlarged[old_dimension:, :old_dimension] = cross
        enlarged[:old_dimension, old_dimension:] = cross.T
        enlarged[old_dimension:, old_dimension:] = (
            f @ self.covariance[-6:, -6:] @ f.T + qd
        )
        self.covariance = 0.5 * (enlarged + enlarged.T)
        self.times_s.append(float(time_s))
        self._record_peak_storage()

    def propagate_to(self, time_s: float) -> None:
        target = float(time_s)
        if not np.isfinite(target) or target < self.times_s[-1]:
            raise ValueError("history propagation must be causal")
        while target - self.times_s[-1] > self.config.history_step_s + 1e-12:
            self._append_one(self.times_s[-1] + self.config.history_step_s)
            # Marginalize as the causal frontier moves.  Keeping the oldest
            # node whose successor is inside the window preserves the boundary
            # pair needed for bridge interpolation without transient growth
            # proportional to the event-free gap.
            self.prune()
        if target > self.times_s[-1] + 1e-12:
            self._append_one(target)
            self.prune()

    def prune(self) -> None:
        while (
            self.node_count > 2
            and self.times_s[-1] - self.times_s[1] > self.config.history_window_s
        ):
            self.times_s.pop(0)
            self.mean = self.mean[6:].copy()
            self.covariance = self.covariance[6:, 6:].copy()

    def evaluate(
        self, time_s: float
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Conditional-mean state and its full augmented-state Jacobian."""

        t = float(time_s)
        if t < self.times_s[0] - 1e-12 or t > self.times_s[-1] + 1e-12:
            raise HistoryObservationError("emission_outside_history")
        index = int(np.searchsorted(self.times_s, t))
        for candidate in (index, index - 1):
            if 0 <= candidate < self.node_count and abs(self.times_s[candidate] - t) <= 1e-11:
                jacobian = np.zeros((6, self.mean.size))
                jacobian[:, 6 * candidate : 6 * candidate + 6] = np.eye(6)
                return self.mean[6 * candidate : 6 * candidate + 6].copy(), jacobian
        left = index - 1
        right = index
        h = self.times_s[right] - self.times_s[left]
        a0, a1, _ = integrated_wiener_bridge(
            t - self.times_s[left], h, self.config.qc_m2_s3
        )
        jacobian = np.zeros((6, self.mean.size))
        jacobian[:, 6 * left : 6 * left + 6] = a0
        jacobian[:, 6 * right : 6 * right + 6] = a1
        return jacobian @ self.mean, jacobian

    def insert_bridge_node(self, time_s: float) -> None:
        """Augment a conditional Gaussian bridge without losing correlations."""

        t = float(time_s)
        index = int(np.searchsorted(self.times_s, t))
        if any(
            0 <= candidate < self.node_count and abs(self.times_s[candidate] - t) <= 1e-11
            for candidate in (index, index - 1)
        ):
            return
        if index <= 0 or index >= self.node_count:
            raise HistoryObservationError("emission_outside_history")
        left = index - 1
        right = index
        h = self.times_s[right] - self.times_s[left]
        a0, a1, bridge = integrated_wiener_bridge(
            t - self.times_s[left], h, self.config.qc_m2_s3
        )
        d = self.mean.size
        left_slice = slice(6 * left, 6 * left + 6)
        right_slice = slice(6 * right, 6 * right + 6)
        new_mean = a0 @ self.mean[left_slice] + a1 @ self.mean[right_slice]
        new_cross = (
            a0 @ self.covariance[left_slice, :]
            + a1 @ self.covariance[right_slice, :]
        )
        new_variance = (
            a0 @ self.covariance[left_slice, left_slice] @ a0.T
            + a0 @ self.covariance[left_slice, right_slice] @ a1.T
            + a1 @ self.covariance[right_slice, left_slice] @ a0.T
            + a1 @ self.covariance[right_slice, right_slice] @ a1.T
            + bridge
        )
        expanded = np.empty((d + 6, d + 6))
        expanded[:d, :d] = self.covariance
        expanded[d:, :d] = new_cross
        expanded[:d, d:] = new_cross.T
        expanded[d:, d:] = new_variance
        block_order = [*range(index), self.node_count, *range(index, self.node_count)]
        scalar_order = [6 * block + k for block in block_order for k in range(6)]
        self.mean = np.concatenate((self.mean, new_mean))[scalar_order]
        self.covariance = 0.5 * (
            expanded[np.ix_(scalar_order, scalar_order)]
            + expanded[np.ix_(scalar_order, scalar_order)].T
        )
        self.times_s.insert(index, t)
        self._record_peak_storage()

    def emission_time(self, station: StationPose, reception_time_s: float) -> float:
        reception = float(reception_time_s)
        if not np.isfinite(reception) or reception > self.times_s[-1] + 1e-10:
            raise HistoryObservationError("reception_outside_history")

        def equation(candidate: float) -> float:
            state, _ = self.evaluate(candidate)
            return (
                candidate
                + np.linalg.norm(state[:3] - station.position_world_m) / self.sound_speed
                - reception
            )

        lower = equation(self.times_s[0])
        upper = equation(self.times_s[-1])
        if lower > 0.0:
            raise HistoryObservationError("emission_outside_history")
        if upper < 0.0:
            raise HistoryObservationError("reception_outside_history")
        return float(brentq(equation, self.times_s[0], self.times_s[-1], xtol=2e-14))

    def predict_measurement(
        self, station: StationPose, measurement: BearingMeasurement,
        *, insert_emission_node: bool = False,
    ) -> HistoryBearingPrediction:
        if not measurement.valid or measurement.station_id != station.station_id:
            raise HistoryObservationError("invalid_or_mismatched_bearing")
        emission = self.emission_time(station, measurement.reception_center_timestamp_s)
        if insert_emission_node:
            self.insert_bridge_node(emission)
        state, state_jacobian = self.evaluate(emission)
        displacement = state[:3] - station.position_world_m
        range_m = float(np.linalg.norm(displacement))
        if range_m <= 0.0 or not np.isfinite(range_m):
            raise HistoryObservationError("source_at_station_centroid")
        direction_world = displacement / range_m
        speed = float(np.linalg.norm(state[3:]))
        if speed >= self.sound_speed:
            raise HistoryObservationError("supersonic_history_prediction")
        denominator = 1.0 + direction_world @ state[3:] / self.sound_speed
        if denominator <= 0.0:
            raise HistoryObservationError("nonmonotone_retarded_map")
        direct_position_jacobian = state_jacobian[:3, :]
        emission_jacobian = -(
            direction_world @ direct_position_jacobian
        ) / (self.sound_speed * denominator)
        emission_position_jacobian = (
            direct_position_jacobian + np.outer(state[3:], emission_jacobian)
        )
        projector = np.eye(3) - np.outer(direction_world, direction_world)
        direction_local = station.world_to_local_direction(direction_world)
        direction_jacobian = (
            station.rotation_local_to_world.T
            @ projector @ emission_position_jacobian / range_m
        )
        if measurement.tangent_frame == "measurement":
            residual = measurement_anchored_tangent_residual(
                direction_local, measurement.direction_local
            )
            residual_direction_jacobian = measurement_anchored_tangent_residual_jacobian(
                direction_local, measurement.direction_local
            )
        else:
            if np.hypot(*direction_local[:2]) <= 1e-10:
                raise HistoryObservationError("prediction_tangent_frame_undefined_at_pole")
            residual = tangent_residual(direction_local, measurement.direction_local)
            residual_direction_jacobian = tangent_residual_jacobian_wrt_true_direction(
                direction_local, measurement.direction_local
            )
        return HistoryBearingPrediction(
            emission_time_s=emission,
            range_m=range_m,
            direction_local=direction_local,
            residual_tangent_rad=residual - measurement.calibration_bias_tangent_rad,
            jacobian_augmented=residual_direction_jacobian @ direction_jacobian,
        )

    def update_bearing(
        self, station: StationPose, measurement: BearingMeasurement,
        maximum_pre_update_nis: float,
    ) -> ManoeuvreUpdateDiagnostic:
        started = time.perf_counter()
        identity = bearing_event_id(measurement)
        emission = float("nan")
        nis = float("nan")
        applied = False
        reason: str | None = None
        try:
            r = np.asarray(measurement.covariance_tangent_rad2, dtype=float)
            np.linalg.cholesky(0.5 * (r + r.T))
            predicted = self.predict_measurement(
                station, measurement, insert_emission_node=True
            )
            emission = predicted.emission_time_s
            h = predicted.jacobian_augmented
            s = h @ self.covariance @ h.T + r
            np.linalg.cholesky(0.5 * (s + s.T))
            nis = float(predicted.residual_tangent_rad @ np.linalg.solve(
                s, predicted.residual_tangent_rad
            ))
            if nis > maximum_pre_update_nis:
                reason = "pre_update_nis_gate"
            else:
                gain = np.linalg.solve(s, h @ self.covariance).T
                mean_after = self.mean - gain @ predicted.residual_tangent_rad
                identity_matrix = np.eye(self.mean.size)
                joseph = identity_matrix - gain @ h
                p_after = (
                    joseph @ self.covariance @ joseph.T + gain @ r @ gain.T
                )
                if not np.all(np.isfinite(mean_after)) or not np.all(np.isfinite(p_after)):
                    raise HistoryObservationError("nonfinite_history_posterior")
                self.mean = mean_after
                self.covariance = 0.5 * (p_after + p_after.T)
                applied = True
        except HistoryObservationError as error:
            reason = error.reason
        except (ValueError, AntipodalDirectionError, np.linalg.LinAlgError):
            reason = "invalid_history_measurement_linearization"
        return ManoeuvreUpdateDiagnostic(
            event_id=identity,
            processing_time_s=self.times_s[-1],
            reception_time_s=measurement.reception_center_timestamp_s,
            emission_time_s=emission,
            pre_update_nis=nis,
            update_applied=applied,
            failure_reason=reason,
            runtime_s=time.perf_counter() - started,
            history_node_count=self.node_count,
            history_memory_bytes=self.memory_bytes,
        )


class CausalManoeuvreRetardedTimeEKF(CausalConfirmedRetardedTimeEKF):
    """Confirmed-recovery lifecycle with an opt-in Qc>0 augmented update."""

    def __init__(
        self, stations: Sequence[StationPose], events: Sequence,
        *, estimator_variant: str,
        history_config: ManoeuvreHistoryConfig,
        sound_speed: float = DEFAULT_SOUND_SPEED,
        recovery_config: InitializationRecoveryConfig | None = None,
    ) -> None:
        super().__init__(
            stations, events, estimator_variant=estimator_variant,
            sound_speed=sound_speed, recovery_config=recovery_config,
        )
        history_config.validate_for_sound_speed(sound_speed)
        self._history_config = history_config
        self._history: AugmentedMotionHistory | None = None
        self._manoeuvre_publications: list[ManoeuvrePublication] = []
        self._maximum_history_memory_bytes = 0
        self._maximum_history_node_count = 0

    @property
    def publications(self) -> tuple[ManoeuvrePublication, ...]:
        return tuple(self._manoeuvre_publications)

    @property
    def maximum_history_memory_bytes(self) -> int:
        """Peak owned numerical history payload across all generations."""

        return self._maximum_history_memory_bytes

    @property
    def maximum_history_node_count(self) -> int:
        """Peak retained nodes, including transient bridge nodes."""

        return self._maximum_history_node_count

    def _record_history_peak(self) -> None:
        if self._history is None:
            return
        self._maximum_history_memory_bytes = max(
            self._maximum_history_memory_bytes,
            self._history.maximum_memory_bytes,
        )
        self._maximum_history_node_count = max(
            self._maximum_history_node_count,
            self._history.maximum_node_count,
        )

    def _invalidate_state(
        self, processing_time_s: float, *, action: str, reason: str,
        event_ids: Sequence[str], failure_reason: str | None = None,
    ) -> RecoveryLifecycleDiagnostic:
        self._history = None
        return super()._invalidate_state(
            processing_time_s, action=action, reason=reason,
            event_ids=event_ids, failure_reason=failure_reason,
        )

    def _process_group(
        self, processing_time_s: float, prefix: BearingEventPrefix,
    ) -> tuple[list[ManoeuvreUpdateDiagnostic], list[RecoveryHypothesisDiagnostic], list[RecoveryLifecycleDiagnostic]]:
        if self._state is None:
            updates, hypotheses, lifecycle = super()._process_group(processing_time_s, prefix)
            if self._state is not None:
                assert self._covariance is not None
                initial_state = self._state
                initial_covariance = self._covariance
                if not np.any(self._history_config.qc_m2_s3):
                    # Only at Q=0 is reverse CV propagation an exact
                    # deterministic transformation of the same state.
                    age = (
                        self._history_config.maximum_range_m / self._sound_speed
                        + self._history_config.maximum_transport_delay_s
                    )
                    f_back = np.eye(6)
                    f_back[:3, 3:] = -age * np.eye(3)
                    vector = f_back @ initial_state.vector
                    initial_state = ConstantVelocityState(
                        vector[:3], vector[3:], initial_state.reference_time_s - age
                    )
                    initial_covariance = f_back @ initial_covariance @ f_back.T
                self._history = AugmentedMotionHistory(
                    initial_state, initial_covariance, self._history_config, self._sound_speed
                )
                if initial_state.reference_time_s < processing_time_s:
                    self._history.propagate_to(processing_time_s)
                self._record_history_peak()
            return list(updates), hypotheses, lifecycle

        conflict_hypotheses, conflict_lifecycle, invalidated = self._handle_conflicts(
            processing_time_s, prefix
        )
        if invalidated:
            return [], conflict_hypotheses, conflict_lifecycle
        assert self._history is not None
        self._history.propagate_to(processing_time_s)
        updates: list[ManoeuvreUpdateDiagnostic] = []
        lifecycle = list(conflict_lifecycle)
        pending = self._eligible(prefix)
        for index, measurement in enumerate(pending):
            identity = bearing_event_id(measurement)
            update = self._history.update_bearing(
                self._station_map[measurement.station_id], measurement,
                self._config.pre_update_nis_threshold,
            )
            self._record_history_peak()
            updates.append(update)
            self._processed_ids.add(identity)
            if update.update_applied:
                self._applied_ids.add(identity)
                self._active_generation_ids.add(identity)
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
                lifecycle.append(self._reset_for_recovery(processing_time_s))
                self._exclude_remainder_after_reset(pending[index + 1 :])
                break
        if self._history is not None:
            self._state, self._covariance = self._history.latest_state()
            self._record_history_peak()
        return updates, conflict_hypotheses, lifecycle

    def advance_to(self, processing_time_s: float) -> ManoeuvrePublication:
        base = super().advance_to(processing_time_s)
        if self._history is None:
            state, covariance = base.state, base.covariance_state
            count, oldest, memory = 0, float("nan"), 0
        else:
            state, covariance = self._history.predict_latest_to(processing_time_s)
            count = self._history.node_count
            oldest = self._history.oldest_time_s
            memory = self._history.memory_bytes
        publication = ManoeuvrePublication(
            base=base, state=state, covariance_state=np.array(covariance, copy=True),
            history_node_count=count, history_oldest_time_s=oldest,
            history_memory_bytes=memory,
            qc_m2_s3=np.array(self._history_config.qc_m2_s3, copy=True),
        )
        self._manoeuvre_publications.append(publication)
        return publication


__all__ = [
    "AugmentedMotionHistory", "CausalManoeuvreRetardedTimeEKF",
    "HistoryBearingPrediction", "HistoryObservationError",
    "ManoeuvreHistoryConfig", "ManoeuvrePublication", "ManoeuvreUpdateDiagnostic",
]
