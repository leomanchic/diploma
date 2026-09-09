"""Causal retarded-time EKF baseline under strict constant velocity.

The six-dimensional state is ``x(T)=[q(T), v(T)]`` in SI units.  Between
publication epochs it follows exact constant velocity with ``Q=0``.  Each
bearing is evaluated at its physical reception timestamp through the existing
retarded-time model, while availability controls only causal scheduling.

This C1 baseline supports positive-definite tangent covariances and a
positive-definite initialized state covariance.  Singular observation
covariances remain supported by the independent batch estimator, but are
reported as ``unsupported_singular_covariance`` here; no epsilon loading or
pseudoinverse is used.
"""

from __future__ import annotations

import time
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from estimators.retarded_state_batch import (
    RetardedBatchResult,
    estimate_retarded_constant_velocity_batch,
)
from model.bearing_events import (
    BearingEventPrefix,
    CausalBearingEventStream,
    ScheduledBearingEvent,
    bearing_event_id,
)
from model.bearing_statistics import AntipodalDirectionError
from model.dynamic_state import (
    ConstantVelocityState,
    constant_velocity_transition_jacobian,
    rebase_constant_velocity_state,
)
from model.geometry import DEFAULT_SOUND_SPEED
from model.measurements import BearingMeasurement
from model.retarded_bearing import (
    retarded_bearing_residual,
    retarded_bearing_residual_jacobian,
)
from model.station import StationPose


def _readonly(value: ArrayLike) -> NDArray[np.float64]:
    result = np.array(value, dtype=float, copy=True)
    result.setflags(write=False)
    return result


def _symmetric(value: ArrayLike, shape: tuple[int, int], *, name: str) -> NDArray[np.float64]:
    matrix = np.asarray(value, dtype=float)
    if matrix.shape != shape or not np.all(np.isfinite(matrix)):
        raise ValueError(f"{name} must be a finite matrix with shape {shape}")
    if not np.allclose(matrix, matrix.T, rtol=0.0, atol=1e-14):
        raise ValueError(f"{name} must be symmetric")
    return 0.5 * (matrix + matrix.T)


def _positive_definite(
    value: ArrayLike, shape: tuple[int, int], *, name: str
) -> NDArray[np.float64]:
    matrix = _symmetric(value, shape, name=name)
    try:
        np.linalg.cholesky(matrix)
    except np.linalg.LinAlgError as error:
        raise ValueError(f"{name} must be positive definite") from error
    return matrix


@dataclass(frozen=True, slots=True)
class LinearizedResidualUpdate:
    """One linear-Gaussian update for residual ``e(x)+H dx``."""

    posterior_vector: NDArray[np.float64]
    posterior_covariance: NDArray[np.float64]
    innovation_covariance: NDArray[np.float64]
    kalman_gain: NDArray[np.float64]
    normalized_innovation_squared: float
    covariance_symmetry_error: float
    covariance_minimum_eigenvalue: float


def joseph_residual_update(
    prior_vector: ArrayLike,
    prior_covariance: ArrayLike,
    residual: ArrayLike,
    residual_jacobian: ArrayLike,
    observation_covariance: ArrayLike,
) -> LinearizedResidualUpdate:
    """Apply the EKF algebra for ``e(x+dx)=e(x)+H dx``.

    Because ``H`` differentiates the residual itself, the correction is
    ``x_post=x_prior-K e``.  The covariance uses the Joseph form.  Linear
    systems are solved directly; no matrix inverse or pseudoinverse is used.
    """

    vector = np.asarray(prior_vector, dtype=float)
    if vector.ndim != 1 or not np.all(np.isfinite(vector)):
        raise ValueError("prior_vector must be a finite vector")
    dimension = vector.size
    covariance = _positive_definite(
        prior_covariance, (dimension, dimension), name="prior_covariance"
    )
    innovation = np.asarray(residual, dtype=float)
    jacobian = np.asarray(residual_jacobian, dtype=float)
    if innovation.ndim != 1 or not np.all(np.isfinite(innovation)):
        raise ValueError("residual must be a finite vector")
    if jacobian.shape != (innovation.size, dimension) or not np.all(
        np.isfinite(jacobian)
    ):
        raise ValueError("residual_jacobian has incompatible shape or values")
    observation = _positive_definite(
        observation_covariance,
        (innovation.size, innovation.size),
        name="observation_covariance",
    )
    innovation_covariance = jacobian @ covariance @ jacobian.T + observation
    innovation_covariance = _positive_definite(
        innovation_covariance,
        (innovation.size, innovation.size),
        name="innovation_covariance",
    )
    solved_cross = np.linalg.solve(
        innovation_covariance, jacobian @ covariance
    )
    gain = solved_cross.T
    posterior_vector = vector - gain @ innovation
    identity_minus = np.eye(dimension) - gain @ jacobian
    posterior_covariance = (
        identity_minus @ covariance @ identity_minus.T
        + gain @ observation @ gain.T
    )
    symmetry_error = float(
        np.max(np.abs(posterior_covariance - posterior_covariance.T), initial=0.0)
    )
    posterior_covariance = 0.5 * (
        posterior_covariance + posterior_covariance.T
    )
    _positive_definite(
        posterior_covariance,
        (dimension, dimension),
        name="posterior_covariance",
    )
    nis = float(innovation @ np.linalg.solve(innovation_covariance, innovation))
    return LinearizedResidualUpdate(
        posterior_vector=_readonly(posterior_vector),
        posterior_covariance=_readonly(posterior_covariance),
        innovation_covariance=_readonly(innovation_covariance),
        kalman_gain=_readonly(gain),
        normalized_innovation_squared=nis,
        covariance_symmetry_error=symmetry_error,
        covariance_minimum_eigenvalue=float(
            np.min(np.linalg.eigvalsh(posterior_covariance))
        ),
    )


def propagate_constant_velocity_estimate(
    state: ConstantVelocityState,
    covariance_state: ArrayLike,
    new_reference_time_s: float,
) -> tuple[ConstantVelocityState, NDArray[np.float64], NDArray[np.float64]]:
    """Propagate state and covariance exactly under CV with ``Q=0``."""

    if not isinstance(state, ConstantVelocityState):
        raise TypeError("state must be ConstantVelocityState")
    covariance = _positive_definite(
        covariance_state, (6, 6), name="covariance_state"
    )
    new_time = float(new_reference_time_s)
    if not np.isfinite(new_time):
        raise ValueError("new_reference_time_s must be finite")
    if new_time < state.reference_time_s:
        raise ValueError("EKF state time cannot move backwards")
    transition = constant_velocity_transition_jacobian(
        new_time - state.reference_time_s
    )
    propagated = transition @ covariance @ transition.T
    propagated = 0.5 * (propagated + propagated.T)
    _positive_definite(propagated, (6, 6), name="propagated_covariance")
    return (
        rebase_constant_velocity_state(state, new_time),
        _readonly(propagated),
        _readonly(transition),
    )


@dataclass(frozen=True, slots=True)
class RetardedEKFUpdateResult:
    """Result of one attempted bearing update; truth is deliberately absent."""

    event_id: str
    update_applied: bool
    valid: bool
    failure_reason: str | None
    prior_state: ConstantVelocityState
    posterior_state: ConstantVelocityState
    prior_covariance: NDArray[np.float64]
    posterior_covariance: NDArray[np.float64]
    residual_tangent_rad: NDArray[np.float64]
    residual_jacobian_state: NDArray[np.float64]
    innovation_covariance: NDArray[np.float64]
    kalman_gain: NDArray[np.float64]
    normalized_innovation_squared: float
    covariance_rank: int
    covariance_condition_number: float
    covariance_symmetry_error: float
    covariance_minimum_eigenvalue: float
    runtime_s: float


def _rejected_update(
    *,
    started: float,
    measurement: BearingMeasurement,
    state: ConstantVelocityState,
    covariance: NDArray[np.float64],
    reason: str,
) -> RetardedEKFUpdateResult:
    return RetardedEKFUpdateResult(
        event_id=bearing_event_id(measurement),
        update_applied=False,
        valid=False,
        failure_reason=reason,
        prior_state=state,
        posterior_state=state,
        prior_covariance=_readonly(covariance),
        posterior_covariance=_readonly(covariance),
        residual_tangent_rad=_readonly(np.full(2, np.nan)),
        residual_jacobian_state=_readonly(np.full((2, 6), np.nan)),
        innovation_covariance=_readonly(np.full((2, 2), np.nan)),
        kalman_gain=_readonly(np.full((6, 2), np.nan)),
        normalized_innovation_squared=float("nan"),
        covariance_rank=6,
        covariance_condition_number=float(np.linalg.cond(covariance)),
        covariance_symmetry_error=float(
            np.max(np.abs(covariance - covariance.T), initial=0.0)
        ),
        covariance_minimum_eigenvalue=float(np.min(np.linalg.eigvalsh(covariance))),
        runtime_s=time.perf_counter() - started,
    )


def update_retarded_ekf(
    prior_state: ConstantVelocityState,
    prior_covariance: ArrayLike,
    station: StationPose,
    measurement: BearingMeasurement,
    *,
    sound_speed: float = DEFAULT_SOUND_SPEED,
) -> RetardedEKFUpdateResult:
    """Apply one retarded-time bearing update at the state's current epoch."""

    started = time.perf_counter()
    covariance = _positive_definite(
        prior_covariance, (6, 6), name="prior_covariance"
    )
    if not measurement.valid:
        return _rejected_update(
            started=started,
            measurement=measurement,
            state=prior_state,
            covariance=covariance,
            reason="invalid_measurement",
        )
    if station.station_id != measurement.station_id:
        return _rejected_update(
            started=started,
            measurement=measurement,
            state=prior_state,
            covariance=covariance,
            reason="station_measurement_id_mismatch",
        )
    observation = _symmetric(
        measurement.covariance_tangent_rad2,
        (2, 2),
        name="observation_covariance",
    )
    try:
        np.linalg.cholesky(observation)
    except np.linalg.LinAlgError:
        return _rejected_update(
            started=started,
            measurement=measurement,
            state=prior_state,
            covariance=covariance,
            reason="unsupported_singular_covariance",
        )
    try:
        residual = retarded_bearing_residual(
            prior_state, station, measurement, sound_speed
        )
        jacobian = retarded_bearing_residual_jacobian(
            prior_state, station, measurement, sound_speed
        )
        linear = joseph_residual_update(
            prior_state.vector,
            covariance,
            residual,
            jacobian,
            observation,
        )
        posterior = ConstantVelocityState(
            linear.posterior_vector[:3],
            linear.posterior_vector[3:],
            prior_state.reference_time_s,
        )
        speed = float(sound_speed)
        if not np.isfinite(speed) or speed <= 0.0:
            raise ValueError("sound_speed must be finite and positive")
        if float(np.linalg.norm(posterior.velocity_world_mps)) >= speed:
            return _rejected_update(
                started=started,
                measurement=measurement,
                state=prior_state,
                covariance=covariance,
                reason="non_subsonic_posterior",
            )
    except (ValueError, AntipodalDirectionError, np.linalg.LinAlgError) as error:
        return _rejected_update(
            started=started,
            measurement=measurement,
            state=prior_state,
            covariance=covariance,
            reason=f"measurement_update_error:{type(error).__name__}",
        )
    eigenvalues = np.linalg.eigvalsh(linear.posterior_covariance)
    return RetardedEKFUpdateResult(
        event_id=bearing_event_id(measurement),
        update_applied=True,
        valid=True,
        failure_reason=None,
        prior_state=prior_state,
        posterior_state=posterior,
        prior_covariance=_readonly(covariance),
        posterior_covariance=linear.posterior_covariance,
        residual_tangent_rad=_readonly(residual),
        residual_jacobian_state=_readonly(jacobian),
        innovation_covariance=linear.innovation_covariance,
        kalman_gain=linear.kalman_gain,
        normalized_innovation_squared=linear.normalized_innovation_squared,
        covariance_rank=int(np.linalg.matrix_rank(linear.posterior_covariance)),
        covariance_condition_number=float(np.max(eigenvalues) / np.min(eigenvalues)),
        covariance_symmetry_error=linear.covariance_symmetry_error,
        covariance_minimum_eigenvalue=linear.covariance_minimum_eigenvalue,
        runtime_s=time.perf_counter() - started,
    )


@dataclass(frozen=True, slots=True)
class RetardedEKFInitializationCriteria:
    """Fixed, pre-evaluation conditions for causal batch initialization."""

    minimum_measurement_count: int = 6
    required_local_rank: int = 6
    maximum_scaled_condition_number: float = 1e9
    maximum_angular_residual_rad: float = 0.1
    maximum_scaled_kkt_residual: float = 1e-6

    def __post_init__(self) -> None:
        if self.minimum_measurement_count < 3:
            raise ValueError("minimum_measurement_count must be at least three")
        if self.required_local_rank != 6:
            raise ValueError("C1 requires initialization rank 6")
        for value, name in (
            (self.maximum_scaled_condition_number, "maximum_scaled_condition_number"),
            (self.maximum_angular_residual_rad, "maximum_angular_residual_rad"),
            (self.maximum_scaled_kkt_residual, "maximum_scaled_kkt_residual"),
        ):
            if not np.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive")


@dataclass(frozen=True, slots=True)
class RetardedEKFPublication:
    """Immutable causal publication at one processor timestamp."""

    processing_time_s: float
    initialized: bool
    valid: bool
    failure_reason: str | None
    state: ConstantVelocityState | None
    covariance_state: NDArray[np.float64]
    prefix: BearingEventPrefix
    initialization_event_ids: tuple[str, ...]
    applied_event_ids: tuple[str, ...]
    rejected_event_ids: tuple[str, ...]
    update_diagnostics: tuple[RetardedEKFUpdateResult, ...]
    propagation_dt_s: float
    initialization_runtime_s: float
    measurement_update_runtime_s: float
    total_runtime_s: float
    initialization_batch: RetardedBatchResult | None


class CausalRetardedTimeEKF:
    """Causal event processor for the strict-CV retarded-time EKF baseline."""

    def __init__(
        self,
        stations: Sequence[StationPose],
        events: Sequence[ScheduledBearingEvent | BearingMeasurement],
        *,
        estimator_variant: str,
        sound_speed: float = DEFAULT_SOUND_SPEED,
        initialization_criteria: RetardedEKFInitializationCriteria | None = None,
    ) -> None:
        station_map = {station.station_id: station for station in stations}
        if len(station_map) != len(stations) or not station_map:
            raise ValueError("station ids must be nonempty and unique")
        self._stations = tuple(stations)
        self._station_map = station_map
        self._stream = CausalBearingEventStream(
            events, estimator_variant=estimator_variant
        )
        self._sound_speed = float(sound_speed)
        if not np.isfinite(self._sound_speed) or self._sound_speed <= 0.0:
            raise ValueError("sound_speed must be finite and positive")
        self._criteria = initialization_criteria or RetardedEKFInitializationCriteria()
        self._state: ConstantVelocityState | None = None
        self._covariance: NDArray[np.float64] | None = None
        self._processed_ids: set[str] = set()
        self._initialization_ids: set[str] = set()
        self._applied_ids: set[str] = set()
        self._rejected_ids: set[str] = set()
        self._recovery_pending = False
        self._last_processing_time = float("-inf")
        self._publications: list[RetardedEKFPublication] = []

    @property
    def publications(self) -> tuple[RetardedEKFPublication, ...]:
        return tuple(self._publications)

    def _publish(
        self,
        *,
        started: float,
        processing_time: float,
        prefix: BearingEventPrefix,
        valid: bool,
        failure_reason: str | None,
        updates: Sequence[RetardedEKFUpdateResult] = (),
        propagation_dt_s: float = 0.0,
        initialization_runtime_s: float = 0.0,
        initialization_batch: RetardedBatchResult | None = None,
    ) -> RetardedEKFPublication:
        publication = RetardedEKFPublication(
            processing_time_s=processing_time,
            initialized=self._state is not None,
            valid=valid,
            failure_reason=failure_reason,
            state=self._state,
            covariance_state=(
                _readonly(self._covariance)
                if self._covariance is not None
                else _readonly(np.full((6, 6), np.nan))
            ),
            prefix=prefix,
            initialization_event_ids=tuple(sorted(self._initialization_ids)),
            applied_event_ids=tuple(sorted(self._applied_ids)),
            rejected_event_ids=tuple(sorted(self._rejected_ids)),
            update_diagnostics=tuple(updates),
            propagation_dt_s=float(propagation_dt_s),
            initialization_runtime_s=float(initialization_runtime_s),
            measurement_update_runtime_s=float(sum(item.runtime_s for item in updates)),
            total_runtime_s=time.perf_counter() - started,
            initialization_batch=initialization_batch,
        )
        self._publications.append(publication)
        self._last_processing_time = processing_time
        return publication

    def advance_to(self, processing_time_s: float) -> RetardedEKFPublication:
        """Advance causally and publish a new immutable filter result."""

        started = time.perf_counter()
        processing_time = float(processing_time_s)
        if not np.isfinite(processing_time):
            raise ValueError("processing_time_s must be finite")
        if processing_time < self._last_processing_time:
            raise ValueError("processing time cannot move backwards")
        prefix = self._stream.advance_to(processing_time)
        active_ids = set(prefix.accepted_event_ids)
        invalidated_used = (
            self._initialization_ids | self._applied_ids
        ) - active_ids
        if self._state is not None and invalidated_used:
            self._state = None
            self._covariance = None
            self._processed_ids.clear()
            self._initialization_ids.clear()
            self._applied_ids.clear()
            self._rejected_ids.clear()
            self._recovery_pending = True
            return self._publish(
                started=started,
                processing_time=processing_time,
                prefix=prefix,
                valid=False,
                failure_reason="conflicted_used_event_requires_reinitialization",
            )

        if self._state is None:
            if self._recovery_pending:
                self._recovery_pending = False
            for measurement in prefix.measurements:
                try:
                    _positive_definite(
                        measurement.covariance_tangent_rad2,
                        (2, 2),
                        name="observation_covariance",
                    )
                except ValueError:
                    return self._publish(
                        started=started,
                        processing_time=processing_time,
                        prefix=prefix,
                        valid=False,
                        failure_reason="unsupported_singular_covariance",
                    )
            if len(prefix.measurements) < self._criteria.minimum_measurement_count:
                return self._publish(
                    started=started,
                    processing_time=processing_time,
                    prefix=prefix,
                    valid=False,
                    failure_reason="not_initialized",
                )
            initialization_started = time.perf_counter()
            batch = estimate_retarded_constant_velocity_batch(
                self._stations,
                prefix.measurements,
                reference_time_s=processing_time,
                sound_speed=self._sound_speed,
            )
            initialization_runtime = time.perf_counter() - initialization_started
            acceptable = (
                batch.valid
                and batch.local_observability_rank
                == self._criteria.required_local_rank
                and batch.scaled_information_condition_number
                <= self._criteria.maximum_scaled_condition_number
                and batch.maximum_angular_residual_rad
                <= self._criteria.maximum_angular_residual_rad
                and batch.scaled_projected_kkt_residual
                <= self._criteria.maximum_scaled_kkt_residual
            )
            if not acceptable or batch.state is None:
                reason = batch.failure_reason or "initialization_quality_gate_failed"
                return self._publish(
                    started=started,
                    processing_time=processing_time,
                    prefix=prefix,
                    valid=False,
                    failure_reason=f"initialization_failed:{reason}",
                    initialization_runtime_s=initialization_runtime,
                    initialization_batch=batch,
                )
            try:
                covariance = _positive_definite(
                    batch.covariance_state_linearization,
                    (6, 6),
                    name="initialization_covariance",
                )
            except ValueError:
                return self._publish(
                    started=started,
                    processing_time=processing_time,
                    prefix=prefix,
                    valid=False,
                    failure_reason="initialization_failed:non_positive_definite_covariance",
                    initialization_runtime_s=initialization_runtime,
                    initialization_batch=batch,
                )
            self._state = batch.state
            self._covariance = _readonly(covariance)
            self._initialization_ids = set(prefix.accepted_event_ids)
            self._processed_ids = set(prefix.accepted_event_ids)
            return self._publish(
                started=started,
                processing_time=processing_time,
                prefix=prefix,
                valid=True,
                failure_reason=None,
                initialization_runtime_s=initialization_runtime,
                initialization_batch=batch,
            )

        assert self._covariance is not None
        previous_time = self._state.reference_time_s
        self._state, self._covariance, _ = propagate_constant_velocity_estimate(
            self._state, self._covariance, processing_time
        )
        pending = sorted(
            (
                item
                for item in prefix.measurements
                if bearing_event_id(item) not in self._processed_ids
            ),
            key=lambda item: (
                item.available_timestamp_s,
                bearing_event_id(item),
                item.reception_center_timestamp_s,
            ),
        )
        diagnostics: list[RetardedEKFUpdateResult] = []
        for measurement in pending:
            identity = bearing_event_id(measurement)
            station = self._station_map[measurement.station_id]
            update = update_retarded_ekf(
                self._state,
                self._covariance,
                station,
                measurement,
                sound_speed=self._sound_speed,
            )
            diagnostics.append(update)
            self._processed_ids.add(identity)
            if update.update_applied:
                self._state = update.posterior_state
                self._covariance = update.posterior_covariance
                self._applied_ids.add(identity)
            else:
                self._rejected_ids.add(identity)
        rejected = [item.failure_reason for item in diagnostics if not item.valid]
        return self._publish(
            started=started,
            processing_time=processing_time,
            prefix=prefix,
            valid=not rejected,
            failure_reason=rejected[0] if rejected else None,
            updates=diagnostics,
            propagation_dt_s=processing_time - previous_time,
        )


__all__ = [
    "CausalRetardedTimeEKF",
    "LinearizedResidualUpdate",
    "RetardedEKFInitializationCriteria",
    "RetardedEKFPublication",
    "RetardedEKFUpdateResult",
    "joseph_residual_update",
    "propagate_constant_velocity_estimate",
    "update_retarded_ekf",
]
