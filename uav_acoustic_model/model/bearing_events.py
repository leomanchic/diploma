"""Deterministic causal replay of asynchronous bearing measurements.

The stream is deliberately truth-free.  Physical prediction uses the
``reception_center_timestamp_s`` stored in :class:`BearingMeasurement`, while
``available_timestamp_s`` only controls when the central processor may see an
event.  Prefixes are cumulative: a late event can affect later batch results,
but previously returned prefixes are immutable and are never rewritten.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Sequence

import numpy as np

from model.measurements import BearingMeasurement


EventAction = Literal[
    "accepted",
    "duplicate_exact",
    "excluded_conflict",
    "excluded_conflicted_id",
    "excluded_dropped",
    "excluded_invalid",
    "excluded_wrong_estimator_variant",
]


def bearing_event_id(measurement: BearingMeasurement) -> str:
    """Return the stable identity of one estimator output for one audio frame."""

    if not isinstance(measurement, BearingMeasurement):
        raise TypeError("measurement must be a BearingMeasurement")
    return "|".join(
        (
            measurement.sequence_id,
            measurement.station_id,
            str(measurement.frame_index),
            measurement.estimator_variant,
        )
    )


def audio_frame_id(measurement: BearingMeasurement) -> str:
    """Return frame identity without estimator variant.

    Different estimators of this same frame are not automatically independent.
    A stream therefore selects exactly one ``estimator_variant``.
    """

    if not isinstance(measurement, BearingMeasurement):
        raise TypeError("measurement must be a BearingMeasurement")
    return "|".join(
        (
            measurement.sequence_id,
            measurement.station_id,
            str(measurement.frame_index),
        )
    )


def _payload_signature(measurement: BearingMeasurement) -> tuple[object, ...]:
    quality = tuple(sorted(measurement.quality_metadata.items()))
    return (
        measurement.station_id,
        measurement.sequence_id,
        measurement.frame_index,
        measurement.reception_center_timestamp_s,
        measurement.available_timestamp_s,
        tuple(np.asarray(measurement.direction_local).tolist()),
        tuple(np.asarray(measurement.covariance_tangent_rad2).ravel().tolist()),
        tuple(np.asarray(measurement.calibration_bias_tangent_rad).tolist()),
        measurement.estimator_variant,
        quality,
        measurement.valid,
        measurement.invalid_reason,
    )


def measurements_are_exact_duplicates(
    first: BearingMeasurement, second: BearingMeasurement
) -> bool:
    """Compare complete immutable payloads, treating matching NaNs as equal."""

    if not isinstance(first, BearingMeasurement) or not isinstance(
        second, BearingMeasurement
    ):
        raise TypeError("both values must be BearingMeasurement instances")
    scalar_equal = (
        first.station_id == second.station_id
        and first.sequence_id == second.sequence_id
        and first.frame_index == second.frame_index
        and first.reception_center_timestamp_s
        == second.reception_center_timestamp_s
        and first.available_timestamp_s == second.available_timestamp_s
        and first.estimator_variant == second.estimator_variant
        and dict(first.quality_metadata) == dict(second.quality_metadata)
        and first.valid == second.valid
        and first.invalid_reason == second.invalid_reason
    )
    return bool(
        scalar_equal
        and np.array_equal(
            first.direction_local, second.direction_local, equal_nan=True
        )
        and np.array_equal(
            first.covariance_tangent_rad2,
            second.covariance_tangent_rad2,
            equal_nan=True,
        )
        and np.array_equal(
            first.calibration_bias_tangent_rad,
            second.calibration_bias_tangent_rad,
            equal_nan=True,
        )
    )


@dataclass(frozen=True, slots=True)
class ScheduledBearingEvent:
    """One transport outcome for a truth-free bearing measurement."""

    measurement: BearingMeasurement
    dropped: bool = False
    drop_reason: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.measurement, BearingMeasurement):
            raise TypeError("measurement must be a BearingMeasurement")
        dropped = bool(self.dropped)
        reason = None if self.drop_reason is None else str(self.drop_reason)
        if dropped and not reason:
            raise ValueError("a dropped event requires drop_reason")
        if not dropped and reason:
            raise ValueError("a delivered event cannot have drop_reason")
        object.__setattr__(self, "dropped", dropped)
        object.__setattr__(self, "drop_reason", reason)


@dataclass(frozen=True, slots=True)
class BearingEventLogEntry:
    """Auditable disposition of one event at the central processor."""

    processing_time_s: float
    event_id: str
    audio_frame_id: str
    station_id: str
    frame_index: int
    reception_timestamp_s: float
    available_timestamp_s: float
    action: EventAction
    reason: str


@dataclass(frozen=True, slots=True)
class BearingEventPrefix:
    """Immutable causal snapshot returned after advancing processing time."""

    processing_time_s: float
    measurements: tuple[BearingMeasurement, ...]
    journal: tuple[BearingEventLogEntry, ...]
    accepted_event_ids: tuple[str, ...]
    conflicted_event_ids: tuple[str, ...]


class CausalBearingEventStream:
    """Replay a fixed schedule into monotonically increasing causal prefixes.

    Equal-availability events use a canonical identity/payload ordering.  An
    exact duplicate is logged but never reweighted.  If two distinct payloads
    share an event identity, the identity is quarantined and neither payload is
    exposed to subsequent estimates.
    """

    def __init__(
        self,
        events: Sequence[ScheduledBearingEvent | BearingMeasurement],
        *,
        estimator_variant: str,
    ) -> None:
        variant = str(estimator_variant)
        if not variant:
            raise ValueError("estimator_variant is required")
        normalized: list[ScheduledBearingEvent] = []
        for value in events:
            if isinstance(value, BearingMeasurement):
                normalized.append(ScheduledBearingEvent(value))
            elif isinstance(value, ScheduledBearingEvent):
                normalized.append(value)
            else:
                raise TypeError(
                    "events must contain ScheduledBearingEvent or BearingMeasurement"
                )
        sequence_ids = {item.measurement.sequence_id for item in normalized}
        if len(sequence_ids) > 1:
            raise ValueError("one event stream must contain one sequence_id")
        self._events = tuple(
            sorted(
                normalized,
                key=lambda item: (
                    item.measurement.available_timestamp_s,
                    bearing_event_id(item.measurement),
                    repr(_payload_signature(item.measurement)),
                    item.dropped,
                ),
            )
        )
        self._estimator_variant = variant
        self._cursor = 0
        self._processing_time_s = float("-inf")
        self._active: dict[str, BearingMeasurement] = {}
        self._conflicted: set[str] = set()
        self._journal: list[BearingEventLogEntry] = []

    @property
    def estimator_variant(self) -> str:
        return self._estimator_variant

    def _log(
        self,
        measurement: BearingMeasurement,
        processing_time_s: float,
        action: EventAction,
        reason: str,
    ) -> None:
        self._journal.append(
            BearingEventLogEntry(
                processing_time_s=float(processing_time_s),
                event_id=bearing_event_id(measurement),
                audio_frame_id=audio_frame_id(measurement),
                station_id=measurement.station_id,
                frame_index=measurement.frame_index,
                reception_timestamp_s=measurement.reception_center_timestamp_s,
                available_timestamp_s=measurement.available_timestamp_s,
                action=action,
                reason=str(reason),
            )
        )

    def _process(
        self, event: ScheduledBearingEvent, processing_time_s: float
    ) -> None:
        measurement = event.measurement
        identity = bearing_event_id(measurement)
        if event.dropped:
            self._log(
                measurement,
                processing_time_s,
                "excluded_dropped",
                event.drop_reason or "transport_drop",
            )
            return
        if measurement.estimator_variant != self._estimator_variant:
            self._log(
                measurement,
                processing_time_s,
                "excluded_wrong_estimator_variant",
                "one estimator variant is selected per experiment",
            )
            return
        if not measurement.valid:
            self._log(
                measurement,
                processing_time_s,
                "excluded_invalid",
                measurement.invalid_reason or "invalid_measurement",
            )
            return
        if identity in self._conflicted:
            self._log(
                measurement,
                processing_time_s,
                "excluded_conflicted_id",
                "event identity was previously quarantined",
            )
            return
        existing = self._active.get(identity)
        if existing is None:
            self._active[identity] = measurement
            self._log(measurement, processing_time_s, "accepted", "accepted")
            return
        if measurements_are_exact_duplicates(existing, measurement):
            self._log(
                measurement,
                processing_time_s,
                "duplicate_exact",
                "exact duplicate does not increase weight",
            )
            return
        del self._active[identity]
        self._conflicted.add(identity)
        self._log(
            measurement,
            processing_time_s,
            "excluded_conflict",
            "conflicting payload quarantined together with prior payload",
        )

    def advance_to(self, processing_time_s: float) -> BearingEventPrefix:
        """Advance monotonically and expose only events available by ``T``."""

        processing_time = float(processing_time_s)
        if not np.isfinite(processing_time):
            raise ValueError("processing_time_s must be finite")
        if processing_time < self._processing_time_s:
            raise ValueError("processing time cannot move backwards")
        while self._cursor < len(self._events):
            event = self._events[self._cursor]
            if event.measurement.available_timestamp_s > processing_time:
                break
            self._process(event, processing_time)
            self._cursor += 1
        self._processing_time_s = processing_time
        measurements = tuple(
            sorted(
                self._active.values(),
                key=lambda item: (
                    item.reception_center_timestamp_s,
                    item.station_id,
                    item.frame_index,
                    item.estimator_variant,
                ),
            )
        )
        return BearingEventPrefix(
            processing_time_s=processing_time,
            measurements=measurements,
            journal=tuple(self._journal),
            accepted_event_ids=tuple(bearing_event_id(item) for item in measurements),
            conflicted_event_ids=tuple(sorted(self._conflicted)),
        )


__all__ = [
    "BearingEventLogEntry",
    "BearingEventPrefix",
    "CausalBearingEventStream",
    "EventAction",
    "ScheduledBearingEvent",
    "audio_frame_id",
    "bearing_event_id",
    "measurements_are_exact_duplicates",
]
